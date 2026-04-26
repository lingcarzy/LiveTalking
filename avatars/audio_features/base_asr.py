###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import time
import numpy as np

import queue
from queue import Queue
from numpy.typing import NDArray
import torch.multiprocessing as mp

from avatars.base_avatar import BaseAvatar,AudioFrameData


class BaseASR:
    def __init__(self, opt, parent:BaseAvatar = None):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms * 16000 / 1000)
        self.queue:Queue[AudioFrameData] = Queue(maxsize=max(8, opt.batch_size * 8))
        self.output_queue:Queue[AudioFrameData] = Queue(maxsize=max(16, opt.batch_size * 12))
        self.play_queue:Queue[AudioFrameData] = Queue(maxsize=max(100, opt.batch_size * 24))

        self.batch_size = opt.batch_size
        self._streaming_audio_active = False
        # Keep active-stream timeout close to one audio chunk to avoid starving playback.
        self._streaming_timeout_sec = max(0.02, min(0.05, self.chunk / float(self.sample_rate)))
        # Idle timeout should also follow chunk cadence; 50ms idle waits cap render cadence too aggressively.
        self._idle_timeout_sec = max(0.01, min(0.02, self.chunk / float(self.sample_rate)))

        self.frames: list[NDArray[np.float32]] = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        #self.context_size = 10
        self.feat_queue = Queue(maxsize=max(2, int(getattr(opt, 'asr_feat_queue_size', max(4, opt.batch_size)))))

        #self.warm_up()

    def flush_talk(self):
        self._streaming_audio_active = False
        self._drain_queue(self.queue)
        self._drain_queue(self.output_queue)
        self._drain_queue(self.play_queue)
        self._drain_queue(self.feat_queue)

    @staticmethod
    def _drain_queue(q: Queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _put_with_drop_oldest(q: Queue, item):
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    return

    @staticmethod
    def _is_silence_like(frame: AudioFrameData) -> bool:
        if frame.type != 0:
            return True
        return bool(frame.userdata.get('_skip_playback'))

    def _put_preserve_speech(self, q: Queue, frame: AudioFrameData, *, timeout: float = 0.2) -> None:
        """Prefer preserving speech order; only drop oldest when the incoming frame is silence-like."""
        if self._is_silence_like(frame):
            self._put_with_drop_oldest(q, frame)
            return

        deadline = time.perf_counter() + max(0.01, timeout)
        while True:
            try:
                q.put(frame, block=True, timeout=0.02)
                return
            except queue.Full:
                if time.perf_counter() >= deadline:
                    # Backpressure still too high, degrade gracefully to keep system alive.
                    self._put_with_drop_oldest(q, frame)
                    return

    def _make_silence_frame(self, *, skip_playback: bool = False) -> AudioFrameData:
        userdata = {'_skip_playback': True} if skip_playback else {}
        return AudioFrameData(data=np.zeros(self.chunk, dtype=np.float32), type=1, userdata=userdata)

    def put_audio_frame(self,audio_chunk:NDArray[np.float32],datainfo:dict): #16khz 20ms pcm
        status = datainfo.get('status')
        if status == 'start':
            self._streaming_audio_active = True

        frame_type = 1 if np.count_nonzero(audio_chunk) == 0 else 0
        self._put_preserve_speech(self.queue, AudioFrameData(data=audio_chunk, type=frame_type, userdata=datainfo))

        if status == 'end':
            self._streaming_audio_active = False
            # Pre-fill tail silence for lip-sync context instead of waiting on repeated timeout-generated silence.
            padding_chunks = max(self.stride_right_size, self.batch_size * 2 - 1)
            for _ in range(padding_chunks):
                self._put_with_drop_oldest(self.queue, self._make_silence_frame(skip_playback=True))

    #return frame:audio pcm; type: 0-normal speak, 1-silence; eventpoint:custom event sync with audio
    def get_audio_frame(self)->AudioFrameData:        
        try:
            if self.parent and self.parent.custom_audiotype>1: #播放自定义音频,优先播放完自定义动作,可以通过interrupt打断动作播放
                frame = self.parent.get_custom_audio_stream(self.parent.custom_audiotype)
                type = self.parent.custom_audiotype
                return AudioFrameData(data=frame, type=type, userdata={})
            else:
                timeout = self._streaming_timeout_sec if self._streaming_audio_active else self._idle_timeout_sec
                frame = self.queue.get(block=True,timeout=timeout)
                return frame
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            return self._make_silence_frame()


    #return frame:audio pcm; type: 0-normal speak, 1-silence; eventpoint:custom event sync with audio
    def get_audio_out(self)->AudioFrameData: 
        return self.output_queue.get()

    def get_play_audio_out(self, block=True, timeout=None)->AudioFrameData:
        return self.play_queue.get(block, timeout)

    def publish_audio_frame(self, audio_frame: AudioFrameData) -> None:
        self._put_preserve_speech(self.output_queue, audio_frame)
        self._put_preserve_speech(self.play_queue, audio_frame)

    def report_feature_stats(self, *, step_sec: float, feat_batches: int = 0, feat_chunks: int = 0) -> None:
        if not self.parent or not hasattr(self.parent, '_perf'):
            return
        perf = self.parent._perf
        perf['asr_steps'] += 1
        perf['asr_step_sec'] += step_sec
        perf['asr_feat_batches'] += feat_batches
        perf['asr_feat_chunks'] += feat_chunks
    
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame=self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.publish_audio_frame(audio_frame)
        for _ in range(self.stride_left_size):
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def run_step(self):
        pass

    def get_next_feat(self,block,timeout):        
        return self.feat_queue.get(block,timeout)

    #分割音频特征，子类调用
    def _get_sliced_feature(self, feature_array, 
                        vid_idx,  
                        audio_feat_win,  
                        feature_idx_multiplier=1.0):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param vid_idx: 视频帧在一个batch内编号
        :param audio_feat_win: 音频特征窗口大小，通常为 [左侧窗口大小, 右侧窗口大小]，单位为视频帧数
        :param feature_idx_multiplier: 用于将视频帧索引转换为特征索引的乘数，通常为 (特征提取的宽度 / 视频帧率)
        :return: 
        """
        length = feature_array.shape[0] #len(feature_array)
        selected_feature = []
        selected_idx = []
        
        center_idx = int(vid_idx * feature_idx_multiplier) 
        left = int(center_idx - audio_feat_win[0]*feature_idx_multiplier)
        right = int(center_idx + audio_feat_win[1]*feature_idx_multiplier)
        # pad_left = 0
        # pad_right = 0
        # if left < 0:
        #     pad_left = -left
        #     left = 0
        # if right > feature_array.shape[0]:
        #     pad_right = right - feature_array.shape[0]
        #     right = feature_array.shape[0]
        # auds = feature_array[left:right]
        # if pad_left > 0:
        #     auds = np.concatenate([feature_array[left]*pad_left, auds], axis=0)
        # if pad_right > 0:
        #     auds = np.concatenate([auds, feature_array[right-1]*pad_right], axis=0) # [8, 16]
        
        for idx in range(left,right):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        # selected_feature = np.concatenate(selected_feature, axis=0)
        # selected_feature = selected_feature.reshape(-1, 256)# 20*256
        return np.asarray(selected_feature),selected_idx

    #参数定义 
    def _feature2chunks(self,feature_array,batch_size,audio_feat_win=[8,8],start=0,feature_idx_multiplier=1.0):
        """
        :param feature_array: 
        :param batch_size: batch大小
        :param audio_feat_win: 音频特征窗口大小，通常为 [左侧窗口大小, 右侧窗口大小]，单位为视频帧数
        :param start: 起始帧索引，通常为 stride_left_size/2
        :param feature_idx_multiplier: 用于将视频帧索引转换为特征索引的乘数，通常为 (特征提取的宽度 / 视频帧率)
        :return: 
        """
        feature_chunks = []
        #start += 10
        #feature_idx_multiplier = 50./fps 
        for i in range(batch_size):
            # start_idx = int(i * whisper_idx_multiplier)
            # if start_idx>=len(feature_array):
            #     break
            selected_feature,selected_idx = self._get_sliced_feature(
                feature_array=feature_array, vid_idx=i+start,
                audio_feat_win=audio_feat_win, feature_idx_multiplier=feature_idx_multiplier)
            #print(f"i:{i},selected_idx {selected_idx},feature_idx_multiplier:{feature_idx_multiplier}")
            feature_chunks.append(selected_feature)
        return feature_chunks
