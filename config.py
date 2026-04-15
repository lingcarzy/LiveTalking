###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def load_env_file(env_file: str = ".env"):
    """Load KEY=VALUE pairs from .env without overriding existing environment variables."""
    env_path = os.path.join(os.path.dirname(__file__), env_file)
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            # Support inline comments like: FOO=bar  # comment
            if value and value[0] in ("'", '"') and value[-1:] == value[0]:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()

            os.environ.setdefault(key, value)


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数"""
    load_env_file(".env")
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default=os.getenv('TTS', 'edgetts'),
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default=os.getenv('REF_FILE', "zh-CN-YunxiaNeural"),
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=os.getenv('REF_TEXT', None))
    parser.add_argument('--TTS_SERVER', type=str, default=os.getenv('TTS_SERVER', 'http://127.0.0.1:9880'))

    # ─── LLM ───────────────────────────────────────────────────────────
    parser.add_argument('--llm_provider', type=str,
                        default=os.getenv('LLM_PROVIDER', os.getenv('LIVETALKING_LLM_PROVIDER', 'grok')),
                        help="llm provider: dashscope/grok/deepseek/ollama/openai")
    parser.add_argument('--llm_model', type=str, default=os.getenv('LLM_MODEL', ''),
                        help="llm model name, empty uses provider default")
    parser.add_argument('--llm_api_key', type=str,
                        default=os.getenv('LLM_API_KEY', os.getenv('LIVETALKING_LLM_API_KEY', '')),
                        help="llm api key, empty uses provider env var")
    parser.add_argument('--llm_base_url', type=str,
                        default=os.getenv('LLM_BASE_URL', os.getenv('LIVETALKING_LLM_BASE_URL', '')),
                        help="llm base url, empty uses provider default")
    parser.add_argument('--llm_proxy', type=str,
                        default=os.getenv('LLM_PROXY', os.getenv('GROK_PROXY', '')),
                        help="optional proxy url for llm requests")
    parser.add_argument('--llm_system_prompt', type=str,
                        default=os.getenv('LLM_SYSTEM_PROMPT', '你是一个知识助手，尽量以简短、口语化的方式输出'),
                        help="llm system prompt")
    parser.add_argument('--llm_history_turns', type=int, default=int(os.getenv('LLM_HISTORY_TURNS', '6')),
                        help="number of recent conversation turns to keep")
    parser.add_argument('--llm_request_timeout', type=float, default=float(os.getenv('LLM_REQUEST_TIMEOUT', '60')),
                        help="llm request timeout seconds")

    # ─── 安全/性能限制 ─────────────────────────────────────────────────
    parser.add_argument('--max_chat_chars', type=int, default=int(os.getenv('MAX_CHAT_CHARS', '4000')),
                        help="max chars for /human text")
    parser.add_argument('--max_audio_upload_mb', type=int, default=int(os.getenv('MAX_AUDIO_UPLOAD_MB', '20')),
                        help="max upload size for /humanaudio in MB")
    parser.add_argument('--max_custom_config_chars', type=int, default=int(os.getenv('MAX_CUSTOM_CONFIG_CHARS', '50000')),
                        help="max chars for custom_config json payload")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")
    parser.add_argument('--ice_server_urls', type=str, default=os.getenv('ICE_SERVER_URLS', ''),
                        help="comma-separated ICE urls, empty disables STUN/TURN")
    parser.add_argument('--skip_model_warmup', type=str_to_bool,
                        default=False,
                        help="deprecated, ignored: model warmup is always enabled")

    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    if opt.max_chat_chars < 1:
        opt.max_chat_chars = 1
    if opt.max_audio_upload_mb < 1:
        opt.max_audio_upload_mb = 1
    if opt.max_custom_config_chars < 100:
        opt.max_custom_config_chars = 100

    # keep backward compatibility for existing scripts/env, but always warm up models
    opt.skip_model_warmup = False

    return opt
