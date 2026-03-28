# services/llm/openai_llm.py
import time
import os
import random
import asyncio
from logger import logger
from openai import OpenAI, APIConnectionError, RateLimitError

class OpenAILLM:
    def __init__(self, config):
        self.config = config
        api_key = self.config.llm.api_key
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "ollama")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.llm.base_url
        )
    async def chat_stream(self, message: str):
        """
        改造为异步生成器，直接 yield 文本片段
        """
        try:
            stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.llm.model,
                messages=[
                    {'role': 'system', 'content': self.config.llm.system_msg},
                    {'role': 'user', 'content': message}
                ],
                stream=True
            )
            
            result = ""
            async for chunk in stream:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if not delta or not delta.content: continue
                
                msg = delta.content
                result += msg
                
                # 简单的分句逻辑
                for i, char in enumerate(result):
                    if char in "，。！？,." and i >= 5:
                        yield result[:i+1]
                        result = result[i+1:]
                        break
            
            if result:
                yield result

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            yield "抱歉，服务出现异常。"
    def chat(self, message, callback):
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                return self._execute_chat(message, callback)
            except (APIConnectionError, RateLimitError) as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 指数退避
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                else:
                    logger.error("LLM API call failed after multiple retries.")
                    callback("抱歉，连接模型服务失败，请稍后重试。")
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}")
                callback("抱歉，发生了未知错误。")
                break

    def _execute_chat(self, message, callback):
        start = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.config.llm.model,
            messages=[
                {'role': 'system', 'content': self.config.llm.system_msg},
                {'role': 'user', 'content': message}
            ],
            stream=True
        )
        
        result = ""
        first = True
        
        for chunk in completion:
            # ... (原有流式处理逻辑保持不变) ...
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            if not delta or not delta.content: continue
            
            msg = delta.content
            if first:
                logger.info(f"LLM Time to first chunk: {time.perf_counter()-start:.3f}s")
                first = False
            
            msg = msg.replace('*', '').replace('#', '').replace('`', '')
            result += msg
            
            # 分句逻辑
            for i, char in enumerate(result):
                if char in "，。！？,." and i >= 5:
                    send_part = result[:i+1]
                    result = result[i+1:]
                    callback(send_part)
                    break
            
            if len(result) > 30:
                callback(result)
                result = ""
        
        if result:
            callback(result)