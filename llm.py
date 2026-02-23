import time
import os
import re
import yaml
from basereal import BaseReal
from logger import logger


def load_config(config_path: str = "llm_config.yaml"):
    """加载配置"""
    def resolve_env(obj):
        if isinstance(obj, dict):
            return {k: resolve_env(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.getenv(obj[2:-1], "")
        return obj
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return resolve_env(config)


def llm_response(message, nerfreal: BaseReal, config_path: str = "llm_config.yaml"):
    """根据配置文件执行LLM回复"""
    start = time.perf_counter()
    
    # 读取配置
    config = load_config(config_path)
    provider = config['llm']['provider']
    pcfg = config['llm'][provider]
    
    from openai import OpenAI
    client = OpenAI(
        api_key=pcfg.get('api_key', 'ollama'),
        base_url=pcfg['base_url']
    )
    
    # 系统提示词
    system_msg = "你是真人主播，说话自然口语化，像朋友聊天。禁止用星号、markdown、序号。多用语气词呢、呀、啦。简短2-3句话。"
    
    # 创建请求
    completion = client.chat.completions.create(
        model=pcfg['model'],
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': message}
        ],
        stream=True
    )
    
    # 处理流式输出
    result = ""
    first = True
    
    for chunk in completion:
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        
        msg = delta.content
        
        if first:
            logger.info(f"llm Time to first chunk: {time.perf_counter()-start:.3f}s")
            first = False
        
        # 简单清洗
        msg = msg.replace('*', '').replace('#', '').replace('`', '')
        
        # 累积并查找断句点
        result += msg
        
        for i, char in enumerate(result):
            if char in "，。！？,." and i >= 5:
                send_part = result[:i+1]
                result = result[i+1:]
                logger.info(f"llm: {send_part}")
                nerfreal.put_msg_txt(send_part)
                break
        
        if len(result) > 30:
            logger.info(f"llm: {result}")
            nerfreal.put_msg_txt(result)
            result = ""
    
    # 发送剩余
    if result:
        result = result.replace('*', '').replace('#', '').replace('`', '')
        if result.strip():
            logger.info(f"llm: {result}")
            nerfreal.put_msg_txt(result)
    
    logger.info(f"llm Time to last chunk: {time.perf_counter()-start:.3f}s")