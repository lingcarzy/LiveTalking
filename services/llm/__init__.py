# services/llm/__init__.py
from .openai_llm import OpenAILLM
def create_llm_service(config):
    # 目前仅支持 OpenAI 兼容接口，后续可扩展
    return OpenAILLM(config)