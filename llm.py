import os
import time
from typing import TYPE_CHECKING

from utils.logger import logger

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


def _provider_defaults(provider: str) -> tuple[str, str, str]:
    provider = (provider or "dashscope").lower()
    if provider == "grok":
        return (
            os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", ""),
            "https://api.x.ai/v1",
            "grok-4.20-fast-non-reasoning",
        )
    if provider == "deepseek":
        return (
            os.getenv("DEEPSEEK_API_KEY", ""),
            "https://api.deepseek.com/v1",
            "deepseek-chat",
        )
    if provider == "ollama":
        return (
            os.getenv("OLLAMA_API_KEY", "ollama"),
            "http://127.0.0.1:11434/v1",
            "llama3.1:8b",
        )
    if provider == "openai":
        return (
            os.getenv("OPENAI_API_KEY", ""),
            "https://api.openai.com/v1",
            "gpt-4.1-mini",
        )
    return (
        os.getenv("DASHSCOPE_API_KEY", ""),
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
    )


def _build_client(opt):
    from openai import OpenAI

    provider = getattr(opt, "llm_provider", "dashscope").lower()
    default_key, default_base_url, default_model = _provider_defaults(provider)

    api_key = getattr(opt, "llm_api_key", "") or os.getenv("LIVETALKING_LLM_API_KEY", "") or default_key
    base_url = getattr(opt, "llm_base_url", "") or default_base_url
    model = getattr(opt, "llm_model", "") or default_model
    timeout = float(getattr(opt, "llm_request_timeout", 60))
    proxy = getattr(opt, "llm_proxy", "") or os.getenv("LLM_PROXY", "")
    if provider == "grok":
        proxy = proxy or os.getenv("GROK_PROXY", "")

    if not api_key:
        raise ValueError(f"LLM API key not configured for provider: {provider}")

    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
    }
    if proxy:
        try:
            import httpx

            try:
                http_client = httpx.Client(proxy=proxy, timeout=timeout)
            except TypeError:
                http_client = httpx.Client(proxies={"http://": proxy, "https://": proxy}, timeout=timeout)
            client_kwargs["http_client"] = http_client
        except Exception:
            logger.warning("llm proxy configured but httpx client init failed, fallback without explicit proxy")

    client = OpenAI(**client_kwargs)
    return provider, model, client


def _get_history(avatar_session: "BaseAvatar") -> list[dict]:
    history = getattr(avatar_session, "_llm_history", None)
    if history is None:
        history = []
        setattr(avatar_session, "_llm_history", history)
    return history


def _trim_history(history: list[dict], max_turns: int) -> None:
    keep_messages = max(0, int(max_turns)) * 2
    if keep_messages <= 0:
        history.clear()
        return
    if len(history) > keep_messages:
        del history[:-keep_messages]


def _pop_sentence(buffer: str):
    for i, ch in enumerate(buffer):
        if ch in ",.!;:，。！？：；\n" and i >= 5:
            return buffer[:i + 1].strip(), buffer[i + 1:]
    return None, buffer


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = None):
    datainfo = datainfo or {}
    try:
        text = str(message or "").strip()
        if not text:
            return

        opt = avatar_session.opt
        start = time.perf_counter()
        provider, model, client = _build_client(opt)

        history = _get_history(avatar_session)
        _trim_history(history, getattr(opt, "llm_history_turns", 6))

        system_prompt = getattr(
            opt,
            "llm_system_prompt",
            "你是一个知识助手，尽量以简短、口语化的方式输出",
        )
        messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": text}]

        logger.info(
            "llm init done provider=%s model=%s elapsed=%.3fs",
            provider,
            model,
            time.perf_counter() - start,
        )

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

        response_parts = []
        pending = ""
        first = True

        for chunk in completion:
            if not chunk.choices:
                continue
            msg = chunk.choices[0].delta.content
            if msg is None:
                continue

            if first:
                first = False
                logger.info("llm first chunk provider=%s elapsed=%.3fs", provider, time.perf_counter() - start)

            response_parts.append(msg)
            pending += msg

            while True:
                sentence, pending = _pop_sentence(pending)
                if sentence is None:
                    break
                avatar_session.put_msg_txt(sentence, datainfo)

        full_response = "".join(response_parts).strip()
        if pending.strip():
            avatar_session.put_msg_txt(pending.strip(), datainfo)

        if full_response:
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": full_response})
            _trim_history(history, getattr(opt, "llm_history_turns", 6))

        logger.info("llm finished provider=%s elapsed=%.3fs", provider, time.perf_counter() - start)

    except Exception as e:
        logger.exception("llm exception:")
        avatar_session.put_msg_txt(f"LLM调用失败: {e}", datainfo)
