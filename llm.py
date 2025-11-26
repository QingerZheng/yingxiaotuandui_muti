import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, Field
from langchain_core.utils.utils import secret_from_env

class ChatOpenRouter(ChatOpenAI):
    """
    一个专门用于连接 OpenRouter 的 LangChain ChatOpenAI 子类。
    它会自动处理 API 密钥和基础 URL 的配置。
    """
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        # 优先使用传入的key，否则从环境变量获取
        api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required. Please set the OPENROUTER_API_KEY environment variable.")
        
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            **kwargs
        )


class ChatAiHubMix(ChatOpenAI):
    """专门用于连接 AiHubMix 的 LangChain ChatOpenAI 子类"""
    
    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        api_key = openai_api_key or os.environ.get("AIHUBMIX_API_KEY")
        if not api_key:
            raise ValueError("AiHubMix API key is required. Please set the AIHUBMIX_API_KEY environment variable.")
        
        super().__init__(
            base_url="https://aihubmix.com/v1",
            openai_api_key=api_key,
            **kwargs
        )

def create_llm(model_provider: str, model_name: str, **kwargs: Any) -> ChatOpenAI:
    """
    LLM 工厂函数，根据提供商创建并返回一个语言模型实例。

    Args:
        model_provider (str): 模型提供商， 'openai' 或 'openrouter'.
        model_name (str): 模型的具体名称, e.g., 'gpt-4o' or 'x-ai/grok-4'.
        **kwargs: 传递给模型构造函数的其他参数 (例如, temperature).

    Returns:
        ChatOpenAI: 一个配置好的 ChatOpenAI 或其子类的实例。

    Raises:
        ValueError: 如果提供了不支持的 model_provider.
    """
    # 设置稳健的默认网络参数，避免在云端长时间挂起
    # timeout: 单请求超时（秒）；max_retries: 失败重试次数（设为0避免阻塞）
    if "timeout" not in kwargs:
        kwargs["timeout"] = 30
    if "max_retries" not in kwargs:
        kwargs["max_retries"] = 0

    # 调试：打印传入的provider与model
    try:
        print(f"[LLM] request -> provider: {model_provider}, model: {model_name}")
    except Exception:
        pass

    # 针对特定模型做安全路由修正
    # deepseek 模型应通过 OpenRouter 访问，避免误走 AiHubMix（会导致 400 或空响应）
    preferred_provider = model_provider
    try:
        if isinstance(model_name, str) and model_name.startswith("deepseek/") and model_provider != "openrouter":
            print("[LLM] routing fix: deepseek/* 强制使用 OpenRouter 网关 (忽略 aihubmix)")
            preferred_provider = "openrouter"
    except Exception:
        preferred_provider = model_provider

    # 统一路由到 OpenRouter：即使传入 "openai"，也走 OpenRouter 网关
    if preferred_provider in ("openrouter", "openai"):
        # 规范化模型名：对于 OpenAI 家族模型在 OpenRouter 需加 "openai/" 前缀
        normalized_model_name = model_name
        try:
            if isinstance(normalized_model_name, str) and normalized_model_name.startswith("gpt-") and "/" not in normalized_model_name:
                normalized_model_name = f"openai/{normalized_model_name}"
        except Exception:
            normalized_model_name = model_name
        referer = os.environ.get("HTTP_REFERER", "")
        title = os.environ.get("X_TITLE", "")
        
        try:
            print(f"[LLM] routing -> OpenRouter | model: {normalized_model_name} | base_url: https://openrouter.ai/api/v1")
        except Exception:
            pass

        return ChatOpenRouter(
            model=normalized_model_name,
            default_headers={
                "HTTP-Referer": referer,
                "X-Title": title,
            },
            **kwargs
        )
    elif preferred_provider == "aihubmix":
        try:
            print(f"[LLM] routing -> AiHubMix | model: {model_name} | base_url: https://aihubmix.com/v1")
        except Exception:
            pass
        return ChatAiHubMix(model=model_name, **kwargs)
    else:
        raise ValueError(f"不支持的模型提供商: '{model_provider}'. 请选择 'openai'、'openrouter' 或 'aihubmix'。")