from __future__ import annotations

from src.config import ModelConfig, require_env
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMProvider
from src.providers.deepseek_provider import DeepSeekProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.xai_provider import XAIProvider

_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}


def create_provider(model: ModelConfig) -> LLMProvider:
    env_var = _ENV_BY_PROVIDER.get(model.provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {model.provider}")

    require_env(env_var)

    if model.provider == "openai":
        return OpenAIProvider(model.model_id)
    if model.provider == "anthropic":
        return AnthropicProvider(model.model_id)
    if model.provider == "google":
        return GeminiProvider(model.model_id)
    if model.provider == "deepseek":
        return DeepSeekProvider(model.model_id)
    if model.provider == "xai":
        return XAIProvider(model.model_id)

    raise ValueError(f"Unsupported provider: {model.provider}")
