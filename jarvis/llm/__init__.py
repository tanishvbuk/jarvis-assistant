"""One way into any model: make_backend(). The provider is a config setting."""

from __future__ import annotations

from ..config import ConfigError, ProviderConfig
from .base import AuthFailed, Backend, LLMError, RateLimited, ToolCall, ToolSpec, Unavailable, Usage

__all__ = ["make_backend", "Backend", "Usage", "ToolSpec", "ToolCall", "LLMError", "RateLimited",
           "Unavailable", "AuthFailed"]


def make_backend(p: ProviderConfig) -> Backend:
    if p.kind == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(name=p.name, model=p.model, api_key=p.api_key(), base_url=p.base_url,
                                  extra=p.extra)
    if p.kind == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=p.model, api_key=p.api_key(), effort=p.effort, thinking=p.thinking)
    raise ConfigError(f"Unknown provider kind {p.kind!r} for {p.name}")

