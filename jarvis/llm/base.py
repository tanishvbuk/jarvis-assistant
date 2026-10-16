"""Shared types for LLM backends.

Every backend turns its SDK's errors into the three exceptions below, so the
rest of Jarvis never imports a provider SDK.

Messages are plain dicts in one neutral shape, which each backend converts:
    {"role": "user" | "assistant", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolSpec:
    """What the model is told about a tool."""
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})  # JSON schema


@dataclass
class ToolCall:
    """The model asking for a tool to be run."""
    id: str
    name: str
    arguments: dict


class LLMError(Exception):
    """Something went wrong that retrying won't fix (bad key, bad request)."""


class RateLimited(LLMError):
    """The provider said slow down (HTTP 429). retry_after: seconds it asked us to wait, if it said."""

    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def retry_after_from(error) -> float | None:
    """Seconds to wait, from a 429's Retry-After header or a "try again in 5.5s" / "82ms" message."""
    import re

    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        value = headers.get("retry-after")
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)\b", str(error))
    if m:
        return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
    return None


class Unavailable(LLMError):
    """Network trouble, timeout or a provider outage (5xx). Worth one retry."""


class AuthFailed(LLMError):
    """The API key was rejected."""


class Backend(Protocol):
    name: str
    model: str
    last_usage: Usage
    last_tool_calls: list[ToolCall]

    def stream(self, messages: list[dict], *, system: str, max_tokens: int,
               tools: list[ToolSpec] | None = None) -> Iterator[str]:
        """Yield reply text chunks. Sets last_usage and last_tool_calls when the reply finishes."""
        ...
