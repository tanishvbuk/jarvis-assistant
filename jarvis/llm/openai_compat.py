"""Any OpenAI-style chat API: Groq, OpenRouter and others (change base_url, key, model)."""

from __future__ import annotations

import json
from typing import Iterator

import openai

from .base import AuthFailed, LLMError, RateLimited, ToolCall, ToolSpec, Unavailable, Usage, retry_after_from


def to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m["tool_calls"]]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _parse_arguments(raw: str) -> dict:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


class OpenAICompatBackend:
    def __init__(self, *, name: str, model: str, api_key: str, base_url: str, extra: dict | None = None,
                 client=None):
        self.name = name
        self.model = model
        self.extra = extra or {}
        self.last_usage = Usage()
        self.last_tool_calls: list[ToolCall] = []
        # Retries are handled by Jarvis so he can speak while waiting.
        self._client = client or openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=30.0)

    def stream(self, messages: list[dict], *, system: str, max_tokens: int,
               tools: list[ToolSpec] | None = None) -> Iterator[str]:
        self.last_usage = Usage()
        self.last_tool_calls = []
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
        calls: dict[int, dict] = {}  # tool calls arrive in pieces, keyed by index
        try:
            chunks = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *to_openai(messages)],
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=self.extra or None,
                **kwargs,
            )
            for chunk in chunks:
                usage = getattr(chunk, "usage", None)
                if usage is None:  # Groq has also reported usage under x_groq
                    usage = getattr(getattr(chunk, "x_groq", None), "usage", None)
                if usage is not None:
                    self.last_usage = Usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    for tc in getattr(delta, "tool_calls", None) or []:
                        c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        c["id"] = tc.id or c["id"]
                        if tc.function is not None:
                            c["name"] += tc.function.name or ""
                            c["args"] += tc.function.arguments or ""
                    if delta.content:
                        yield delta.content
        except openai.RateLimitError as e:
            raise RateLimited(str(e), retry_after_from(e)) from e
        except openai.AuthenticationError as e:
            raise AuthFailed(str(e)) from e
        except (openai.APIConnectionError, openai.InternalServerError) as e:  # includes timeouts
            raise Unavailable(str(e)) from e
        except openai.APIStatusError as e:
            raise LLMError(f"{e.status_code}: {e}") from e
        except openai.APIError as e:  # errors sent mid-stream, e.g. Groq's "model called a tool" refusals
            raise LLMError(str(e)) from e
        self.last_tool_calls = [ToolCall(c["id"] or f"call_{i}", c["name"], _parse_arguments(c["args"]))
                                for i, c in sorted(calls.items()) if c["name"]]
