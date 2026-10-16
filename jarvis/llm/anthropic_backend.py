"""Claude via the official Anthropic SDK (Messages API, streaming)."""

from __future__ import annotations

from typing import Iterator

import anthropic

from .base import AuthFailed, LLMError, RateLimited, ToolCall, ToolSpec, Unavailable, Usage, retry_after_from

REFUSAL_LINE = " I'm afraid I can't help with that one, sir."


def to_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)  # results of parallel calls share one user turn
            else:
                out.append({"role": "user", "content": [block]})
        elif m.get("tool_calls"):
            blocks: list[dict] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class AnthropicBackend:
    def __init__(self, *, model: str, api_key: str, effort: str | None = "low",
                 thinking: str | None = "disabled", client=None):
        self.name = "anthropic"
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.last_usage = Usage()
        self.last_tool_calls: list[ToolCall] = []
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=30.0)

    def _request(self, messages: list[dict], system: str, max_tokens: int, tools: list[ToolSpec] | None) -> dict:
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, system=system, messages=to_anthropic(messages))
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.thinking:
            # Thinking tokens count towards max_tokens and add delay, so voice runs with it off.
            kwargs["thinking"] = {"type": self.thinking}
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                               for t in tools]
        return kwargs

    def stream(self, messages: list[dict], *, system: str, max_tokens: int,
               tools: list[ToolSpec] | None = None) -> Iterator[str]:
        self.last_usage = Usage()
        self.last_tool_calls = []
        try:
            with self._client.messages.stream(**self._request(messages, system, max_tokens, tools)) as s:
                for text in s.text_stream:
                    yield text
                final = s.get_final_message()
            self.last_usage = Usage(final.usage.input_tokens, final.usage.output_tokens)
            self.last_tool_calls = [ToolCall(b.id, b.name, dict(b.input or {}))
                                    for b in getattr(final, "content", None) or [] if b.type == "tool_use"]
            if final.stop_reason == "refusal":
                yield REFUSAL_LINE
        except anthropic.RateLimitError as e:
            raise RateLimited(str(e), retry_after_from(e)) from e
        except anthropic.AuthenticationError as e:
            raise AuthFailed(str(e)) from e
        except (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.OverloadedError) as e:
            raise Unavailable(str(e)) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"{e.status_code}: {e}") from e
