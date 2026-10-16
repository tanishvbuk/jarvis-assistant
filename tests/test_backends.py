"""Both LLM backends against mocked SDK clients (no network, no keys)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import anthropic
import httpx2
import openai
import pytest

from jarvis.llm import AuthFailed, LLMError, RateLimited, ToolCall, ToolSpec, Unavailable
from jarvis.llm.anthropic_backend import REFUSAL_LINE, AnthropicBackend, to_anthropic
from jarvis.llm.openai_compat import OpenAICompatBackend, to_openai


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://example.invalid/v1"))


# ------------------------------------------------------------ openai_compat

def _chunk(text=None, usage=None, tool_calls=None):
    choices = [NS(delta=NS(content=text, tool_calls=tool_calls))] if text is not None or tool_calls else []
    return NS(choices=choices, usage=usage)


def _openai_backend(create):
    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return OpenAICompatBackend(name="groq", model="m", api_key="k", base_url="https://x", client=client), client


def test_openai_compat_streams_text_and_usage():
    chunks = [_chunk("Good "), _chunk("evening, sir."), _chunk(usage=NS(prompt_tokens=42, completion_tokens=7))]
    backend, client = _openai_backend(lambda **kw: iter(chunks))

    out = list(backend.stream([{"role": "user", "content": "hi"}], system="SYS", max_tokens=200))

    assert "".join(out) == "Good evening, sir."
    assert (backend.last_usage.input_tokens, backend.last_usage.output_tokens) == (42, 7)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert kwargs["max_tokens"] == 200 and kwargs["stream"] is True
    assert kwargs["extra_body"] is None


def test_openai_compat_sends_provider_extras():
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: iter([_chunk("Hi.")])
    backend = OpenAICompatBackend(name="groq", model="m", api_key="k", base_url="https://x",
                                  extra={"reasoning_effort": "low"}, client=client)
    list(backend.stream([], system="", max_tokens=10))
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning_effort": "low"}


def test_openai_compat_reads_groq_x_groq_usage():
    chunks = [_chunk("Hi."), NS(choices=[], usage=None, x_groq=NS(usage=NS(prompt_tokens=3, completion_tokens=2)))]
    backend, _ = _openai_backend(lambda **kw: iter(chunks))
    list(backend.stream([], system="", max_tokens=10))
    assert backend.last_usage.output_tokens == 2


@pytest.mark.parametrize("exc, expected", [
    (openai.RateLimitError("slow down", response=_response(429), body=None), RateLimited),
    (openai.AuthenticationError("bad key", response=_response(401), body=None), AuthFailed),
    (openai.InternalServerError("boom", response=_response(503), body=None), Unavailable),
    (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), Unavailable),
    (openai.APIError("Tool choice is none, but model called a tool", request=httpx2.Request("POST", "https://x"),
                     body=None), LLMError),
])
def test_openai_compat_maps_errors(exc, expected):
    def create(**kw):
        raise exc

    backend, _ = _openai_backend(create)
    with pytest.raises(expected):
        list(backend.stream([], system="", max_tokens=10))


def test_retry_after_from_header_or_message():
    from jarvis.llm.base import retry_after_from

    resp = httpx2.Response(429, headers={"retry-after": "7"}, request=httpx2.Request("POST", "https://x"))
    assert retry_after_from(openai.RateLimitError("slow", response=resp, body=None)) == 7.0
    assert retry_after_from(Exception("Please try again in 5.5275s. Need more")) == 5.5275
    assert retry_after_from(Exception("Please try again in 82.4ms.")) == 0.0824
    assert retry_after_from(Exception("nope")) is None


# ------------------------------------------------------------ anthropic

def _anthropic_backend(texts=(), stop_reason="end_turn", raises=None, content=(), **kw):
    client = MagicMock()
    final = NS(usage=NS(input_tokens=100, output_tokens=20), stop_reason=stop_reason, content=list(content))

    @contextmanager
    def stream(**params):
        if raises:
            raise raises
        yield NS(text_stream=iter(texts), get_final_message=lambda: final)

    client.messages.stream.side_effect = stream
    return AnthropicBackend(model="claude-haiku-5-5", api_key="k", client=client, **kw), client


def test_anthropic_streams_text_and_usage():
    backend, client = _anthropic_backend(["Quite ", "so, sir."])
    out = "".join(backend.stream([{"role": "user", "content": "hi"}], system="SYS", max_tokens=200))
    assert out == "Quite so, sir."
    assert (backend.last_usage.input_tokens, backend.last_usage.output_tokens) == (100, 20)
    params = client.messages.stream.call_args.kwargs
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"] == "SYS" and params["max_tokens"] == 200
    assert params["output_config"] == {"effort": "low"}
    assert params["thinking"] == {"type": "disabled"}


def test_anthropic_adaptive_thinking_setting_passes_through():
    backend, client = _anthropic_backend(["ok"], thinking="adaptive", effort="medium")
    list(backend.stream([], system="", max_tokens=10))
    params = client.messages.stream.call_args.kwargs
    assert params["thinking"] == {"type": "adaptive"} and params["output_config"] == {"effort": "medium"}


def test_anthropic_refusal_is_spoken_politely():
    backend, _ = _anthropic_backend([], stop_reason="refusal")
    assert "".join(backend.stream([], system="", max_tokens=10)) == REFUSAL_LINE


@pytest.mark.parametrize("exc, expected", [
    (anthropic.RateLimitError("slow down", response=_response(429), body=None), RateLimited),
    (anthropic.AuthenticationError("bad key", response=_response(401), body=None), AuthFailed),
    (anthropic.InternalServerError("boom", response=_response(500), body=None), Unavailable),
    (anthropic.OverloadedError("overloaded", response=_response(529), body=None), Unavailable),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")), Unavailable),
])
def test_anthropic_maps_errors(exc, expected):
    backend, _ = _anthropic_backend(raises=exc)
    with pytest.raises(expected):
        list(backend.stream([], system="", max_tokens=10))


# ------------------------------------------------------------ tools

SPEC = ToolSpec("check_email", "Read email", {"type": "object", "properties": {"account": {"type": "string"}}})
CONVERSATION = [
    {"role": "user", "content": "Any email?"},
    {"role": "assistant", "content": "Checking.", "tool_calls": [ToolCall("c1", "check_email", {"account": "a"}),
                                                                 ToolCall("c2", "check_email", {})]},
    {"role": "tool", "tool_call_id": "c1", "name": "check_email", "content": "R1"},
    {"role": "tool", "tool_call_id": "c2", "name": "check_email", "content": "R2"},
]


def _tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def test_openai_compat_assembles_streamed_tool_calls():
    chunks = [_chunk(tool_calls=[_tc(0, "call_1", "check_email", '{"acc')]),
              _chunk(tool_calls=[_tc(0, None, None, 'ount": "personal"}')]),
              _chunk(tool_calls=[_tc(1, "call_2", "check_email", "{}")]),
              _chunk(usage=NS(prompt_tokens=5, completion_tokens=3))]
    backend, client = _openai_backend(lambda **kw: iter(chunks))
    assert list(backend.stream([], system="", max_tokens=10, tools=[SPEC])) == []
    assert backend.last_tool_calls == [ToolCall("call_1", "check_email", {"account": "personal"}),
                                       ToolCall("call_2", "check_email", {})]
    sent = client.chat.completions.create.call_args.kwargs["tools"]
    assert sent == [{"type": "function", "function": {"name": "check_email", "description": "Read email",
                                                      "parameters": SPEC.parameters}}]


def test_openai_compat_bad_tool_arguments_become_empty():
    backend, _ = _openai_backend(lambda **kw: iter([_chunk(tool_calls=[_tc(0, "c", "check_email", "{not json")])]))
    list(backend.stream([], system="", max_tokens=10, tools=[SPEC]))
    assert backend.last_tool_calls[0].arguments == {}


def test_openai_compat_sends_no_tools_key_without_tools():
    backend, client = _openai_backend(lambda **kw: iter([_chunk("Hi.")]))
    list(backend.stream([], system="", max_tokens=10))
    assert "tools" not in client.chat.completions.create.call_args.kwargs


def test_to_openai_messages():
    out = to_openai(CONVERSATION)
    assert out[1]["tool_calls"][0] == {"id": "c1", "type": "function",
                                       "function": {"name": "check_email", "arguments": '{"account": "a"}'}}
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "R1"}


def test_anthropic_reads_tool_use_blocks():
    blocks = [NS(type="text", text="Checking."), NS(type="tool_use", id="tu_1", name="check_email",
                                                    input={"account": "shop"})]
    backend, client = _anthropic_backend(["Checking."], stop_reason="tool_use", content=blocks)
    assert "".join(backend.stream([], system="", max_tokens=10, tools=[SPEC])) == "Checking."
    assert backend.last_tool_calls == [ToolCall("tu_1", "check_email", {"account": "shop"})]
    assert client.messages.stream.call_args.kwargs["tools"] == [
        {"name": "check_email", "description": "Read email", "input_schema": SPEC.parameters}]


def test_to_anthropic_messages_group_tool_results():
    out = to_anthropic(CONVERSATION)
    assert out[1] == {"role": "assistant", "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "c1", "name": "check_email", "input": {"account": "a"}},
        {"type": "tool_use", "id": "c2", "name": "check_email", "input": {}}]}
    assert out[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "R1"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "R2"}]}
    assert len(out) == 3
