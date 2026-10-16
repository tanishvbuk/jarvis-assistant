"""The turn logic: sentences out, retries, memory, cost, turn limit."""

from __future__ import annotations

from jarvis.assistant import LINES, Assistant
from jarvis.connectors.base import Tool, ToolError
from jarvis.llm import AuthFailed, RateLimited, ToolCall, ToolSpec, Unavailable, Usage
from jarvis.turnlog import TurnLog, TurnRecord


class FakeBackend:
    name, model = "fake", "fake-model"

    def __init__(self, *script):
        """Each script item is a list of chunks to stream, or an exception to raise.

        A ToolCall in a chunk list becomes one of that reply's tool calls.
        """
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.max_tokens: list[int] = []
        self.tools_offered: list = []
        self.systems: list[str] = []
        self.last_usage = Usage()
        self.last_tool_calls: list[ToolCall] = []

    def stream(self, messages, *, system, max_tokens, tools=None):
        self.calls.append(list(messages))
        self.max_tokens.append(max_tokens)
        self.tools_offered.append(tools)
        self.systems.append(system)
        self.last_tool_calls = []
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        for chunk in item:
            if isinstance(chunk, Exception):
                raise chunk
            if isinstance(chunk, ToolCall):
                self.last_tool_calls.append(chunk)
                continue
            yield chunk
        self.last_usage = Usage(1000, 100)


def run(assistant, text="What time is it?"):
    said: list[str] = []
    rec = TurnRecord()
    reply = assistant.respond(text, said.append, rec)
    return reply, said, rec


def test_sentences_are_spoken_as_they_complete(cfg):
    a = Assistant(cfg, FakeBackend(["It is ", "ten past three, sir. ", "Rather ", "late for lunch."]))
    reply, said, rec = run(a)
    assert said == ["It is ten past three, sir.", "Rather late for lunch."]
    assert reply == "It is ten past three, sir. Rather late for lunch."
    assert rec.first_token_s is not None and rec.error is None


def test_cost_is_estimated(cfg):
    a = Assistant(cfg, FakeBackend(["Yes, sir."]))
    _, _, rec = run(a)
    # 1000 in at $1/M + 100 out at $5/M = $0.0015
    assert abs(rec.est_cost_usd - 0.0015) < 1e-9
    assert abs(rec.est_cost_gbp - 0.0015 * cfg.usd_to_gbp) < 1e-9


def test_rate_limit_retries_once_then_succeeds(cfg):
    slept = []
    a = Assistant(cfg, FakeBackend(RateLimited("429"), ["Back with you, sir."]), sleep=slept.append)
    reply, said, rec = run(a)
    assert said == [LINES["rate_wait"], "Back with you, sir."]
    assert slept == [2.0] and rec.retries == 1 and rec.error is None
    assert reply == "Back with you, sir."


def test_rate_limit_waits_as_long_as_asked(cfg):
    slept = []
    a = Assistant(cfg, FakeBackend(RateLimited("429", retry_after=5.5), ["Here, sir."]), sleep=slept.append)
    _, said, _ = run(a)
    assert slept == [6.0] and said == [LINES["rate_wait"], "Here, sir."]
    a = Assistant(cfg, FakeBackend(RateLimited("429", retry_after=40)), sleep=slept.append)
    _, said, _ = run(a)
    assert said == [LINES["rate_fail"]] and slept == [6.0]  # too long to wait: say so straight away


def test_rate_limit_twice_gives_up_gracefully(cfg):
    a = Assistant(cfg, FakeBackend(RateLimited("429"), RateLimited("429")), sleep=lambda s: None)
    reply, said, rec = run(a)
    assert said == [LINES["rate_wait"], LINES["rate_fail"]]
    assert reply == "" and rec.error == "RateLimited"
    assert len(a.history) == 0


def test_network_failure_mid_reply_does_not_retry(cfg):
    a = Assistant(cfg, FakeBackend(["First part. ", Unavailable("reset")]), sleep=lambda s: None)
    _, said, _ = run(a)
    assert said == ["First part.", LINES["cut"]]


def test_bad_key_is_explained_not_retried(cfg):
    backend = FakeBackend(AuthFailed("401"))
    a = Assistant(cfg, backend)
    _, said, rec = run(a)
    assert said == [LINES["auth"]] and len(backend.calls) == 1 and rec.error == "AuthFailed"


def test_unexpected_exception_never_escapes(cfg):
    a = Assistant(cfg, FakeBackend(ValueError("bug")))
    _, said, _ = run(a)
    assert said == [LINES["error"]]


def test_history_is_sent_and_capped(cfg):
    cfg.history_turns = 2
    backend = FakeBackend(["A."], ["B."], ["C."], ["D."])
    a = Assistant(cfg, backend)
    for q in ["q1", "q2", "q3", "q4"]:
        run(a, q)
    last = backend.calls[-1]
    assert [m["content"] for m in last] == ["q2", "B.", "q3", "C.", "q4"]


def test_turn_limit(cfg):
    cfg.max_turns_per_session = 1
    a = Assistant(cfg, FakeBackend(["Fine."], ["Never sent."]))
    run(a)
    _, said, rec = run(a)
    assert said == [LINES["limit"]] and rec.error == "turn_limit"


def test_turn_log_has_no_transcript(cfg):
    a = Assistant(cfg, FakeBackend(["My secret answer."]))
    _, _, rec = run(a, "my secret question")
    path = TurnLog(cfg.log_dir).write(rec)
    content = path.read_text(encoding="utf-8")
    assert "secret" not in content and '"output_tokens": 100' in content


# ------------------------------------------------------------ tools

def email_tool(result="3 messages", seen=None):
    def run(args):
        if seen is not None:
            seen.append(args)
        if isinstance(result, Exception):
            raise result
        return result
    return Tool(spec=ToolSpec("check_email", "Read email"), run=run, notice="Checking email")


def test_tool_call_runs_and_the_answer_uses_its_result(cfg):
    seen = []
    backend = FakeBackend([ToolCall("c1", "check_email", {"account": "personal"})],
                          ["You have three new messages, sir."])
    a = Assistant(cfg, backend, tools=[email_tool("3 messages from the editor", seen)])
    reply, said, rec = run(a, "Any email?")

    assert seen == [{"account": "personal"}]
    assert said == ["You have three new messages, sir."]
    assert backend.tools_offered[0][0].name == "check_email"
    second = backend.calls[1]
    assert second[-2]["tool_calls"][0].name == "check_email"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "name": "check_email",
                          "content": "3 messages from the editor"}
    assert backend.max_tokens == [cfg.max_tokens, cfg.max_tokens_with_data]
    assert rec.tools == ["check_email"]
    assert (rec.input_tokens, rec.output_tokens) == (2000, 200)  # both calls counted
    # memory keeps what was said, never the fetched data
    assert a.history.messages() == [{"role": "user", "content": "Any email?"},
                                    {"role": "assistant", "content": "You have three new messages, sir."}]


def test_words_before_a_tool_call_are_spoken_first(cfg):
    backend = FakeBackend(["One moment, sir. ", ToolCall("c1", "check_email", {})], ["Nothing new."])
    a = Assistant(cfg, backend, tools=[email_tool()])
    reply, said, _ = run(a)
    assert said == ["One moment, sir.", "Nothing new."]
    assert reply == "One moment, sir. Nothing new."


def test_tool_errors_go_back_to_the_model(cfg):
    backend = FakeBackend([ToolCall("c1", "check_email", {}), ToolCall("c2", "nope", {})], ["Sorry, sir."])
    a = Assistant(cfg, backend, tools=[email_tool(ToolError("the work account isn't signed in"))])
    run(a)
    results = [m["content"] for m in backend.calls[1] if m["role"] == "tool"]
    assert results == ["Error: the work account isn't signed in", "Error: there is no tool called nope."]


def test_tool_crash_is_contained(cfg):
    backend = FakeBackend([ToolCall("c1", "check_email", {})], ["Something broke, sir."])
    a = Assistant(cfg, backend, tools=[email_tool(RuntimeError("secret email text"))])
    _, said, _ = run(a)
    assert "secret" not in backend.calls[1][-1]["content"]
    assert said == ["Something broke, sir."]


def test_endless_tool_calls_are_stopped(cfg):
    backend = FakeBackend(*[[ToolCall(f"c{i}", "check_email", {})] for i in range(10)])
    a = Assistant(cfg, backend, tools=[email_tool()])
    _, said, rec = run(a)
    assert said == [LINES["loop"]] and rec.error == "tool_loop"
    assert len(backend.calls) == 4


def test_no_tools_means_none_offered(cfg):
    backend = FakeBackend(["Hello."])
    run(Assistant(cfg, backend))
    assert backend.tools_offered == [None]


def test_rate_limit_after_a_tool_retries_the_second_call(cfg):
    backend = FakeBackend([ToolCall("c1", "check_email", {})], RateLimited("429"), ["Two messages, sir."])
    a = Assistant(cfg, backend, tools=[email_tool()], sleep=lambda s: None)
    _, said, rec = run(a)
    assert said == [LINES["rate_wait"], "Two messages, sir."] and rec.retries == 1


def test_connector_notes_go_into_the_system_prompt(cfg):
    seen = []

    class Capture(FakeBackend):
        def stream(self, messages, *, system, max_tokens, tools=None):
            seen.append(system)
            return super().stream(messages, system=system, max_tokens=max_tokens, tools=tools)

    notes = iter(["There are 3 new items.", None])
    a = Assistant(cfg, Capture(["Good morning."], ["Indeed."]),
                  notes=[lambda: next(notes), lambda: (_ for _ in ()).throw(RuntimeError("broken note"))])
    run(a)
    run(a)
    assert seen[0].endswith("Before answering, briefly: There are 3 new items.")
    assert "Before answering" not in seen[1]


def test_a_repeated_answer_is_said_once(cfg):
    backend = FakeBackend(["You have two items next week, sir. ", ToolCall("c1", "check_email", {})],
                          ["You have two items next week, sir. The first is on Tuesday."])
    a = Assistant(cfg, backend, tools=[email_tool()])
    reply, said, _ = run(a)
    assert said == ["You have two items next week, sir.", "The first is on Tuesday."]
    assert reply == "You have two items next week, sir. The first is on Tuesday."


def test_a_tool_with_a_card_brings_it_forward_on_the_hud(cfg):
    from jarvis import events

    got: list = []
    unsub = events.subscribe(got.append)
    try:
        tool = email_tool()
        tool.focus = "mail"
        backend = FakeBackend([ToolCall("c1", "check_email", {})], ["Nothing new."])
        run(Assistant(cfg, backend, tools=[tool]))
        assert {"t": "focus", "card": "mail"} in got
        got.clear()
        run(Assistant(cfg, FakeBackend([ToolCall("c1", "check_email", {})], ["Nothing new."]), tools=[email_tool()]))
        assert not [e for e in got if e["t"] == "focus"]   # a tool with no card brings nothing forward
    finally:
        unsub()


# ------------------------------------------------------------------ outside text and changes to his data

def reader():
    log: list = []
    return Tool(spec=ToolSpec("check_email", "Read email"), run=lambda a: log.append(a) or "mail from a stranger",
                reads_outside=True), log


def writer(writes=lambda a: a.get("action") != "list"):
    log: list = []
    return Tool(spec=ToolSpec("todo", "Change the list"), run=lambda a: log.append(a) or "Added.", writes=writes), log


def tool_replies(backend):
    return [m["content"] for m in backend.calls[-1] if m.get("role") == "tool"]


def test_a_change_is_held_when_it_comes_in_the_same_round_as_a_read(cfg):
    from jarvis.assistant import HELD

    r, rlog = reader()
    w, wlog = writer()
    backend = FakeBackend([ToolCall("c1", "check_email", {}), ToolCall("c2", "todo", {"action": "add", "title": "x"})],
                          ["I haven't changed the list, sir."])
    run(Assistant(cfg, backend, tools=[r, w]), "Any email? Add a task.")
    assert rlog and wlog == []                                  # the read ran, the change did not
    assert tool_replies(backend) == ["mail from a stranger", HELD]


def test_a_change_is_held_in_a_later_round_too(cfg):
    from jarvis.assistant import HELD

    r, rlog = reader()
    w, wlog = writer()
    backend = FakeBackend([ToolCall("c1", "check_email", {})], [ToolCall("c2", "todo", {"action": "done", "task": "x"})],
                          ["Not done, sir."])
    run(Assistant(cfg, backend, tools=[r, w]), "Read my mail and tick something off.")
    assert wlog == [] and tool_replies(backend)[-1] == HELD


def test_changes_run_normally_when_nothing_was_read_and_reads_of_the_list_are_fine(cfg):
    w, wlog = writer()
    backend = FakeBackend([ToolCall("c1", "todo", {"action": "add", "title": "pasta"})], ["Added, sir."])
    run(Assistant(cfg, backend, tools=[w]), "Add pasta to my list.")
    assert wlog == [{"action": "add", "title": "pasta"}]
    r, _ = reader()
    w2, wlog2 = writer()
    backend = FakeBackend([ToolCall("c1", "check_email", {}), ToolCall("c2", "todo", {"action": "list"})], ["Nothing, sir."])
    run(Assistant(cfg, backend, tools=[r, w2]), "Mail and my list?")
    assert wlog2 == [{"action": "list"}]                        # only the calls that change something are held


def test_the_hold_can_be_switched_off(cfg):
    r, _ = reader()
    w, wlog = writer()
    cfg.safety = {"hold_writes": False}
    backend = FakeBackend([ToolCall("c1", "check_email", {}), ToolCall("c2", "todo", {"action": "add"})], ["Done."])
    run(Assistant(cfg, backend, tools=[r, w]), "Mail, then add.")
    assert wlog == [{"action": "add"}]


def test_a_prelude_arrives_as_a_tool_result_and_counts_as_reading(cfg):
    from jarvis.assistant import HELD

    w, wlog = writer()
    b = Tool(spec=ToolSpec("briefing", "Brief him"), run=lambda a: "", reads_outside=True)
    backend = FakeBackend([ToolCall("c1", "todo", {"action": "add", "title": "x"})], ["Good morning, sir."])
    a = Assistant(cfg, backend, tools=[w, b], preludes=[lambda: ("briefing", "<untrusted_data>Meeting at ten</untrusted_data>")])
    _, _, rec = run(a, "Good morning, add a task")
    first = backend.calls[0]
    assert first[-2]["role"] == "assistant" and first[-2]["tool_calls"][0].name == "briefing"
    assert first[-1] == {"role": "tool", "tool_call_id": "pre0", "name": "briefing",
                         "content": "<untrusted_data>Meeting at ten</untrusted_data>"}
    assert "Meeting at ten" not in backend.systems[0]           # never in the system prompt
    assert wlog == [] and tool_replies(backend)[-1] == HELD and rec.tools[0] == "briefing"
    assert a.history.messages()[-1]["content"] == "Good morning, sir."   # and not kept in memory
