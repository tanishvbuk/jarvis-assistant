"""One conversational turn: text in, sentences out, with tools, retries and memory."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

from . import events
from .config import Config
from .connectors.base import Tool, ToolError, set_turn
from .llm import AuthFailed, Backend, LLMError, RateLimited, ToolCall, Unavailable
from .memory import History
from .persona import system_prompt
from .text import SentenceSplitter
from .turnlog import TurnRecord

log = logging.getLogger(__name__)

RETRY_DELAY_S = 2.0
MAX_RETRY_DELAY_S = 12.0  # if the provider asks for longer, give up and say so
MAX_TOOL_ROUNDS = 3  # model calls that may ask for tools before he must answer

LINES = {
    "rate_wait": "I'm being rate-limited, sir. One moment.",
    "rate_fail": "Still rate-limited, I'm afraid. Do try again shortly.",
    "down_wait": "I seem to have lost my connection, sir. Trying again.",
    "down_fail": "I can't reach my brain at the moment, sir. The network or the provider may be down.",
    "auth": "My credentials were refused, sir. The API key in the dot env file needs checking.",
    "error": "Something went wrong on my side, sir. The details are in the log.",
    "cut": "I appear to have lost my train of thought, sir.",
    "limit": "We've reached this session's turn limit, sir. Restart me if you'd like to carry on.",
    "loop": "I seem to be going round in circles, sir. Perhaps ask me that another way.",
}

HELD = ("Error: nothing was changed. This turn has already read text from outside (email, web pages, feeds, "
        "invites or sheets), and changes to his data are held back after that as a safety rule. Tell him plainly "
        "that you haven't made the change, and ask him to say it again as a separate request.")


class Assistant:
    def __init__(self, cfg: Config, backend: Backend, *, tools: list[Tool] | None = None,
                 notes: list[Callable[[], str | None]] | None = None,
                 preludes: list[Callable[[], tuple[str, str] | None]] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.backend = backend
        self.tools = {t.spec.name: t for t in tools or []}
        self.notes = list(notes or [])  # e.g. a once-a-day mention
        # Fetched material that belongs at the start of a turn (the morning briefing), given to the model as the
        # result of a tool call, never in the system prompt: (tool name, result text), or None.
        self.preludes = list(preludes or [])
        self.history = History(cfg.history_turns)
        self.turns = 0
        self._sleep = sleep

    def respond(self, user_text: str, say: Callable[[str], None], rec: TurnRecord,
                cancel: threading.Event | None = None) -> str:
        """Stream a reply to `user_text`, passing each complete sentence to `say`.

        If the model asks for tools, run them and ask again, up to MAX_TOOL_ROUNDS times.
        Tool results are used for this turn only: memory keeps just what was said.
        Never raises for provider problems: speaks an in-character line instead.
        """
        cancel = cancel or threading.Event()
        rec.provider, rec.model = self.backend.name, self.backend.model
        if self.turns >= self.cfg.max_turns_per_session:
            say(LINES["limit"])
            rec.error = "turn_limit"
            return ""
        self.turns += 1
        set_turn(self.turns, user_text)  # confirmations check the user's own words, not fetched text

        messages = self.history.messages() + [{"role": "user", "content": user_text}]
        system = system_prompt() + self._notes()
        tainted = False  # this turn has read text from outside
        for n, prelude in enumerate(self.preludes):
            try:
                got = prelude()
            except Exception:
                log.exception("A prelude failed")
                continue
            if got:
                name, result = got
                call = ToolCall(f"pre{n}", name, {})
                messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                             {"role": "tool", "tool_call_id": call.id, "name": name, "content": result}]
                rec.tools.append(name)
                tainted = tainted or self._reads_outside(name)
        specs = [t.spec for t in self.tools.values()] or None
        start = time.perf_counter()
        spoken: list[str] = []  # the model's sentences, as said
        heard: set[str] = set()

        def emit(sentence: str) -> None:
            # Some models answer, check with a tool, then give the same answer again: say it once.
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if key and key in heard:
                return
            heard.add(key)
            spoken.append(sentence)
            say(sentence)

        for round_ in range(MAX_TOOL_ROUNDS + 1):
            # Answers built on fetched data (a list of emails) need more room than chat.
            max_tokens = self.cfg.max_tokens if round_ == 0 else self.cfg.max_tokens_with_data
            result = self._call(messages, system, specs, max_tokens, say, emit, rec, start, cancel, bool(spoken))
            if result is None:
                break
            text, calls = result
            if not calls or cancel.is_set():
                break
            if round_ == MAX_TOOL_ROUNDS:
                say(LINES["loop"])
                rec.error = "tool_loop"
                break
            messages = messages + [{"role": "assistant", "content": text, "tool_calls": calls}]
            # Once a turn has read text from outside, it may not change his data (the model could have been
            # told to by whoever wrote that text), even by a call made in the same round as the reading.
            tainted = tainted or any(self._reads_outside(c.name) for c in calls)
            hold = tainted and self.cfg.safety.get("hold_writes", True)
            for call in calls:
                rec.tools.append(call.name)  # the name only: arguments may be personal
                if hold and self._writes(call):
                    log.info("Held a %s change: this turn had read text from outside", call.name)
                    result = HELD
                else:
                    result = self._run_tool(call)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})

        p = self.cfg.provider
        rec.est_cost_usd = (rec.input_tokens * p.price_in_per_mtok + rec.output_tokens * p.price_out_per_mtok) / 1e6
        rec.est_cost_gbp = rec.est_cost_usd * self.cfg.usd_to_gbp
        reply = " ".join(spoken)
        if reply:
            self.history.add(user_text, reply)
        return reply

    def _call(self, messages, system, specs, max_tokens, say, emit, rec, start, cancel,
              spoken_before: bool) -> tuple[str, list[ToolCall]] | None:
        """One model call, retried once on rate limits or network trouble.

        The model's sentences go to emit() as they complete; Jarvis's own lines (apologies) to say().
        Returns the text and any tool calls, or None if it failed (after apologising).
        """
        for attempt in range(2):
            text = ""
            splitter = SentenceSplitter()
            try:
                for chunk in self.backend.stream(messages, system=system, max_tokens=max_tokens, tools=specs):
                    if rec.first_token_s is None:
                        rec.first_token_s = time.perf_counter() - start
                    text += chunk
                    for sentence in splitter.feed(chunk):
                        emit(sentence)
                    if cancel.is_set():
                        break
                if not cancel.is_set():
                    for sentence in splitter.flush():
                        emit(sentence)
                rec.error = None  # a retry may have succeeded
                self._count_usage(rec)
                return text, ([] if cancel.is_set() else list(self.backend.last_tool_calls))
            except (RateLimited, Unavailable) as e:
                kind = "rate" if isinstance(e, RateLimited) else "down"
                log.warning("%s on attempt %d: %s", type(e).__name__, attempt + 1, e)
                rec.error = type(e).__name__
                if text or spoken_before:  # failed part-way through a spoken answer
                    say(LINES["cut"])
                    return text, []
                asked = getattr(e, "retry_after", None)
                if attempt == 0 and (asked is None or asked <= MAX_RETRY_DELAY_S):
                    rec.retries += 1
                    say(LINES[f"{kind}_wait"])
                    # Wait as long as the provider asked (Groq says "try again in 5.5s"), plus a little.
                    self._sleep(max(RETRY_DELAY_S, (asked or 0) + 0.5))
                    continue
                say(LINES[f"{kind}_fail"])
            except AuthFailed as e:
                log.error("Authentication failed: %s", e)
                rec.error = "AuthFailed"
                say(LINES["auth"])
            except LLMError as e:
                log.error("Provider error: %s", e)
                rec.error = "LLMError"
                say(LINES["error"])
            except Exception:
                log.exception("Unexpected error during reply")
                rec.error = "Exception"
                say(LINES["error"])
            return None
        return None

    def _notes(self) -> str:
        """Things a connector wants mentioned at the start of this reply (each decides how often)."""
        out = []
        for note in self.notes:
            try:
                text = note()
            except Exception:
                log.exception("A connector note failed")
                continue
            if text:
                out.append(text)
        return "".join(f"\n\nBefore answering, briefly: {t}" for t in out)

    def _count_usage(self, rec: TurnRecord) -> None:
        usage = self.backend.last_usage
        rec.input_tokens += usage.input_tokens
        rec.output_tokens += usage.output_tokens

    def _reads_outside(self, name: str) -> bool:
        tool = self.tools.get(name)
        return bool(tool and tool.reads_outside)

    def _writes(self, call: ToolCall) -> bool:
        tool = self.tools.get(call.name)
        try:
            return bool(tool and tool.writes and tool.writes(call.arguments))
        except Exception:
            return True  # can't tell: treat it as a change

    def _run_tool(self, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            return f"Error: there is no tool called {call.name}."
        if tool.notice:
            events.publish("notice", text=tool.notice)
        if tool.focus:
            events.publish("focus", card=tool.focus)
        try:
            return tool.run(call.arguments)
        except ToolError as e:
            return f"Error: {e}"
        except Exception:
            log.exception("Tool %s failed", call.name)  # the traceback, never the fetched content
            return "Error: the tool failed unexpectedly. The details are in Jarvis's log."
