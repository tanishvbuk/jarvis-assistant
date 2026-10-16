"""The morning briefing: everything he'd want to know at the start of the day, in one go.

Two ways in, both ending in the same compact text for the model to speak:

* The first time he speaks to Jarvis in the morning (between `from_hour` and `until_hour`, by the wake word,
  the key or the HUD ring) the voice loop calls activate(). That takes over the HUD with the briefing screen at
  once, and the first turn's system prompt carries the briefing (daily_note()), together with every other
  once-a-day mention (gym, tasks, embargoes), so they're said once, as part of it, not one after another.
* The `briefing` tool, for "good morning", "brief me" or "what's on today", any time.

The data is what the HUD cards already hold: each connector's card() (cached underneath), so nothing new is
fetched here. It is prepared in the background (tick()) so the first turn doesn't wait on the network.
Meals are included as one next meal per plan; [briefing] meals = false leaves them out.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import Tool, read_state, untrusted, write_state

log = logging.getLogger(__name__)

PREPARE_S = 900     # how old prepared text may be before it's put together again
NOTE_HEAD = ("This is his first activation today, so give his morning briefing first, from the briefing result "
             "that comes with his message: three to five short spoken sentences, most important first (anything "
             "overdue or due today, the first event, then a clause "
             "each for gym and mail (say how many are from people)). No lists, and don't read it all out: it is also on his screen. Then deal "
             "with what he said, if anything.")


class Briefing:
    name = "briefing"

    def __init__(self, parts: dict[str, object], *, data_dir: Path | None = None, enabled: bool = True,
                 from_hour: int = 5, until_hour: int = 12, meals: bool = True, others: list | None = None,
                 now: Callable[[], datetime] | None = None, clock: Callable[[], float] = time.monotonic):
        self.parts = parts              # connectors by name: calendar, tasks, meal_plan, gmail, opengym, timers
        self.enabled = enabled
        self.from_hour, self.until_hour = from_hour, until_hour
        self.meals = meals
        self.others = others or []      # the other connectors, whose once-a-day mentions are folded into the briefing
        self._now = now or (lambda: datetime.now(LOCAL))
        self._clock = clock
        self.state_file = data_dir / "briefing_state.json" if data_dir else None
        self._armed = False
        self._deliver = False           # the briefing goes to the model with this turn (see prelude())
        self._ready: tuple[float, str] | None = None
        self._lock = threading.Lock()

    # ---- the tool

    def tools(self) -> list[Tool]:
        spec = ToolSpec(
            name="briefing",
            description=("His briefing for the day: calendar, to-do, meals, gym and mail. "
                         "For 'good morning', 'brief me', 'what's on today'. Speak it in a few sentences."),
            parameters={"type": "object", "properties": {}})
        return [Tool(spec=spec, run=self.briefing, notice="Putting the briefing together", focus="briefing", reads_outside=True)]

    def briefing(self, args: dict) -> str:
        self._armed = False
        self._mark()
        return self._text()

    # ---- the first activation of the day

    def window(self) -> bool:
        return self.from_hour <= self._now().hour < self.until_hour

    def delivered(self) -> bool:
        return read_state(self.state_file).get("day") == self._now().date().isoformat()

    def _mark(self) -> None:
        write_state(self.state_file, {**read_state(self.state_file), "day": self._now().date().isoformat()})

    def activate(self) -> bool:
        """He has just spoken to Jarvis. If it's the first time this morning, put the briefing on screen and arm it
        for this turn. True if that happened."""
        if not self.enabled or self._armed or not self.window() or self.delivered():
            return False
        self._armed = True
        events.publish("briefing")
        return True

    def daily_note(self) -> str | None:
        """What to do on the armed turn (once). The briefing itself goes in as a tool result (prelude()), not here:
        it holds calendar titles, task names and headlines that other people wrote, which don't belong in the system
        prompt. The other connectors' once-a-day mentions are used up here so nothing repeats the briefing; those that
        carry fetched text are dropped, since the briefing already holds that, and the rest (a
        count of embargoes, a weigh-in copied from the sheet) are added."""
        if not self._armed:
            return None
        self._armed = False
        self._deliver = True
        self._mark()
        extras = []
        for c in self.others:
            try:
                text = c.daily_note()
            except Exception:
                log.exception("A connector note failed")
                continue
            if text and "<untrusted_data" not in text:
                extras.append(text)
        if extras:
            return NOTE_HEAD + "\nAlso work these in, briefly:\n" + "\n".join(extras)
        return NOTE_HEAD

    def prelude(self) -> tuple[str, str] | None:
        """The briefing as the result of a `briefing` tool call, for the turn daily_note() armed."""
        if not self._deliver:
            return None
        self._deliver = False
        return "briefing", self._text()

    # ---- background preparation

    def tick(self) -> None:
        if not self.enabled or not self.window() or self.delivered():
            return
        with self._lock:
            fresh = self._ready and self._clock() - self._ready[0] < PREPARE_S
        if not fresh:
            text = self.compile()
            with self._lock:
                self._ready = (self._clock(), text)

    def _text(self) -> str:
        with self._lock:
            if self._ready and self._clock() - self._ready[0] < PREPARE_S:
                return self._ready[1]
        text = self.compile()
        with self._lock:
            self._ready = (self._clock(), text)
        return text

    # ---- putting it together

    def _card(self, name: str) -> dict | None:
        part = self.parts.get(name)
        if part is None or not hasattr(part, "card"):
            return None
        try:
            return part.card()
        except Exception as e:
            log.warning("briefing: the %s card failed: %s", name, type(e).__name__)
            return None

    def compile(self) -> str:
        now = self._now()
        lines = [f"Briefing for {spoken(now, '%A %d %B')}; it is {now:%H:%M} local time."]
        for build in (self._calendar, self._tasks, self._gym, self._meals, self._mail, self._timers):
            try:
                line = build(now)
            except Exception as e:  # one section never spoils the rest
                log.warning("briefing section failed: %s", type(e).__name__)
                continue
            if line:
                lines.append(line)
        return untrusted("Jarvis's own cards (calendar, tasks, sheets, feeds)", "\n".join(lines))

    def _calendar(self, now) -> str | None:
        c = self._card("calendar")
        if not c:
            return None
        if c.get("error") and not c.get("items"):
            return f"Calendar: unavailable ({c['error'].lower()})."
        items = c.get("items") or []
        if not items:
            return f"Calendar: nothing on {c.get('label', 'today').lower()}."
        shown = "; ".join(f"{i['time']} {i['title']}" for i in items[:5])
        more = len(items) - 5 + (c.get("more") or 0)
        return f"Calendar {c.get('label', 'today').lower()}: {shown}" + (f" (and {more} more)" if more > 0 else "") + "."

    def _tasks(self, now) -> str | None:
        c = self._card("tasks")
        if not c:
            return None
        if c.get("error"):
            return f"To do: unavailable ({c['error'].lower()})."
        items = c.get("items") or []
        late = [i["title"] for i in items if i.get("late")]
        today = [i["title"] + (f" ({i['when']})" if re.match(r"\d\d:\d\d$", i.get("when", "")) else "")
                 for i in items if not i.get("late") and (i.get("when") == "TODAY" or re.match(r"\d\d:\d\d$", i.get("when", "")))]
        parts = []
        if late:
            parts.append("overdue: " + "; ".join(late[:4]))
        if today:
            parts.append("due today: " + "; ".join(today[:4]))
        rest = len(items) + (c.get("more") or 0) - len(late) - len(today)
        if rest > 0:
            parts.append(f"{rest} other open task{'s' * (rest != 1)}")
        return "To do: " + ("; ".join(parts) if parts else "nothing open") + "."

    def _gym(self, now) -> str | None:
        c = self._card("opengym")
        if not c or c.get("error"):
            return None
        if c["status"] == "PLANNED":
            first = (c.get("items") or [{}])[0]
            out = f"Gym today: {c['title']}" + (f", first exercise {first['name']}" + (f" (last time {first['last']})" if first.get("last") else "")
                                                 if first.get("name") else "") + "."
        elif c["status"] == "DONE":
            out = f"Gym: already done today ({c.get('detail', '')})."
        else:
            out = f"Gym: a rest day. {c.get('detail', '')}".strip()
        w = c.get("weight")
        if w:
            out += f" His body weight is {w['latest']} {w['unit']}" + (f", {w['trend']} a week" if w.get("trend") not in (None, "0", "+0", "-0") else "") + \
                (f", goal {w['goal']}" if w.get("goal") else "") + \
                (f", last weighed {w['days']} days ago" if w["days"] >= 7 else "") + "."
        return out

    def _meals(self, now) -> str | None:
        if not self.meals:
            return None
        c = self._card("meal_plan")
        if not c:
            return None
        minutes = now.hour * 60 + now.minute
        out = []
        for s in c.get("sections") or []:
            if s.get("error") or not s.get("items"):
                continue
            nxt = next((m for m in s["items"] if _mins(m.get("time")) is not None and _mins(m["time"]) >= minutes), s["items"][0])
            out.append(f"{s['label'].title()}: {nxt['name']} at {nxt['time']}, {nxt['food']}")
        return ("Next meal: " + "; ".join(out) + ".") if out else None

    def _mail(self, now) -> str | None:
        c = self._card("gmail")
        if not c:
            return None
        plus = "+" if c.get("capped") else ""
        out = f"Mail, last {c['days']} days: {c['people']}{plus} from real people, {c['auto']}{plus} automated notifications."
        if c.get("problems"):
            out += " Needs signing in again: " + ", ".join(c["problems"]) + "."
        return out


    def _timers(self, now) -> str | None:
        c = self._card("timers")
        if not c or not c.get("items"):
            return None
        return "Timers and reminders: " + "; ".join(f"{t['label']} at {t['due'][11:16]}" for t in c["items"][:3]) + "."


def _mins(t: str | None) -> int | None:
    m = re.match(r"(\d{1,2}):(\d{2})$", t or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def briefing_from_config(cfg, connectors: list) -> Briefing | None:
    b = cfg.briefing
    if not b.get("enabled", True):
        return None
    parts = {c.name: c for c in connectors}
    return Briefing(parts, data_dir=cfg.root / "data", from_hour=int(b.get("from_hour", 5)),
                    until_hour=int(b.get("until_hour", 12)), meals=bool(b.get("meals", True)),
                    others=[c for c in connectors if hasattr(c, "daily_note")])
