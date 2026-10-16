"""Timers and reminders: "set a 40-minute timer for the pasta", "remind me at 6 to post the parcels".

Local only: nothing is sent anywhere, so no confirmation is asked. Kept in
data/timers.json, so they survive a restart; anything that fell due while
Jarvis was off is announced as missed when he starts.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .. import events
from ..connectors.base import Tool, ToolError
from ..llm import ToolSpec
from ..persona import LOCAL, spoken

log = logging.getLogger(__name__)

MAX_ACTIVE = 20
MAX_AHEAD = timedelta(days=365)


def _spoken_duration(minutes: float) -> str:
    if minutes < 1:
        return f"{round(minutes * 60)}-second"
    if minutes < 60 or minutes % 60:
        return f"{minutes:g}-minute"
    return f"{minutes / 60:g}-hour"


class Timers:
    name = "timers"

    def __init__(self, store: Path, *, now: Callable[[], datetime] | None = None):
        self.store = store
        self._now = now or (lambda: datetime.now(LOCAL))
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._on_due: Callable[[str], None] | None = None
        self.state = self._load()

    # ---- storage

    def _load(self) -> dict:
        try:
            return json.loads(self.store.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as e:
            log.warning("timers: couldn't read %s (%s); starting afresh", self.store, type(e).__name__)
        return {"timers": [], "next_id": 1}

    def _save(self) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.store)

    # ---- the tool

    def tools(self) -> list[Tool]:
        spec = ToolSpec(
            name="timer",
            description=(
                "Timers and reminders on this laptop. set: minutes (a countdown) or at (a reminder), plus a short "
                "label in his words; list; cancel by label, or 'all'."),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["set", "list", "cancel"]},
                    "minutes": {"type": "number"},
                    "at": {"type": "string", "description": "YYYY-MM-DDTHH:MM local time"},
                    "label": {"type": "string"},
                },
                "required": ["action"],
            },
        )
        return [Tool(spec=spec, run=self.timer)]

    def timer(self, args: dict) -> str:
        action = args.get("action")
        if action == "set":
            return self._set(args)
        if action == "list":
            return self._list()
        if action == "cancel":
            return self._cancel(str(args.get("label") or ""))
        raise ToolError("action must be set, list or cancel.")

    def _set(self, args: dict) -> str:
        now = self._now()
        label = " ".join(str(args.get("label") or "").split())[:80]
        if args.get("at"):
            try:
                due = datetime.fromisoformat(str(args["at"]))
            except ValueError:
                raise ToolError(f"{args['at']!r} isn't a time; use YYYY-MM-DDTHH:MM.") from None
            due = due.replace(tzinfo=LOCAL) if due.tzinfo is None else due.astimezone(LOCAL)
            kind, minutes = "reminder", None
        elif args.get("minutes") is not None:
            try:
                minutes = float(args["minutes"])
            except (TypeError, ValueError):
                raise ToolError("minutes must be a number.") from None
            if not 0 < minutes <= 7 * 24 * 60:
                raise ToolError("a timer must be between a few seconds and a week.")
            due, kind = now + timedelta(minutes=minutes), "timer"
        else:
            raise ToolError("say how long (minutes) or when (at).")
        if due <= now:
            raise ToolError(f"{due:%H:%M on %d %b} has already passed.")
        if due - now > MAX_AHEAD:
            raise ToolError("that's more than a year away.")
        with self._lock:
            if len(self.state["timers"]) >= MAX_ACTIVE:
                raise ToolError(f"there are already {MAX_ACTIVE} timers running; cancel some first.")
            item = {"id": self.state["next_id"], "kind": kind, "label": label, "due": due.isoformat(),
                    "minutes": minutes, "created": now.isoformat()}
            if args.get("key"):  # set by other connectors (not the model), to cancel it later
                item["key"] = str(args["key"])
            self.state["next_id"] += 1
            self.state["timers"].append(item)
            self._save()
        self._changed()
        what = (f"{_spoken_duration(minutes)} timer" if kind == "timer" else "reminder") + (
            f" for {label}" if label else "")
        return f"Set: {what}, due {self._say(due)}."

    def _list(self) -> str:
        with self._lock:
            items = sorted(self.state["timers"], key=lambda t: t["due"])
        if not items:
            return "No timers or reminders are set."
        now = self._now()
        lines = [f"{len(items)} set:"]
        for t in items:
            due = datetime.fromisoformat(t["due"])
            left = due - now
            mins = max(0, round(left.total_seconds() / 60))
            what = t["kind"] + (f" for {t['label']}" if t["label"] else "")
            lines.append(f"- {what}: due {self._say(due)}" + (f" (in {mins} min)" if left < timedelta(hours=3) else ""))
        return "\n".join(lines)

    def _cancel(self, label: str) -> str:
        want = label.lower().strip()
        with self._lock:
            items = self.state["timers"]
            if want in ("all", "everything"):
                gone = list(items)
            else:
                gone = [t for t in items if want and want in t["label"].lower()]
                if not gone and len(items) == 1 and not want:
                    gone = list(items)
            if not gone:
                names = ", ".join(t["label"] or t["kind"] for t in items) or "none"
                raise ToolError(f"no timer matches {label!r}; current ones: {names}.")
            if len(gone) > 1 and want not in ("all", "everything"):
                raise ToolError("more than one matches: " + ", ".join(t["label"] for t in gone) + ". Be more specific.")
            self.state["timers"] = [t for t in items if t not in gone]
            self._save()
        self._changed()
        return "Cancelled: " + ", ".join(t["label"] or t["kind"] for t in gone) + "."

    def cancel_key(self, key: str, label: str = "") -> int:
        """Quietly cancel what another connector set with this key (or, for older ones, this exact label)."""
        label = " ".join(label.split())[:80]
        with self._lock:
            items = self.state["timers"]
            keep = [t for t in items if t.get("key") != key and not (label and "key" not in t and t["label"] == label)]
            gone = len(items) - len(keep)
            if gone:
                self.state["timers"] = keep
                self._save()
        if gone:
            self._changed()
        return gone

    def _say(self, due: datetime) -> str:
        now = self._now()
        if due.date() == now.date():
            return f"at {due:%H:%M}"
        if due.date() == (now + timedelta(days=1)).date():
            return f"tomorrow at {due:%H:%M}"
        return spoken(due, "on %A %d %B at %H:%M")

    # ---- firing

    def start(self, on_due: Callable[[str], None]) -> None:
        """Run the alarm clock on a background thread; on_due(text) speaks or prints the alert."""
        self._on_due = on_due
        events.publish("card", **self.card())  # timers left from before a restart
        threading.Thread(target=self._run, name="timers", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _changed(self) -> None:
        self._wake.set()
        events.publish("card", **self.card())

    def _run(self) -> None:
        started = self._now()
        self.fire_due(missed_before=started - timedelta(minutes=1))
        while not self._stop.is_set():
            with self._lock:
                nxt = min((datetime.fromisoformat(t["due"]) for t in self.state["timers"]), default=None)
            wait = 30.0 if nxt is None else max(0.0, min(30.0, (nxt - self._now()).total_seconds()))
            self._wake.wait(wait)
            self._wake.clear()
            self.fire_due()

    def fire_due(self, missed_before: datetime | None = None) -> list[str]:
        """Announce and remove everything that's due. Returns the alerts (for tests)."""
        now = self._now()
        with self._lock:
            due = sorted((t for t in self.state["timers"] if datetime.fromisoformat(t["due"]) <= now),
                         key=lambda t: t["due"])
            if not due:
                return []
            self.state["timers"] = [t for t in self.state["timers"] if t not in due]
            self._save()
        alerts = []
        for t in due:
            when = datetime.fromisoformat(t["due"])
            if missed_before and when < missed_before:
                alerts.append(f"While I was off, sir, your {t['kind']}"
                              + (f" for {t['label']}" if t["label"] else "") + f" was due {self._say(when)}.")
            elif t["kind"] == "timer":
                alerts.append(f"Sir, your {t['label'] + ' ' if t['label'] else ''}timer is done.")
            else:
                alerts.append(f"Sir, a reminder: {t['label']}." if t["label"] else "Sir, your reminder is due.")
        events.publish("card", **self.card())
        for text in alerts:
            log.info("timer alert")
            if self._on_due:
                try:
                    self._on_due(text)
                except Exception:
                    log.exception("Timer alert failed")
        return alerts

    def card(self) -> dict:
        with self._lock:
            items = sorted(self.state["timers"], key=lambda t: t["due"])[:4]
        return {"id": "timers", "items": [{"label": t["label"] or t["kind"], "kind": t["kind"], "due": t["due"]}
                                          for t in items]}
