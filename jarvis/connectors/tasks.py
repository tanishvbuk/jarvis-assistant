"""The to-do list, in Google Tasks (so it's on the user's phone too).

"Add to my list: pitch the cosy games feature, due Friday" adds straight away and
is read back; "undo that" reverses the last add, tick-off or move. Deleting a
task needs a spoken yes in the next turn (as calendar changes do).

Google Tasks keeps only the date of a deadline, so a time ("by 5pm Friday") is
written into the task's notes and a spoken reminder is set an hour before
through the timers skill.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import DailyNote, Tool, ToolError, current_turn, is_clear_yes, untrusted
from .google import Clients, GoogleAccount, GoogleAuth, NeedsLogin

log = logging.getLogger(__name__)

CARD_TASKS = 12   # tasks sent to the HUD card (the page shows the first few until the card is brought forward)

LIST = "@default"  # "My Tasks"
TIME_NOTE = re.compile(r"Deadline: (\d{2}:\d{2})")
REMIND_BEFORE = timedelta(hours=1)


class TasksClient:
    """The Tasks API calls Jarvis makes. Tests replace this with a fake."""

    def __init__(self, creds):
        from googleapiclient.discovery import build

        self._svc = build("tasks", "v1", credentials=creds, cache_discovery=False)

    def open_tasks(self) -> list[dict]:
        out, token = [], None
        while True:
            r = self._svc.tasks().list(tasklist=LIST, showCompleted=False, maxResults=100, pageToken=token).execute()
            out += r.get("items", [])
            token = r.get("nextPageToken")
            if not token:
                return out

    def insert(self, body: dict) -> dict:
        return self._svc.tasks().insert(tasklist=LIST, body=body).execute()

    def patch(self, task_id: str, body: dict) -> dict:
        return self._svc.tasks().patch(tasklist=LIST, task=task_id, body=body).execute()

    def delete(self, task_id: str) -> None:
        self._svc.tasks().delete(tasklist=LIST, task=task_id).execute()


@dataclass
class Task:
    id: str
    title: str
    due: date | None
    time: str = ""   # "17:00" if a time was given
    notes: str = ""


def _task(raw: dict) -> Task:
    notes = raw.get("notes") or ""
    m = TIME_NOTE.search(notes)
    due = date.fromisoformat(raw["due"][:10]) if raw.get("due") else None
    return Task(id=raw["id"], title=" ".join((raw.get("title") or "").split()), due=due,
                time=m.group(1) if m else "", notes=notes)


def _parse_due(value: str) -> tuple[date, str]:
    """'2026-10-16' -> (date, ''); '2026-10-16T17:00' -> (date, '17:00')."""
    value = str(value).strip()
    try:
        if "T" in value or " " in value:
            d = datetime.fromisoformat(value.replace(" ", "T"))
            return d.date(), d.strftime("%H:%M")
        return date.fromisoformat(value), ""
    except ValueError:
        raise ToolError(f"{value!r} isn't a date; use YYYY-MM-DD, or YYYY-MM-DDTHH:MM for a time.") from None


def _api_due(d: date) -> str:
    return f"{d.isoformat()}T00:00:00.000Z"  # Google Tasks stores the date only


class TasksConnector:
    name = "tasks"

    def __init__(self, account: GoogleAccount, auth: GoogleAuth, *, timers=None, state_file: Path | None = None,
                 hud_card: bool = True, client_factory: Callable | None = None,
                 now: Callable[[], datetime] | None = None, clock: Callable[[], float] = time.monotonic):
        self.account = account
        self.auth = auth
        self.timers = timers            # the timers skill, for reminders before timed deadlines
        self.hud_card = hud_card
        self._clients = Clients(auth, client_factory or TasksClient)
        self._now = now or (lambda: datetime.now(LOCAL))
        self._clock = clock
        self._lock = threading.Lock()
        self.undo: tuple | None = None   # ("added", task) / ("done", task) / ("moved", task_before)
        self.pending_delete: tuple[int, Task] | None = None
        self.last: Task | None = None    # "that"
        self._card_at = float("-inf")
        self._due_note = DailyNote(state_file, "day", self._due_today, lambda: self._now().date(), clock)

    # ---- plumbing

    def client(self):
        if not self.auth.has_scope(self.account, "tasks"):
            raise NeedsLogin(f"the {self.account.name} account needs signing in again for the to-do list "
                             f"(run: python -m jarvis --google-login {self.account.name}).")
        return self._clients.get(self.account)

    def _api(self, fn, *args):
        try:
            return fn(*args)
        except ToolError:
            raise
        except Exception as e:
            status = getattr(getattr(e, "resp", None), "status", None)
            log.warning("tasks call failed: %s %s", type(e).__name__, status)
            if status == 403 and "has not been used" in str(e):
                raise ToolError("the Google Tasks API isn't switched on in the Google Cloud project yet.") from e
            if status == 403:
                raise NeedsLogin(f"the {self.account.name} account needs signing in again for the to-do list.") from e
            raise ToolError("Google Tasks didn't respond properly; nothing was changed.") from e

    def open_tasks(self) -> list[Task]:
        tasks = [_task(t) for t in self._api(self.client().open_tasks)]
        far = date.max
        return sorted(tasks, key=lambda t: (t.due or far, t.time or "99:99", t.title.lower()))

    def find(self, what: str | None) -> Task:
        """A task by (part of) its title, or "that" (the last one mentioned)."""
        if not what or what.lower().strip() in ("that", "it", "this", "the last one"):
            if self.last:
                return self.last
            raise ToolError("say which task.")
        want = what.lower().strip()
        tasks = self.open_tasks()
        exact = [t for t in tasks if t.title.lower() == want]
        hits = exact or [t for t in tasks if want in t.title.lower()]
        if not hits:
            words = [w for w in re.findall(r"\w+", want) if len(w) > 2]
            hits = [t for t in tasks if words and all(w in t.title.lower() for w in words)]
        if not hits:
            raise ToolError(f"no open task matches {what!r}.")
        if len(hits) > 1:
            raise ToolError("more than one task matches: " + "; ".join(t.title for t in hits[:5]) + ". Which one?")
        return hits[0]

    def _say_due(self, t: Task) -> str:
        if not t.due:
            return "no deadline"
        today = self._now().date()
        if t.due == today:
            day = "today"
        elif t.due == today + timedelta(days=1):
            day = "tomorrow"
        elif t.due < today:
            day = f"overdue since {t.due.strftime('%A %d %B').replace(' 0', ' ')}"
        else:
            day = spoken(t.due, "%A %d %B")
        return f"due {day}" + (f" at {t.time}" if t.time else "")

    def _remind(self, t: Task) -> None:
        """A spoken reminder an hour before a timed deadline (if the timers skill is there and it's ahead)."""
        if not (self.timers and t.due and t.time):
            return
        at = datetime.combine(t.due, datetime.strptime(t.time, "%H:%M").time(), LOCAL) - REMIND_BEFORE
        if at > self._now():
            try:
                self.timers.timer({"action": "set", "at": at.strftime("%Y-%m-%dT%H:%M"),
                                   "label": f"{t.title} is due at {t.time}", "key": f"task:{t.id}"})
            except ToolError as e:
                log.info("task reminder not set: %s", e)

    def _unremind(self, t: Task) -> None:
        if self.timers:
            self.timers.cancel_key(f"task:{t.id}", f"{t.title} is due at {t.time}" if t.time else "")

    def _changed(self) -> None:
        """Refresh the HUD card now, off the turn's thread (it costs one more API call)."""
        if not (self.hud_card and events.watched()):
            self._card_at = float("-inf")  # stale: refresh as soon as a page is back on screen
            return
        self._card_at = self._clock()
        threading.Thread(target=lambda: events.publish("card", **self.card()), daemon=True).start()

    # ---- the tool

    def tools(self) -> list[Tool]:
        spec = ToolSpec(
            name="todo",
            description=(
                "His to-do list (Google Tasks, also on his phone). add (done at once; read it back), list "
                "(soonest first), done, move (new deadline), undo (the last add, done or move), delete (only "
                "proposes; confirm_delete after he says yes in his next reply). task: part of its title; omit "
                "for the one just mentioned."),
            parameters={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["add", "list", "done", "move", "undo", "delete", "confirm_delete"]},
                "title": {"type": "string", "description": "in his words, tidied"},
                "due": {"type": "string", "description": "YYYY-MM-DD, or YYYY-MM-DDTHH:MM if he gave a time"},
                "task": {"type": "string"},
                "scope": {"type": "string", "enum": ["all", "week", "overdue"]}},
                "required": ["action"]})
        return [Tool(spec=spec, run=self.todo, notice="Updating the list", focus="tasks",
                     writes=lambda a: a.get("action") != "list")]

    def todo(self, args: dict) -> str:
        action = args.get("action")
        handler = {"add": self._add, "list": self._list, "done": self._done, "move": self._move,
                   "undo": self._undo, "delete": self._delete, "confirm_delete": self._confirm_delete}.get(action)
        if handler is None:
            raise ToolError("action must be add, list, done, move, undo, delete or confirm_delete.")
        return handler(args)

    def _add(self, args: dict) -> str:
        title = " ".join(str(args.get("title") or "").split())[:200]
        if not title:
            raise ToolError("what's the task?")
        body: dict = {"title": title, "notes": "Added by Jarvis."}
        due, at = (None, "")
        if args.get("due"):
            due, at = _parse_due(args["due"])
            body["due"] = _api_due(due)
            if at:
                body["notes"] = f"Deadline: {at}\nAdded by Jarvis."
        t = _task(self._api(self.client().insert, body))
        with self._lock:
            self.undo, self.last = ("added", t), t
        self._remind(t)
        self._changed()
        return f"Added: {t.title}, {self._say_due(t)}. (He can say 'undo that'.)"

    def _list(self, args: dict) -> str:
        tasks = self.open_tasks()
        today = self._now().date()
        scope = args.get("scope") or "all"
        if scope == "week":
            tasks = [t for t in tasks if t.due and t.due <= today + timedelta(days=7)]
        elif scope == "overdue":
            tasks = [t for t in tasks if t.due and t.due < today]
        if not tasks:
            return {"week": "Nothing due in the next week.", "overdue": "Nothing overdue."}.get(scope, "The list is empty.")
        lines = [f"{len(tasks)} open task{'s' if len(tasks) != 1 else ''}" + {"week": " due within a week",
                                                                             "overdue": " overdue"}.get(scope, "") + ":"]
        lines += [f"- {t.title} ({self._say_due(t)})" for t in tasks[:25]]
        if len(tasks) > 25:
            lines.append(f"(and {len(tasks) - 25} more)")
        return untrusted("Google Tasks", "\n".join(lines))  # titles may have been typed on the phone

    def _done(self, args: dict) -> str:
        t = self.find(args.get("task"))
        self._api(self.client().patch, t.id, {"status": "completed"})
        with self._lock:
            self.undo, self.last = ("done", t), t
        self._unremind(t)
        self._changed()
        return f"Ticked off: {t.title}."

    def _move(self, args: dict) -> str:
        t = self.find(args.get("task"))
        if not args.get("due"):
            raise ToolError("move it to when?")
        due, at = _parse_due(args["due"])
        notes = TIME_NOTE.sub("", t.notes).strip()
        if at:
            notes = f"Deadline: {at}\n{notes}".strip()
        self._api(self.client().patch, t.id, {"due": _api_due(due), "notes": notes})
        moved = Task(t.id, t.title, due, at, notes)
        with self._lock:
            self.undo, self.last = ("moved", t), moved
        self._unremind(t)
        self._remind(moved)
        self._changed()
        return f"Moved: {t.title}, now {self._say_due(moved)}."

    def _undo(self, args: dict) -> str:
        with self._lock:
            undo, self.undo = self.undo, None
        if undo is None:
            raise ToolError("there's nothing to undo.")
        kind, t = undo
        client = self.client()
        if kind == "added":
            self._api(client.delete, t.id)
            self._unremind(t)
            self.last = None
            msg = f"Undone: removed {t.title}."
        elif kind == "done":
            self._api(client.patch, t.id, {"status": "needsAction"})
            self._remind(t)
            self.last = t
            msg = f"Undone: {t.title} is back on the list."
        else:
            self._api(client.patch, t.id, {"due": _api_due(t.due) if t.due else None, "notes": t.notes})
            self._unremind(t)  # the moved deadline's reminder
            self._remind(t)    # and the original one back
            self.last = t
            msg = f"Undone: {t.title} is back to {self._say_due(t)}."
        self._changed()
        return msg

    def _delete(self, args: dict) -> str:
        t = self.find(args.get("task"))
        with self._lock:
            self.pending_delete = (current_turn().number, t)
        return (f"Proposed, NOT done yet: delete '{t.title}' ({self._say_due(t)}). Ask him to confirm; it's deleted "
                "only if he says yes in his next reply.")

    def _confirm_delete(self, args: dict) -> str:
        turn = current_turn()
        with self._lock:
            pending, self.pending_delete = self.pending_delete, None
        if pending is None:
            raise ToolError("there's no deletion waiting to be confirmed.")
        made_at, t = pending
        if turn.number == made_at:
            with self._lock:
                self.pending_delete = pending
            raise ToolError("he hasn't answered yet: ask him, and confirm in the next turn.")
        if turn.number != made_at + 1 or not is_clear_yes(turn.user_text):
            raise ToolError("he didn't clearly say yes straight away, so nothing was deleted.")
        self._api(self.client().delete, t.id)
        self._unremind(t)
        with self._lock:
            self.last, self.undo = None, None
        self._changed()
        return f"Deleted: {t.title}."

    # ---- once a day, the waiting deletion, and the HUD

    def daily_note(self) -> str | None:
        """Once a day: what's due today or overdue (looked up in the background)."""
        return self._due_note.take()

    def _due_today(self) -> str | None:
        today = self._now().date()
        due = [t for t in self.open_tasks() if t.due and t.due <= today]
        if not due:
            return None
        listed = "; ".join(f"{t.title} ({self._say_due(t)})" for t in due[:5])
        return (f"From his to-do list, due today or overdue: {listed}. Mention this briefly, in a sentence.")

    def pending_note(self) -> str | None:
        with self._lock:
            pending = self.pending_delete
        if pending is None or current_turn().number != pending[0] + 1:
            return None
        return (f"Waiting for his answer: delete the task '{pending[1].title}'? If he's just said a plain yes, call "
                "todo with action confirm_delete; otherwise don't.")

    def tick(self) -> None:
        self._due_note.prepare()
        if not self.hud_card or not events.watched() or self._clock() - self._card_at < 600:
            return
        self._card_at = self._clock()
        events.publish("card", **self.card())

    def card(self) -> dict:
        try:
            tasks = self.open_tasks()
        except ToolError as e:
            return {"id": "tasks", "items": [], "error": "SIGN-IN NEEDED" if isinstance(e, NeedsLogin) else "UNAVAILABLE"}
        today = self._now().date()
        items = []
        for t in tasks[:CARD_TASKS]:
            if not t.due:
                when = ""
            elif t.due < today:
                when = "OVERDUE"
            elif t.due == today:
                when = t.time or "TODAY"
            elif t.due == today + timedelta(days=1):
                when = "TOMORROW"
            else:
                when = spoken(t.due, "%a %d").upper()
            items.append({"title": t.title, "when": when, "late": bool(t.due and t.due < today)})
        return {"id": "tasks", "items": items, "more": max(0, len(tasks) - CARD_TASKS), "error": None}
