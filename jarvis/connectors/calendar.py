"""Google Calendar: read every account's calendars, and add, move or delete events after a spoken yes.

Reading covers every calendar of every account with "calendar" in its services
(plus subscribed ones such as UK holidays), merged into one list.

Changing is a two-step affair, and the second step is enforced here, not left
to the model:
  1. calendar_change proposes an add, move or delete and changes nothing; the
     model reads it back and asks.
  2. calendar_change with action confirm carries it out only if the proposal was made in the
     previous turn and the user's own words this turn are a clear yes
     (connectors.base.is_clear_yes on the transcribed speech; text from emails
     or web pages can never count).
Only accounts with "calendar_write" can be changed, and only events Jarvis
created himself (tagged jarvis=1) can be moved or deleted.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import Tool, ToolError, current_turn, is_clear_yes, untrusted
from .google import Clients, GoogleAccount, GoogleAuth, NeedsLogin

log = logging.getLogger(__name__)

CARD_EVENTS = 10   # events sent to the HUD card (the page shows the first few until the card is brought forward)

TZ = LOCAL.key
MAX_DAYS = 31
CALENDAR_LIST_S = 3600  # which calendars an account has rarely changes


class CalendarClient:
    """The Calendar API calls Jarvis makes. Tests replace this with a fake."""

    def __init__(self, creds):
        from googleapiclient.discovery import build

        self._svc = build("calendar", "v3", credentials=creds, cache_discovery=False)

    def calendars(self) -> list[dict]:
        return self._svc.calendarList().list().execute().get("items", [])

    def events(self, cal_id: str, start: datetime, end: datetime) -> list[dict]:
        return self._svc.events().list(calendarId=cal_id, timeMin=start.isoformat(), timeMax=end.isoformat(),
                                       singleEvents=True, orderBy="startTime", maxResults=250).execute().get("items", [])

    def insert(self, cal_id: str, body: dict) -> dict:
        return self._svc.events().insert(calendarId=cal_id, body=body).execute()

    def patch(self, cal_id: str, event_id: str, body: dict) -> dict:
        return self._svc.events().patch(calendarId=cal_id, eventId=event_id, body=body).execute()

    def delete(self, cal_id: str, event_id: str) -> None:
        self._svc.events().delete(calendarId=cal_id, eventId=event_id).execute()


@dataclass
class Event:
    account: str
    cal_id: str
    cal_name: str
    id: str
    title: str
    start: datetime | date
    end: datetime | date
    location: str = ""
    mine: bool = False   # created by Jarvis: may be moved or deleted
    uid: str = ""

    @property
    def all_day(self) -> bool:
        return not isinstance(self.start, datetime)


@dataclass
class Proposal:
    action: str                      # "add", "move" or "delete"
    turn: int
    text: str                        # the read-back
    account: str
    cal_id: str = "primary"
    event_id: str = ""
    body: dict = field(default_factory=dict)
    title: str = ""


def _parse_when(value: str) -> datetime | date:
    """'2026-10-11' -> a date (all day); '2026-10-11T09:00' -> a local time."""
    value = str(value).strip()
    try:
        if "T" in value or " " in value:
            d = datetime.fromisoformat(value.replace(" ", "T"))
            return d.replace(tzinfo=LOCAL) if d.tzinfo is None else d.astimezone(LOCAL)
        return date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"{value!r} isn't a date or time; use YYYY-MM-DD or YYYY-MM-DDTHH:MM.") from None


def _api_time(v: datetime | date) -> dict:
    return {"dateTime": v.isoformat(), "timeZone": TZ} if isinstance(v, datetime) else {"date": v.isoformat()}


def _from_api(t: dict) -> datetime | date:
    if "dateTime" in t:
        return datetime.fromisoformat(t["dateTime"].replace("Z", "+00:00")).astimezone(LOCAL)
    return date.fromisoformat(t["date"])


def say_when(start: datetime | date, end: datetime | date, today: date) -> str:
    def day(d: date) -> str:
        if d == today:
            return "today"
        if d == today + timedelta(days=1):
            return "tomorrow"
        return spoken(d, "%A %d %B")

    if not isinstance(start, datetime):
        last = end - timedelta(days=1)  # Google's all-day end date is exclusive
        return f"{day(start)}, all day" if last <= start else f"{day(start)} to {day(last)}, all day"
    if start.date() == end.date():
        return f"{day(start.date())}, {start:%H:%M} to {end:%H:%M}"
    return f"{day(start.date())} {start:%H:%M} to {day(end.date())} {end:%H:%M}"


class CalendarConnector:
    name = "calendar"

    def __init__(self, accounts: list[GoogleAccount], auth: GoogleAuth, *, hud_card: bool = True,
                 card_minutes: float = 10, client_factory: Callable | None = None,
                 now: Callable[[], datetime] | None = None, clock: Callable[[], float] = time.monotonic):
        self.accounts = {a.name: a for a in accounts if "calendar" in a.services}
        self.writable = [a.name for a in accounts if "calendar_write" in a.services]
        self.auth = auth
        self.hud_card = hud_card
        self.card_every = card_minutes * 60
        self._clients = Clients(auth, client_factory or CalendarClient)
        self._now = now or (lambda: datetime.now(LOCAL))
        self._clock = clock
        self._lock = threading.Lock()
        self.pending: Proposal | None = None
        self.refs: dict[str, Event] = {}   # short refs ("e1") for Jarvis's own events the model has seen
        self.last_ref: str | None = None
        self._refs_made = 0  # refs are never reused, even after a delete
        self._card_at = float("-inf")
        self._card_day: date | None = None
        self._lists: dict[str, tuple[float, list[dict]]] = {}  # account -> (fetched at, its calendars)

    # ------------------------------------------------------------------ reading

    def _client(self, account: GoogleAccount):
        return self._clients.get(account)

    def _account_events(self, account: GoogleAccount, start: datetime, end: datetime) -> list[Event]:
        client = self._client(account)
        out = []
        for cal in self._calendar_list(account, client):
            for e in client.events(cal["id"], start, end):
                if e.get("status") == "cancelled" or "start" not in e:
                    continue
                private = (e.get("extendedProperties") or {}).get("private") or {}
                out.append(Event(account=account.name, cal_id=cal["id"],
                                 cal_name="holidays" if "holiday" in cal["id"] else account.name,
                                 id=e["id"], title=" ".join((e.get("summary") or "(no title)").split()),
                                 start=_from_api(e["start"]), end=_from_api(e.get("end", e["start"])),
                                 location=" ".join((e.get("location") or "").split()),
                                 mine=private.get("jarvis") == "1", uid=e.get("iCalUID", e["id"])))
        return out

    def _calendar_list(self, account: GoogleAccount, client) -> list[dict]:
        hit = self._lists.get(account.name)
        if hit and self._clock() - hit[0] < CALENDAR_LIST_S:
            return hit[1]
        cals = client.calendars()
        self._lists[account.name] = (self._clock(), cals)
        return cals

    def events_between(self, start: datetime, end: datetime) -> tuple[list[Event], list[str]]:
        """Every account's events, merged, sorted, de-duplicated; plus problems to mention."""
        problems: list[str] = []

        def one(a: GoogleAccount):
            try:
                return self._account_events(a, start, end)
            except ToolError as e:
                problems.append(f"{a.name}: {e}")
            except Exception as e:
                log.warning("calendar %s failed: %s", a.name, type(e).__name__)
                problems.append(f"{a.name}: Google Calendar didn't respond")
            return []

        with ThreadPoolExecutor(max_workers=max(1, len(self.accounts))) as pool:
            found = [e for batch in pool.map(one, self.accounts.values()) for e in batch]
        seen, out = set(), []
        for e in sorted(found, key=lambda e: (self._sort_key(e.start), e.title)):
            key = (e.uid, str(e.start))
            if key not in seen:  # the same invite on two accounts
                seen.add(key)
                out.append(e)
        return out, problems

    @staticmethod
    def _sort_key(v: datetime | date) -> datetime:
        return v if isinstance(v, datetime) else datetime(v.year, v.month, v.day, tzinfo=LOCAL)

    def _ref(self, e: Event) -> str:
        for r, known in self.refs.items():
            if known.id == e.id:
                self.refs[r] = e
                return r
        self._refs_made += 1
        r = f"e{self._refs_made}"
        self.refs[r] = e
        return r

    # ------------------------------------------------------------------ the tools

    def tools(self) -> list[Tool]:
        if not self.accounts:
            return []
        read = ToolSpec(
            name="calendar",
            description="His calendars (every account plus UK holidays), merged. Events Jarvis added carry a ref like [e3].",
            parameters={"type": "object", "properties": {
                "start": {"type": "string", "description": "YYYY-MM-DD; default today"},
                "days": {"type": "integer", "minimum": 1, "maximum": MAX_DAYS, "description": "default 1"},
                "search": {"type": "string", "description": "title contains"}}})
        tools = [Tool(spec=read, run=self.calendar, notice="Checking the calendar", focus="calendar", reads_outside=True)]
        if self.writable:
            change = ToolSpec(
                name="calendar_change",
                description=(
                    "Add, move or delete an event in two steps: propose it (changes nothing; read it back and ask), "
                    "then action confirm only after he says yes in his next reply. Calendars: "
                    f"{', '.join(self.writable)} (default personal; the small side-business shop, e.g. markets, "
                    "stalls, orders, stock, goes on shop). Only events with a ref can be moved or deleted."),
                parameters={"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["add", "move", "delete", "confirm"]},
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "YYYY-MM-DDTHH:MM local time, or YYYY-MM-DD all day"},
                    "end": {"type": "string", "description": "optional; default 1 hour or 1 day"},
                    "calendar": {"type": "string", "enum": self.writable},
                    "location": {"type": "string"},
                    "ref": {"type": "string", "description": "e.g. e3; omit for the one just added"}},
                    "required": ["action"]})
            tools.append(Tool(spec=change, run=self._change_or_confirm, notice="Updating the calendar", focus="calendar"))
        return tools

    def _change_or_confirm(self, args: dict) -> str:
        if args.get("action") == "confirm":
            return self.calendar_confirm(args)
        return self.calendar_change(args)

    def calendar(self, args: dict) -> str:
        today = self._now().date()
        first = today
        if args.get("start"):
            when = _parse_when(args["start"])
            first = when.date() if isinstance(when, datetime) else when
        try:
            days = max(1, min(MAX_DAYS, int(args.get("days") or 1)))
        except (TypeError, ValueError):
            days = 1
        start = datetime(first.year, first.month, first.day, tzinfo=LOCAL)
        evs, problems = self.events_between(start, start + timedelta(days=days))
        search = str(args.get("search") or "").lower().strip()
        if search:
            evs = [e for e in evs if search in e.title.lower()]
        span = say_when(first, first + timedelta(days=days), today).replace(", all day", "")
        lines = [f"Calendar, {span}: " + (f"{len(evs)} event{'s' if len(evs) != 1 else ''}." if evs else "nothing.")]
        with self._lock:
            for e in evs[:40]:
                ref = f" [{self._ref(e)}]" if e.mine else ""
                where = f" at {e.location}" if e.location else ""
                lines.append(f"- {say_when(e.start, e.end, today)}: {e.title}{where} ({e.cal_name}){ref}")
        if problems:
            lines.append("Couldn't read: " + "; ".join(problems))
        return untrusted("Google Calendar", "\n".join(lines))

    def calendar_change(self, args: dict) -> str:
        action = args.get("action")
        turn = current_turn().number
        today = self._now().date()
        if action == "add":
            title = " ".join(str(args.get("title") or "").split())[:200]
            if not title:
                raise ToolError("an event needs a title.")
            cal = args.get("calendar") or ("personal" if "personal" in self.writable else self.writable[0])
            account = self._writable_account(cal)
            start = _parse_when(args.get("start") or "")
            end = _parse_when(args["end"]) if args.get("end") else (
                start + timedelta(hours=1) if isinstance(start, datetime) else start + timedelta(days=1))
            if isinstance(start, datetime) != isinstance(end, datetime):
                raise ToolError("start and end must both be times, or both be dates.")
            if not isinstance(start, datetime) and args.get("end"):
                end = end + timedelta(days=1)  # "Saturday to Sunday" means through Sunday
            if end <= start:
                raise ToolError("the end must be after the start.")
            body = {"summary": title, "start": _api_time(start), "end": _api_time(end),
                    "description": "Added by Jarvis.", "extendedProperties": {"private": {"jarvis": "1"}}}
            if args.get("location"):
                body["location"] = " ".join(str(args["location"]).split())[:200]
            where = f" at {body['location']}" if "location" in body else ""
            text = f"add '{title}'{where}, {say_when(start, end, today)}, to the {cal} calendar"
            proposal = Proposal("add", turn, text, account.name, body=body, title=title)
        elif action in ("move", "delete"):
            ref = args.get("ref") or self.last_ref
            with self._lock:
                e = self.refs.get(str(ref)) if ref else None
            if e is None or not e.mine:
                raise ToolError("only events Jarvis added can be moved or deleted, and I need one I've listed "
                                "(its ref, e.g. e3); check the calendar first.")
            if e.account not in self.writable:
                raise ToolError(f"the {e.account} calendar can't be changed.")
            if action == "delete":
                text = f"delete '{e.title}', {say_when(e.start, e.end, today)}, from the {e.cal_name} calendar"
                proposal = Proposal("delete", turn, text, e.account, e.cal_id, e.id, title=e.title)
            else:
                if not args.get("start"):
                    raise ToolError("say when to move it to.")
                start = _parse_when(args["start"])
                length = (e.end - e.start) if type(e.start) is type(start) else (
                    timedelta(hours=1) if isinstance(start, datetime) else timedelta(days=1))
                end = _parse_when(args["end"]) if args.get("end") else start + length
                if isinstance(start, datetime) != isinstance(end, datetime):
                    raise ToolError("start and end must both be times, or both be dates.")
                if not isinstance(start, datetime) and args.get("end"):
                    end = end + timedelta(days=1)  # as for add: "to Sunday" means through Sunday
                if end <= start:
                    raise ToolError("the end must be after the start.")
                body = {"start": _api_time(start), "end": _api_time(end)}
                text = f"move '{e.title}' to {say_when(start, end, today)}"
                proposal = Proposal("move", turn, text, e.account, e.cal_id, e.id, body=body, title=e.title)
        else:
            raise ToolError("action must be add, move or delete.")
        with self._lock:
            self.pending = proposal
        return (f"Proposed, NOT done yet: {proposal.text}. Read this back and ask him to confirm; "
                "it happens only if he says yes in his next reply.")

    def calendar_confirm(self, args: dict) -> str:
        turn = current_turn()
        with self._lock:
            p = self.pending
            if p is None:
                raise ToolError("there's no calendar change waiting to be confirmed.")
            if turn.number != p.turn + 1:
                if turn.number == p.turn:
                    raise ToolError("the user hasn't answered yet: ask them, and confirm in the next turn.")
                self.pending = None
                raise ToolError("that proposal has lapsed; propose it again if they still want it.")
            if not is_clear_yes(turn.user_text):
                self.pending = None
                raise ToolError("the user didn't clearly say yes, so nothing was changed. Ask again if needed.")
            self.pending = None
        account = self.accounts[p.account]
        client = self._client(account)
        try:
            if p.action == "add":
                created = client.insert(p.cal_id, p.body)
                e = Event(account=p.account, cal_id=p.cal_id, cal_name=p.account, id=created["id"], title=p.title,
                          start=_from_api(p.body["start"]), end=_from_api(p.body["end"]), mine=True,
                          location=p.body.get("location", ""), uid=created.get("iCalUID", created["id"]))
                with self._lock:
                    self.last_ref = self._ref(e)
            elif p.action == "move":
                client.patch(p.cal_id, p.event_id, p.body)
                with self._lock:
                    for r, known in self.refs.items():
                        if known.id == p.event_id:
                            known.start, known.end = _from_api(p.body["start"]), _from_api(p.body["end"])
                            self.last_ref = r
            else:
                client.delete(p.cal_id, p.event_id)
                with self._lock:
                    self.refs = {r: e for r, e in self.refs.items() if e.id != p.event_id}
                    self.last_ref = None
        except ToolError:
            raise
        except Exception as e:
            status = getattr(getattr(e, "resp", None), "status", None)
            log.warning("calendar %s failed: %s %s", p.action, type(e).__name__, status)
            if status == 403:
                raise NeedsLogin(f"the {p.account} account needs signing in again to allow calendar changes "
                                 f"(run: python -m jarvis --google-login {p.account}).") from e
            raise ToolError("Google Calendar refused the change; nothing was done.") from e
        log.info("calendar %s done", p.action)
        self._card_at = float("-inf")  # refresh the HUD card at the next tick
        return f"Done: {p.text.replace('add ', 'added ', 1).replace('move ', 'moved ', 1).replace('delete ', 'deleted ', 1)}."

    def _writable_account(self, name: str) -> GoogleAccount:
        if name not in self.writable or name not in self.accounts:
            raise ToolError(f"the {name} calendar can't be changed; use one of: {', '.join(self.writable)}.")
        account = self.accounts[name]
        if not self.auth.has_scope(account, "calendar_write"):
            raise NeedsLogin(f"the {name} account needs signing in again to allow adding events "
                             f"(run: python -m jarvis --google-login {name}).")
        return account

    # ------------------------------------------------------------------ the waiting proposal and the HUD

    def pending_note(self) -> str | None:
        """Reminds the model, on the next turn only, that a change is waiting for a yes."""
        with self._lock:
            p = self.pending
        if p is None or current_turn().number != p.turn + 1:
            return None
        return (f"A calendar change is waiting for the user's answer: {p.text}. If they've just said a plain yes, "
                "call calendar_change with action confirm. If they want it different (another time, title or calendar), call "
                "calendar_change again with the new details and ask again; never say a change is prepared "
                "without calling it. Otherwise leave it.")

    def tick(self) -> None:
        if not self.hud_card or not events.watched():
            return
        day = self._now().date()
        if day == self._card_day and self._clock() - self._card_at < self.card_every:
            return
        self._card_day, self._card_at = day, self._clock()
        events.publish("card", **self.card())

    def card(self) -> dict:
        """Today's remaining events, or tomorrow's if today's are done. Screen only."""
        now = self._now()
        today = now.date()
        start = datetime(today.year, today.month, today.day, tzinfo=LOCAL)
        try:
            evs, problems = self.events_between(start, start + timedelta(days=2))
        except Exception as e:
            log.warning("calendar card failed: %s", type(e).__name__)
            return {"id": "calendar", "error": "UNAVAILABLE"}
        left = [e for e in evs if self._sort_key(e.end) > now
                and (e.start if not isinstance(e.start, datetime) else e.start.date()) <= today]
        label = "TODAY"
        if not left:
            tomorrow = today + timedelta(days=1)
            left = [e for e in evs if (e.start if not isinstance(e.start, datetime) else e.start.date()) == tomorrow]
            label = "TOMORROW"
        items = [{"time": "ALL DAY" if e.all_day else f"{e.start:%H:%M}", "title": e.title, "cal": e.cal_name,
                  "start": e.start.isoformat() if isinstance(e.start, datetime) else None} for e in left[:CARD_EVENTS]]
        return {"id": "calendar", "label": label, "items": items, "more": max(0, len(left) - CARD_EVENTS),
                "error": "SIGN-IN NEEDED" if problems and not evs else None}
