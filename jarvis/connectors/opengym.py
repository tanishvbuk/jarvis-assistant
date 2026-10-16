"""openGym: his self-hosted gym tracker (plan, sessions, body weight), through its own web API.

Reads his weekly plan, logged sessions and weigh-ins. The one thing it writes is
body weight: a weigh-in he says out loud, and the weigh-ins the meal-plan sheet logs in
the meal plan sheet's tracker tab, copied across every few hours. It never
touches his plan or sessions.

openGym keeps the profile as one document with a revision number. Every save
quotes the revision Jarvis read (baseRev); if the phone saved in between, the
server refuses it (409) and sends the current copy, and Jarvis makes its change
again on that and retries (see OpenGymConnector._update). The server also stamps
what each save changed, so the phone merges Jarvis's weigh-in in rather than
saving over it, and an "undo" is a deletion the phone respects. Sheet weigh-ins
are copied once each: one he deletes in the app stays deleted.

Jarvis signs in like the phone app: in openGym, Settings > Account > "Pair the
mobile app" shows a five-minute code, and `python -m jarvis --opengym-pair CODE`
swaps it for a bearer token (saved in tokens/opengym.json). Once the token is past
half its life, GET /api/me hands back a fresh one (checked daily in tick()).
"Sign out everywhere" in the app ends it. Settings: [opengym] in config.toml; the
exercise catalogue path is in config.local.toml.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import DailyNote, Tool, ToolError, read_state, untrusted, write_state
from .web import request_json

log = logging.getLogger(__name__)

LB = 0.45359237
LOGIN_HINT = ("in openGym: Settings > Account > Pair the mobile app, then run: "
              "python -m jarvis --opengym-pair CODE")
RETRIES = 3                 # saves refused because the phone saved first
RENEW_S = 86400             # how often the token's renewal is checked
CARD_S = 300                # how often the HUD's gym card is refreshed (while a page is on screen)
CARD_ITEMS = 12             # exercises sent to the card (the page shows the first few until the card is brought forward)
PACE_DAYS = 42              # the weigh-ins the pace is worked out from
GAIN_BAND = (0.25, 0.5)     # kg a week: a usual lean bulk
LOSS_BAND = (0.25, 0.75)    # kg a week: a usual steady cut


def short(d: date, year: bool = False) -> str:
    """ "8 Oct", or "8 Oct 2026"."""
    return f"{d.day} {d.strftime('%b')}" + (f" {d.year}" if year else "")


def ago(days: int) -> str:
    return {0: "today", 1: "yesterday"}.get(days, f"{days} days ago")


def fmt(x: float) -> str:
    """57.80 -> "57.8", 60.0 -> "60"."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ the API

def token_expiry(token: str) -> str:
    """The day a token runs out, from its signed payload (`<uid>:<expiry ms>:<version>.<mac>`), or ""."""
    try:
        ms = int(token.rsplit(".", 1)[0].split(":")[1])
    except (IndexError, ValueError):
        return ""
    return datetime.fromtimestamp(ms / 1000, LOCAL).date().isoformat()


class Conflict(Exception):
    """The phone saved since Jarvis read: the server's current copy and its revision."""

    def __init__(self, state: dict, rev: int):
        super().__init__("openGym revision conflict")
        self.state, self.rev = state, rev


class OpenGymClient:
    """openGym's web API with the paired bearer token. Tests pass a fake `request`."""

    def __init__(self, url: str, session_file: Path, request: Callable = request_json):
        self.url = url.rstrip("/")
        self.session_file = session_file
        self._request = request

    def session(self) -> dict:
        return read_state(self.session_file)

    def _call(self, method: str, path: str, body=None, *, signed_in: bool = True):
        headers = {}
        if signed_in:
            token = self.session().get("token")
            if not token:
                raise ToolError(f"Jarvis isn't paired with openGym yet ({LOGIN_HINT}).")
            headers["Authorization"] = f"Bearer {token}"
        try:
            return self._request(method, self.url + path, body, headers)
        except urllib.error.HTTPError as e:
            if e.code == 409 and path == "/api/data":
                try:
                    current = json.loads(e.read() or b"{}")
                except ValueError:
                    current = {}
                raise Conflict(current.get("state") or {}, int(current.get("rev") or 0)) from None
            log.warning("openGym %s %s failed: HTTP %s", method, path, e.code)
            if e.code == 401:
                raise ToolError(f"Jarvis's openGym sign-in has expired or was signed out ({LOGIN_HINT}).") from e
            if e.code == 400 and path == "/api/pair/redeem":
                raise ToolError("openGym didn't accept that code (they last five minutes and work once).") from e
            raise ToolError("openGym answered with an error; nothing was changed.") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("openGym %s %s failed: %s", method, path, type(e).__name__)
            raise ToolError("openGym isn't answering (is Docker Desktop running on the PC?).") from e

    def _keep(self, token: str, user: str) -> dict:
        session = {"token": token, "user": user, "expires": token_expiry(token)}
        write_state(self.session_file, session)
        return session

    def pair(self, code: str) -> dict:
        """Swap a pairing code from the app for a token, and save it."""
        answer = self._call("POST", "/api/pair/redeem", {"code": code.strip()}, signed_in=False) or {}
        if not answer.get("token"):
            raise ToolError("openGym's answer had no token in it.")
        return self._keep(answer["token"], (answer.get("user") or {}).get("name", ""))

    def renew(self) -> bool:
        """Ask /api/me, which hands back a fresh token once the old one is past half its life. True if renewed."""
        answer = self._call("GET", "/api/me") or {}
        if not answer.get("token"):
            return False
        self._keep(answer["token"], (answer.get("user") or {}).get("name", "") or self.session().get("user", ""))
        return True

    def get_state(self) -> tuple[dict, int]:
        """The profile document and its revision."""
        answer = self._call("GET", "/api/data") or {}
        return answer.get("state") or {}, int(answer.get("rev") or 0)

    def put_state(self, state: dict, rev: int) -> int:
        """Save over revision `rev`; the new revision. Raises Conflict if the phone saved since."""
        body = {"state": state, "baseRev": rev}
        if isinstance(state.get("_wid"), str):
            body["baseWid"] = state["_wid"]
        return int((self._call("PUT", "/api/data", body) or {}).get("rev") or 0)


# ------------------------------------------------------------------ reading his data

@dataclass
class Weigh:
    day: date
    w: float   # in his profile's unit


def weighs(state: dict) -> list[Weigh]:
    out = []
    for b in state.get("bodyweight") or []:
        try:
            out.append(Weigh(date.fromisoformat(str(b["d"])), float(b["w"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: x.day)


def pace(points: list[Weigh]) -> float | None:
    """Units a week over the last PACE_DAYS (least squares), or None if they span under two weeks."""
    if not points:
        return None
    recent = [p for p in points if (points[-1].day - p.day).days <= PACE_DAYS]
    if len(recent) < 2 or (recent[-1].day - recent[0].day).days < 14:
        return None
    xs = [(p.day - recent[0].day).days for p in recent]
    mx, my = sum(xs) / len(xs), sum(p.w for p in recent) / len(recent)
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (p.w - my) for x, p in zip(xs, recent)) / var * 7 if var else None


def set_text(s: dict, unit: str) -> str:
    if "min" in s:
        return f"{fmt(s.get('min') or 0)} min"
    if "sec" in s:
        return f"{int(s.get('sec') or 0)} s" + (f" with {fmt(s['w'])} {unit}" if s.get("w") else "")
    w = s.get("w") or 0
    return (f"{fmt(w)} {unit} x {s.get('r') or 0}") if w else f"{s.get('r') or 0} reps"


def top_set(sets: list[dict]) -> dict | None:
    done = [s for s in sets if s.get("done", True)]
    return max(done, key=lambda s: (s.get("w") or 0, s.get("r") or 0, s.get("sec") or 0, s.get("min") or 0),
               default=None)


def e1rm(s: dict) -> float | None:
    """Epley estimate, only where it means something (weighted sets of 1 to 12 reps)."""
    w, r = s.get("w") or 0, s.get("r") or 0
    return w * (1 + r / 30) if w > 0 and 1 <= r <= 12 else None


def _js_weekday(d: date) -> str:
    return str((d.weekday() + 1) % 7)  # openGym keys its week by JavaScript's getDay(): Sunday is 0


def routine_for(state: dict, day: date) -> dict | None:
    """The session planned for a day: a one-off change for that date first, else the weekly plan."""
    routines = {r.get("id"): r for r in state.get("routines") or []}
    over = (state.get("dayPlan") or {}).get(day.isoformat())
    if over == "rest":
        return None
    if over in routines:
        return routines[over]
    return routines.get((state.get("week") or {}).get(_js_weekday(day)))


def resolve_days(entries: list, today: date) -> list[tuple]:
    """Each sheet weigh-in with the date to file it under, or None if it fits nowhere.

    The log runs in order, one row per weigh-in. Where a date reads both ways (see WeighIn.alt),
    take the readings that keep every date in order and not in the future, with the whole log
    spanning as short a time as it can. A row that can't fit at all is left out (None)."""
    skip = 10 ** 6
    states: dict[date, tuple[int, list]] = {date.min: (0, [])}   # last date used -> (cost, choices so far)
    for e in entries:
        options = [d for d in dict.fromkeys((e.day, e.alt)) if d and d <= today]
        nxt: dict[date, tuple[int, list]] = {}
        for last, (cost, path) in states.items():
            moves = [(last, cost + skip, path + [None])]
            moves += [(d, cost + (0 if last == date.min else (d - last).days), path + [d]) for d in options if d >= last]
            for d, c, pth in moves:
                if d not in nxt or c < nxt[d][0]:
                    nxt[d] = (c, pth)
        states = nxt
    return list(zip(entries, min(states.values(), key=lambda v: v[0])[1]))


# ------------------------------------------------------------------ the connector

class OpenGymConnector:
    name = "opengym"

    def __init__(self, client: OpenGymClient, *, library: Path | None = None, data_dir: Path | None = None,
                 meal_plan=None, sheet_plan: str = "", sync_hours: float = 3, cache_seconds: float = 60,
                 hud_card: bool = True, now: Callable[[], datetime] | None = None, clock: Callable[[], float] = time.monotonic):
        self.client = client
        self.library = library
        self.meal_plan = meal_plan      # the meal plan connector, for the meal-plan sheet's weigh-ins
        self.sheet_plan = sheet_plan
        self.sync_s = sync_hours * 3600
        self.hud_card = hud_card
        self._card_at = float("-inf")
        self.cache_s = cache_seconds
        self._now = now or (lambda: datetime.now(LOCAL))
        self._clock = clock
        self._lock = threading.RLock()  # one read-modify-write at a time
        self._cache: tuple[float, dict] | None = None
        self._names: dict[str, str] | None = None
        self.undo: tuple[str, dict | None, dict] | None = None   # (day, entry it replaced, entry logged)
        self._news: list[str] = []
        self._sync_at = float("-inf")
        self._renew_at = float("-inf")
        self.sync_file = data_dir / "opengym_sync.json" if data_dir else None
        self.names_file = data_dir / "opengym_names.json" if data_dir else None
        self._nudge = DailyNote(data_dir / "opengym_notes.json" if data_dir else None, "nudge", self._nudge_text,
                                lambda: self._now().date(), clock)

    # ---- plumbing

    def state(self, fresh: bool = False) -> dict:
        with self._lock:
            if not fresh and self._cache and self._clock() - self._cache[0] < self.cache_s:
                return self._cache[1]
            s, _ = self.client.get_state()
            self._cache = (self._clock(), s)
            return s

    def _update(self, change: Callable[[dict], bool]) -> dict:
        """Read, `change` the copy (True if it changed anything), save over the revision read. If the phone saved
        first, make the change again on the server's current copy, up to RETRIES times. Returns the copy saved."""
        with self._lock:
            s, rev = self.client.get_state()
            for _ in range(RETRIES):
                if not change(s):
                    self._cache = (self._clock(), s)
                    return s
                try:
                    self.client.put_state(s, rev)
                except Conflict as c:
                    log.info("openGym: the phone saved first; making the change again on its copy")
                    s, rev = c.state, c.rev
                    continue
                self._cache = None   # the server may have merged in more than this copy holds
                return s
            raise ToolError("openGym kept changing while Jarvis saved (the phone syncing?); nothing was saved. Try again.")

    def _catalogue(self) -> dict[str, str]:
        """Exercise names from openGym's catalogue: one <id>.json per exercise with its English "name" (ids never
        change). Reading ~6,500 files cold can take most of a minute on Windows, so the names are kept in
        data/opengym_names.json and the folder is only read again when it changes (files added or replaced)."""
        if not self.library or not self.library.is_dir():
            if self.library:
                log.warning("openGym exercise catalogue not found: %s", self.library)
            return {}
        stamp = f"{self.library}|{self.library.stat().st_mtime_ns}"
        kept = read_state(self.names_file)
        if kept.get("stamp") == stamp and kept.get("names"):
            return kept["names"]
        names = {}
        for f in self.library.glob("*.json"):
            try:
                e = json.loads(f.read_text(encoding="utf-8"))
                if e.get("id") and e.get("name"):
                    names[str(e["id"])] = e["name"]
            except (OSError, ValueError, AttributeError) as e:
                log.debug("openGym catalogue entry unreadable (%s): %s", f.name, type(e).__name__)
        if names:
            write_state(self.names_file, {"stamp": stamp, "names": names})
        else:
            log.warning("openGym exercise catalogue is empty: %s", self.library)
        return names

    def names(self, state: dict) -> dict[str, str]:
        if self._names is None:
            self._names = self._catalogue()
        custom = {c.get("id"): c.get("n") for c in state.get("customEx") or [] if c.get("id") and c.get("n")}
        return {**self._names, **custom}

    def _today(self) -> date:
        return self._now().date()

    def _unit(self, state: dict) -> str:
        return "lb" if state.get("unit") == "lb" else "kg"

    # ---- the tool

    def tools(self) -> list[Tool]:
        spec = ToolSpec(
            name="gym",
            description=(
                "His gym tracker (openGym, on his phone). today: the session planned for a date; recent: his last "
                "sessions; exercise: progress on one exercise; weight: weigh-ins and pace towards his goal; "
                "log_weight: save a weigh-in at once (read it back; undo removes it)."),
            parameters={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["today", "recent", "exercise", "weight", "log_weight", "undo"]},
                "exercise": {"type": "string", "description": "part of its name, e.g. bench"},
                "weight": {"type": "number"},
                "unit": {"type": "string", "enum": ["kg", "lb"]},
                "date": {"type": "string", "description": "YYYY-MM-DD; default today"}},
                "required": ["action"]})
        return [Tool(spec=spec, run=self.gym, notice="Checking the gym log", focus="gym",
                     writes=lambda a: a.get("action") in ("log_weight", "undo"))]

    def gym(self, args: dict) -> str:
        handler = {"today": self._day, "recent": self._recent, "exercise": self._exercise, "weight": self._weight,
                   "log_weight": self._log_weight, "undo": self._undo}.get(args.get("action"))
        if handler is None:
            raise ToolError("action must be today, recent, exercise, weight, log_weight or undo.")
        return handler(args)

    def _date_arg(self, args: dict) -> date:
        if not args.get("date"):
            return self._today()
        try:
            return date.fromisoformat(str(args["date"])[:10])
        except ValueError:
            raise ToolError(f"{args['date']!r} isn't a date; use YYYY-MM-DD.") from None

    def _last_entry(self, state: dict, ex_id: str, before: date) -> tuple[date, dict] | None:
        for w in reversed(state.get("workouts") or []):
            if str(w.get("d", "")) >= before.isoformat():
                continue
            for e in w.get("entries") or []:
                if e.get("id") == ex_id and top_set(e.get("sets") or []):
                    return date.fromisoformat(w["d"]), e
        return None

    @staticmethod
    def _plan_text(ex: dict) -> str:
        if "min" in ex:
            return f"{fmt(ex['min'])} min"
        if "sec" in ex:
            return f"{ex.get('sets', 1)} x {int(ex['sec'])} s"
        return f"{ex.get('sets', 1)} x {ex.get('reps', '?')}"

    def _session_lines(self, state: dict, routine: dict, day: date) -> list[str]:
        names, unit = self.names(state), self._unit(state)
        lines = []
        for ex in routine.get("ex") or []:
            name = names.get(ex.get("id"), f"exercise {ex.get('id')}")
            plan = self._plan_text(ex)
            last = self._last_entry(state, ex.get("id"), day)
            tail = f" (last time, {short(last[0])}: {set_text(top_set(last[1]['sets']), unit)})" if last else ""
            lines.append(f"- {name}: {plan}{tail}")
        return lines

    def _day(self, args: dict) -> str:
        s, day = self.state(), self._date_arg(args)
        when = "today" if day == self._today() else spoken(day)
        done = [w for w in s.get("workouts") or [] if w.get("d") == day.isoformat()]
        routine = routine_for(s, day)
        lines = []
        if done:
            lines.append(f"Already done {when}: " + "; ".join(self._workout_line(s, w) for w in done) + ".")
        if routine and not done:
            lines.append(f"Planned {when}: {routine.get('name') or 'a session'}.")
            lines += self._session_lines(s, routine, day)
        elif not routine and not done:
            nxt = next(((d, r) for d in (day + timedelta(days=i) for i in range(1, 8)) if (r := routine_for(s, d))), None)
            lines.append(f"Nothing planned {when} (a rest day)."
                         + (f" Next: {nxt[1].get('name') or 'a session'} on {spoken(nxt[0])}." if nxt else
                            " Nothing planned in the next week either."))
        if not s.get("routines"):
            lines.append("He hasn't set up a weekly plan in openGym yet (the AI Coach in the app can make one).")
        return untrusted("openGym", "\n".join(lines))

    def _workout_line(self, state: dict, w: dict) -> str:
        names = self.names(state)
        mins = round((w["end"] - w["start"]) / 60000) if w.get("end") and w.get("start") is not None else None
        prs = [names.get(i, "an exercise") for i in w.get("prs") or []]
        n = len(w.get("entries") or [])
        return (f"{w.get('name') or 'a session'}" + (f", {mins} min" if mins else "") + f", {n} exercise{'s' * (n != 1)}"
                + (f", personal bests on {', '.join(prs)}" if prs else ""))

    def _recent(self, args: dict) -> str:
        s = self.state()
        ws = (s.get("workouts") or [])[-5:]
        if not ws:
            return untrusted("openGym", "No sessions logged in openGym yet.")
        today = self._today()
        lines = [f"His last {len(ws)} session{'s' * (len(ws) != 1)}, most recent last:"]
        for w in ws:
            d = date.fromisoformat(w["d"])
            lines.append(f"- {d.strftime('%a') + ' ' + short(d)} ({ago((today - d).days)}): {self._workout_line(s, w)}")
        week_ago = (today - timedelta(days=7)).isoformat()
        lines.append(f"Sessions in the last 7 days: {sum(1 for w in s.get('workouts') or [] if w.get('d', '') > week_ago)}.")
        return untrusted("openGym", "\n".join(lines))

    def _exercise(self, args: dict) -> str:
        s = self.state()
        want = str(args.get("exercise") or "").lower().strip()
        if not want:
            raise ToolError("which exercise?")
        names, unit = self.names(s), self._unit(s)
        trained: dict[str, list[tuple[date, dict]]] = {}
        for w in s.get("workouts") or []:
            for e in w.get("entries") or []:
                t = top_set(e.get("sets") or [])
                if t:
                    trained.setdefault(e.get("id"), []).append((date.fromisoformat(w["d"]), t))
        words = [x for x in re.findall(r"\w+", want) if len(x) > 2] or [want]
        hits = [i for i in trained if all(x in names.get(i, "").lower() for x in words)]
        if not hits:
            raise ToolError(f"he hasn't logged anything matching {want!r} in openGym.")
        hits.sort(key=lambda i: trained[i][-1][0], reverse=True)  # the most recently trained match first
        best_id = hits[0]
        history = trained[best_id][-6:]
        lines = [f"{names.get(best_id)}: his top set in each of the last {len(history)} sessions:"]
        lines += [f"- {short(d)}: {set_text(t, unit)}" for d, t in history]
        ests = [(d, e1rm(t)) for d, t in trained[best_id] if e1rm(t)]
        if ests:
            best = max(ests, key=lambda x: x[1])
            lines.append(f"Estimated one-rep max: {fmt(round(ests[-1][1], 1))} {unit} now; best "
                         f"{fmt(round(best[1], 1))} {unit} on {short(best[0])}.")
        if len(hits) > 1:
            lines.append("Other matches: " + ", ".join(names.get(i, i) for i in hits[1:4]) + ".")
        return untrusted("openGym", "\n".join(lines))

    def _weight(self, args: dict) -> str:
        s = self.state()
        unit, points = self._unit(s), weighs(s)
        if not points:
            return "No weigh-ins in openGym yet."
        last, today = points[-1], self._today()
        lines = [f"Latest weigh-in: {fmt(last.w)} {unit} on {spoken(last.day)} ({ago((today - last.day).days)})."]
        lines.append("Recent: " + ", ".join(f"{short(p.day)} {fmt(p.w)}" for p in points[-6:]) + ".")
        if len(points) > 1:
            lines.append(f"Since the first ({short(points[0].day, year=True)}): {self._signed(last.w - points[0].w)} {unit}.")
        lines.append(self._pace_line(s, points))
        return "\n".join(lines)

    @staticmethod
    def _signed(x: float) -> str:
        return ("+" if x > 0 else "") + fmt(round(x, 2))

    def _pace_line(self, s: dict, points: list[Weigh]) -> str:
        unit, goal, rate = self._unit(s), s.get("targetW"), pace(points)
        per = 1 / LB if unit == "lb" else 1.0
        parts = []
        if rate is not None:
            parts.append(f"Trend over the last {PACE_DAYS // 7} weeks: {self._signed(round(rate, 2))} {unit} a week.")
        else:
            parts.append("Too few recent weigh-ins for a trend (needs two at least a fortnight apart).")
        if goal:
            left = float(goal) - points[-1].w
            band = GAIN_BAND if left > 0 else LOSS_BAND
            parts.append(f"Goal: {fmt(float(goal))} {unit} ({self._signed(round(left, 2))} to go). A usual steady pace "
                         f"is {fmt(band[0] * per)} to {fmt(band[1] * per)} {unit} a week"
                         + (f", so about {round(abs(left) / (sum(band) / 2 * per))} weeks." if abs(left) > 0.2 else "."))
        else:
            parts.append("No goal weight set in openGym.")
        return " ".join(parts)

    def _log_weight(self, args: dict) -> str:
        try:
            value = float(args.get("weight"))
        except (TypeError, ValueError):
            raise ToolError("what did he weigh?") from None
        day = self._date_arg(args)
        if day > self._today():
            raise ToolError("that date is in the future.")
        got: dict = {}

        def change(s: dict) -> bool:
            unit = self._unit(s)
            said = args.get("unit") or unit
            w = value if said == unit else (value * LB if said == "lb" else value / LB)
            kg = w * LB if unit == "lb" else w
            if not 25 <= kg <= 300:
                raise ToolError(f"{fmt(value)} {said} doesn't look like a body weight; ask him to repeat it.")
            entry = {"d": day.isoformat(), "w": round(w, 2), "t": int(self._now().timestamp() * 1000)}
            bw = s.get("bodyweight") or []
            prev = next((b for b in bw if b.get("d") == entry["d"]), None)
            s["bodyweight"] = sorted([b for b in bw if b.get("d") != entry["d"]] + [entry], key=lambda b: b.get("d", ""))
            got.update(unit=unit, entry=entry, prev=prev)
            return True

        with self._lock:
            s = self._update(change)
            unit, entry, prev = got["unit"], got["entry"], got["prev"]
            self.undo = (entry["d"], prev, entry)
        when = "today" if day == self._today() else spoken(day)
        msg = f"Logged {fmt(entry['w'])} {unit} for {when} in openGym."
        if prev:
            msg += f" It replaces {fmt(float(prev.get('w', 0)))} {unit} logged earlier that day."
        else:
            earlier = [p for p in weighs(s) if p.day < day]
            if earlier:
                msg += f" Previous: {fmt(earlier[-1].w)} {unit} on {spoken(earlier[-1].day)}."
        self._changed()
        return msg + " (He can say 'undo that'.)"

    def _undo(self, args: dict) -> str:
        with self._lock:
            if self.undo is None:
                raise ToolError("there's no weigh-in to undo.")
            day, prev, entry = self.undo

            def change(s: dict) -> bool:
                bw = [b for b in s.get("bodyweight") or [] if not (b.get("d") == day and b.get("w") == entry["w"])]
                if prev:
                    restored = {**prev, "t": int(self._now().timestamp() * 1000)}  # newer than the one it replaces
                    bw = sorted([b for b in bw if b.get("d") != day] + [restored], key=lambda b: b.get("d", ""))
                s["bodyweight"] = bw
                return True

            self._update(change)
            self.undo = None
        self._changed()
        return f"Undone: the weigh-in of {fmt(entry['w'])} is removed" + (", and the earlier one is back." if prev else ".")

    # ---- background: renew the token, copy the meal-plan sheet's weigh-ins, prepare the notes

    def tick(self) -> None:
        if self._clock() >= self._renew_at and self.client.session().get("token"):
            try:
                if self.client.renew():
                    log.info("openGym: token renewed (until %s)", self.client.session().get("expires"))
                self._renew_at = self._clock() + RENEW_S
            except ToolError as e:
                log.info("openGym token check skipped: %s", e)
                self._renew_at = self._clock() + 1800
        if self.meal_plan and self.sheet_plan and self._clock() >= self._sync_at:
            try:
                self.sync_sheet()
                self._sync_at = self._clock() + self.sync_s
            except ToolError as e:
                log.info("openGym sheet sync not done: %s", e)
                self._sync_at = self._clock() + 1800
        self._nudge.prepare()
        try:
            self._check_pace()
        except ToolError as e:
            log.info("openGym pace check skipped: %s", e)
        if self.hud_card and events.watched() and self._clock() - self._card_at >= CARD_S:
            self._card_at = self._clock()
            events.publish("card", **self.card())

    def _changed(self) -> None:
        """Refresh the HUD card now, off the turn's thread (a weigh-in changed what it shows)."""
        if not (self.hud_card and events.watched()):
            self._card_at = float("-inf")  # stale: refresh as soon as a page is back on screen
            return
        self._card_at = self._clock()
        threading.Thread(target=lambda: events.publish("card", **self.card()), daemon=True).start()

    def card(self) -> dict:
        """Today's session (planned, done or a rest day) and the weight trend, for the HUD's gym card. Screen only."""
        try:
            s = self.state()
        except ToolError as e:
            if not self.client.session().get("token"):
                return {"id": "gym", "error": "NOT PAIRED"}
            return {"id": "gym", "error": "SIGN-IN NEEDED" if "expired" in str(e) else "OFFLINE"}
        today, unit, names = self._today(), self._unit(s), self.names(s)
        done = [w for w in s.get("workouts") or [] if w.get("d") == today.isoformat()]
        routine = routine_for(s, today)
        card: dict = {"id": "gym", "error": None, "items": [], "more": 0, "detail": ""}
        if done:
            w = done[-1]
            mins = round((w["end"] - w["start"]) / 60000) if w.get("end") and w.get("start") is not None else None
            prs = [names.get(i, "") for i in w.get("prs") or [] if names.get(i)]
            n = len(w.get("entries") or [])
            card.update(status="DONE", title=w.get("name") or "Session",
                        detail=" · ".join(filter(None, [f"{mins} min" if mins else "", f"{n} exercise{'s' * (n != 1)}",
                                                        ("PB " + ", ".join(prs[:2])) if prs else ""])))
        elif routine:
            exs = routine.get("ex") or []
            items = []
            for ex in exs[:CARD_ITEMS]:
                last = self._last_entry(s, ex.get("id"), today)
                items.append({"name": names.get(ex.get("id"), f"exercise {ex.get('id')}"), "plan": self._plan_text(ex),
                              "last": set_text(top_set(last[1]["sets"]), unit) if last else ""})
            card.update(status="PLANNED", title=routine.get("name") or "Session", items=items,
                        more=max(0, len(exs) - CARD_ITEMS))
        else:
            nxt = next(((d, r) for d in (today + timedelta(days=i) for i in range(1, 8)) if (r := routine_for(s, d))), None)
            card.update(status="REST", title="Rest day",
                        detail=f"Next: {nxt[1].get('name') or 'a session'}, {nxt[0].strftime('%a')}" if nxt else "")
        points = weighs(s)
        if points:
            rate, goal = pace(points), s.get("targetW")
            card["weight"] = {"latest": fmt(points[-1].w), "unit": unit, "days": (today - points[-1].day).days,
                              "trend": self._signed(round(rate, 2)) if rate is not None else None,
                              "goal": fmt(float(goal)) if goal else None,
                              "left": self._signed(round(float(goal) - points[-1].w, 1)) if goal else None}
        return card

    def sync_sheet(self) -> int:
        """Copy the meal-plan sheet's new weigh-ins into openGym. Each sheet row is copied once (rows are known by what's
        in their cells, not by the date worked out for them); days that already have a weigh-in are left alone.
        Returns how many were added."""
        entries = self.meal_plan.weigh_ins(self.sheet_plan)
        today = self._today()
        with self._lock:
            ledger = read_state(self.sync_file)
            seen: dict = ledger.setdefault("synced", {})
            fresh = []
            for e, day in resolve_days(entries, today):
                key = f"{e.raw_date}|{fmt(e.kg)}"
                if key in seen:
                    continue
                if day is None:
                    seen[key] = None  # never copied, even once a future date has passed: it's a typo
                    what = (spoken(e.day, "%A %d %B %Y") + (f" or {spoken(e.alt, '%A %d %B %Y')}" if e.alt else "")
                            if e.day else repr(e.raw_date))
                    self._news.append(f"His meal-plan sheet has a weigh-in ({fmt(e.kg)} kg) dated {what}, which "
                                      "doesn't fit the log (in the future, or out of order), so it wasn't copied into "
                                      "openGym (probably a typo). Mention it once, briefly.")
                else:
                    fresh.append((key, day, e))
            added: list = []
            unit = "kg"
            if fresh:
                def change(s: dict) -> bool:
                    nonlocal unit
                    unit = self._unit(s)
                    have = {b.get("d") for b in s.get("bodyweight") or []}
                    added.clear()
                    for _, day, e in fresh:
                        if day.isoformat() in have:
                            continue
                        have.add(day.isoformat())
                        noon = datetime.combine(day, dtime(12), LOCAL)
                        w = e.kg / LB if unit == "lb" else e.kg
                        added.append({"d": day.isoformat(), "w": round(w, 2), "t": int(noon.timestamp() * 1000)})
                    if added:
                        s["bodyweight"] = sorted((s.get("bodyweight") or []) + added, key=lambda b: b.get("d", ""))
                    return bool(added)

                self._update(change)
                for key, day, _ in fresh:
                    seen[key] = day.isoformat()
                if added:
                    last = max(added, key=lambda b: b["d"])
                    self._news.append(f"Jarvis copied {len(added)} weigh-in{'s' * (len(added) != 1)} from his "
                                      f"meal-plan sheet into openGym (latest {fmt(last['w'])} {unit} on "
                                      f"{spoken(date.fromisoformat(last['d']))}). Mention it once, briefly.")
            write_state(self.sync_file, ledger)
        return len(added)

    def _nudge_text(self) -> str | None:
        s, today = self.state(), self._today()
        routine = routine_for(s, today)
        if not routine or any(w.get("d") == today.isoformat() for w in s.get("workouts") or []):
            return None
        lines = self._session_lines(s, routine, today)[:3]
        return untrusted("openGym", f"Gym today: {routine.get('name') or 'a session'}.\n" + "\n".join(lines)) + \
            " Mention today's session in one short sentence (the first exercise and last time's numbers at most)."

    def _check_pace(self) -> None:
        """At most once a week: a note if his weight is moving too slowly, too fast or the wrong way for his goal."""
        today = self._today()
        ledger = read_state(self.sync_file)
        last = ledger.get("pace_checked")
        if last and (today - date.fromisoformat(last)).days < 7:
            return
        s = self.state()
        points, goal = weighs(s), s.get("targetW")
        ledger["pace_checked"] = today.isoformat()
        write_state(self.sync_file, ledger)
        if not goal or not points:
            return
        unit, left = self._unit(s), float(goal) - points[-1].w
        per = 1 / LB if unit == "lb" else 1.0
        if abs(left) < 0.5 * per:
            return
        stale = (today - points[-1].day).days
        if stale > 14:
            self._news.append(f"His last weigh-in in openGym was {stale} days ago. Suggest, once and briefly, that he "
                              "logs one (he can just tell Jarvis his weight).")
            return
        rate = pace(points)
        if rate is None:
            return
        lo, hi = (GAIN_BAND if left > 0 else LOSS_BAND)
        towards = rate * (1 if left > 0 else -1)  # positive when heading for the goal
        if lo * per <= towards <= hi * per:
            return
        how = "too slowly" if towards < lo * per else "faster than a steady pace"
        if towards <= 0:
            how = "away from his goal"
        self._news.append(
            f"Weight check: over the last {PACE_DAYS // 7} weeks he's moved {self._signed(round(rate, 2))} {unit} a week, "
            f"{how} (goal {fmt(float(goal))} {unit}; a usual steady pace is {fmt(lo * per)} to {fmt(hi * per)} {unit} "
            "a week). Mention it once, briefly. If he wants to change it, his meals are the lever: you can look at "
            "his meal plan with him, but suggest he takes changes to whoever set the plan rather than prescribing food.")

    def daily_note(self) -> str | None:
        with self._lock:
            news, self._news = self._news, []
        parts = [p for p in [self._nudge.take(), *news] if p]
        return "\n".join(parts) or None

    def status(self) -> str:
        sess = self.client.session()
        if not sess.get("token"):
            return f"openGym (not paired: {LOGIN_HINT})"
        exp = sess.get("expires", "")
        if exp and date.fromisoformat(exp) < self._today():
            return f"openGym (sign-in expired: {LOGIN_HINT})"
        return f"openGym ({sess.get('user') or 'signed in'}" + (f", until {exp}" if exp else "") + \
            (f"; weigh-ins from the {self.sheet_plan} sheet" if self.meal_plan and self.sheet_plan else "") + ")"


def opengym_from_config(cfg, meal_plan=None) -> OpenGymConnector | None:
    og = cfg.opengym
    if not og.get("enabled", True) or not og.get("url"):
        return None
    library = og.get("library")
    return OpenGymConnector(
        OpenGymClient(og["url"], cfg.root / og.get("session_file", "tokens/opengym.json")),
        library=Path(library) if library else None, data_dir=cfg.root / "data",
        meal_plan=meal_plan, sheet_plan=og.get("weigh_ins_from", "") if meal_plan else "",
        sync_hours=float(og.get("sync_hours", 3)), cache_seconds=float(og.get("cache_seconds", 60)),
        hud_card=bool(og.get("hud_card", True)))
