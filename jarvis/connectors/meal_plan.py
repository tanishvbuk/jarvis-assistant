"""Meal plans from Google Sheets, read-only.

The meal-plan sheets have one tab per week, a row of dates across the top and
the meals down the side, but the meal names and times change from week to week.
So nothing here assumes cell positions: it looks through every tab for the
requested date (as a real date value, so day/month order can't be misread) and
reads that date's column downwards until the first blank row.

Only that one day's meals ever leave this module: never the recipes, grocery
list or dashboard. The one exception is the tracker tab's weight log (date and
weight only), which the openGym connector copies into his gym tracker; its
measurements and remarks are never read. Plans are listed in config.local.toml:

    [meal_plan.plans.alex]
    sheet_id = "..."
    account = "personal"      # the Google account the sheet is shared with
    who = "the user himself"  # how the model should think of this person
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import Tool, ToolError, untrusted
from .google import Clients, GoogleAccount, GoogleAuth, NeedsLogin

log = logging.getLogger(__name__)

SCAN_RANGE = "A1:Z40"           # top-left corner of each tab; a week plan is about 18 rows by 10 columns
TRACKER_RANGE = "A1:Z150"       # the tracker tab's weight log runs about 60 rows below its header
EPOCH = date(1899, 12, 30)      # Google Sheets date serial 0


@dataclass
class Plan:
    name: str
    sheet_id: str
    account: str
    who: str = ""


@dataclass
class Meal:
    time: str   # "19:30", or "" if the sheet gives none
    name: str   # "Dinner"
    food: str


class SheetsClient:
    """The two Sheets API calls Jarvis makes. Tests replace this with a fake."""

    def __init__(self, creds):
        from googleapiclient.discovery import build

        self._svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

    def grids(self, sheet_id: str) -> dict[str, list[list]]:
        """{tab title: rows of raw cell values} for every visible tab. Dates come back as serial numbers."""
        meta = self._svc.spreadsheets().get(spreadsheetId=sheet_id,
                                            fields="sheets(properties(title,hidden))").execute()
        titles = [s["properties"]["title"] for s in meta.get("sheets", []) if not s["properties"].get("hidden")]
        if not titles:
            return {}
        ranges = ["'" + t.replace("'", "''") + "'!" + SCAN_RANGE for t in titles]
        res = self._svc.spreadsheets().values().batchGet(
            spreadsheetId=sheet_id, ranges=ranges, valueRenderOption="UNFORMATTED_VALUE",
            dateTimeRenderOption="SERIAL_NUMBER").execute()
        return {t: vr.get("values", []) for t, vr in zip(titles, res.get("valueRanges", []))}

    def tracker_grids(self, sheet_id: str) -> dict[str, list[list]]:
        """{tab title: rows} for the visible tabs with TRACKER in their name (the weight log is far below SCAN_RANGE)."""
        meta = self._svc.spreadsheets().get(spreadsheetId=sheet_id,
                                            fields="sheets(properties(title,hidden))").execute()
        titles = [s["properties"]["title"] for s in meta.get("sheets", [])
                  if not s["properties"].get("hidden") and "tracker" in s["properties"]["title"].lower()]
        if not titles:
            return {}
        ranges = ["'" + t.replace("'", "''") + "'!" + TRACKER_RANGE for t in titles]
        res = self._svc.spreadsheets().values().batchGet(
            spreadsheetId=sheet_id, ranges=ranges, valueRenderOption="UNFORMATTED_VALUE",
            dateTimeRenderOption="SERIAL_NUMBER").execute()
        return {t: vr.get("values", []) for t, vr in zip(titles, res.get("valueRanges", []))}


@dataclass
class WeighIn:
    day: date | None   # None if the cell isn't a date
    kg: float
    raw_date: str      # as typed, for mentioning a bad one
    alt: date | None = None  # the same date with day and month swapped, when both readings are real dates


def _swapped(d: date) -> date | None:
    try:
        return date(d.year, d.day, d.month) if d.day <= 12 and d.day != d.month else None
    except ValueError:
        return None


def _cell_date(v) -> date | None:
    """A date cell: a serial number, or text typed day first ("26/07/2026", "10/7/2026").

    The meal-plan sheets are set to US order, so a date typed day first comes out swapped
    whenever both numbers are 12 or under ("10/7" became 7 October). WeighIn.alt keeps the
    other reading; resolve_days() in the openGym connector picks between them."""
    if _is_num(v) and v > 20000:
        return EPOCH + timedelta(days=int(v))
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", str(v).strip())
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def _cell_number(v) -> float | None:
    if _is_num(v):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return None


def find_weigh_ins(grids: dict[str, list[list]]) -> list[WeighIn]:
    """The weight log: rows under a header with DATE and WEIGHT cells, in whichever tab has one."""
    for rows in grids.values():
        for r, row in enumerate(rows):
            labels = [str(x).strip().upper() for x in row]
            if "DATE" not in labels or "WEIGHT" not in labels:
                continue
            dc, wc = labels.index("DATE"), labels.index("WEIGHT")
            uc = labels.index("UNIT") if "UNIT" in labels else None
            out = []
            for below in rows[r + 1:]:
                cell = lambda c: below[c] if c is not None and c < len(below) else ""  # noqa: E731
                kg = _cell_number(cell(wc))
                if kg is None or str(cell(dc)).strip() == "":
                    continue
                if str(cell(uc)).strip().lower() in ("lb", "lbs"):
                    kg *= 0.45359237
                day = _cell_date(cell(dc))
                out.append(WeighIn(day, round(kg, 2), str(cell(dc)).strip(), _swapped(day) if day else None))
            return out
    return []


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _serial(d: date) -> int:
    return (d - EPOCH).days


def _time_text(v) -> str:
    """A time cell: a day fraction (0.354 -> "08:30") or text ("8:30 AM" -> "08:30")."""
    if _is_num(v) and 0 <= v < 1:
        minutes = round(v * 24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    s = str(v).strip()
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%I %p"):
        try:
            return datetime.strptime(s.upper(), fmt).strftime("%H:%M")
        except ValueError:
            pass
    return ""


TYPOS = {"Evevning": "Evening"}  # seen in the real sheets


def _label(v) -> str:
    s = " ".join(str(v).split())
    s = s.capitalize() if s.isupper() else s
    return TYPOS.get(s, s)


def find_day(grids: dict[str, list[list]], day: date) -> tuple[str, list[Meal]] | None:
    """The tab holding `day` and that day's meals, top to bottom; None if no tab has it."""
    target = _serial(day)
    for tab, rows in grids.items():
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                if not (_is_num(v) and abs(v - target) < 1e-6):
                    continue
                # A real header row has other dates near this one (the rest of the week).
                if sum(1 for x in row if _is_num(x) and 0 < abs(x - target) <= 14) < 2:
                    continue
                first_date_col = min(i for i, x in enumerate(row) if _is_num(x) and abs(x - target) <= 14)
                return tab, _meals_below(rows, r, c, first_date_col)
    return None


def _meals_below(rows: list[list], header: int, col: int, label_cols: int) -> list[Meal]:
    meals: list[Meal] = []
    for row in rows[header + 1:]:
        cells = [x for x in row if str(x).strip()]
        if not cells:
            if meals:
                break  # the blank row after the plan; below it are recipes and the grocery list
            continue
        food = " ".join(str(row[col]).split()) if col < len(row) else ""
        side = [x for x in row[:label_cols] if str(x).strip()]
        t = next((_time_text(x) for x in side if _time_text(x)), "")
        names = [_label(x) for x in side if not _time_text(x)]
        if food or names:
            meals.append(Meal(time=t, name=" ".join(names), food=food))
    return meals


class MealPlanConnector:
    name = "meal_plan"

    def __init__(self, plans: list[Plan], accounts: list[GoogleAccount], auth: GoogleAuth, *,
                 cache_seconds: float = 600, hud_card: bool = True, client_factory: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic, today: Callable[[], date] | None = None):
        self.plans = {p.name: p for p in plans}
        self.accounts = {a.name: a for a in accounts}
        self.auth = auth
        self.cache_seconds = cache_seconds
        self.hud_card = hud_card
        self._card_day: date | None = None
        self._card_at = float("-inf")
        self._clients = Clients(auth, client_factory or SheetsClient)
        self._cache: dict[str, tuple[float, dict]] = {}
        self._clock = clock
        self._today = today or (lambda: datetime.now(LOCAL).date())

    def tools(self) -> list[Tool]:
        if not self.plans:
            return []
        people = "; ".join(f"{p.name}" + (f" ({p.who})" if p.who else "") for p in self.plans.values())
        spec = ToolSpec(
            name="meal_plan",
            description=f"The meal plan for one day, every meal with its time. Plans: {people}.",
            parameters={
                "type": "object",
                "properties": {
                    "person": {"type": "string", "enum": list(self.plans)},
                    "date": {"type": "string", "description": "YYYY-MM-DD; default today"},
                },
                "required": ["person"],
            },
        )
        return [Tool(spec=spec, run=self.meal_plan, notice="Checking the meal plan", focus="meals", reads_outside=True)]

    # ------------------------------------------------------------------ the tool

    def meal_plan(self, args: dict) -> str:
        name = args.get("person") or (next(iter(self.plans)) if len(self.plans) == 1 else None)
        if name not in self.plans:
            raise ToolError(f"say whose plan: {', '.join(self.plans)}.")
        day = self._today()
        if args.get("date"):
            try:
                day = date.fromisoformat(str(args["date"]))
            except ValueError:
                raise ToolError(f"{args['date']!r} isn't a date; use YYYY-MM-DD.") from None
        plan = self.plans[name]
        who = f"{name}" + (f" ({plan.who})" if plan.who else "")
        when = spoken(day, "%A %d %B %Y")
        found = find_day(self.grids(plan), day)
        if found is None:
            return untrusted("the meal plan sheet",
                             f"Meal plan for {who}: {when} isn't in the plan (the plan may not have added "
                             "that week yet).")
        tab, meals = found
        lines = [f"Meal plan for {who}, {when} (sheet tab {tab}):"]
        lines += [f"{m.time or '--:--'} {m.name or 'Meal'}: {m.food or '(nothing listed)'}" for m in meals]
        return untrusted("the meal plan sheet", "\n".join(lines))

    # ------------------------------------------------------------------ the HUD card

    def tick(self) -> None:
        """Refresh the HUD card every cache_seconds and just after midnight. Called from a background thread."""
        day = self._today()
        if not self.hud_card or not events.watched() or (day == self._card_day and self._clock() - self._card_at < self.cache_seconds):
            return
        events.publish("card", **self.card())
        self._card_day, self._card_at = day, self._clock()

    def card(self) -> dict:
        """Today's meals for every plan, for the HUD. Display only: this never goes to the model."""
        day = self._today()
        sections = []
        for plan in self.plans.values():
            try:
                found = find_day(self.grids(plan), day)
            except ToolError as e:
                sections.append({"label": plan.name.upper(), "error": "SIGN-IN NEEDED"
                                 if isinstance(e, NeedsLogin) else "UNAVAILABLE"})
                continue
            except Exception as e:
                log.warning("Meal plan card for %s failed: %s", plan.name, type(e).__name__)
                sections.append({"label": plan.name.upper(), "error": "UNAVAILABLE"})
                continue
            if found is None:
                sections.append({"label": plan.name.upper(), "error": "NOT IN PLAN YET"})
            else:
                sections.append({"label": plan.name.upper(),
                                 "items": [{"time": m.time, "name": m.name, "food": m.food} for m in found[1]]})
        return {"id": "meals", "date": day.isoformat(), "sections": sections}

    # ------------------------------------------------------------------ fetching

    def grids(self, plan: Plan) -> dict[str, list[list]]:
        hit = self._cache.get(plan.name)
        if hit and self._clock() - hit[0] < self.cache_seconds:
            return hit[1]
        grids = self._fetch(plan, "grids")
        self._cache[plan.name] = (self._clock(), grids)
        return grids

    def weigh_ins(self, name: str) -> list[WeighIn]:
        """The weight log from one plan's tracker tab (not cached: it's read every few hours)."""
        if name not in self.plans:
            raise ToolError(f"there's no meal plan called {name!r}.")
        return find_weigh_ins(self._fetch(self.plans[name], "tracker_grids"))

    def _fetch(self, plan: Plan, what: str) -> dict[str, list[list]]:
        account = self.accounts.get(plan.account)
        if account is None or "sheets" not in account.services:
            raise ToolError(f"the {plan.name} meal plan needs a Google account with 'sheets' access "
                            f"(set account = \"{plan.account}\" in config.local.toml).")
        from googleapiclient.errors import HttpError

        try:
            return getattr(self._clients.get(account), what)(plan.sheet_id)
        except HttpError as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "resp", None), "status", None)
            log.warning("Meal plan %s: Sheets error %s", plan.name, status)
            if status in (403, 404):
                raise ToolError(f"the {plan.name} meal plan sheet can't be opened by the {account.name} "
                                "account (moved, or no longer shared).") from e
            raise ToolError("Google Sheets didn't respond properly.") from e


def plans_from_config(meal_plan: dict) -> list[Plan]:
    return [Plan(name=n, sheet_id=p.get("sheet_id", ""), account=p.get("account", "personal"), who=p.get("who", ""))
            for n, p in (meal_plan.get("plans") or {}).items() if p.get("sheet_id")]
