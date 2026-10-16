"""Meal plan connector against fake sheets shaped like the dietitian's (invented meals)."""

from __future__ import annotations

from datetime import date

import pytest

from jarvis.connectors.base import ToolError
from jarvis.connectors.google import GoogleAccount, NeedsLogin
from jarvis.connectors.meal_plan import Meal, MealPlanConnector, Plan, find_day, plans_from_config

EPOCH = date(1899, 12, 30)


def serial(d: date) -> int:
    return (d - EPOCH).days


def week(start: date, rows: list[tuple]) -> list[list]:
    """A week tab: title, DAY 1..7, dates as serial numbers, then (time, label, meal-per-day) rows."""
    days = [serial(date.fromordinal(start.toordinal() + i)) for i in range(7)]
    grid = [[], ["", "", "", "", "", "WEEKLY DIET PLAN"], [],
            ["", ""] + [f"DAY {i}" for i in range(1, 8)],
            ["", ""] + days]
    for t, label, food in rows:
        grid.append([t, label] + ([food(i) for i in range(7)] if callable(food) else [food] * 7))
    grid += [[], ["RECIPES", "", "", "", "MAJOR GROCERY LIST", "", "", "DIETITIAN ASSIGNED :"], [],
             ["", "", "", "", "secret grocery list", "", "", "dietitian phone"]]
    return grid


MINE = {
    "TRANSFORMATION TRACKER": [["weight"], [serial(date(2026, 10, 10)), 81.5]],  # a lone date: not a week header
    "WEEK 10": week(date(2026, 9, 28), [(0.3333333, "BREAKFAST", "porridge"),
                                         (0.8125, "DINNER", lambda i: f"old dinner {i}")]),
    "WEEK 11": week(date(2026, 10, 5), [(1 / 3, "EARLY MORNING", "warm water"),
                                         (0.3541667, "BREAKFAST", lambda i: f"eggs day {i + 1}"),
                                         (13 / 24, "LUNCH", "dal and rice"),
                                         (2 / 3, "EVEVNING", "makhana"),
                                         (0.8125, "DINNER", lambda i: f"paneer tikka day {i + 1}"),
                                         (22 / 24, "POST DINNER", "turmeric milk")]),
}
SAM = {
    "WEEK 11": week(date(2026, 10, 5), [("7:30 AM", "EARLY MORNING", "jeera water"),
                                         ("11:00 AM", "BRUNCH", "poha"),
                                         ("", "AFTER 30 MIN", "green tea"),
                                         ("7:00 PM", "DINNER", lambda i: f"khichdi {i}"),
                                         ("9:00 PM", "IF HUNGRY", "milk")]),
}
SHEETS = {"S1": MINE, "S2": SAM}


def test_finds_the_day_and_reads_its_column():
    tab, meals = find_day(MINE, date(2026, 10, 10))  # Saturday, DAY 6
    assert tab == "WEEK 11"
    assert meals[0] == Meal("08:00", "Early morning", "warm water")
    assert meals[1] == Meal("08:30", "Breakfast", "eggs day 6")
    assert meals[4] == Meal("19:30", "Dinner", "paneer tikka day 6")
    assert [m.name for m in meals] == ["Early morning", "Breakfast", "Lunch", "Evening", "Dinner", "Post dinner"]


def test_text_times_and_rows_without_a_time():
    tab, meals = find_day(SAM, date(2026, 10, 11))
    assert [(m.time, m.name) for m in meals] == [("07:30", "Early morning"), ("11:00", "Brunch"),
                                                ("", "After 30 min"), ("19:00", "Dinner"), ("21:00", "If hungry")]
    assert meals[3].food == "khichdi 6"


def test_older_weeks_and_missing_days():
    assert find_day(MINE, date(2026, 9, 30))[1][1].food == "old dinner 2"
    assert find_day(MINE, date(2026, 10, 19)) is None


def test_recipes_and_grocery_list_are_never_included():
    _, meals = find_day(MINE, date(2026, 10, 5))
    assert not any("grocery" in m.food or "dietitian" in m.food for m in meals)


class FakeSheets:
    calls: list[str] = []

    def __init__(self, creds):
        self.creds = creds

    def grids(self, sheet_id):
        FakeSheets.calls.append(sheet_id)
        return SHEETS[sheet_id]


class FakeAuth:
    def __init__(self, fail=False):
        self.fail = fail

    def credentials(self, account):
        if self.fail:
            raise NeedsLogin("the personal Google account isn't signed in yet")
        return "creds"


PERSONAL = GoogleAccount("personal", "me@example.com", "", ["gmail", "sheets"])
PLANS = [Plan("alex", "S1", "personal", "the user himself"), Plan("sam", "S2", "personal", "Sam")]


@pytest.fixture
def meals():
    FakeSheets.calls = []
    clock = [0.0]
    c = MealPlanConnector(PLANS, [PERSONAL], FakeAuth(), client_factory=FakeSheets, clock=lambda: clock[0],
                          today=lambda: date(2026, 10, 10))
    c.clock = clock
    return c


def test_tool_answers_for_today_by_default(meals):
    (tool,) = meals.tools()
    assert tool.spec.parameters["properties"]["person"]["enum"] == ["alex", "sam"]
    out = tool.run({"person": "alex"})
    assert out.startswith('<untrusted_data source="the meal plan sheet">')
    assert "Meal plan for alex (the user himself), Saturday 10 October 2026 (sheet tab WEEK 11):" in out
    assert "19:30 Dinner: paneer tikka day 6" in out
    assert "grocery" not in out


def test_tool_other_dates_and_people(meals):
    assert "19:00 Dinner: khichdi 6" in meals.meal_plan({"person": "sam", "date": "2026-10-11"})
    assert "isn't in the plan" in meals.meal_plan({"person": "alex", "date": "2026-12-25"})
    with pytest.raises(ToolError, match="isn't a date"):
        meals.meal_plan({"person": "alex", "date": "tomorrow"})
    with pytest.raises(ToolError, match="whose plan"):
        meals.meal_plan({})


def test_sheet_fetched_once_per_cache_period(meals):
    meals.meal_plan({"person": "alex"})
    meals.meal_plan({"person": "alex", "date": "2026-10-11"})
    assert FakeSheets.calls == ["S1"]
    meals.clock[0] = 601
    meals.meal_plan({"person": "alex"})
    assert FakeSheets.calls == ["S1", "S1"]


def test_account_without_sheets_access_is_explained():
    c = MealPlanConnector(PLANS, [GoogleAccount("personal", "", "", ["gmail"])], FakeAuth(),
                          client_factory=FakeSheets)
    with pytest.raises(ToolError, match="'sheets' access"):
        c.meal_plan({"person": "alex"})


def test_hud_card_has_both_plans_and_survives_errors(meals):
    card = meals.card()
    assert card["id"] == "meals" and card["date"] == "2026-10-10"
    mine, sam = card["sections"]
    assert mine["label"] == "ALEX" and mine["items"][4] == {"time": "19:30", "name": "Dinner",
                                                            "food": "paneer tikka day 6"}
    assert sam["items"][3]["food"] == "khichdi 5"
    signed_out = MealPlanConnector(PLANS, [PERSONAL], FakeAuth(fail=True), client_factory=FakeSheets)
    assert [s["error"] for s in signed_out.card()["sections"]] == ["SIGN-IN NEEDED", "SIGN-IN NEEDED"]


def test_plans_from_config_skips_blank_ids():
    plans = plans_from_config({"plans": {"a": {"sheet_id": "X", "who": "me"}, "b": {"sheet_id": ""}}})
    assert plans == [Plan("a", "X", "personal", "me")]


def test_hud_card_refreshes_on_its_own_timetable(meals):
    from jarvis import events

    seen = []
    off = events.subscribe(seen.append)
    try:
        meals.tick()                       # no HUD on screen: nothing fetched
        assert FakeSheets.calls == [] and seen == []
        events.set_watched(True)
        meals.tick()
        meals.tick()                       # too soon: nothing
        meals.clock[0] = 601
        meals.tick()
        meals._today = lambda: date(2026, 10, 11)
        meals.clock[0] = 602               # a new day refreshes straight away
        meals.tick()
    finally:
        events.set_watched(False)
        off()
    assert [e["date"] for e in seen if e["t"] == "card"] == ["2026-10-10", "2026-10-10", "2026-10-11"]
