"""openGym connector against a fake instance (an in-memory /api/data) and a fake diet sheet."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import date, datetime

import pytest

from jarvis.connectors.base import ToolError
from jarvis.connectors.meal_plan import WeighIn, find_weigh_ins
from jarvis.connectors.opengym import (OpenGymClient, OpenGymConnector, Weigh, pace, resolve_days, routine_for,
                                       token_expiry, weighs)
from jarvis.persona import LOCAL

NOW = datetime(2026, 10, 15, 8, 0, tzinfo=LOCAL)  # a Thursday


def plan_state() -> dict:
    push = {"id": "r1", "name": "Push Day", "ex": [{"id": "0025", "sets": 4, "reps": 8}, {"id": "0047", "sets": 3, "reps": 10}]}
    legs = {"id": "r2", "name": "Leg Day", "ex": [{"id": "0043", "sets": 4, "reps": 8}, {"id": "9001", "sets": 3, "sec": 45}]}
    return {
        "unit": "kg", "targetW": 65, "_ts": 1,
        "routines": [push, legs],
        "week": {"4": "r1", "6": "r2"},            # Thursday push, Saturday legs (JavaScript getDay numbers)
        "dayPlan": {"2026-10-17": "rest"},          # this Saturday moved to a rest day
        "customEx": [{"id": "9001", "n": "Wall sit"}],
        "workouts": [
            {"id": "w1", "d": "2026-10-08", "name": "Push Day", "start": 0, "end": 52 * 60000, "prs": ["0025"],
             "entries": [{"id": "0025", "sets": [{"w": 40, "r": 8, "done": True}, {"w": 42.5, "r": 6, "done": True},
                                                 {"w": 45, "r": 3, "done": False}]},
                         {"id": "0047", "sets": [{"w": 14, "r": 10, "done": True}]}]},
            {"id": "w2", "d": "2026-10-10", "name": "Leg Day", "start": 0, "end": 40 * 60000, "prs": [],
             "entries": [{"id": "0043", "sets": [{"w": 50, "r": 8, "done": True}]},
                         {"id": "9001", "sets": [{"sec": 45, "w": 0, "done": True}]}]},
        ],
        "bodyweight": [{"d": "2026-09-01", "w": 57.5, "t": 1}, {"d": "2026-09-20", "w": 57.6, "t": 1},
                       {"d": "2026-10-10", "w": 57.8, "t": 1}],
        "exWeights": {"0025": {"w": 42.5}},
        "settingsWeDontKnow": {"keep": True},
    }


TOKEN = "uid:1792016989384:0.mac"   # runs out on 14 Oct 2026 (the payload's middle field, in ms)


class FakeServer:
    """openGym's API as upstream has it: the document has a revision and a save quoting an older one is refused
    with 409 and the current copy; a pairing code swaps for a token; /api/me can hand back a fresh token."""

    def __init__(self, state: dict | None = None):
        self.state = state
        self.rev = 1
        self.calls: list = []
        self.down = False
        self.status = 200
        self.phone_saves_first = 0    # how many of the next saves the phone gets in ahead of
        self.renewed = None           # a token /api/me hands back

    def phone_save(self):
        """The phone saves a change of its own."""
        self.state["dayPlan"] = {**(self.state.get("dayPlan") or {}), "2026-10-20": "r2"}
        self.rev += 1

    def __call__(self, method, url, body, headers):
        path = url.split("gym.test", 1)[1]
        self.calls.append((method, path, headers.get("Authorization")))
        if self.down:
            raise urllib.error.URLError("refused")
        if self.status != 200:
            raise urllib.error.HTTPError(url, self.status, "no", {}, io.BytesIO(b""))
        if path == "/api/pair/redeem":
            if body["code"] != "ABCD1234":
                raise urllib.error.HTTPError(url, 400, "no", {}, io.BytesIO(b'{"code":"pair-invalid"}'))
            return {"token": TOKEN, "user": {"id": "uid", "name": "Alex"}}
        if path == "/api/me":
            return {"user": {"id": "uid", "name": "Alex"}, **({"token": self.renewed} if self.renewed else {})}
        if method == "GET":
            return {"state": json.loads(json.dumps(self.state)) if self.state is not None else None, "rev": self.rev}
        if self.phone_saves_first:
            self.phone_saves_first -= 1
            self.phone_save()
        if body.get("baseRev") != self.rev:
            current = json.dumps({"error": "conflict", "rev": self.rev, "state": self.state}).encode()
            raise urllib.error.HTTPError(url, 409, "conflict", {}, io.BytesIO(current))
        body = json.loads(json.dumps(body))
        body["state"].pop("active", None)
        self.state = body["state"]
        self.rev += 1
        return {"ok": True, "rev": self.rev}


class FakeMeals:
    def __init__(self, entries):
        self.entries = entries
        self.fail = False

    def weigh_ins(self, name):
        if self.fail:
            raise ToolError("Google Sheets didn't respond properly.")
        assert name == "alex"
        return list(self.entries)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def gym(tmp_path):
    server = FakeServer(plan_state())
    session = tmp_path / "tokens" / "opengym.json"
    session.parent.mkdir()
    session.write_text(json.dumps({"token": TOKEN, "user": "Alex", "expires": "2027-04-13"}))
    lib = tmp_path / "catalogue" / "exercises"   # openGym's catalogue: one file per exercise
    lib.mkdir(parents=True)
    for i, name in [("0025", "barbell bench press"), ("0047", "dumbbell incline bench press"),
                    ("0043", "barbell full squat")]:
        (lib / f"{i}.json").write_text(json.dumps({"id": i, "name": name, "bodyPart": "chest"}))
    (lib / "broken.json").write_text("{not json")
    clock = Clock()
    c = OpenGymConnector(OpenGymClient("http://gym.test/", session, request=server), library=lib, data_dir=tmp_path,
                         meal_plan=FakeMeals([]), sheet_plan="alex", now=lambda: NOW, clock=clock)
    c.server, c.clock, c.tmp = server, clock, tmp_path
    return c


# ------------------------------------------------------------------ reading

def test_today_lists_the_session_with_last_times_numbers(gym):
    out = gym.gym({"action": "today"})
    assert "Planned today: Push Day." in out
    assert "- barbell bench press: 4 x 8 (last time, 8 Oct: 42.5 kg x 6)" in out  # the unfinished 45 kg set is ignored
    assert "- dumbbell incline bench press: 3 x 10 (last time, 8 Oct: 14 kg x 10)" in out
    assert "untrusted_data" in out  # routine and exercise names are his own typing


def test_rest_day_override_and_next_session(gym):
    out = gym.gym({"action": "today", "date": "2026-10-17"})
    assert "Nothing planned Saturday 17 October (a rest day). Next: Push Day on Thursday 22 October." in out


def test_a_day_already_trained_says_so(gym):
    out = gym.gym({"action": "today", "date": "2026-10-10"})
    assert "Already done Saturday 10 October: Leg Day, 40 min, 2 exercises." in out


def test_catalogue_names_are_kept_until_the_folder_changes(gym):
    assert gym.names({})["0025"] == "barbell bench press"
    kept = json.loads((gym.tmp / "opengym_names.json").read_text())
    assert kept["names"]["0043"] == "barbell full squat" and "broken" not in kept["names"]
    (gym.library / "0025.json").write_text(json.dumps({"id": "0025", "name": "renamed"}))  # an edit: folder unchanged
    fresh = OpenGymConnector(gym.client, library=gym.library, data_dir=gym.tmp, now=lambda: NOW, clock=gym.clock)
    assert fresh.names({})["0025"] == "barbell bench press"   # read from the kept copy
    (gym.library / "0999.json").write_text(json.dumps({"id": "0999", "name": "new one"}))  # a file added
    again = OpenGymConnector(gym.client, library=gym.library, data_dir=gym.tmp, now=lambda: NOW, clock=gym.clock)
    assert again.names({})["0999"] == "new one" and again.names({})["0025"] == "renamed"


def test_routine_for_uses_javascript_weekday_numbers():
    s = plan_state()
    assert routine_for(s, date(2026, 10, 15))["name"] == "Push Day"   # Thursday = 4
    assert routine_for(s, date(2026, 10, 24))["name"] == "Leg Day"    # Saturday = 6
    assert routine_for(s, date(2026, 10, 17)) is None                  # moved to rest
    assert routine_for(s, date(2026, 10, 18)) is None                  # Sunday = 0, nothing planned


def test_recent_sessions(gym):
    out = gym.gym({"action": "recent"})
    assert "- Thu 8 Oct (7 days ago): Push Day, 52 min, 2 exercises, personal bests on barbell bench press" in out
    assert "Sessions in the last 7 days: 1." in out


def test_exercise_progress_matches_part_of_the_name(gym):
    out = gym.gym({"action": "exercise", "exercise": "bench"})
    assert out.count("barbell bench press") >= 1 and "8 Oct: 42.5 kg x 6" in out
    assert "Estimated one-rep max: 51 kg now" in out  # 42.5 x 6 by Epley = 51.0
    assert "Other matches: dumbbell incline bench press." in out
    assert "Wall sit" not in out
    with pytest.raises(ToolError, match="matching 'deadlift'"):
        gym.gym({"action": "exercise", "exercise": "deadlift"})


def test_timed_sets_and_custom_exercise_names(gym):
    out = gym.gym({"action": "today", "date": "2026-10-24"})
    assert "- Wall sit: 3 x 45 s (last time, 10 Oct: 45 s)" in out


def test_weight_trend_and_goal(gym):
    out = gym.gym({"action": "weight"})
    assert "Latest weigh-in: 57.8 kg on Saturday 10 October (5 days ago)." in out
    assert "Since the first (1 Sep 2026): +0.3 kg." in out
    assert "Goal: 65 kg (+7.2 to go). A usual steady pace is 0.25 to 0.5 kg a week, so about 19 weeks." in out


def test_pace_needs_two_weeks_of_points():
    assert pace([Weigh(date(2026, 10, 1), 57), Weigh(date(2026, 10, 10), 58)]) is None
    assert pace([Weigh(date(2026, 9, 1), 57), Weigh(date(2026, 9, 15), 58)]) == pytest.approx(0.5)


# ------------------------------------------------------------------ writing weigh-ins

def test_log_weight_saves_at_once_and_keeps_everything_else(gym):
    out = gym.gym({"action": "log_weight", "weight": 58.1})
    assert out == ("Logged 58.1 kg for today in openGym. Previous: 57.8 kg on Saturday 10 October. "
                   "(He can say 'undo that'.)")
    s = gym.server.state
    assert s["bodyweight"][-1]["d"] == "2026-10-15" and s["bodyweight"][-1]["w"] == 58.1
    assert s["bodyweight"][-1]["t"] == int(NOW.timestamp() * 1000)
    assert s["workouts"] == plan_state()["workouts"] and s["settingsWeDontKnow"] == {"keep": True}
    assert gym.server.calls[-1] == ("PUT", "/api/data", f"Bearer {TOKEN}")
    assert gym.server.rev == 2                                   # saved over the revision it read


def test_log_weight_replaces_the_same_day_and_undo_restores_it(gym):
    out = gym.gym({"action": "log_weight", "weight": 58.0, "date": "2026-10-10"})
    assert "It replaces 57.8 kg logged earlier that day." in out
    assert [b["w"] for b in gym.server.state["bodyweight"] if b["d"] == "2026-10-10"] == [58.0]
    assert gym.gym({"action": "undo"}) == "Undone: the weigh-in of 58 is removed, and the earlier one is back."
    assert [b["w"] for b in gym.server.state["bodyweight"] if b["d"] == "2026-10-10"] == [57.8]
    with pytest.raises(ToolError, match="no weigh-in to undo"):
        gym.gym({"action": "undo"})


def test_log_weight_converts_units_and_rejects_nonsense(gym):
    gym.gym({"action": "log_weight", "weight": 128, "unit": "lb"})
    assert gym.server.state["bodyweight"][-1]["w"] == 58.06
    with pytest.raises(ToolError, match="doesn't look like a body weight"):
        gym.gym({"action": "log_weight", "weight": 5.81})
    with pytest.raises(ToolError, match="future"):
        gym.gym({"action": "log_weight", "weight": 58, "date": "2026-10-16"})


def test_a_save_the_phone_beat_is_made_again_on_its_copy(gym):
    gym.server.phone_saves_first = 2         # the phone saves between Jarvis's read and save, twice
    out = gym.gym({"action": "log_weight", "weight": 58.1})
    assert out.startswith("Logged 58.1 kg for today")
    s = gym.server.state
    assert s["dayPlan"]["2026-10-20"] == "r2"                    # the phone's change is kept
    assert s["bodyweight"][-1]["d"] == "2026-10-15" and s["bodyweight"][-1]["w"] == 58.1
    assert [c[0] for c in gym.server.calls].count("PUT") == 3


def test_gives_up_when_the_phone_keeps_saving_first(gym):
    gym.server.phone_saves_first = 5
    with pytest.raises(ToolError, match="nothing was saved"):
        gym.gym({"action": "log_weight", "weight": 58.1})
    assert all(b["d"] != "2026-10-15" for b in gym.server.state["bodyweight"])
    assert gym.undo is None


def test_undo_after_the_phone_saved_keeps_the_phones_change(gym):
    gym.gym({"action": "log_weight", "weight": 58.1})
    gym.server.phone_save()
    gym.gym({"action": "undo"})
    s = gym.server.state
    assert all(b["d"] != "2026-10-15" for b in s["bodyweight"]) and s["dayPlan"]["2026-10-20"] == "r2"


# ------------------------------------------------------------------ the dietitian's sheet

def test_sheet_weigh_ins_are_copied_once_and_typos_flagged(gym):
    gym.meal_plan.entries = [WeighIn(date(2026, 7, 6), 57.1, "06/07/2026"), WeighIn(date(2026, 9, 1), 57.4, "x"),
                             WeighIn(date(2026, 12, 9), 57.8, "09/12/2026"), WeighIn(None, 57.9, "sometime")]
    assert gym.sync_sheet() == 1                                   # 1 Sep already has one: left alone
    days = {b["d"]: b["w"] for b in gym.server.state["bodyweight"]}
    assert days["2026-07-06"] == 57.1 and days["2026-09-01"] == 57.5 and "2026-12-09" not in days
    note = gym.daily_note()
    assert "dated Wednesday 9 December 2026" in note and "dated 'sometime'" in note
    assert "copied 1 weigh-in from his dietitian's sheet into openGym (latest 57.1 kg on Monday 6 July)" in note
    # Deleted in the app afterwards: not copied again. And the December typo stays skipped after December.
    gym.server.state["bodyweight"] = [b for b in gym.server.state["bodyweight"] if b["d"] != "2026-07-06"]
    assert gym.sync_sheet() == 0
    assert "2026-07-06" not in {b["d"] for b in gym.server.state["bodyweight"]}
    assert gym.daily_note() is None or "dietitian" not in gym.daily_note()


def test_sync_runs_from_tick_and_waits_after_a_failure(gym):
    gym.meal_plan.entries = [WeighIn(date(2026, 7, 6), 57.1, "")]
    gym.meal_plan.fail = True
    gym.tick()
    assert "2026-07-06" not in {b["d"] for b in gym.server.state["bodyweight"]}
    gym.meal_plan.fail = False
    gym.tick()                                # within half an hour: not tried again
    assert "2026-07-06" not in {b["d"] for b in gym.server.state["bodyweight"]}
    gym.clock.t += 1801
    gym.tick()
    assert "2026-07-06" in {b["d"] for b in gym.server.state["bodyweight"]}


# ------------------------------------------------------------------ notes

def test_morning_nudge_once_a_day_and_not_on_rest_days(gym):
    gym.tick()
    note = gym.daily_note()
    assert "Gym today: Push Day." in note and "barbell bench press: 4 x 8" in note
    assert "Gym today" not in (gym.daily_note() or "")
    rest = OpenGymConnector(gym.client, data_dir=gym.tmp / "other", now=lambda: datetime(2026, 10, 18, 8, tzinfo=LOCAL),
                            clock=gym.clock)
    rest.tick()
    assert "Gym today" not in (rest.daily_note() or "")


def test_pace_note_when_gain_is_too_slow_at_most_weekly(gym):
    gym.server.state["bodyweight"] = [{"d": "2026-09-10", "w": 57.6}, {"d": "2026-09-24", "w": 57.7},
                                      {"d": "2026-10-12", "w": 57.8}]
    gym.tick()
    note = gym.daily_note()
    assert "Weight check:" in note and "too slowly (goal 65 kg; a usual steady pace is 0.25 to 0.5 kg a week)" in note
    assert "dietitian" in note
    gym.tick()
    assert "Weight check" not in (gym.daily_note() or "")


def test_on_pace_says_nothing_and_stale_weigh_ins_get_a_prompt(gym):
    gym.server.state["bodyweight"] = [{"d": "2026-09-14", "w": 57.0}, {"d": "2026-10-12", "w": 58.4}]
    gym.tick()
    assert "Weight check" not in (gym.daily_note() or "")
    stale = OpenGymConnector(gym.client, data_dir=gym.tmp / "stale", now=lambda: datetime(2026, 11, 20, 8, tzinfo=LOCAL),
                             clock=gym.clock)
    stale.tick()
    assert "last weigh-in in openGym was 39 days ago" in stale.daily_note()


# ------------------------------------------------------------------ signing in and errors

def test_errors_are_explained(gym):
    gym.server.status = 401
    with pytest.raises(ToolError, match="sign-in has expired.*--opengym-pair"):
        gym.gym({"action": "weight"})
    gym.server.status, gym.server.down = 200, True
    with pytest.raises(ToolError, match="Docker Desktop"):
        gym.gym({"action": "log_weight", "weight": 58})
    gym.client.session_file.unlink()
    with pytest.raises(ToolError, match="isn't paired"):
        gym.state(fresh=True)
    assert "not paired" in gym.status()


def test_pairing_saves_the_token(gym):
    gym.client.session_file.unlink()
    with pytest.raises(ToolError, match="didn't accept that code"):
        gym.client.pair("WRONG")
    s = gym.client.pair(" ABCD1234 ")
    assert s == {"token": TOKEN, "user": "Alex", "expires": "2026-10-14"}
    assert json.loads(gym.client.session_file.read_text()) == s
    assert gym.server.calls[-1] == ("POST", "/api/pair/redeem", None)   # the code is the credential


def test_token_renewed_from_api_me_once_a_day(gym):
    gym.tick()
    assert json.loads(gym.client.session_file.read_text())["token"] == TOKEN   # not past half its life yet
    gym.server.renewed = "uid:1807568989384:0.mac2"
    gym.tick()                                                                # checked once a day only
    assert json.loads(gym.client.session_file.read_text())["token"] == TOKEN
    gym.clock.t += 86401
    gym.tick()
    assert json.loads(gym.client.session_file.read_text()) == \
        {"token": "uid:1807568989384:0.mac2", "user": "Alex", "expires": "2027-04-12"}
    gym.state(fresh=True)
    assert gym.server.calls[-1] == ("GET", "/api/data", "Bearer uid:1807568989384:0.mac2")


def test_token_expiry_reads_the_payload():
    assert token_expiry(TOKEN) == "2026-10-14" and token_expiry("junk") == ""


# ------------------------------------------------------------------ the sheet's weight log

def test_find_weigh_ins_reads_the_log_under_its_header():
    serial = lambda d: (d - date(1899, 12, 30)).days  # noqa: E731
    tracker = [["", "Start Date:", "", "07/06/2026"], ["", "Goal Weight:", "", 70],
               [], ["", "", "", "WEIGHT LOG BOOK"],
               ["", "", "", "WEEK", "", "DATE", "", "WEIGHT", "", "UNIT", "", "BMI"],
               ["", "", "", 0, "", serial(date(2026, 7, 6)), "", 57.1, "", "Kg", "", 18.6],
               ["", "", "", 1, "", "10/7/2026", "", "57.10", "", "kg"],
               ["", "", "", 2, "", "26/07/2026", "", 126, "", "lb"],
               ["", "", "", 3, "", "", "", "", "", "Kg"],
               ["", "", "", 4, "", "31/02/2026", "", 57.8, "", "Kg"]]
    got = find_weigh_ins({"TRANSFORMATION TRACKER": tracker})
    assert got == [WeighIn(date(2026, 7, 6), 57.1, str(serial(date(2026, 7, 6))), date(2026, 6, 7)),
                   WeighIn(date(2026, 7, 10), 57.1, "10/7/2026", date(2026, 10, 7)),
                   WeighIn(date(2026, 7, 26), 57.15, "26/07/2026"),
                   WeighIn(None, 57.8, "31/02/2026")]
    assert find_weigh_ins({"WEEK 1": [["DATE", "MEAL"]]}) == []


def test_weighs_skips_broken_entries():
    assert weighs({"bodyweight": [{"d": "2026-10-01", "w": 57}, {"d": "bad", "w": 1}, {"w": 3}]}) == \
        [Weigh(date(2026, 10, 1), 57.0)]


def test_swapped_dates_are_put_back_in_order():
    """The real sheet: typed day first into a sheet set to US order, so 6 July became 7 June and 10 July
    became 7 October; 18 and 26 July stayed text (they can't be months); the fifth reads 12 Sep or 9 Dec."""
    rows = [WeighIn(date(2026, 6, 7), 57.1, "a", date(2026, 7, 6)), WeighIn(date(2026, 10, 7), 57.1, "b", date(2026, 7, 10)),
            WeighIn(date(2026, 7, 18), 57.5, "c"), WeighIn(date(2026, 7, 26), 57.65, "d"),
            WeighIn(date(2026, 9, 12), 57.8, "e", date(2026, 12, 9)), WeighIn(date(2026, 7, 1), 58, "f")]
    got = [d for _, d in resolve_days(rows, date(2026, 10, 15))]
    assert got == [date(2026, 7, 6), date(2026, 7, 10), date(2026, 7, 18), date(2026, 7, 26), date(2026, 9, 12), None]


def test_sync_files_swapped_dates_correctly_and_never_twice(gym):
    gym.meal_plan.entries = [WeighIn(date(2026, 6, 7), 57.1, "46180", date(2026, 7, 6)),
                             WeighIn(date(2026, 10, 7), 57.1, "46302", date(2026, 7, 10)),
                             WeighIn(date(2026, 7, 18), 57.5, "18/07/2026")]
    assert gym.sync_sheet() == 3
    days = {b["d"]: b["w"] for b in gym.server.state["bodyweight"]}
    assert days["2026-07-06"] == 57.1 and days["2026-07-10"] == 57.1 and "2026-10-07" not in days
    assert gym.sync_sheet() == 0


# ------------------------------------------------------------------ the HUD card

def test_card_shows_the_planned_session_and_weight(gym):
    c = gym.card()
    assert c["id"] == "gym" and c["error"] is None and c["status"] == "PLANNED" and c["title"] == "Push Day"
    assert c["items"][0] == {"name": "barbell bench press", "plan": "4 x 8", "last": "42.5 kg x 6"}
    assert c["items"][1]["name"] == "dumbbell incline bench press" and c["more"] == 0
    assert c["weight"] == {"latest": "57.8", "unit": "kg", "days": 5, "trend": "+0.05", "goal": "65", "left": "+7.2"}


def test_card_after_a_session_is_done_and_on_a_rest_day(gym):
    gym.server.state["workouts"].append({"id": "w3", "d": "2026-10-15", "name": "Push Day", "start": 0, "end": 45 * 60000,
                                         "prs": ["0025"], "entries": [{"id": "0025", "sets": [{"w": 45, "r": 5}]}]})
    done = gym.card()
    assert done["status"] == "DONE" and done["items"] == []
    assert done["detail"] == "45 min · 1 exercise · PB barbell bench press"
    gym.server.state["dayPlan"]["2026-10-15"] = "rest"
    gym.server.state["workouts"].pop()
    gym.clock.t += 120  # past the cache
    rest = gym.card()
    assert rest["status"] == "REST" and rest["detail"] == "Next: Push Day, Thu"


def test_card_says_why_it_has_nothing(gym):
    gym.server.down = True
    gym.clock.t += 120
    assert gym.card() == {"id": "gym", "error": "OFFLINE"}
    gym.server.down = False
    gym.server.status = 401
    assert gym.card()["error"] == "SIGN-IN NEEDED"
    gym.client.session_file.write_text("{}")
    assert gym.card()["error"] == "NOT PAIRED"


def test_card_is_only_published_while_a_page_is_watching(gym):
    from jarvis import events

    got: list = []
    unsub = events.subscribe(got.append)
    try:
        gym.tick()
        assert not [e for e in got if e["t"] == "card"]
        events.set_watched(True)
        gym.tick()
        cards = [e for e in got if e["t"] == "card"]
        assert len(cards) == 1 and cards[0]["id"] == "gym"
        gym.tick()
        assert len([e for e in got if e["t"] == "card"]) == 1   # not again for five minutes
    finally:
        events.set_watched(False)
        unsub()
