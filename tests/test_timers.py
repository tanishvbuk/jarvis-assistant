"""Timers and reminders with a fake clock."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from jarvis.connectors.base import ToolError
from jarvis.persona import LOCAL
from jarvis.skills.timers import Timers

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=LOCAL)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


@pytest.fixture
def timers(tmp_path):
    clock = Clock()
    t = Timers(tmp_path / "timers.json", now=clock)
    t.clock = clock
    return t


def test_countdown_goes_off_on_time(timers):
    assert timers.timer({"action": "set", "minutes": 40, "label": "pasta"}) == "Set: 40-minute timer for pasta, due at 14:40."
    timers.clock.t += timedelta(minutes=39)
    assert timers.fire_due() == []
    timers.clock.t += timedelta(minutes=1)
    assert timers.fire_due() == ["Sir, your pasta timer is done."]
    assert timers.fire_due() == []  # only once


def test_reminder_at_a_clock_time(timers):
    out = timers.timer({"action": "set", "at": "2026-10-11T09:00", "label": "chase the invoice"})
    assert out == "Set: reminder for chase the invoice, due tomorrow at 09:00."
    timers.clock.t = datetime(2026, 10, 11, 9, 0, 5, tzinfo=LOCAL)
    assert timers.fire_due() == ["Sir, a reminder: chase the invoice."]


def test_list_and_cancel(timers):
    timers.timer({"action": "set", "minutes": 120, "label": "cure"})
    timers.timer({"action": "set", "minutes": 10, "label": "tea"})
    listing = timers.timer({"action": "list"})
    assert listing.splitlines() == ["2 set:", "- timer for tea: due at 14:10 (in 10 min)",
                                    "- timer for cure: due at 16:00 (in 120 min)"]
    assert timers.timer({"action": "cancel", "label": "Tea"}) == "Cancelled: tea."
    with pytest.raises(ToolError, match="no timer matches 'pasta'; current ones: cure"):
        timers.timer({"action": "cancel", "label": "pasta"})
    assert timers.timer({"action": "cancel", "label": "all"}) == "Cancelled: cure."
    assert timers.timer({"action": "list"}) == "No timers or reminders are set."


def test_bad_requests(timers):
    for args, msg in [({"action": "set"}, "how long"), ({"action": "set", "minutes": -5}, "between"),
                      ({"action": "set", "at": "2026-10-10T13:00"}, "already passed"),
                      ({"action": "set", "at": "teatime"}, "isn't a time"), ({"action": "dance"}, "action must")]:
        with pytest.raises(ToolError, match=msg):
            timers.timer(args)


def test_survives_restart_and_reports_what_was_missed(timers, tmp_path):
    timers.timer({"action": "set", "at": "2026-10-10T18:00", "label": "post the parcels"})
    timers.timer({"action": "set", "minutes": 600, "label": "cure"})
    clock = Clock()
    clock.t = datetime(2026, 10, 10, 19, 30, tzinfo=LOCAL)  # Jarvis was off at 18:00
    again = Timers(tmp_path / "timers.json", now=clock)
    said = again.fire_due(missed_before=clock.t - timedelta(minutes=1))
    assert said == ["While I was off, sir, your reminder for post the parcels was due at 18:00."]
    assert [t["label"] for t in again.state["timers"]] == ["cure"]


def test_background_thread_calls_back(tmp_path):
    import threading
    import time as _time

    got = threading.Event()
    t = Timers(tmp_path / "t.json")
    t.timer({"action": "set", "minutes": 0.02, "label": "quick"})  # 1.2 seconds
    t.start(lambda text: got.set())
    try:
        assert got.wait(5)
    finally:
        t.stop()
    _time.sleep(0.05)


def test_card(timers):
    timers.timer({"action": "set", "minutes": 5, "label": ""})
    card = timers.card()
    assert card["id"] == "timers" and card["items"][0]["label"] == "timer"
    assert card["items"][0]["due"] == (T0 + timedelta(minutes=5)).isoformat()
