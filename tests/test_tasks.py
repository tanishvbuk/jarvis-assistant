"""The to-do list with a fake Google Tasks."""

from __future__ import annotations

from datetime import datetime

import pytest

from jarvis.connectors.base import ToolError, set_turn
from jarvis.connectors.google import GoogleAccount, NeedsLogin
from jarvis.connectors.tasks import TasksConnector
from jarvis.persona import LOCAL
from jarvis.skills.timers import Timers

NOW = datetime(2026, 10, 10, 9, 0, tzinfo=LOCAL)  # Saturday


class FakeTasks:
    def __init__(self, creds):
        self.items: dict[str, dict] = {}
        self.calls: list = []
        self.n = 0

    def open_tasks(self):
        return [dict(t) for t in self.items.values() if t.get("status") != "completed"]

    def insert(self, body):
        self.n += 1
        t = dict(body, id=f"t{self.n}", status="needsAction")
        self.items[t["id"]] = t
        self.calls.append(("insert", body))
        return dict(t)

    def patch(self, task_id, body):
        self.items[task_id].update(body)
        self.calls.append(("patch", task_id, body))
        return dict(self.items[task_id])

    def delete(self, task_id):
        del self.items[task_id]
        self.calls.append(("delete", task_id))


class FakeAuth:
    def __init__(self, ok=True):
        self.ok = ok

    def credentials(self, account):
        return "creds"

    def has_scope(self, account, service):
        return self.ok


ACCOUNT = GoogleAccount("personal", "", "", ["tasks"])


@pytest.fixture
def todo(tmp_path):
    fake = FakeTasks("creds")
    timers = Timers(tmp_path / "timers.json", now=lambda: NOW)
    c = TasksConnector(ACCOUNT, FakeAuth(), timers=timers, state_file=tmp_path / "tasks.json", hud_card=False,
                       client_factory=lambda creds: fake, now=lambda: NOW)
    c.fake, c.timer_skill = fake, timers
    set_turn(1, "add things")
    return c


def test_add_with_and_without_a_deadline(todo):
    assert todo.todo({"action": "add", "title": "Order more pasta"}) == \
        "Added: Order more pasta, no deadline. (He can say 'undo that'.)"
    out = todo.todo({"action": "add", "title": "Pitch the cosy games feature", "due": "2026-10-16"})
    assert out.startswith("Added: Pitch the cosy games feature, due Friday 16 October.")
    body = todo.fake.calls[-1][1]
    assert body["due"] == "2026-10-16T00:00:00.000Z" and "Added by Jarvis." in body["notes"]


def test_timed_deadline_gets_a_reminder_an_hour_before(todo):
    out = todo.todo({"action": "add", "title": "File the review", "due": "2026-10-12T17:00"})
    assert "due Monday 12 October at 17:00" in out
    assert "Deadline: 17:00" in todo.fake.calls[-1][1]["notes"]
    (reminder,) = todo.timer_skill.state["timers"]
    assert reminder["label"] == "File the review is due at 17:00" and reminder["due"].startswith("2026-10-12T16:00")
    todo.todo({"action": "done", "task": "review"})
    assert todo.timer_skill.state["timers"] == []  # ticked off: no reminder


def test_reminders_follow_the_task_whatever_its_title(todo):
    title = "Send the long-form feature on the history of match-three puzzle games to the commissioning editor"
    todo.todo({"action": "add", "title": title, "due": "2026-10-12T17:00"})
    assert len(todo.timer_skill.state["timers"]) == 1  # its label is cut to 80 characters
    todo.todo({"action": "done", "task": "that"})
    assert todo.timer_skill.state["timers"] == []


def test_undoing_a_move_puts_the_reminder_back(todo):
    todo.todo({"action": "add", "title": "File the review", "due": "2026-10-12T17:00"})
    todo.todo({"action": "move", "task": "review", "due": "2026-10-14T12:00"})
    assert [t["due"][:16] for t in todo.timer_skill.state["timers"]] == ["2026-10-14T11:00"]
    todo.todo({"action": "undo"})
    assert [t["due"][:16] for t in todo.timer_skill.state["timers"]] == ["2026-10-12T16:00"]


def test_older_reminders_without_a_key_are_still_cancelled(todo):
    todo.timer_skill.timer({"action": "set", "at": "2026-10-12T16:00", "label": "File the review is due at 17:00"})
    todo.fake.items["old"] = {"id": "old", "title": "File the review", "due": "2026-10-12T00:00:00.000Z",
                              "notes": "Deadline: 17:00", "status": "needsAction"}
    todo.todo({"action": "done", "task": "review"})
    assert todo.timer_skill.state["timers"] == []


def test_list_soonest_first_and_scopes(todo):
    todo.todo({"action": "add", "title": "No rush"})
    todo.todo({"action": "add", "title": "Invoice Steel Media", "due": "2026-10-08"})
    todo.todo({"action": "add", "title": "Pitch", "due": "2026-10-11"})
    todo.todo({"action": "add", "title": "Christmas cards", "due": "2026-12-01"})
    out = todo.todo({"action": "list"})
    lines = out.split("\n")[1:6]
    assert lines == ["4 open tasks:", "- Invoice Steel Media (due overdue since Thursday 8 October)",
                     "- Pitch (due tomorrow)", "- Christmas cards (due Tuesday 1 December)", "- No rush (no deadline)"]
    assert "Christmas" not in todo.todo({"action": "list", "scope": "week"})
    assert "Pitch" not in todo.todo({"action": "list", "scope": "overdue"})


def test_done_move_and_undo(todo):
    todo.todo({"action": "add", "title": "Pitch the cosy games feature", "due": "2026-10-16"})
    assert todo.todo({"action": "move", "task": "cosy", "due": "2026-10-19"}) == \
        "Moved: Pitch the cosy games feature, now due Monday 19 October."
    assert todo.todo({"action": "undo"}) == "Undone: Pitch the cosy games feature is back to due Friday 16 October."
    assert todo.todo({"action": "done", "task": "that"}) == "Ticked off: Pitch the cosy games feature."
    assert todo.fake.open_tasks() == []
    assert todo.todo({"action": "undo"}) == "Undone: Pitch the cosy games feature is back on the list."
    assert len(todo.fake.open_tasks()) == 1
    with pytest.raises(ToolError, match="nothing to undo"):
        todo.todo({"action": "undo"})


def test_undo_an_add(todo):
    todo.todo({"action": "add", "title": "Wrong thing"})
    assert todo.todo({"action": "undo"}) == "Undone: removed Wrong thing."
    assert todo.fake.items == {}


def test_delete_needs_a_yes_next_turn(todo):
    todo.todo({"action": "add", "title": "Old task"})
    set_turn(2, "delete the old task")
    assert todo.todo({"action": "delete", "task": "old"}).startswith("Proposed, NOT done yet: delete 'Old task'")
    with pytest.raises(ToolError, match="hasn't answered yet"):
        todo.todo({"action": "confirm_delete"})
    assert "delete the task 'Old task'" in (set_turn(3, "Yes, go ahead.") or todo.pending_note())
    assert todo.todo({"action": "confirm_delete"}) == "Deleted: Old task."
    assert todo.fake.items == {}


def test_delete_refused_without_a_clear_yes(todo):
    todo.todo({"action": "add", "title": "Keep me"})
    set_turn(2, "delete keep me")
    todo.todo({"action": "delete", "task": "keep"})
    set_turn(3, "no, actually leave it")
    with pytest.raises(ToolError, match="didn't clearly say yes"):
        todo.todo({"action": "confirm_delete"})
    assert len(todo.fake.items) == 1


def test_finding_tasks(todo):
    todo.todo({"action": "add", "title": "Email the editor"})
    todo.todo({"action": "add", "title": "Email the printer"})
    with pytest.raises(ToolError, match="more than one task matches"):
        todo.todo({"action": "done", "task": "email"})
    assert todo.todo({"action": "done", "task": "the printer"}) == "Ticked off: Email the printer."
    with pytest.raises(ToolError, match="no open task matches"):
        todo.todo({"action": "done", "task": "dentist"})


def test_daily_note_and_card(todo):
    todo.todo({"action": "add", "title": "Invoice", "due": "2026-10-08"})
    todo.todo({"action": "add", "title": "Pitch", "due": "2026-10-10T17:00"})
    todo.todo({"action": "add", "title": "Later", "due": "2026-10-20"})
    assert todo.daily_note() is None  # not looked up yet: the turn never waits for Google
    todo.tick()
    note = todo.daily_note()
    assert note.startswith("From his to-do list, due today or overdue: Invoice (due overdue since Thursday 8 "
                           "October); Pitch (due today at 17:00).")
    assert todo.daily_note() is None
    card = todo.card()
    assert [(i["title"], i["when"], i["late"]) for i in card["items"]] == [
        ("Invoice", "OVERDUE", True), ("Pitch", "17:00", False), ("Later", "TUE 20", False)]


def test_needs_sign_in_and_bad_dates(tmp_path):
    c = TasksConnector(ACCOUNT, FakeAuth(ok=False), client_factory=FakeTasks, now=lambda: NOW)
    with pytest.raises(NeedsLogin, match="--google-login personal"):
        c.todo({"action": "list"})
    assert c.card()["error"] == "SIGN-IN NEEDED"
    ok = TasksConnector(ACCOUNT, FakeAuth(), client_factory=FakeTasks, now=lambda: NOW)
    with pytest.raises(ToolError, match="isn't a date"):
        ok.todo({"action": "add", "title": "x", "due": "next week sometime"})


def test_daily_note_waits_ten_minutes_after_a_failure(tmp_path):
    from datetime import date

    from jarvis.connectors.base import DailyNote

    calls, clock, fail = [], [0.0], [True]

    def compute():
        calls.append(clock[0])
        if fail[0]:
            raise NeedsLogin("signed out")
        return "Two things due."

    note = DailyNote(tmp_path / "s.json", "day", compute, lambda: date(2026, 10, 10), clock=lambda: clock[0])
    note.prepare()
    clock[0] = 30
    note.prepare()  # the next tick: too soon to try again
    assert calls == [0.0] and note.take() is None
    clock[0], fail[0] = 601, False
    note.prepare()
    assert note.take() == "Two things due." and note.take() is None
