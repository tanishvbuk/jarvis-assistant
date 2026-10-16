from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest

from jarvis import events
from jarvis.hud import Hud
from jarvis.hud.widget import WidgetHost, is_show_me
from jarvis.hud.window import load_place, parse_hotkey, save_place, start_position, widget_args


def test_hotkey_parsing():
    assert parse_hotkey("ctrl+alt+j") == (0x2 | 0x1, ord("J"))
    assert parse_hotkey("Ctrl+Shift+F9") == (0x2 | 0x4, 0x78)
    assert parse_hotkey("j") is None            # a bare key would steal typing
    assert parse_hotkey("ctrl+alt+banana") is None


def test_widget_args_follow_the_config(tmp_path):
    args = widget_args({"on_top": True, "hotkey": "ctrl+alt+k"}, tmp_path / "w.json")
    assert args[:2] == ["--mode", "widget"]
    assert "--on-top" in args and "--transparent" not in args
    assert "--collapse" in args and args[args.index("--orb") + 1] == "320"
    assert widget_args({"orb": 260}, tmp_path / "w.json")[widget_args({"orb": 260}, tmp_path / "w.json").index("--orb") + 1] == "260" and args[args.index("--hotkey") + 1] == "ctrl+alt+k"


def test_place_is_remembered_and_defaults_to_top_right(tmp_path):
    f = tmp_path / "widget.json"
    assert load_place(f) == {}
    assert start_position(380, 24, {}, 1920) == (1516, 24)
    save_place(f, {"r": 1000, "y": 20})
    assert start_position(380, 24, load_place(f), 1920) == (620, 20)   # the right edge is what's kept
    assert start_position(150, 24, load_place(f), 1920) == (850, 20)
    f.write_text("not json")
    assert load_place(f) == {}


@pytest.mark.parametrize("text", ["Show me", "show me that.", "Jarvis, show me that please", "show it", "Put that on screen",
                                  "Let me see it"])
def test_show_me_phrases(text):
    assert is_show_me(text)


@pytest.mark.parametrize("text", ["show me the weather in Paris", "show me my calendar", "what's on my calendar"])
def test_other_requests_go_to_the_model(text):
    assert not is_show_me(text)


class FakeHud:
    url = "http://127.0.0.1:1/"
    pending_focus = None
    on_open = on_close = None


class FakeWindow:
    def __init__(self, url):
        self.url, self.running = url, True

    def close(self):
        self.running = False


def make_host(clock):
    made: list[FakeWindow] = []
    hud = FakeHud()
    host = WidgetHost(hud, lambda u: made.append(FakeWindow(u)) or made[-1], clock=clock)
    return host, hud, made


def test_show_me_opens_the_hud_with_the_last_card():
    now = [100.0]
    host, hud, made = make_host(lambda: now[0])
    try:
        assert host.show_me("show me") == "There's nothing on screen to show, sir."
        events.publish("focus", card="calendar")
        assert host.show_me("show me that") == "Of course, sir."
        assert hud.pending_focus == "calendar" and made[0].url.endswith("?auto=1")
        assert host.show_me("what's the weather") is None
        got: list[dict] = []
        unsub = events.subscribe(got.append)
        host.open("tasks")                        # already open: tells the page instead of a second window
        unsub()
        assert len(made) == 1 and {"t": "show", "card": "tasks"} in got
        host.close()
        assert not made[0].running
    finally:
        host.stop()


def test_old_focus_is_forgotten():
    now = [0.0]
    host, hud, made = make_host(lambda: now[0])
    try:
        events.publish("focus", card="gym")
        now[0] = 10_000
        assert host.show_me("show me") == "There's nothing on screen to show, sir."
    finally:
        host.stop()


def test_briefing_opens_the_hud_by_itself():
    host, hud, made = make_host(lambda: 0.0)
    try:
        events.publish("briefing")
        assert hud.pending_focus == "briefing" and len(made) == 1
    finally:
        host.stop()


def test_server_open_message_and_claim():
    opened: list[str] = []
    closed: list[bool] = []
    hud = Hud(port=0, on_open=opened.append, on_close=lambda: closed.append(True))
    hud.start()

    async def run():
        base = f"http://127.0.0.1:{hud.port}"
        async with aiohttp.ClientSession() as s:
            async with s.get(base + "/widget") as r:
                assert r.status == 200 and "Jarvis widget" in await r.text()
            async with s.ws_connect(base + "/ws", headers={"Origin": base}) as ws:
                await ws.send_str(json.dumps({"t": "open", "card": "calendar"}))
                await ws.send_str(json.dumps({"t": "close"}))
                await asyncio.sleep(0.3)
            async with s.ws_connect(base + "/ws") as plain:               # the widget: gets no focus
                assert "focus" not in json.loads((await plain.receive()).data)
            async with s.ws_connect(base + "/ws?claim=1") as hudpage:     # the HUD opened for it
                assert json.loads((await hudpage.receive()).data)["focus"] == "calendar"
            async with s.ws_connect(base + "/ws?claim=1") as again:       # handed over once only
                assert "focus" not in json.loads((await again.receive()).data)

    try:
        asyncio.run(run())
        assert opened == ["calendar"] and closed == [True]
    finally:
        hud.stop()
