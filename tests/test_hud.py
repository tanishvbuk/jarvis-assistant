from __future__ import annotations

import asyncio
import queue
import threading

import aiohttp
import numpy as np
import pytest

from jarvis import events
from jarvis.hud import Hud
from jarvis.tts.base import SpeechQueue


@pytest.fixture
def captured():
    got: list[dict] = []
    unsubscribe = events.subscribe(got.append)
    yield got
    unsubscribe()


def test_publish_reaches_subscribers_until_unsubscribed():
    got: list[dict] = []
    unsubscribe = events.subscribe(got.append)
    events.state("thinking")
    unsubscribe()
    events.state("idle")
    assert got == [{"t": "state", "s": "thinking"}]


def test_publish_survives_a_broken_subscriber(captured):
    def broken(_):
        raise RuntimeError("boom")

    unsubscribe = events.subscribe(broken)
    events.publish("say", text="Still here.")
    unsubscribe()
    assert captured == [{"t": "say", "text": "Still here."}]


def test_level_scale():
    assert events.level_of(np.zeros(800, dtype=np.int16)) == 0.0
    assert events.level_of(np.full(800, 32767, dtype=np.int16)) == 1.0
    quiet = events.level_of(np.full(800, 100, dtype=np.int16))  # about -50 dBFS
    assert 0.1 < quiet < 0.3


class FakeVoice:
    name = "fake"

    def __init__(self, log):
        self.log = log

    def speak(self, text, cancel):
        self.log.append({"t": "spoken", "text": text})


def test_speech_queue_announces_each_sentence_then_drains(captured):
    drained = threading.Event()
    speech = SpeechQueue(lambda: FakeVoice(captured), on_drained=drained.set)
    speech.say("Good evening, sir.")
    speech.say("All systems nominal.")
    assert speech.wait(timeout=5) and drained.wait(timeout=5)
    speech.close()
    says = [e for e in captured if e["t"] in ("say", "spoken")]
    assert says == [
        {"t": "say", "text": "Good evening, sir."}, {"t": "spoken", "text": "Good evening, sir."},
        {"t": "say", "text": "All systems nominal."}, {"t": "spoken", "text": "All systems nominal."}]
    assert {"t": "state", "s": "speaking"} in captured


# ------------------------------------------------------------------ the server

@pytest.fixture
def hud():
    talk: queue.Queue[str] = queue.Queue()
    server = Hud(port=0, on_talk=talk.put, info={"brain": "fake / model"})
    server.start()
    server.talk = talk
    yield server
    server.stop()


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=10))


async def receive(ws, kind):
    while True:
        msg = await ws.receive_json()
        if msg["t"] == kind:
            return msg


def origin(server):
    return {"Origin": f"http://127.0.0.1:{server.port}"}


def test_serves_the_page_on_localhost_only(hud):
    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.get(hud.url) as r:
                assert r.status == 200 and "<title>Jarvis</title>" in await r.text()
            async with s.get(hud.url, headers={"Host": "evil.example"}) as r:
                assert r.status == 403
    run(go())


def test_websocket_refuses_other_websites(hud):
    async def go():
        async with aiohttp.ClientSession() as s:
            with pytest.raises(aiohttp.WSServerHandshakeError) as e:
                await s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers={"Origin": "https://evil.example"})
            assert e.value.status == 403
    run(go())


def test_websocket_gets_snapshot_then_live_events(hud):
    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers=origin(hud)) as ws:
                hello = await receive(ws, "hello")
                assert hello["state"] == "idle" and hello["talk"] is True
                assert hello["info"] == {"brain": "fake / model"}
                events.publish("heard", text="what time is it")
                events.state("speaking")
                events.publish("say", text="It is five past seven, sir.")
                assert (await receive(ws, "heard"))["text"] == "what time is it"
                assert (await receive(ws, "state"))["s"] == "speaking"
                assert (await receive(ws, "say"))["text"] == "It is five past seven, sir."
            # a page opened now sees the turn so far
            async with s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers=origin(hud)) as ws:
                hello = await receive(ws, "hello")
                assert hello["heard"] == "what time is it"
                assert hello["reply"] == ["It is five past seven, sir."]
    run(go())


def test_cards_reach_pages_live_and_on_reconnect(hud):
    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers=origin(hud)) as ws:
                assert (await receive(ws, "hello"))["cards"] == {}
                events.publish("card", id="meals", sections=[{"label": "ALEX", "items": []}])
                assert (await receive(ws, "card"))["sections"][0]["label"] == "ALEX"
            async with s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers=origin(hud)) as ws:
                assert (await receive(ws, "hello"))["cards"]["meals"]["sections"][0]["label"] == "ALEX"
    run(go())


def test_hold_to_talk_and_release_on_disconnect(hud):
    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers=origin(hud)) as ws:
                await receive(ws, "hello")
                await ws.send_json({"t": "talk", "a": "down"})
                await ws.send_json({"t": "talk", "a": "down"})  # repeats are ignored
                await ws.send_json({"t": "talk", "a": "up"})
                await ws.send_json({"t": "talk", "a": "down"})
            # closed while held: the server lets go for it

    run(go())
    got = [hud.talk.get(timeout=5) for _ in range(4)]
    assert got == ["down", "up", "down", "up"]
    assert hud.talk.empty()


def test_talk_is_ignored_without_voice_mode():
    server = Hud(port=0, on_talk=None)
    server.start()
    try:
        async def go():
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(f"ws://127.0.0.1:{server.port}/ws", headers=origin(server)) as ws:
                    assert (await receive(ws, "hello"))["talk"] is False
                    await ws.send_json({"t": "talk", "a": "down"})
        run(go())
    finally:
        server.stop()


def test_busy_port_raises_oserror(hud):
    with pytest.raises(OSError):
        Hud(port=hud.port).start()


def test_window_process_does_not_load_the_web_server():
    import subprocess
    import sys

    code = "import sys, jarvis.hud.window; print('aiohttp' in sys.modules, 'jarvis.hud.server' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]


def test_window_falls_back_when_it_cannot_start():
    import threading

    from jarvis.hud import window

    failed = threading.Event()
    w = window.HudWindow.__new__(window.HudWindow)
    w._proc = __import__("subprocess").Popen([window.sys.executable, "-c", "import sys; sys.exit(3)"])
    w._started = __import__("time").monotonic()
    w._on_failed = failed.set
    w._watch()
    assert failed.is_set()


def test_window_closed_by_user_tells_jarvis():
    import subprocess
    import sys
    import threading
    import time

    from jarvis.hud import window

    closed, failed = threading.Event(), threading.Event()
    w = window.HudWindow.__new__(window.HudWindow)
    w._proc = subprocess.Popen([sys.executable, "-c", "pass"])  # exits 0, like the user closing it
    w._started, w._on_failed, w._on_closed = time.monotonic(), failed.set, closed.set
    w._watch()
    assert closed.is_set() and not failed.is_set()


def test_window_notices_when_jarvis_goes_away():
    import subprocess
    import sys
    import threading

    from jarvis.hud.window import _wait_for_parent

    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    gone = threading.Event()
    t = threading.Thread(target=_wait_for_parent, args=(parent.pid, gone.set), daemon=True)
    t.start()
    assert gone.wait(10)


def test_show_existing_without_a_window(monkeypatch):
    from jarvis.hud import window

    monkeypatch.setattr(window, "SHOW_EVENT", r"Local\JarvisShowHudTest")  # never the real, running window
    assert window.show_existing() is False


def test_icon_is_a_valid_ico(tmp_path):
    import struct

    from jarvis.launcher import ICON_SIZES, make_icon

    data = make_icon(tmp_path / "j.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind, count) == (0, 1, len(ICON_SIZES))
    for i, size in enumerate(ICON_SIZES):
        w, h, _, _, planes, bits, length, offset = struct.unpack("<BBBBHHII", data[6 + 16 * i:22 + 16 * i])
        assert (w or 256) == size and bits == 32 and data[offset:offset + 8] == b"\x89PNG\r\n\x1a\n"
        assert offset + length <= len(data)


def test_cards_are_watched_only_while_a_page_is_on_screen(hud):
    import time

    async def go():
        assert not events.watched()
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"ws://127.0.0.1:{hud.port}/ws", headers=origin(hud)) as ws:
                await receive(ws, "hello")
                assert (await receive(ws, "watched"))["v"] is True  # on screen as it connects
                assert events.watched()
                await ws.send_json({"t": "seen", "v": False})  # minimised
                assert (await receive(ws, "watched"))["v"] is False
                await ws.send_json({"t": "seen", "v": True})   # restored
                assert (await receive(ws, "watched"))["v"] is True
        for _ in range(50):  # closed: the server notices the socket went
            if not events.watched():
                break
            await asyncio.sleep(0.05)
        assert not events.watched()

    run(go())


def test_jobs_run_at_once_when_a_page_comes_on_screen():
    from jarvis.__main__ import start_jobs

    ticks = threading.Semaphore(0)

    class Card:
        name = "card"

        def tick(self):
            ticks.release()

    jobs = start_jobs([Card()])
    try:
        assert ticks.acquire(timeout=5)       # the first round, straight away
        events.set_watched(True)
        assert ticks.acquire(timeout=5)       # not 30 seconds later
    finally:
        events.set_watched(False)
        jobs.set()
