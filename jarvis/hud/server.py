"""Local web server for the HUD: one HTML page plus a WebSocket.

Runs on its own thread with its own asyncio loop. Events from jarvis.events are
handed to that loop with call_soon_threadsafe and pushed to every open page.

Security: binds to 127.0.0.1 only, and checks the Host and Origin headers, so
neither another machine nor some other website open in the browser can connect
to the socket and work the microphone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Callable

from aiohttp import WSMsgType, web

from .. import events

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
CLIENT_QUEUE = 256  # events buffered per page before level updates start being dropped


def _quiet_resets(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """A browser closing or reloading the page makes Windows' asyncio print a
    ConnectionResetError traceback. It's harmless; keep it out of the console."""
    if isinstance(context.get("exception"), ConnectionResetError):
        log.debug("HUD connection reset: %s", context.get("message"))
        return
    loop.default_exception_handler(context)


class Hud:
    """The HUD web server.

    on_talk receives "down"/"up" when the on-screen ring is held and released
    (None disables click-to-talk, e.g. in text mode).

    The desktop widget (/widget) asks for the full HUD with {"t":"open","card":...}: on_open(card) runs, and
    `pending_focus` is what the full HUD page, which connects a moment later with ?claim=1, shows first. A
    full HUD that opened on demand says {"t":"close"} when it is done: on_close() runs.
    """

    def __init__(self, port: int = 8765, on_talk: Callable[[str], None] | None = None, info: dict | None = None,
                 on_open: Callable[[str], None] | None = None, on_close: Callable[[], None] | None = None):
        self.port = port
        self.on_talk = on_talk
        self.on_open = on_open
        self.on_close = on_close
        self.pending_focus: str | None = None
        self._snapshot = {"state": "idle", "heard": "", "reply": [], "info": dict(info or {}),
                          "talk": on_talk is not None, "cards": {}}
        self._clients: set[asyncio.Queue] = set()
        self._seen: dict[asyncio.Queue, bool] = {}  # whether each page is on screen (it says when that changes)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._allowed_hosts: set[str] = set()
        self.window = None  # the HUD's own window (hud/window.py), if one was opened
        self.widget_host = None  # with window = "widget": opens the full HUD on request (hud/widget.py)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> str:
        """Start serving; returns the URL. Raises OSError if the port is taken."""
        ready = threading.Event()
        error: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.set_exception_handler(_quiet_resets)
            try:
                loop.run_until_complete(self._serve())
            except BaseException as e:  # reported to start()
                error.append(e)
                ready.set()
                loop.close()
                return
            ready.set()
            loop.run_forever()
            loop.run_until_complete(self._runner.cleanup())
            loop.close()

        self._thread = threading.Thread(target=run, name="hud", daemon=True)
        self._thread.start()
        ready.wait(timeout=10)
        if error:
            raise error[0]
        self._unsubscribe = events.subscribe(self._on_event)
        return self.url

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def stop(self) -> None:
        events.set_watched(False)
        if self.widget_host:
            self.widget_host.stop()
        if self.window:
            self.window.close()
        if self._unsubscribe:
            self._unsubscribe()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    async def _serve(self) -> None:
        app = web.Application(middlewares=[self._check_host])
        app.router.add_get("/", self._index)
        app.router.add_get("/widget", self._widget)
        app.router.add_get("/ws", self._ws)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        self.port = self._runner.addresses[0][1]  # the real port when 0 was asked for
        self._allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    # ------------------------------------------------------------ events in

    def _on_event(self, event: dict) -> None:
        """Called on whichever thread published; hops onto the server loop."""
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._broadcast, event)

    def _broadcast(self, event: dict) -> None:
        t = event["t"]
        snap = self._snapshot
        if t == "state":
            snap["state"] = event["s"]
            if event["s"] == "listening":
                snap["heard"], snap["reply"] = "", []
        elif t == "heard":
            snap["heard"], snap["reply"] = event["text"], []
        elif t == "say":
            snap["reply"].append(event["text"])
        elif t == "info":
            snap["info"].update({k: v for k, v in event.items() if k != "t"})
        elif t == "card":
            snap["cards"][event.get("id", "")] = event  # the latest of each card, for pages that (re)connect
        message = json.dumps(event)
        for q in self._clients:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass  # a stalled page; it gets a fresh snapshot when it reconnects

    # ------------------------------------------------------------ HTTP

    @web.middleware
    async def _check_host(self, request: web.Request, handler):
        # Rejecting other Host names blocks DNS-rebinding tricks from web pages.
        if request.host not in self._allowed_hosts:
            raise web.HTTPForbidden(text="Jarvis HUD only answers on 127.0.0.1")
        return await handler(request)

    async def _index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def _widget(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "widget.html", headers={"Cache-Control": "no-store"})

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        origin = request.headers.get("Origin")
        if origin is not None and origin.rstrip("/") not in {f"http://{h}" for h in self._allowed_hosts}:
            log.warning("HUD refused a WebSocket from origin %r", origin)
            raise web.HTTPForbidden(text="Wrong origin")

        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=CLIENT_QUEUE)
        hello = {"t": "hello", **self._snapshot}
        if request.query.get("claim") and self.pending_focus:
            hello["focus"], self.pending_focus = self.pending_focus, None  # handed to the HUD that was opened for it
        await ws.send_str(json.dumps(hello))
        self._clients.add(q)
        self._set_seen(q, True)  # a page that just loaded is on screen until it says otherwise
        sender = asyncio.create_task(self._send_loop(ws, q))
        held = False
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                if data.get("t") == "seen":
                    self._set_seen(q, bool(data.get("v")))
                elif data.get("t") == "open" and self.on_open:
                    card = str(data.get("card") or "")[:20]
                    self.pending_focus = card or None
                    self.on_open(card)
                elif data.get("t") == "close" and self.on_close:
                    self.on_close()
                elif data.get("t") == "talk" and self.on_talk:
                    action = data.get("a")
                    if action == "down" and not held:
                        held = True
                        self.on_talk("down")
                    elif action == "up" and held:
                        held = False
                        self.on_talk("up")
        finally:
            self._clients.discard(q)
            self._set_seen(q, None)
            sender.cancel()
            if held and self.on_talk:  # page closed or connection dropped mid-press
                self.on_talk("up")
        return ws

    def _set_seen(self, q: asyncio.Queue, seen: bool | None) -> None:
        if seen is None:
            self._seen.pop(q, None)
        else:
            self._seen[q] = seen
        events.set_watched(any(self._seen.values()))

    @staticmethod
    async def _send_loop(ws: web.WebSocketResponse, q: asyncio.Queue) -> None:
        try:
            while True:
                await ws.send_str(await q.get())
        except (ConnectionResetError, RuntimeError):
            pass
