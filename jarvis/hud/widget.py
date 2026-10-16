"""The desktop widget's side in the Jarvis process: opening the full HUD on request.

With `[hud] window = "widget"` the everyday window is the small widget (hud/window.py --mode widget, page
/widget). Showing a card is on request: the widget offers a chip after a tool answer, and a click on it, or
"show me", comes here. The full HUD is then started as a second window (/?auto=1) with the card forward, and
closes itself after the answer. The morning briefing is the one automatic exception.

This module is also where the "show me" phrase is recognised, in code and before the model (no tokens).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

from .. import events

log = logging.getLogger(__name__)

RECENT_S = 600  # "show me" refers to a card shown in the last ten minutes

_FILLER = re.compile(r"\b(jarvis|please|sir|would you|could you|can you|will you)\b")
_SHOW = re.compile(r"^(show( it| that| this)?( to)?( me| us)?( that| it| this| again)?|put (it|that) (up )?on( the)? screen|"
                   r"(open|bring up|pull up) (it|that|the (hud|display|screen))|let me see( it| that)?)$")


def is_show_me(text: str) -> bool:
    """Only the short, plain request counts: "show me", "show me that", "show it", "put that on screen"."""
    t = _FILLER.sub(" ", re.sub(r"[^a-z' ]", " ", text.lower()))
    return bool(_SHOW.match(" ".join(t.split())))


class WidgetHost:
    """Opens and closes the on-demand full HUD window. `make_window(url)` starts one (a HudWindow)."""

    def __init__(self, hud, make_window: Callable[[str], object], clock: Callable[[], float] = time.monotonic):
        self.hud = hud
        self._make = make_window
        self._clock = clock
        self._window = None
        self._lock = threading.Lock()
        self._last: tuple[str, float] | None = None  # (card, when) of the last tool-driven focus
        hud.on_open = self.open
        hud.on_close = self.close
        self._unsubscribe = events.subscribe(self._on_event)

    def stop(self) -> None:
        self._unsubscribe()
        self.close()

    # ---- events: remember what was last shown; the briefing opens the HUD by itself
    def _on_event(self, event: dict) -> None:
        t = event["t"]
        if t == "focus":
            self._last = (event.get("card", ""), self._clock())
        elif t == "briefing":
            self.open("briefing")

    # ---- the HUD window
    @property
    def is_open(self) -> bool:
        return bool(self._window and getattr(self._window, "running", False))

    def open(self, card: str = "") -> None:
        """Start the full HUD with `card` forward (or show it on the one already open)."""
        card = card or (self._last[0] if self._last and self._clock() - self._last[1] < RECENT_S else "")
        with self._lock:
            if self.is_open:
                events.publish("show", card=card)  # only the full HUD page listens to this
                return
            self.hud.pending_focus = card or None
            self._window = self._make(self.hud.url + "?auto=1")

    def close(self) -> None:
        with self._lock:
            window, self._window = self._window, None
        if window:
            window.close()

    # ---- "show me"
    def show_me(self, text: str) -> str | None:
        """If `text` is a plain "show me", open the HUD and return what to say; otherwise None (the model answers)."""
        if not is_show_me(text):
            return None
        if not (self._last and self._clock() - self._last[1] < RECENT_S) and not self.is_open:
            return "There's nothing on screen to show, sir."
        self.open()
        return "Of course, sir."
