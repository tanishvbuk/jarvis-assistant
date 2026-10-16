"""A tiny publish/subscribe bus so the voice loop can tell the HUD what's happening.

The voice loop only ever calls publish(); it never imports web code. With nobody
subscribed (text mode, or the HUD switched off) publish() does nothing.

Events are plain dicts with a "t" (type) key:
    {"t": "state", "s": "idle" | "listening" | "thinking" | "speaking"}
    {"t": "heard", "text": "..."}         what the user said (shown, never logged)
    {"t": "say", "text": "..."}           a sentence, sent as Jarvis starts speaking it
    {"t": "level", "v": 0..1, "src": "mic" | "voice"}
    {"t": "info", ...}                    brain, ears, voice names
    {"t": "turn", ...}                    timings at the end of a turn
    {"t": "card", "id": "meals", ...}     data for a HUD card (screen only, never sent to the model)
    {"t": "watched", "v": bool}           a HUD page has come on screen (True) or none is (False)
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

Subscriber = Callable[[dict], None]

_subscribers: list[Subscriber] = []
_lock = threading.Lock()


def subscribe(fn: Subscriber) -> Callable[[], None]:
    """Register fn; returns a function that unsubscribes it."""
    with _lock:
        _subscribers.append(fn)

    def unsubscribe() -> None:
        with _lock:
            if fn in _subscribers:
                _subscribers.remove(fn)

    return unsubscribe


def active() -> bool:
    return bool(_subscribers)


def publish(t: str, **data) -> None:
    if not _subscribers:
        return
    event = {"t": t, **data}
    with _lock:
        subs = list(_subscribers)
    for fn in subs:
        try:
            fn(event)
        except Exception:
            log.exception("Event subscriber failed on %s", t)


_watched = False


def set_watched(v: bool) -> None:
    """The HUD server says whether any page is actually on screen (not minimised, hidden or closed)."""
    global _watched
    if v != _watched:
        _watched = v
        publish("watched", v=v)


def watched() -> bool:
    """Whether a HUD card refresh would be seen. False with no HUD at all, so cards cost nothing then."""
    return _watched


def state(s: str) -> None:
    publish("state", s=s)


def level_of(samples: np.ndarray) -> float:
    """Loudness of int16 audio as 0..1 on a log scale (-60 dBFS -> 0, -10 dBFS -> 1)."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0
    if rms <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, (20 * math.log10(rms) + 60) / 50))
