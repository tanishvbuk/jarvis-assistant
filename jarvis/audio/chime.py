"""A short two-note chime for timer alerts, made on the fly (no sound files)."""

from __future__ import annotations

import threading

import numpy as np

RATE = 24000


def chime_audio() -> np.ndarray:
    def note(freq: float, secs: float) -> np.ndarray:
        t = np.arange(int(RATE * secs)) / RATE
        env = np.exp(-t * 7) * np.minimum(1, t * 200)  # quick attack, soft decay
        return np.sin(2 * np.pi * freq * t) * env

    tone = np.concatenate([note(880, 0.22), note(1318.5, 0.42)])
    return (tone * 0.35 * 32767).astype(np.int16)


def play_chime() -> None:
    from .playback import play

    play(chime_audio(), RATE, threading.Event())
