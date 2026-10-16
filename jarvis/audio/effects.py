"""Audio effects applied to Jarvis's voice before playback."""

from __future__ import annotations

import numpy as np


def film(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The 'voice in the walls' treatment: a thin comb filter, a little room, and the boom taken out.

    Takes and returns mono int16. Vectorised, so a sentence costs a few milliseconds.
    """
    if audio.size == 0:
        return audio
    x = audio.astype(np.float32) / 32768
    # comb filter: an 11 ms echo gives the slightly metallic, processed colour
    d = int(sample_rate * 0.011)
    y = x.copy()
    y[d:] += 0.32 * x[:-d]
    # three early reflections: a small hard room
    room = y.copy()
    for ms, gain in ((23, 0.18), (37, 0.12), (53, 0.08)):
        k = int(sample_rate * ms / 1000)
        room[k:] += gain * y[:-k]
    # high-pass at roughly 180 Hz: subtract a short moving average
    n = max(1, sample_rate // 180)
    room -= np.convolve(room, np.ones(n, dtype=np.float32) / n, mode="same")
    # the echoes add up; bring peaks back under full scale without changing quiet sentences
    peak = float(np.abs(room).max())
    if peak > 0.95:
        room *= 0.95 / peak
    return (room * 32767).astype(np.int16)


EFFECTS = {"none": None, "film": film}


def get(name: str):
    """Look up an effect by its config name ("none" or "film")."""
    name = (name or "none").strip().lower()
    if name not in EFFECTS:
        raise ValueError(f"Unknown voice effect {name!r}. Use one of: {', '.join(EFFECTS)}.")
    return EFFECTS[name]
