"""Play raw audio, stoppable mid-sentence."""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from .. import events

BLOCK_SECONDS = 0.05


def play(audio: np.ndarray, sample_rate: int, cancel: threading.Event) -> None:
    """Play mono int16 audio, checking `cancel` every 50 ms."""
    block = int(sample_rate * BLOCK_SECONDS)
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as out:
        for i in range(0, len(audio), block):
            if cancel.is_set():
                return
            chunk = audio[i:i + block]
            events.publish("level", v=round(events.level_of(chunk), 3), src="voice")
            out.write(chunk.reshape(-1, 1))
