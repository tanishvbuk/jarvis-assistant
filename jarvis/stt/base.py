from __future__ import annotations

import io
import wave
from typing import Protocol

import numpy as np


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """audio: mono int16 samples. Returns the recognised text ('' if nothing)."""
        ...


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Wrap raw int16 samples in a WAV header, in memory (never written to disk)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()
