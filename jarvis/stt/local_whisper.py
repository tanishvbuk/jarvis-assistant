"""Offline speech-to-text with faster-whisper. Install with: pip install -e .[local]

The model (about 150 MB for base.en) downloads on first use.
"""

from __future__ import annotations

import numpy as np


class LocalWhisper:
    name = "local"

    def __init__(self, *, model: str = "base.en", language: str = "en"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("faster-whisper isn't installed. Run: pip install -e .[local]") from e
        self.language = language
        self._model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio; set audio.sample_rate = 16000")
        samples = audio.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(samples, language=self.language, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
