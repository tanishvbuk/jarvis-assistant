"""Local neural voice with Piper (https://github.com/OHF-Voice/piper1-gpl).

Voice files (.onnx plus .onnx.json) live in models/piper/, which is gitignored.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio.playback import play


class PiperVoiceTTS:
    def __init__(self, model_path: Path, speed: float = 1.0,
                 effect: Callable[[np.ndarray, int], np.ndarray] | None = None):
        from piper import PiperVoice, SynthesisConfig

        if not model_path.exists():
            raise FileNotFoundError(f"Piper voice not found: {model_path}")
        self.name = f"piper:{model_path.stem}"
        self._voice = PiperVoice.load(model_path)
        self._syn = SynthesisConfig(length_scale=speed)
        self._effect = effect

    def speak(self, text: str, cancel: threading.Event) -> None:
        for chunk in self._voice.synthesize(text, self._syn):
            if cancel.is_set():
                return
            audio = chunk.audio_int16_array
            if self._effect:
                audio = self._effect(audio, chunk.sample_rate)
            play(audio, chunk.sample_rate, cancel)
