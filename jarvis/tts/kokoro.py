"""Local neural voice with Kokoro (https://github.com/thewh1teagle/kokoro-onnx).

Kokoro-82M is Apache-2.0; kokoro-onnx is MIT and runs it on onnxruntime (no PyTorch).
Model files live in models/kokoro/, which is gitignored.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio.playback import play


class KokoroVoice:
    def __init__(self, model_path: Path, voices_path: Path, voice: str = "bm_daniel", speed: float = 1.0,
                 lang: str = "en-gb", effect: Callable[[np.ndarray, int], np.ndarray] | None = None):
        from kokoro_onnx import Kokoro

        for p in (model_path, voices_path):
            if not p.exists():
                raise FileNotFoundError(f"Kokoro file not found: {p}")
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        if voice not in self._kokoro.get_voices():
            raise ValueError(f"Kokoro has no voice {voice!r}")
        self.name = f"kokoro:{voice}"
        self._voice, self._speed, self._lang, self._effect = voice, speed, lang, effect
        # Kokoro takes 0.3-1.2 s per sentence, so sentences are rendered on one worker thread
        # as soon as they're queued, while the one before is still playing.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")
        self._ahead: dict[str, list[Future]] = {}
        self._lock = threading.Lock()
        self._render("Ready.")  # the first call is slow; pay for it at startup, not on the greeting

    def _render(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._kokoro.create(text, voice=self._voice, speed=self._speed, lang=self._lang)
        audio = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        if self._effect:
            audio = self._effect(audio, sr)
        return audio, sr

    def prefetch(self, text: str) -> None:
        """Start rendering a sentence that will be spoken soon."""
        with self._lock:
            self._ahead.setdefault(text, []).append(self._worker.submit(self._render, text))

    def discard(self) -> None:
        """Forget sentences rendered ahead (Jarvis was interrupted)."""
        with self._lock:
            for futures in self._ahead.values():
                for f in futures:
                    f.cancel()
            self._ahead.clear()

    def prepare(self, text: str) -> None:
        """Block until `text` is rendered, so the caller knows sound will start immediately."""
        with self._lock:
            waiting = self._ahead.get(text)
            future = waiting.pop(0) if waiting else None
            if waiting == []:
                del self._ahead[text]
        if future is None or future.cancelled():
            future = self._worker.submit(self._render, text)
        self._ready = (text, future.result())

    def speak(self, text: str, cancel: threading.Event) -> None:
        ready, self._ready = getattr(self, "_ready", None), None
        if ready is None or ready[0] != text:
            self.prepare(text)
            ready, self._ready = self._ready, None
        if cancel.is_set():
            return
        audio, sr = ready[1]
        play(audio, sr, cancel)
