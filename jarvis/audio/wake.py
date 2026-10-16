"""Hands-free: listen for "hey Jarvis", then record until the user stops talking.

Everything here runs on the laptop: the microphone audio goes to the wake word
model (openWakeWord, ONNX) and nowhere else until the phrase is heard. Then the
request is recorded until about a second of silence and handed over like a
push-to-talk recording.

Detection pauses while Jarvis is busy (speaking, thinking, or a push-to-talk
recording), so his own voice can't wake him; F9 still interrupts.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from .. import events

log = logging.getLogger(__name__)

RATE = 16000
FRAME = 1280                  # 80 ms, what openWakeWord expects
FRAME_S = FRAME / RATE
PREROLL_FRAMES = 20           # keep 1.6 s before the detection: the transcript then includes "hey Jarvis",
                              # which confirms the wake-up (see heard_wake_phrase)
COOLDOWN_FRAMES = 8           # ignore 0.6 s after Jarvis stops talking (room echo)

WAKE_PHRASE = re.compile(r"^\W*(?:(?:hey|hi|okay|ok|a)\W+)?(?:jarvis|jervis|javis|travis)\b\W*", re.I)


def heard_wake_phrase(text: str) -> bool:
    """Second opinion from the transcript: does it really start with "hey Jarvis"? (Rejects false wake-ups.)"""
    return bool(WAKE_PHRASE.match(text)) or bool(re.search(r"\b(jarvis|jervis)\b", " ".join(text.split()[:5]), re.I))


def strip_wake_phrase(text: str) -> str:
    """'Hey Jarvis, what time is it?' -> 'what time is it?' (the transcript usually keeps the phrase)."""
    rest = WAKE_PHRASE.sub("", text, count=1).strip()
    return rest[:1].upper() + rest[1:] if rest else ""


class Endpointer:
    """Decides when a spoken request has finished, from loudness alone.

    feed() returns None (keep going), "done" (speech then enough silence, or too long)
    or "nothing" (no speech at all within start_timeout).
    """

    def __init__(self, floor: float, *, start_timeout: float = 3.5, silence: float = 0.9,
                 max_seconds: float = 12.0, min_speech: float = 0.24, margin: float = 0.12):
        self.threshold = max(floor + margin, 0.25)  # level_of() scale: 0.02 per dB
        self.start_frames = round(start_timeout / FRAME_S)
        self.silence_frames = round(silence / FRAME_S)
        self.max_frames = round(max_seconds / FRAME_S)
        self.min_speech = round(min_speech / FRAME_S)
        self.frames = self.speech = self.quiet = 0

    def feed(self, level: float) -> str | None:
        self.frames += 1
        if level >= self.threshold:
            self.speech += 1
            self.quiet = 0
        else:
            self.quiet += 1
        if self.frames >= self.max_frames:
            return "done" if self.speech >= self.min_speech else "nothing"
        if self.speech < self.min_speech:
            return "nothing" if self.frames >= self.start_frames else None
        return "done" if self.quiet >= self.silence_frames else None


class OwwDetector:
    """openWakeWord's "hey jarvis" model, ONNX, from models/openwakeword/."""

    def __init__(self, model_dir: Path):
        from openwakeword.model import Model

        self.model = Model(wakeword_models=[str(model_dir / "hey_jarvis_v0.1.onnx")], inference_framework="onnx",
                           melspec_model_path=str(model_dir / "melspectrogram.onnx"),
                           embedding_model_path=str(model_dir / "embedding_model.onnx"))

    def score(self, frame: np.ndarray) -> float:
        return float(max(self.model.predict(frame).values()))

    def reset(self) -> None:
        self.model.reset()


class WakeListener:
    def __init__(self, *, on_wake: Callable[[], None], on_utterance: Callable[[np.ndarray, float, bool], None],
                 on_nothing: Callable[[], None], busy: Callable[[], bool], detector=None,
                 model_dir: Path | None = None, threshold: float = 0.5, device=None):
        self.detector = detector or OwwDetector(model_dir)
        self.threshold = threshold
        self.device = device
        self._on_wake, self._on_utterance, self._on_nothing, self._busy = on_wake, on_utterance, on_nothing, busy
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=200)
        self._stream = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.state = "armed"  # "armed" (listening for the phrase) or "capturing" (recording the request)
        self._preroll: deque[np.ndarray] = deque(maxlen=PREROLL_FRAMES)
        self._by_phrase = True  # this recording began with the wake phrase (not a "Yes, sir?" follow-up)
        self._audio: list[np.ndarray] = []
        self._endpoint: Endpointer | None = None
        self._floor = 0.2
        self._was_busy = False
        self._cooldown = 0
        self._pending = np.zeros(0, dtype=np.int16)

    # ---- lifecycle

    def start(self) -> None:
        import sounddevice as sd

        self._thread = threading.Thread(target=self._run, name="wake-word", daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME,
                                      device=self.device, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._q.put(None)

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # the worker is behind; dropping a frame is harmless

    def _run(self) -> None:
        while True:
            chunk = self._q.get()
            if chunk is None:
                return
            try:
                self.process(chunk)
            except Exception:
                log.exception("Wake word processing failed")

    # ---- control from the voice loop

    def listen_now(self) -> None:
        """Record a request without the wake phrase (after "Yes, sir?")."""
        with self._lock:
            self._start_capture(preroll=False)

    def cancel(self) -> None:
        """Push-to-talk took over: drop any request being recorded."""
        with self._lock:
            self.state, self._audio, self._endpoint = "armed", [], None

    # ---- the state machine (fed 80 ms frames; public for tests)

    def process(self, chunk: np.ndarray) -> None:
        # Frames may arrive in odd sizes; the model wants exactly 1280 samples.
        data = np.concatenate([self._pending, chunk]) if self._pending.size else chunk
        n = len(data) // FRAME * FRAME
        self._pending = data[n:]
        for i in range(0, n, FRAME):
            self._frame(data[i:i + FRAME])

    def _frame(self, frame: np.ndarray) -> None:
        level = events.level_of(frame)
        with self._lock:
            if self.state == "capturing":
                self._audio.append(frame)
                events.publish("level", v=round(level, 3), src="mic")
                result = self._endpoint.feed(level)
                if result is None:
                    return
                audio = np.concatenate(self._audio)
                self.state, self._audio, self._endpoint = "armed", [], None
                self._was_busy = True  # reset the detector before listening again
                by_phrase = self._by_phrase
                callback = (lambda: self._on_utterance(audio, len(audio) / RATE, by_phrase)) if result == "done" \
                    else self._on_nothing
            else:
                callback = self._armed(frame, level)
        if callback:
            callback()

    def _armed(self, frame: np.ndarray, level: float):
        if self._busy():
            self._was_busy = True
            self._preroll.clear()
            return None
        if self._was_busy:  # Jarvis has just gone quiet: forget what the model heard meanwhile
            self.detector.reset()
            self._was_busy = False
            self._cooldown = COOLDOWN_FRAMES
        self._floor = 0.97 * self._floor + 0.03 * level if level < self._floor + 0.1 else self._floor
        self._preroll.append(frame)
        score = self.detector.score(frame)
        if self._cooldown:
            self._cooldown -= 1
            return None
        if score < self.threshold:
            return None
        log.info("Wake word heard (score %.2f)", score)
        self._start_capture(preroll=True)
        return self._on_wake

    def _start_capture(self, preroll: bool) -> None:
        self.state = "capturing"
        self._by_phrase = preroll
        self._audio = list(self._preroll) if preroll else []
        self._preroll.clear()
        self._endpoint = Endpointer(self._floor)
