"""Microphone recording and the push-to-talk key."""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from .. import events


class Recorder:
    """Records mono int16 audio between start() and stop(). Audio stays in memory."""

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 30, device: str | int | None = None):
        self.sample_rate = sample_rate
        self.max_frames = int(max_seconds * sample_rate)
        self.device = device or None
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            if self._frames < self.max_frames:
                self._chunks.append(indata[:, 0].copy())
                self._frames += frames
        events.publish("level", v=round(events.level_of(indata[:, 0]), 3), src="mic")

    def start(self) -> None:
        self._chunks, self._frames = [], 0
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                                      blocksize=self.sample_rate // 25,  # 40 ms: drives the HUD ring
                                      device=self.device, callback=self._callback)
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)


def parse_key(name: str):
    """'f9' -> Key.f9, 'scroll_lock' -> Key.scroll_lock, 'x' -> KeyCode for x."""
    from pynput import keyboard

    name = name.strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unknown push-to-talk key {name!r}. Try f9, f8, scroll_lock or a single letter.")


class PushToTalk:
    """Calls on_down once when the key goes down and on_up when it's released.

    Uses pynput, which works on Windows without admin rights and while other
    windows have focus. Windows key-repeat is filtered out.
    """

    def __init__(self, key_name: str, on_down: Callable[[], None], on_up: Callable[[], None]):
        from pynput import keyboard

        self._key = parse_key(key_name)
        self._held = False
        self._on_down, self._on_up = on_down, on_up
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)

    def _press(self, key) -> None:
        if key == self._key and not self._held:
            self._held = True
            self._on_down()

    def _release(self, key) -> None:
        if key == self._key and self._held:
            self._held = False
            self._on_up()

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
