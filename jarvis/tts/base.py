"""Text-to-speech interface and the background speech queue."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Protocol

from .. import events

log = logging.getLogger(__name__)


class Voice(Protocol):
    name: str

    def speak(self, text: str, cancel: threading.Event) -> None:
        """Say `text`, blocking until done or until `cancel` is set."""
        ...

    # Optional, for slow voices (Kokoro):
    #   prefetch(text)  start rendering a queued sentence early
    #   prepare(text)   block until it's rendered, so sound starts the moment speak() is called
    #   discard()       forget sentences rendered ahead (after an interruption)


class SpeechQueue:
    """Speaks sentences one after another on a background thread.

    The voice is built inside that thread (Windows COM voices require it).
    """

    def __init__(self, voice_factory: Callable[[], Voice], on_drained: Callable[[], None] | None = None):
        self._factory = voice_factory
        self._on_drained = on_drained  # called on the speech thread whenever the queue empties
        self._q: queue.Queue[str | None] = queue.Queue()
        self._cancel = threading.Event()
        self._pending = 0  # sentences queued or being spoken
        self._cond = threading.Condition()
        self._ready = threading.Event()
        self._voice: Voice | None = None
        self.voice_name = "?"
        self.first_audio_at: float | None = None
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60)

    def _run(self) -> None:
        voice = self._factory()
        self.voice_name = voice.name
        self._voice = voice
        self._ready.set()
        while True:
            text = self._q.get()
            if text is None:
                return
            if not self._cancel.is_set() and hasattr(voice, "prepare"):
                try:
                    voice.prepare(text)
                except Exception:
                    log.exception("Speech failed")
            if not self._cancel.is_set():
                if self.first_audio_at is None:
                    self.first_audio_at = time.perf_counter()
                # Sent as the sentence starts, so the HUD types in step with the voice.
                events.state("speaking")
                events.publish("say", text=text)
                try:
                    voice.speak(text, self._cancel)
                except Exception:
                    log.exception("Speech failed")
            with self._cond:
                self._pending -= 1
                drained = self._pending <= 0
                self._cond.notify_all()
            if drained and self._on_drained:
                self._on_drained()

    def say(self, text: str) -> None:
        with self._cond:
            self._pending += 1
        prefetch = getattr(self._voice, "prefetch", None)
        if prefetch and not self._cancel.is_set():
            prefetch(text)
        self._q.put(text)

    def new_turn(self) -> None:
        self._cancel.clear()
        self.first_audio_at = None

    def stop(self) -> None:
        """Cut Jarvis off: drop queued sentences and stop the current one."""
        self._cancel.set()
        dropped = 0
        try:
            while True:
                if self._q.get_nowait() is not None:
                    dropped += 1
        except queue.Empty:
            pass
        with self._cond:
            self._pending -= dropped
            self._cond.notify_all()
        discard = getattr(self._voice, "discard", None)
        if discard:
            discard()

    @property
    def speaking(self) -> bool:
        with self._cond:
            return self._pending > 0

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken (or cut off)."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending <= 0, timeout)

    def close(self) -> None:
        self.stop()
        self._q.put(None)
