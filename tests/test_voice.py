from __future__ import annotations

import threading

import numpy as np
import pytest

from jarvis import events
from jarvis.audio import effects
from jarvis.tts import voice_factory
from jarvis.tts.base import SpeechQueue


def test_film_effect_keeps_length_and_never_clips():
    sr = 24000
    t = np.arange(sr) / sr
    loud = (np.sin(2 * np.pi * 220 * t) * 32000).astype(np.int16)
    out = effects.film(loud, sr)
    assert out.dtype == np.int16 and len(out) == len(loud)
    assert np.abs(out.astype(np.int32)).max() <= int(0.95 * 32767) + 1
    assert not np.array_equal(out, loud)
    assert effects.film(np.zeros(0, dtype=np.int16), sr).size == 0


def test_effect_names():
    assert effects.get("none") is None and effects.get("Film") is effects.film
    with pytest.raises(ValueError, match="Unknown voice effect"):
        effects.get("echo")


def test_missing_kokoro_falls_back_to_next_voice(cfg, monkeypatch):
    built = []

    class FakePiper:
        def __init__(self, path, speed=1.0, effect=None):
            self.name, self.effect = "piper:fake", effect
            built.append(self)

    import jarvis.tts.piper as piper_mod

    monkeypatch.setattr(piper_mod, "PiperVoiceTTS", FakePiper)
    cfg.tts = {"provider": "kokoro", "effect": "film", "kokoro_model": "missing.onnx", "kokoro_voices": "missing.bin"}
    voice = voice_factory(cfg)()
    assert voice.name == "piper:fake" and voice.effect is effects.film


def test_bad_effect_name_fails_at_startup(cfg):
    cfg.tts = {"provider": "piper", "effect": "reverb"}
    with pytest.raises(ValueError):
        voice_factory(cfg)


class SlowVoice:
    """Records the order of calls, like Kokoro: prefetch on queue, prepare, then speak."""

    name = "slow"

    def __init__(self, log):
        self.log = log
        self.release = threading.Event()

    def prefetch(self, text):
        self.log.append(("prefetch", text))

    def prepare(self, text):
        self.log.append(("prepare", text))

    def speak(self, text, cancel):
        self.log.append(("speak", text))
        if text == "First.":
            self.release.wait(timeout=5)

    def discard(self):
        self.log.append(("discard",))


def test_speech_queue_renders_ahead_and_announces_after_prepare():
    log: list = []
    unsubscribe = events.subscribe(lambda e: log.append(("say-event", e["text"])) if e["t"] == "say" else None)
    voice = SlowVoice(log)
    speech = SpeechQueue(lambda: voice)
    try:
        speech.say("First.")
        speech.say("Second.")
        voice.release.set()
        assert speech.wait(timeout=5)
    finally:
        unsubscribe()
        speech.close()
    # both sentences start rendering as soon as they're queued
    assert log.index(("prefetch", "Second.")) < log.index(("speak", "First."))
    # the HUD hears about a sentence only once its audio is ready
    for text in ("First.", "Second."):
        assert log.index(("prepare", text)) < log.index(("say-event", text)) < log.index(("speak", text))


def test_interrupting_discards_sentences_rendered_ahead():
    log: list = []
    voice = SlowVoice(log)
    speech = SpeechQueue(lambda: voice)
    speech.say("First.")
    speech.say("Second.")
    speech.stop()
    voice.release.set()
    speech.close()
    assert ("discard",) in log
