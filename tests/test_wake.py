"""Wake word: phrase stripping, end-of-speech detection and the listener's state machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.audio.wake import FRAME, PREROLL_FRAMES, Endpointer, WakeListener, heard_wake_phrase, strip_wake_phrase

MODELS = Path(__file__).resolve().parent.parent / "models"


def test_strip_wake_phrase():
    assert strip_wake_phrase("Hey Jarvis, what time is it?") == "What time is it?"
    assert strip_wake_phrase("hey jarvis") == ""
    assert strip_wake_phrase("Jarvis. Set a timer.") == "Set a timer."
    assert strip_wake_phrase("Hey, Jarvis! Any emails?") == "Any emails?"
    assert strip_wake_phrase("What's the weather?") == "What's the weather?"  # no phrase: unchanged


def test_heard_wake_phrase():
    assert all(map(heard_wake_phrase, ["Hey Jarvis, what time is it?", "Hey, Jarvis.", "Jarvis, any emails?",
                                       "Okay Jarvis set a timer", "Hey Travis what's on"]))
    assert not any(map(heard_wake_phrase, ["Claire, that looks so good. I can wipe the dryer.",
                                           "What's the weather like?", "So yesterday I ended up talking about it with Jarvis's dad"
                                           " for ages"]))


def run(ep: Endpointer, levels):
    for i, lv in enumerate(levels):
        r = ep.feed(lv)
        if r:
            return r, i + 1
    return None, len(levels)


def test_endpointer():
    speech, quiet = 0.8, 0.05
    assert run(Endpointer(0.1), [speech] * 10 + [quiet] * 20) == ("done", 10 + 11)   # 0.9 s of quiet ends it
    assert run(Endpointer(0.1), [quiet] * 50) == ("nothing", 44)                     # 3.5 s and no speech
    assert run(Endpointer(0.1), [speech] * 200)[0] == "done"                         # 12 s cap
    assert run(Endpointer(0.1), [speech, quiet] * 2 + [quiet] * 50)[0] == "nothing"  # a couple of blips aren't speech
    assert Endpointer(0.5).threshold == pytest.approx(0.62)                          # noisy room: higher bar


class FakeDetector:
    def __init__(self, fire_at):
        self.fire_at, self.n, self.resets = set(fire_at), 0, 0

    def score(self, frame):
        self.n += 1
        return 0.9 if self.n in self.fire_at else 0.0

    def reset(self):
        self.resets += 1


def frames(level_amp, count):
    return [np.full(FRAME, level_amp, dtype=np.int16) for _ in range(count)]


def listener(detector, busy=lambda: False):
    got = []
    w = WakeListener(on_wake=lambda: got.append("wake"),
                     on_utterance=lambda a, s, by_phrase: got.append(("said", round(s, 2), by_phrase)),
                     on_nothing=lambda: got.append("nothing"), busy=busy, detector=detector)
    return w, got


LOUD, SILENT = 8000, 0


def test_wake_then_request_then_silence():
    w, got = listener(FakeDetector(fire_at={10}))
    for f in frames(SILENT, 10) + frames(LOUD, 15) + frames(SILENT, 12):
        w.process(f)
    # 10 frames of pre-roll (all there was) + 15 of speech + 11 of quiet = 36 frames of 80 ms
    assert PREROLL_FRAMES >= 10
    assert got == ["wake", ("said", 2.88, True)]
    assert w.state == "armed"


def test_wake_with_nothing_said():
    w, got = listener(FakeDetector(fire_at={5}))
    for f in frames(SILENT, 60):
        w.process(f)
    assert got == ["wake", "nothing"]


def test_no_detection_while_busy_and_a_reset_after():
    busy = [True]
    det = FakeDetector(fire_at={1, 2, 3})
    w, got = listener(det, busy=lambda: busy[0])
    for f in frames(SILENT, 5):
        w.process(f)
    assert got == [] and det.n == 0  # the model isn't even run while Jarvis talks
    busy[0] = False
    for f in frames(SILENT, 3):
        w.process(f)
    assert det.resets == 1 and got == []  # and its first frames afterwards are ignored (echo)


def test_listen_now_and_cancel():
    w, got = listener(FakeDetector(fire_at=set()))
    w.listen_now()
    for f in frames(LOUD, 5) + frames(SILENT, 11):
        w.process(f)
    assert got == [("said", 1.28, False)]  # a follow-up: no wake phrase expected
    w.listen_now()
    w.cancel()
    assert w.state == "armed"


def test_odd_sized_chunks_are_reframed():
    det = FakeDetector(fire_at=set())
    w, _ = listener(det)
    w.process(np.zeros(1000, dtype=np.int16))
    w.process(np.zeros(1000, dtype=np.int16))
    assert det.n == 1  # 2000 samples = one full 1280-sample frame, 720 kept for next time


@pytest.mark.skipif(not (MODELS / "openwakeword" / "hey_jarvis_v0.1.onnx").exists()
                    or not (MODELS / "kokoro").exists(), reason="needs the downloaded models")
def test_real_model_on_synthesised_speech():
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly

    tts = Kokoro(str(MODELS / "kokoro" / "kokoro-v1.0.int8.onnx"), str(MODELS / "kokoro" / "voices-v1.0.bin"))
    audio, _ = tts.create("Hey Jarvis, what time is it?", voice="bm_george", speed=1.0, lang="en-gb")
    pcm = np.concatenate([np.zeros(16000), resample_poly(audio, 2, 3), np.zeros(32000)])
    pcm = (np.clip(pcm, -1, 1) * 32767 * 0.6).astype(np.int16)
    got = []
    w = WakeListener(on_wake=lambda: got.append("wake"), on_utterance=lambda a, s, p: got.append(("said", s)),
                     on_nothing=lambda: got.append("nothing"), busy=lambda: False,
                     model_dir=MODELS / "openwakeword")
    for i in range(0, len(pcm), 1600):  # 100 ms chunks, like an odd-sized microphone block
        w.process(pcm[i:i + 1600])
    assert got[0] == "wake" and got[1][0] == "said" and 2.0 < got[1][1] < 4.5  # with the 1.6 s pre-roll
