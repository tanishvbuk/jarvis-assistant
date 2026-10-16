from __future__ import annotations

import logging
from typing import Callable

from ..audio import effects
from ..config import Config
from .base import SpeechQueue, Voice

log = logging.getLogger(__name__)


def voice_factory(cfg: Config) -> Callable[[], Voice]:
    """The configured voice, falling back Kokoro -> Piper -> Windows voice if one won't load."""
    tts = cfg.tts
    effect = effects.get(tts.get("effect", "none"))  # checked now, so a typo fails at startup

    def kokoro() -> Voice:
        from .kokoro import KokoroVoice

        return KokoroVoice(cfg.root / tts.get("kokoro_model", ""), cfg.root / tts.get("kokoro_voices", ""),
                           voice=tts.get("kokoro_voice", "bm_daniel"), speed=float(tts.get("kokoro_speed", 1.0)),
                           lang=tts.get("kokoro_lang", "en-gb"), effect=effect)

    def piper() -> Voice:
        from .piper import PiperVoiceTTS

        return PiperVoiceTTS(cfg.root / tts.get("piper_model", ""), speed=float(tts.get("piper_speed", 1.0)),
                             effect=effect)

    def sapi() -> Voice:
        from .sapi import SapiVoice  # speaks straight to the speakers, so no effect

        return SapiVoice(tts.get("sapi_voice", ""), rate=int(tts.get("sapi_rate", 0)))

    chains = {"kokoro": [kokoro, piper, sapi], "piper": [piper, sapi], "sapi": [sapi]}

    def build() -> Voice:
        chain = chains.get(tts.get("provider", "piper"), [piper, sapi])
        for make in chain[:-1]:
            try:
                return make()
            except Exception as e:
                log.warning("%s voice unavailable (%s); trying the next one.", make.__name__.capitalize(), e)
        return chain[-1]()

    return build


__all__ = ["SpeechQueue", "Voice", "voice_factory"]
