from __future__ import annotations

import os

from ..config import Config, ConfigError
from .base import Transcriber


def make_transcriber(cfg: Config) -> Transcriber:
    stt = cfg.stt
    provider = stt.get("provider", "groq")
    if provider == "groq":
        from .groq_whisper import GroqWhisper

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ConfigError("stt.provider is 'groq' but GROQ_API_KEY is not set in .env.")
        return GroqWhisper(api_key=key, model=stt.get("groq_model", "whisper-large-v3-turbo"),
                           language=stt.get("language", "en"))
    if provider == "local":
        from .local_whisper import LocalWhisper

        return LocalWhisper(model=stt.get("local_model", "base.en"), language=stt.get("language", "en"))
    raise ConfigError(f"Unknown stt.provider {provider!r}")
