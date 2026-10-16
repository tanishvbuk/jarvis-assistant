"""Speech-to-text with Groq's hosted Whisper (OpenAI-compatible audio endpoint)."""

from __future__ import annotations

import numpy as np
import openai

from ..llm.base import AuthFailed, LLMError, RateLimited, Unavailable
from .base import to_wav_bytes


class GroqWhisper:
    name = "groq"

    def __init__(self, *, api_key: str, model: str = "whisper-large-v3-turbo", language: str = "en",
                 base_url: str = "https://api.groq.com/openai/v1"):
        self.model = model
        self.language = language
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=1, timeout=20.0)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            result = self._client.audio.transcriptions.create(
                file=("speech.wav", to_wav_bytes(audio, sample_rate)),
                model=self.model,
                language=self.language,
            )
        except openai.RateLimitError as e:
            raise RateLimited(str(e)) from e
        except openai.AuthenticationError as e:
            raise AuthFailed(str(e)) from e
        except (openai.APIConnectionError, openai.InternalServerError) as e:
            raise Unavailable(str(e)) from e
        except openai.APIStatusError as e:
            raise LLMError(f"{e.status_code}: {e}") from e
        return (result.text or "").strip()
