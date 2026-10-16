"""Windows built-in voices through SAPI 5 (pywin32).

Used directly rather than via pyttsx3, which tends to hang after the first
utterance on Windows. Must be created on the thread that uses it.
"""

from __future__ import annotations

import threading

SVSF_ASYNC = 1
SVSF_PURGE = 2
BRITISH_LCID = "809"  # en-GB


def _voices(sp):
    tokens = sp.GetVoices()
    return [tokens.Item(i) for i in range(tokens.Count)]


def list_voices() -> list[str]:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    sp = win32com.client.Dispatch("SAPI.SpVoice")
    return [f"{t.GetDescription()}  (language {t.GetAttribute('Language')}, {t.GetAttribute('Gender')})"
            for t in _voices(sp)]


def _score(token) -> int:
    """Prefer British male, then British female, then any male."""
    lang = (token.GetAttribute("Language") or "").lower()
    gender = (token.GetAttribute("Gender") or "").lower()
    british = BRITISH_LCID in lang.split(";")
    return (2 if british else 0) + (1 if gender == "male" else 0)


class SapiVoice:
    def __init__(self, preferred: str = "", rate: int = 0):
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self._sp = win32com.client.Dispatch("SAPI.SpVoice")
        voices = _voices(self._sp)
        chosen = None
        if preferred:
            chosen = next((t for t in voices if preferred.lower() in t.GetDescription().lower()), None)
        if chosen is None and voices:
            chosen = max(voices, key=_score)
        if chosen is not None:
            self._sp.Voice = chosen
        self._sp.Rate = rate
        self.name = f"sapi:{self._sp.Voice.GetDescription()}"

    def speak(self, text: str, cancel: threading.Event) -> None:
        self._sp.Speak(text, SVSF_ASYNC)
        while not self._sp.WaitUntilDone(50):
            if cancel.is_set():
                self._sp.Speak("", SVSF_ASYNC | SVSF_PURGE)
                return
