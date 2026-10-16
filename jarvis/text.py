"""Turn streamed text into speakable sentences."""

from __future__ import annotations

import re

# Words ending in a full stop that don't end a sentence.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "prof", "vs", "etc", "e.g", "i.e", "approx", "no"}
# A sentence ends at punctuation followed by a space, or straight by a capitalised word
# ("sir.Your list...", which some models produce when they run two messages together).
_BOUNDARY = re.compile(r"([.!?…]+[\"')\]]*)(?:\s+|(?=[A-Z][a-z]))")


class SentenceSplitter:
    """Feed it chunks; it hands back complete sentences as soon as they exist."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        start = 0
        for m in _BOUNDARY.finditer(self._buf):
            candidate = self._buf[start:m.end(1)]
            last_word = candidate.rstrip(".!?…\"')]").rsplit(maxsplit=1)[-1:] or [""]
            if m.group(1).startswith(".") and last_word[0].lower() in _ABBREVIATIONS:
                continue
            if m.end() == m.end(1) and not self._buf[m.start(1) - 1:m.start(1)].islower():
                continue  # no space: only "word.Next" counts, not "U.S.Army" or "3.Five"
            sentence = clean_for_speech(candidate)
            if sentence:
                out.append(sentence)
            start = m.end()
        self._buf = self._buf[start:]
        return out

    def flush(self) -> list[str]:
        rest = clean_for_speech(self._buf)
        self._buf = ""
        return [rest] if rest else []


def clean_for_speech(text: str) -> str:
    """Strip markdown and URLs that a model might slip in despite the persona."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>|]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()
