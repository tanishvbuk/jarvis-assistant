"""Rolling conversation memory: the last N turns, kept in RAM only."""

from __future__ import annotations

from collections import deque


class History:
    def __init__(self, max_turns: int = 10):
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))

    def messages(self) -> list[dict]:
        out: list[dict] = []
        for user, assistant in self._turns:
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": assistant})
        return out

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
