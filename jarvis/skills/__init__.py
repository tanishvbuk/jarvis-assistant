"""Things Jarvis can do (as opposed to connectors, which read the user's data)."""

from __future__ import annotations

from ..config import Config


def build_skills(cfg: Config) -> list:
    out = []
    if cfg.timers.get("enabled", True):
        from .timers import Timers

        out.append(Timers(cfg.root / cfg.timers.get("store", "data/timers.json")))
    return out
