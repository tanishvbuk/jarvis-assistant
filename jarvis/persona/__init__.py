"""The persona lives in jarvis.md so it can be tuned without touching code."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
from zoneinfo import ZoneInfo

PERSONA_FILE = Path(__file__).with_name("jarvis.md")
LOCAL = ZoneInfo(os.environ.get("JARVIS_TZ") or "UTC")  # set JARVIS_TZ in .env, e.g. America/New_York


def spoken(d, fmt: str = "%A %d %B") -> str:
    """A date as it's said, without leading zeros: "Friday 9 October"."""
    return d.strftime(fmt).replace(" 0", " ")


def system_prompt(now: datetime | None = None) -> str:
    now = now or datetime.now(LOCAL)
    # Day/month order and 24-hour time; the model can say it naturally.
    stamp = spoken(now, "%A %d %B %Y, %H:%M")
    return (PERSONA_FILE.read_text(encoding="utf-8").strip() + f"\n\nCurrent local time: {stamp} ({LOCAL.key}).\n"
            + upcoming_days(now))


def upcoming_days(now: datetime, days: int = 8) -> str:
    """The coming week's dates, so the model reads "Friday" off a list instead of working it out.

    Models are unreliable at date arithmetic: "this Saturday", asked on a Saturday, came back
    as the following Monday or Tuesday. A short list fixes that for about 60 tokens.
    """
    when = [now + timedelta(days=i) for i in range(days)]
    listed = ", ".join(f"{d:%a} {d.date().isoformat()}" + {0: " (today)", 1: " (tomorrow)"}.get(i, "")
                       for i, d in enumerate(when))
    return f"Dates for the coming week: {listed}. Take days of the week from this list; don't work dates out."
