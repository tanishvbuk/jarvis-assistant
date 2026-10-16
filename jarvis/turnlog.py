"""Per-turn timings, token counts and estimated cost, one JSON line per turn.

Never logs audio or what was said.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from .persona import LOCAL


@dataclass
class TurnRecord:
    mode: str = "voice"
    provider: str = ""
    model: str = ""
    stt: str = ""
    voice: str = ""
    record_s: float | None = None
    transcribe_s: float | None = None
    first_token_s: float | None = None   # from end of transcription
    first_audio_s: float | None = None   # from key release (what you actually wait)
    total_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    est_cost_gbp: float = 0.0
    retries: int = 0
    tools: list[str] = field(default_factory=list)  # names of tools called, never their arguments
    error: str | None = None
    ts: str = field(default_factory=lambda: datetime.now(LOCAL).isoformat(timespec="seconds"))


class TurnLog:
    def __init__(self, log_dir: Path):
        self.dir = log_dir

    def write(self, rec: TurnRecord) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"turns-{rec.ts[:10]}.jsonl"
        data = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(rec).items()}
        data["est_cost_usd"] = round(rec.est_cost_usd, 6)
        data["est_cost_gbp"] = round(rec.est_cost_gbp, 6)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
        return path
