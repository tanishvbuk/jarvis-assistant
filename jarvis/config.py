"""Load config.toml (settings) and .env (secrets)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ProviderConfig:
    name: str
    kind: str  # "openai_compat" or "anthropic"
    model: str
    api_key_env: str
    base_url: str | None = None
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    effort: str | None = None
    thinking: str | None = None
    extra: dict = field(default_factory=dict)  # provider-specific request options

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ConfigError(f"{self.api_key_env} is not set. Add it to the .env file (see .env.example).")
        return key


@dataclass
class Config:
    provider: ProviderConfig
    max_tokens: int = 200
    max_tokens_with_data: int = 400
    history_turns: int = 10
    max_turns_per_session: int = 200
    usd_to_gbp: float = 0.75
    stt: dict = field(default_factory=dict)
    tts: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)
    hud: dict = field(default_factory=dict)
    widget: dict = field(default_factory=dict)
    google: dict = field(default_factory=dict)
    gmail: dict = field(default_factory=dict)
    meal_plan: dict = field(default_factory=dict)
    timers: dict = field(default_factory=dict)
    calendar: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    opengym: dict = field(default_factory=dict)
    briefing: dict = field(default_factory=dict)
    websearch: dict = field(default_factory=dict)
    currency: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    log_dir: Path = ROOT / "logs"
    root: Path = ROOT


class ConfigError(Exception):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None = None, env_file: Path | None = None, provider: str | None = None) -> Config:
    """Read config.toml, then config.local.toml beside it if present (private settings, not in Git)."""
    path = path or ROOT / "config.toml"
    load_dotenv(env_file or ROOT / ".env", override=False)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    local = path.with_name("config.local.toml")
    if local.exists():
        with open(local, "rb") as f:
            raw = _merge(raw, tomllib.load(f))

    llm = raw.get("llm", {})
    name = provider or llm.get("provider", "groq")
    providers = llm.get("providers", {})
    if name not in providers:
        raise ConfigError(f"llm.provider = {name!r} but there is no [llm.providers.{name}] block in config.toml.")
    p = providers[name]
    provider = ProviderConfig(
        name=name,
        kind=p.get("kind", "openai_compat"),
        model=p["model"],
        api_key_env=p["api_key_env"],
        base_url=p.get("base_url"),
        price_in_per_mtok=float(p.get("price_in_per_mtok", 0.0)),
        price_out_per_mtok=float(p.get("price_out_per_mtok", 0.0)),
        effort=p.get("effort"),
        thinking=p.get("thinking"),
        extra=dict(p.get("extra", {})),
    )
    root = path.resolve().parent
    return Config(
        provider=provider,
        max_tokens=int(llm.get("max_tokens", 200)),
        max_tokens_with_data=int(llm.get("max_tokens_with_data", 400)),
        history_turns=int(llm.get("history_turns", 10)),
        max_turns_per_session=int(llm.get("max_turns_per_session", 200)),
        usd_to_gbp=float(llm.get("usd_to_gbp", 0.75)),
        stt=raw.get("stt", {}),
        tts=raw.get("tts", {}),
        audio=raw.get("audio", {}),
        hud=raw.get("hud", {}),
        widget=raw.get("widget", {}),
        google=raw.get("google", {}),
        gmail=raw.get("gmail", {}),
        meal_plan=raw.get("meal_plan", {}),
        timers=raw.get("timers", {}),
        calendar=raw.get("calendar", {}),
        tasks=raw.get("tasks", {}),
        opengym=raw.get("opengym", {}),
        briefing=raw.get("briefing", {}),
        websearch=raw.get("websearch", {}),
        currency=raw.get("currency", {}),
        safety=raw.get("safety", {}),
        log_dir=root / raw.get("logging", {}).get("dir", "logs"),
        root=root,
    )
