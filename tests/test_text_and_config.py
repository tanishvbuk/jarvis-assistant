from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.config import ROOT, ConfigError, load_config
from jarvis.persona import LOCAL, system_prompt
from jarvis.text import SentenceSplitter, clean_for_speech


def split_all(chunks):
    s = SentenceSplitter()
    out = [x for c in chunks for x in s.feed(c)]
    return out + s.flush()


def test_splitter_streams_sentences():
    assert split_all(["Good morn", "ing, sir. Three ite", "ms today! Shall I", " begin?"]) == [
        "Good morning, sir.", "Three items today!", "Shall I begin?"]


def test_splitter_keeps_abbreviations_and_decimals():
    assert split_all(["Mr. Smith paid £3.50 today. Dr. Jones did not."]) == [
        "Mr. Smith paid £3.50 today.", "Dr. Jones did not."]


def test_clean_for_speech_strips_markdown_and_urls():
    assert clean_for_speech("**Note:** see https://example.com now") == "Note: see now"
    assert clean_for_speech("- first\n- second") == "first second"


def test_system_prompt_has_persona_and_local_time():
    prompt = system_prompt(datetime(2026, 10, 9, 7, 5, tzinfo=LOCAL))
    assert "sir" in prompt and "no markdown" in prompt.lower()
    assert "Current local time: Friday 9 October 2026, 7:05 (Europe/Lisbon).\n" in prompt
    assert "Fri 2026-10-09 (today), Sat 2026-10-10 (tomorrow), Sun 2026-10-11," in prompt
    assert prompt.rstrip().endswith("don't work dates out.")


def test_coming_week_crosses_the_month_and_the_clocks_going_back():
    from jarvis.persona import upcoming_days

    late = upcoming_days(datetime(2026, 10, 24, 23, 30, tzinfo=LOCAL))  # BST ends on Sunday 25th
    assert "Sat 2026-10-24 (today), Sun 2026-10-25 (tomorrow), Mon 2026-10-26," in late
    assert "Sat 2026-10-31" in late and late.count("2026-") == 8


def test_shipped_config_loads_and_switches_provider(tmp_path):
    cfg = load_config(env_file=tmp_path / "none.env")
    assert cfg.provider.name == "groq" and cfg.provider.kind == "openai_compat"
    assert cfg.provider.extra == {"reasoning_effort": "low", "include_reasoning": False}
    cfg = load_config(env_file=tmp_path / "none.env", provider="anthropic")
    assert cfg.provider.kind == "anthropic" and cfg.provider.model == "claude-haiku-5-5"
    with pytest.raises(ConfigError):
        load_config(env_file=tmp_path / "none.env", provider="nonexistent")


def test_missing_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(env_file=tmp_path / "none.env", provider="anthropic")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        cfg.provider.api_key()


# Patterns for real-looking keys from the providers we use, and Google OAuth secrets and tokens.
SECRET = re.compile(r"(gsk_[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|sk-or-[A-Za-z0-9_-]{20,}"
                    r"|sk-[A-Za-z0-9]{32,}|GOCSPX-[A-Za-z0-9_-]{10,}|1//0[A-Za-z0-9_-]{20,}"
                    r"|ya29\.[A-Za-z0-9_-]{20,})")


def test_no_secrets_in_tracked_or_staged_files():
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert ".env" not in files, ".env must stay gitignored"
    assert "config.local.toml" not in files, "config.local.toml (account addresses) must stay gitignored"
    assert not [f for f in files if f.startswith("tokens/")], "tokens/ (Google sign-ins) must stay gitignored"
    hits = []
    for name in files:
        path = Path(ROOT, name)
        if path.is_file() and path.suffix not in {".onnx"}:
            if SECRET.search(path.read_text(encoding="utf-8", errors="ignore")):
                hits.append(name)
    assert hits == []


def test_splitter_separates_run_together_sentences():
    assert split_all(["It has been undone, sir.Your list now ", "has two items."]) == [
        "It has been undone, sir.", "Your list now has two items."]
    assert split_all(["The U.S.Army and Mr.Smith paid £3.50 today."]) == [
        "The U.S.Army and Mr.Smith paid £3.50 today."]
