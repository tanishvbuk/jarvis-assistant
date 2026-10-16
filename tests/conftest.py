
from __future__ import annotations

import os

os.environ["JARVIS_TZ"] = "Europe/Lisbon"  # fixed zone for the tests (same DST rules as the expected times)

from pathlib import Path

import pytest

from jarvis.config import Config, ProviderConfig


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        provider=ProviderConfig(name="fake", kind="openai_compat", model="fake-model", api_key_env="FAKE_KEY",
                                price_in_per_mtok=1.0, price_out_per_mtok=5.0),
        log_dir=tmp_path / "logs",
        root=tmp_path,
    )
