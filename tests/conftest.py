"""Shared pytest fixtures.

No fixture ever writes into ``data/raw`` — sample CSVs live in pytest's
``tmp_path`` and use generic column names, so nothing here can be mistaken for
the real CIC-IDS2018 data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config


@pytest.fixture(scope="session")
def config() -> Config:
    """The project's default configuration, with directories ensured."""
    cfg = Config.load()
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def empty_data_dir(tmp_path: Path) -> Path:
    """An existing but empty stand-in for the raw data directory."""
    path = tmp_path / "empty_raw"
    path.mkdir()
    return path


@pytest.fixture
def generic_csv(tmp_path: Path) -> Path:
    """A tiny, deliberately generic CSV used to exercise the heuristics.

    Column names are intentionally NOT CIC-IDS2018 names. It contains one
    missing value and one infinity so those code paths are covered.
    """
    path = tmp_path / "generic_traffic_sample.csv"
    path.write_text(
        "event_time,bytes_in,ratio_metric,proto,verdict\n"
        "2024-01-01 00:00:00,100,0.5,tcp,normal\n"
        "2024-01-01 00:00:05,,1.5,udp,normal\n"
        "2024-01-01 00:00:10,300,Infinity,tcp,suspicious\n"
        "2024-01-01 00:00:15,400,2.5,tcp,normal\n",
        encoding="utf-8",
    )
    return path
