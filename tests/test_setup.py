"""Phase 1 setup tests.

Verifies the *foundation* only:
  1. the configuration loads with the agreed initial values,
  2. the required project directories exist,
  3. the inspection script fails gracefully when no dataset is present,
  4. the inspection heuristics work on a throwaway CSV.

Test 4 uses a deliberately generic CSV written into pytest's ``tmp_path`` with
column names that are NOT CIC-IDS2018 names. It exercises the heuristics; it is
never written into ``data/raw`` and is not a stand-in for the real dataset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, Config, ConfigError
from src.preprocessing.inspection import (
    DatasetNotFoundError,
    discover_files,
    inspect_directory,
    inspect_file,
)

INSPECT_SCRIPT = PROJECT_ROOT / "scripts" / "inspect_dataset.py"
DATASET_MISSING_MESSAGE = (
    "Dataset not found. Add CIC-IDS2018 CSV to the configured data/raw directory."
)


# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
def test_default_config_file_exists():
    assert (PROJECT_ROOT / "configs" / "default.yaml").is_file()


def test_config_loads_with_expected_initial_values(config: Config):
    assert config.temporal.window_size == 5, "temporal window T"
    assert config.split.train == pytest.approx(0.70)
    assert config.split.validation == pytest.approx(0.10)
    assert config.split.test == pytest.approx(0.20)
    assert config.split.total == pytest.approx(1.0)
    assert config.debug.row_limit == 100_000
    assert isinstance(config.debug.enabled, bool)
    assert isinstance(config.seed, int)


def test_config_exposes_dataset_and_output_paths(config: Config):
    assert config.dataset_dir.is_absolute()
    assert config.output_dir.is_absolute()
    assert config.dataset_dir == PROJECT_ROOT / "data" / "raw"
    assert config.output_dir == PROJECT_ROOT / "data" / "processed"


def test_debug_row_limit_is_only_applied_in_debug_mode(config: Config):
    config.debug.enabled = False
    assert config.debug.row_limit_or_none is None
    config.debug.enabled = True
    assert config.debug.row_limit_or_none == config.debug.row_limit


def test_config_rejects_split_ratios_that_do_not_sum_to_one():
    cfg = Config()
    cfg.split.train = 0.8
    cfg.split.validation = 0.1
    cfg.split.test = 0.2
    with pytest.raises(ConfigError, match="sum to 1.0"):
        cfg.validate()


def test_config_rejects_invalid_window_size():
    cfg = Config()
    cfg.temporal.window_size = 0
    with pytest.raises(ConfigError, match="window_size"):
        cfg.validate()


def test_config_rejects_unknown_keys():
    with pytest.raises(ConfigError, match="Unknown key"):
        Config.from_dict({"temporal": {"window_size": 5, "typo_key": 1}})


def test_config_summary_mentions_key_settings(config: Config):
    summary = config.summary()
    for expected in ("dataset dir", "temporal window T", "split ratios", "debug mode"):
        assert expected in summary


# --------------------------------------------------------------------------- #
# 2. Directory layout
# --------------------------------------------------------------------------- #
def test_required_directories_exist(config: Config):
    missing = [str(d) for d in config.paths.required_dirs if not d.is_dir()]
    assert not missing, f"missing directories: {missing}"


@pytest.mark.parametrize(
    "relative",
    [
        "configs",
        "data/raw",
        "data/processed",
        "src/preprocessing",
        "src/features",
        "src/graph",
        "src/models",
        "src/pruning",
        "src/training",
        "src/evaluation",
        "src/utils",
        "scripts",
        "tests",
        "checkpoints",
        "results",
        "plots",
        "notebooks",
        "logs",
    ],
)
def test_project_structure_directory_present(relative: str):
    assert (PROJECT_ROOT / relative).is_dir(), f"{relative} is missing"


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "requirements.txt",
        "requirements-gnn.txt",
        "environment.yml",
        "pyproject.toml",
        "configs/default.yaml",
        "src/config.py",
        "src/preprocessing/inspection.py",
        "scripts/inspect_dataset.py",
    ],
)
def test_project_file_present(relative: str):
    assert (PROJECT_ROOT / relative).is_file(), f"{relative} is missing"


def test_ensure_directories_is_idempotent(tmp_path: Path):
    cfg = Config()
    cfg.paths.raw_data_dir = tmp_path / "raw"
    cfg.paths.processed_data_dir = tmp_path / "processed"
    cfg.paths.results_dir = tmp_path / "results"
    cfg.paths.plots_dir = tmp_path / "plots"
    cfg.paths.checkpoints_dir = tmp_path / "checkpoints"
    cfg.paths.logs_dir = tmp_path / "logs"
    for _ in range(2):
        created = cfg.ensure_directories()
    assert all(d.is_dir() for d in created)


# --------------------------------------------------------------------------- #
# 3. Graceful failure when the dataset is absent
# --------------------------------------------------------------------------- #
def test_discover_files_returns_empty_for_empty_dir(tmp_path: Path):
    assert discover_files(tmp_path) == []


def test_discover_files_returns_empty_for_missing_dir(tmp_path: Path):
    assert discover_files(tmp_path / "does-not-exist") == []


def test_inspect_directory_raises_dataset_not_found(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        inspect_directory(tmp_path)


def test_inspection_script_reports_missing_dataset(empty_data_dir: Path, tmp_path: Path):
    """The CLI must exit with code 2, print the guidance and show no traceback."""
    result = subprocess.run(
        [
            sys.executable,
            str(INSPECT_SCRIPT),
            "--data-dir",
            str(empty_data_dir),
            "--no-save",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 2, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert DATASET_MISSING_MESSAGE in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_inspection_script_help_works():
    result = subprocess.run(
        [sys.executable, str(INSPECT_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "--debug" in result.stdout


# --------------------------------------------------------------------------- #
# 4. Inspection heuristics on a throwaway, non-CIC CSV
# --------------------------------------------------------------------------- #
def test_inspect_file_reports_generic_csv(generic_csv: Path):
    report = inspect_file(generic_csv, chunk_size=2)

    assert report.filename == generic_csv.name
    assert report.shape == (4, 5)
    assert report.columns == ["event_time", "bytes_in", "ratio_metric", "proto", "verdict"]

    assert "event_time" in [c.column for c in report.timestamp_candidates]
    assert "verdict" in [c.column for c in report.label_candidates]
    assert report.selected_label_column == "verdict"
    assert report.class_distribution == {"normal": 3, "suspicious": 1}

    assert "bytes_in" in report.numeric_columns
    assert "proto" in report.non_numeric_columns
    assert report.missing_counts.get("bytes_in") == 1
    assert report.total_missing == 1
    assert report.infinity_counts.get("ratio_metric") == 1
    assert report.total_infinities == 1
    assert not report.truncated


def test_inspect_file_honours_row_limit(generic_csv: Path):
    report = inspect_file(generic_csv, chunk_size=10, row_limit=2)
    assert report.n_rows == 2
    assert report.truncated
    assert report.class_distribution_truncated


def test_inspect_file_flags_whitespace_in_column_names(tmp_path: Path):
    path = tmp_path / "spaced.csv"
    path.write_text(" leading,trailing \n1,2\n", encoding="utf-8")
    report = inspect_file(path)
    assert report.columns_with_whitespace == [" leading", "trailing "]
