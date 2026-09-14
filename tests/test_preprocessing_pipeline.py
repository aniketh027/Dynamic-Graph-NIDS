"""Tests for src.preprocessing.pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.labels import LABEL_TO_INT
from src.preprocessing.pipeline import (
    PipelineResult,
    _CORPUS_ORDER,
    discover_csvs,
    run_pipeline,
)
from src.preprocessing.schema import (
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    ORIGINAL_INDEX_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.preprocessing.transforms import (
    FitTransformState,
    fit_transforms,
    fit_transforms_indexed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, n_rows: int, label: str = "Benign", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0.1, 10.0, n_rows) for col in MODEL_FEATURE_COLUMNS}
    data[ORIGINAL_INDEX_COLUMN] = np.arange(n_rows)
    data[TIMESTAMP_COLUMN] = [f"14/02/2018 10:{i % 60:02d}:00" for i in range(n_rows)]
    data[LABEL_COLUMN] = label
    pd.DataFrame(data).to_csv(path, index=False)


def _make_fake_config(tmp_path: Path, csv_dir: Path) -> MagicMock:
    """Return a config mock that points all paths at tmp_path."""
    pc = MagicMock()
    pc.chunk_size = 500
    pc.rare_class_min_test_windows = 1
    pc.output_subdir = "preprocessed"
    pc.save_cleaning_stats = True
    pc.save_split_metadata = True

    paths = MagicMock()
    paths.raw_data_dir = csv_dir.parent
    paths.processed_data_dir = tmp_path / "processed"
    paths.processed_data_dir.mkdir(parents=True, exist_ok=True)

    split = MagicMock()
    split.train = 0.70
    split.validation = 0.10

    cfg = MagicMock()
    cfg.preprocessing = pc
    cfg.paths = paths
    cfg.split = split
    cfg.window_size = 5
    cfg.seed = 42
    cfg.profiling = MagicMock()
    cfg.profiling.csv_subdir = csv_dir.name
    return cfg


# ---------------------------------------------------------------------------
# discover_csvs
# ---------------------------------------------------------------------------

class TestDiscoverCsvs:
    def test_all_files_found(self, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        for name in _CORPUS_ORDER:
            (csv_dir / name).touch()

        cfg = MagicMock()
        cfg.paths.raw_data_dir = tmp_path
        cfg.profiling.csv_subdir = "csvs"
        paths = discover_csvs(cfg)
        assert len(paths) == 10
        assert [p.name for p in paths] == list(_CORPUS_ORDER)

    def test_missing_file_raises(self, tmp_path):
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        # Create only 9 of 10 files.
        for name in _CORPUS_ORDER[:-1]:
            (csv_dir / name).touch()

        cfg = MagicMock()
        cfg.paths.raw_data_dir = tmp_path
        cfg.profiling.csv_subdir = "csvs"
        with pytest.raises(FileNotFoundError, match="03-02-2018.csv"):
            discover_csvs(cfg)

    def test_corpus_order_length(self):
        assert len(_CORPUS_ORDER) == 10

    def test_corpus_order_starts_and_ends(self):
        assert _CORPUS_ORDER[0] == "02-14-2018.csv"
        assert _CORPUS_ORDER[-1] == "03-02-2018.csv"


# ---------------------------------------------------------------------------
# run_pipeline (end-to-end with synthetic mini-corpus)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _setup_corpus(self, tmp_path: Path, n_rows_per_file: int = 50):
        csv_dir = tmp_path / "raw" / "csvs"
        csv_dir.mkdir(parents=True)
        for i, name in enumerate(_CORPUS_ORDER):
            # Use different labels so both experiments have class variety.
            label = "Benign" if i % 2 == 0 else "Bot"
            _write_csv(csv_dir / name, n_rows_per_file, label=label, seed=i)
        return csv_dir

    def test_pipeline_completes(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        assert isinstance(result, PipelineResult)

    def test_exp1_npy_files_written(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        out_dir = result.out_dir
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            assert (out_dir / "exp1" / f"{name}.npy").exists(), f"Missing {name}.npy"

    def test_exp2_npy_files_written(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        out_dir = result.out_dir
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            assert (out_dir / "exp2" / f"{name}.npy").exists(), f"Missing {name}.npy"

    def test_global_artefacts_written(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        out_dir = result.out_dir
        assert (out_dir / "feature_names.json").exists()
        assert (out_dir / "class_mapping.json").exists()
        assert (out_dir / "scaler_state.json").exists()
        assert (out_dir / "windowing_stats.json").exists()

    def test_exp1_X_shape(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        X_train = np.load(result.out_dir / "exp1" / "X_train.npy")
        assert X_train.ndim == 3
        assert X_train.shape[1] == 5   # T
        assert X_train.shape[2] == 78  # features
        assert X_train.dtype == np.float32

    def test_exp1_y_dtype(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        y_train = np.load(result.out_dir / "exp1" / "y_train.npy")
        assert y_train.dtype == np.int16

    def test_split_metadata_json_valid(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        meta = json.loads((result.out_dir / "exp1" / "split_metadata.json").read_text())
        assert meta["experiment"] == "1_global_chronological"
        assert "n_train" in meta

    def test_scaler_state_json_valid(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        state = json.loads((result.out_dir / "scaler_state.json").read_text())
        assert "center" in state
        assert len(state["center"]) == 78

    def test_no_raw_files_modified(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        # Record both metadata and bytes before.  A preserved mtime alone does
        # not prove the raw source remained byte-identical.
        before = {
            p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in csv_dir.glob("*.csv")
        }
        cfg = _make_fake_config(tmp_path, csv_dir)
        run_pipeline(cfg, row_limit=50)
        for p, (mtime_ns, digest) in before.items():
            assert p.stat().st_mtime_ns == mtime_ns, f"Raw mtime changed: {p.name}"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, (
                f"Raw bytes changed: {p.name}"
            )

    def test_summary_keys(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        assert "global" in result.summary
        assert "exp1" in result.summary
        assert "exp2" in result.summary
        assert "exp3" in result.summary

    def test_exp3_npy_files_written(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        out_dir = result.out_dir
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            assert (out_dir / "exp3" / f"{name}.npy").exists(), f"Missing {name}.npy"

    def test_exp3_target_is_binary(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        for split in ("y_train", "y_val", "y_test"):
            y = np.load(result.out_dir / "exp3" / f"{split}.npy")
            assert y.dtype == np.int16
            assert set(np.unique(y).tolist()) <= {0, 1}, split

    def test_exp3_features_match_exp1(self, tmp_path):
        """E3 changes only the target; the scaled window tensor is E1's."""
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        for split in ("X_train", "X_val", "X_test"):
            a = np.load(result.out_dir / "exp1" / f"{split}.npy")
            b = np.load(result.out_dir / "exp3" / f"{split}.npy")
            np.testing.assert_array_equal(a, b)

    def test_exp3_metadata_marks_primary_b(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        meta = json.loads(
            (result.out_dir / "exp3" / "split_metadata.json").read_text()
        )
        assert meta["experiment"] == "3_binary_chronological"
        assert meta["primary_experiment"] == "B"
        assert meta["globally_chronological"] is True

    def test_exp1_metadata_marks_diagnostic_not_benchmark(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        meta = json.loads(
            (result.out_dir / "exp1" / "split_metadata.json").read_text()
        )
        assert meta["is_primary_benchmark"] is False
        assert meta["split_index_overlap"] is False
        assert meta["leakage_free"] is False
        assert "class_availability_matrix" in meta

    def test_exp2_metadata_marks_primary_a_and_not_leakage_free(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        meta = json.loads(
            (result.out_dir / "exp2" / "split_metadata.json").read_text()
        )
        assert meta["primary_experiment"] == "A"
        assert meta["globally_chronological"] is False
        assert meta["leakage_free"] is False
        assert "class_eligibility" in meta

    def test_experiment_specific_transform_fit_provenance(self, tmp_path):
        """E2 must not reuse a state fitted from E1's different membership."""
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)

        dense_fit_inputs = []
        indexed_fit_inputs = []

        def _record_fit(X_train):
            dense_fit_inputs.append(X_train.copy())
            return fit_transforms(X_train)

        def _record_indexed_fit(X, indices):
            indexed_fit_inputs.append((X.copy(), indices.copy()))
            return fit_transforms_indexed(X, indices)

        with patch(
            "src.preprocessing.pipeline.fit_transforms",
            side_effect=_record_fit,
        ), patch(
            "src.preprocessing.pipeline.fit_transforms_indexed",
            side_effect=_record_indexed_fit,
        ):
            result = run_pipeline(cfg, row_limit=50)

        # One dense/view fit for E1/E3 and one memory-safe indexed fit for E2.
        assert len(dense_fit_inputs) == 1
        assert len(indexed_fit_inputs) == 1
        X_all, e2_train_idx = indexed_fit_inputs[0]
        assert dense_fit_inputs[0].shape == X_all[e2_train_idx].shape
        assert not np.array_equal(dense_fit_inputs[0], X_all[e2_train_idx])

        e1_state = json.loads(
            (result.out_dir / "exp1" / "scaler_state.json").read_text()
        )
        e2_state = json.loads(
            (result.out_dir / "exp2" / "scaler_state.json").read_text()
        )
        e3_state = json.loads(
            (result.out_dir / "exp3" / "scaler_state.json").read_text()
        )
        assert e1_state["fit_provenance"]["fit_experiment"] == "E1"
        assert e2_state["fit_provenance"]["fit_experiment"] == "E2"
        assert e3_state == e1_state

        # Extra provenance keys do not prevent the persisted numerical state
        # from being reconstructed and used.
        for state_dict in (e1_state, e2_state, e3_state):
            restored = FitTransformState.from_dict(state_dict)
            assert restored.center.shape == (78,)

        for exp, fit_exp in (("exp1", "E1"), ("exp2", "E2"), ("exp3", "E1")):
            meta = json.loads(
                (result.out_dir / exp / "split_metadata.json").read_text()
            )
            assert meta["preprocessing"]["fit_experiment"] == fit_exp
            assert meta["preprocessing"]["fit_split"] == "train"
            assert meta["preprocessing"]["validation_or_test_used_for_fit"] is False

    def test_two_pipeline_runs_are_byte_reproducible(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=20)
        cfg1 = _make_fake_config(tmp_path, csv_dir)
        cfg1.preprocessing.output_subdir = "run1"
        cfg2 = _make_fake_config(tmp_path, csv_dir)
        cfg2.preprocessing.output_subdir = "run2"

        r1 = run_pipeline(cfg1, row_limit=20)
        r2 = run_pipeline(cfg2, row_limit=20)

        files1 = {
            p.relative_to(r1.out_dir): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in r1.out_dir.rglob("*") if p.is_file()
        }
        files2 = {
            p.relative_to(r2.out_dir): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in r2.out_dir.rglob("*") if p.is_file()
        }
        assert files1 == files2

    def test_stage_c_global_artefacts_written(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        out_dir = result.out_dir
        for fname in (
            "binary_class_mapping.json",
            "duplicate_accounting.json",
            "class_eligibility.json",
            "class_availability.json",
        ):
            assert (out_dir / fname).exists(), fname

    def test_duplicate_accounting_reconciles(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        d = json.loads((result.out_dir / "duplicate_accounting.json").read_text())
        assert d["reconciles"] is True
        assert d["computed_before_windowing"] is True

    def test_cleaning_stats_partition_usable_rows(self, tmp_path):
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        d = json.loads((result.out_dir / "cleaning_stats.json").read_text())
        assert (
            d["conflicting_rows"]
            + d["exact_duplicate_rows_removed"]
            + d["temporal_duplicate_rows_retained"]
            + d["unique_rows"]
        ) == d["usable_rows"]
        assert d["clean_rows"] == (
            d["temporal_duplicate_rows_retained"] + d["unique_rows"]
        )

    def test_raw_row_count_matches_synthetic_corpus(self, tmp_path):
        """10 files × 50 rows, all distinct → nothing excluded."""
        csv_dir = self._setup_corpus(tmp_path, n_rows_per_file=50)
        cfg = _make_fake_config(tmp_path, csv_dir)
        result = run_pipeline(cfg, row_limit=50)
        d = json.loads((result.out_dir / "cleaning_stats.json").read_text())
        assert d["raw_rows"] == 500
        assert d["usable_rows"] == 500
        assert d["conflicting_rows"] == 0
        assert d["exact_duplicate_rows_removed"] == 0
        assert d["clean_rows"] == 500

    def test_missing_corpus_raises_file_not_found(self, tmp_path):
        csv_dir = tmp_path / "raw" / "csvs"
        csv_dir.mkdir(parents=True)
        # Write only one file — rest are missing.
        _write_csv(csv_dir / "02-14-2018.csv", 10)
        cfg = _make_fake_config(tmp_path, csv_dir)
        with pytest.raises(FileNotFoundError):
            run_pipeline(cfg, row_limit=10)
