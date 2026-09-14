"""Tests for src.preprocessing.artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.artifacts import (
    GLOBAL_ARTEFACT_FILES,
    collect_artefact_summary,
    write_binary_class_mapping,
    write_class_availability,
    write_class_eligibility,
    write_class_mapping,
    write_cleaning_stats,
    write_duplicate_accounting,
    write_feature_names,
    write_scaler_state,
    write_split,
    write_split_streaming,
    write_windowing_stats,
)
from src.preprocessing.cleaning import CleaningStats
from src.preprocessing.labels import LABEL_VOCABULARY
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS
from src.preprocessing.transforms import apply_transforms, fit_transforms

T = 5
N_FEAT = 78


def _arrays(n_train=20, n_val=6, n_test=10):
    rng = np.random.default_rng(0)
    def _X(n): return rng.uniform(0, 1, (n, T, N_FEAT)).astype(np.float32)
    def _y(n): return np.zeros(n, dtype=np.int16)
    return _X(n_train), _y(n_train), _X(n_val), _y(n_val), _X(n_test), _y(n_test)


def _consistent_stats() -> CleaningStats:
    """A CleaningStats whose accounting reconciles exactly."""
    stats = CleaningStats(
        raw_rows=120,
        embedded_header_rows=10,
        unknown_label_rows=10,
        usable_rows=100,
        conflicting_rows=6,
        conflicting_groups=2,
        exact_duplicate_rows_removed=4,
        temporal_duplicate_rows_retained=20,
        unique_rows=70,
        clean_rows=90,
    )
    stats.validate()
    return stats


class TestWriteFeatureNames:
    def test_creates_file(self, tmp_path):
        p = write_feature_names(tmp_path)
        assert p.exists()

    def test_content(self, tmp_path):
        p = write_feature_names(tmp_path)
        d = json.loads(p.read_text())
        assert d["n_features"] == 78
        assert d["feature_names"] == list(MODEL_FEATURE_COLUMNS)


class TestWriteClassMapping:
    def test_creates_file(self, tmp_path):
        p = write_class_mapping(tmp_path)
        assert p.exists()

    def test_content(self, tmp_path):
        p = write_class_mapping(tmp_path)
        d = json.loads(p.read_text())
        assert d["n_classes"] == 15
        assert "Heartbleed" in d["absent_from_corpus"]


class TestWriteCleaningStats:
    def test_roundtrip(self, tmp_path):
        stats = _consistent_stats().to_dict()
        p = write_cleaning_stats(stats, tmp_path)
        d = json.loads(p.read_text())
        assert d["raw_rows"] == 120
        assert d["clean_rows"] == 90

    def test_canonical_category_counts_survive_the_round_trip(self, tmp_path):
        p = write_cleaning_stats(_consistent_stats().to_dict(), tmp_path)
        d = json.loads(p.read_text())
        assert d["conflicting_rows"] == 6
        assert d["exact_duplicate_rows_removed"] == 4
        assert d["temporal_duplicate_rows_retained"] == 20
        assert d["unique_rows"] == 70


class TestWriteBinaryClassMapping:
    def test_creates_file(self, tmp_path):
        assert write_binary_class_mapping(tmp_path).exists()

    def test_content(self, tmp_path):
        d = json.loads(write_binary_class_mapping(tmp_path).read_text())
        assert d["n_classes"] == 2
        assert d["benign_label"] == "Benign"
        assert d["n_attack_labels"] == 14
        assert len(d["label_to_binary"]) == len(LABEL_VOCABULARY)

    def test_every_label_maps_to_zero_or_one(self, tmp_path):
        d = json.loads(write_binary_class_mapping(tmp_path).read_text())
        for lbl in LABEL_VOCABULARY:
            assert d["label_to_binary"][lbl] in (0, 1)


class TestWriteDuplicateAccounting:
    def test_creates_file(self, tmp_path):
        stats = _consistent_stats().to_dict()
        assert write_duplicate_accounting(stats, tmp_path).exists()

    def test_reports_reconciliation(self, tmp_path):
        stats = _consistent_stats().to_dict()
        d = json.loads(write_duplicate_accounting(stats, tmp_path).read_text())
        assert d["reconciles"] is True
        rec = d["reconciliation"]
        assert rec["usable_rows"] == rec["sum_of_final_states"] == 100
        assert rec["retained_rows"] == rec["retained_breakdown"] == 90
        assert rec["clean_rows"] == rec["sum_of_retained_categories"] == 90

    def test_table_and_policy_carried_through(self, tmp_path):
        stats = _consistent_stats().to_dict()
        d = json.loads(write_duplicate_accounting(stats, tmp_path).read_text())
        assert d["computed_before_windowing"] is True
        assert d["policy"]["mutually_exclusive"] is True
        quantities = [row["quantity"] for row in d["accounting_table"]]
        for expected in ("raw rows", "conflicting rows"):
            assert any(expected in q for q in quantities), quantities

    def test_detects_a_non_reconciling_report(self, tmp_path):
        """The writer reports the identity result; it does not launder a mismatch."""
        stats = _consistent_stats().to_dict()
        stats["unique_rows"] = 999
        d = json.loads(write_duplicate_accounting(stats, tmp_path).read_text())
        assert d["reconciles"] is False


class TestWriteClassEligibility:
    def test_roundtrip(self, tmp_path):
        eligibility = {
            "thresholds": {"min_train": 100, "min_val": 30, "min_test": 30},
            "eligible_classes": ["Benign"],
            "ineligible_classes": ["Bot"],
            "per_class": {"Bot": {"ineligibility_reason": "too few"}},
        }
        d = json.loads(write_class_eligibility(eligibility, tmp_path).read_text())
        assert d["thresholds"]["min_train"] == 100
        assert d["per_class"]["Bot"]["ineligibility_reason"] == "too few"


class TestWriteClassAvailability:
    def test_roundtrip(self, tmp_path):
        availability = {
            "classes_unseen_during_training": ["Bot"],
            "per_class": {"Bot": {"in_train": False, "in_test": True}},
        }
        d = json.loads(write_class_availability(availability, tmp_path).read_text())
        assert d["classes_unseen_during_training"] == ["Bot"]
        assert d["per_class"]["Bot"]["in_train"] is False


class TestWriteScalerState:
    def test_roundtrip(self, tmp_path):
        state = {"center": [1.0, 2.0], "scale": [0.5, 0.5]}
        p = write_scaler_state(state, tmp_path)
        d = json.loads(p.read_text())
        assert d["center"] == [1.0, 2.0]


class TestWriteWindowingStats:
    def test_roundtrip(self, tmp_path):
        stats = {"n_windows": 42, "window_size": 5}
        p = write_windowing_stats(stats, tmp_path)
        assert json.loads(p.read_text())["n_windows"] == 42


class TestWriteSplit:
    def test_creates_six_npy_and_metadata(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        Xtr, ytr, Xv, yv, Xt, yt = _arrays()
        written = write_split(exp_dir, Xtr, ytr, Xv, yv, Xt, yt, metadata={"exp": "1"})
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            assert written[name].exists()
        assert written["split_metadata"].exists()

    def test_shapes_preserved(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        Xtr, ytr, Xv, yv, Xt, yt = _arrays()
        write_split(exp_dir, Xtr, ytr, Xv, yv, Xt, yt, metadata={})
        assert np.load(exp_dir / "X_train.npy").shape == Xtr.shape
        assert np.load(exp_dir / "y_test.npy").shape == yt.shape

    def test_verify_passes(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        Xtr, ytr, Xv, yv, Xt, yt = _arrays()
        # Should not raise.
        write_split(exp_dir, Xtr, ytr, Xv, yv, Xt, yt, metadata={}, verify=True)

    def test_metadata_json_content(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        Xtr, ytr, Xv, yv, Xt, yt = _arrays()
        write_split(exp_dir, Xtr, ytr, Xv, yv, Xt, yt,
                    metadata={"experiment": "test_exp"})
        d = json.loads((exp_dir / "split_metadata.json").read_text())
        assert d["experiment"] == "test_exp"


class TestWriteSplitStreaming:
    def test_slice_and_index_outputs_match_direct_transform(self, tmp_path):
        X = np.arange(18 * T * N_FEAT, dtype=np.float32).reshape(18, T, N_FEAT)
        state = fit_transforms(X[:9])
        selectors = {
            "train": slice(0, 9),
            "val": np.array([9, 11, 13], dtype=np.int64),
            "test": slice(14, 18),
        }
        y_splits = {
            "train": np.zeros(9, dtype=np.int16),
            "val": np.ones(3, dtype=np.int16),
            "test": np.ones(4, dtype=np.int16),
        }
        written = write_split_streaming(
            tmp_path / "stream",
            X,
            selectors,
            y_splits,
            state,
            metadata={"experiment": "stream-test"},
            chunk_windows=2,
        )
        np.testing.assert_array_equal(
            np.load(written["X_train"]), apply_transforms(X[:9], state)
        )
        np.testing.assert_array_equal(
            np.load(written["X_val"]), apply_transforms(X[[9, 11, 13]], state)
        )
        np.testing.assert_array_equal(
            np.load(written["X_test"]), apply_transforms(X[14:18], state)
        )

    def test_reused_features_are_hardlinked_and_equal(self, tmp_path):
        X = np.ones((9, T, N_FEAT), dtype=np.float32)
        state = fit_transforms(X[:5])
        selectors = {
            "train": slice(0, 5),
            "val": slice(5, 7),
            "test": slice(7, 9),
        }
        ys = {
            "train": np.zeros(5, dtype=np.int16),
            "val": np.zeros(2, dtype=np.int16),
            "test": np.zeros(2, dtype=np.int16),
        }
        first = write_split_streaming(
            tmp_path / "first", X, selectors, ys, state, metadata={}
        )
        second = write_split_streaming(
            tmp_path / "second",
            X,
            selectors,
            ys,
            state,
            metadata={},
            reuse_feature_paths={s: first[f"X_{s}"] for s in selectors},
        )
        for split in selectors:
            a = first[f"X_{split}"]
            b = second[f"X_{split}"]
            assert a.stat().st_ino == b.stat().st_ino
            np.testing.assert_array_equal(np.load(a), np.load(b))


class TestCollectArtefactSummary:
    def test_keys_present(self, tmp_path):
        # Write minimal artefacts so the summary can stat them.
        write_feature_names(tmp_path)
        write_class_mapping(tmp_path)
        exp1_dir = tmp_path / "exp1"
        exp2_dir = tmp_path / "exp2"
        Xtr, ytr, Xv, yv, Xt, yt = _arrays()
        e1 = write_split(exp1_dir, Xtr, ytr, Xv, yv, Xt, yt, metadata={})
        e2 = write_split(exp2_dir, Xtr, ytr, Xv, yv, Xt, yt, metadata={})

        summary = collect_artefact_summary(tmp_path, e1, e2)
        assert "global" in summary
        assert "exp1" in summary
        assert "exp2" in summary
        assert summary["output_root"] == str(tmp_path)

    def test_global_artefacts_listed(self, tmp_path):
        write_feature_names(tmp_path)
        e1 = write_split(tmp_path / "exp1", *_arrays(), metadata={})
        e2 = write_split(tmp_path / "exp2", *_arrays(), metadata={})
        summary = collect_artefact_summary(tmp_path, e1, e2)
        assert "feature_names.json" in summary["global"]

    def test_exp3_omitted_when_not_supplied(self, tmp_path):
        e1 = write_split(tmp_path / "exp1", *_arrays(), metadata={})
        e2 = write_split(tmp_path / "exp2", *_arrays(), metadata={})
        summary = collect_artefact_summary(tmp_path, e1, e2)
        assert "exp3" not in summary

    def test_exp3_listed_when_supplied(self, tmp_path):
        e1 = write_split(tmp_path / "exp1", *_arrays(), metadata={})
        e2 = write_split(tmp_path / "exp2", *_arrays(), metadata={})
        e3 = write_split(tmp_path / "exp3", *_arrays(), metadata={})
        summary = collect_artefact_summary(tmp_path, e1, e2, e3)
        assert "exp3" in summary
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            assert name in summary["exp3"]

    def test_all_stage_c_global_artefacts_are_collected(self, tmp_path):
        write_feature_names(tmp_path)
        write_class_mapping(tmp_path)
        write_binary_class_mapping(tmp_path)
        stats = _consistent_stats().to_dict()
        write_cleaning_stats(stats, tmp_path)
        write_duplicate_accounting(stats, tmp_path)
        write_class_eligibility({"thresholds": {}}, tmp_path)
        write_class_availability({"per_class": {}}, tmp_path)
        write_scaler_state({"center": [], "scale": []}, tmp_path)
        write_windowing_stats({"n_windows": 0}, tmp_path)

        e1 = write_split(tmp_path / "exp1", *_arrays(), metadata={})
        e2 = write_split(tmp_path / "exp2", *_arrays(), metadata={})
        e3 = write_split(tmp_path / "exp3", *_arrays(), metadata={})
        summary = collect_artefact_summary(tmp_path, e1, e2, e3)
        for fname in GLOBAL_ARTEFACT_FILES:
            assert fname in summary["global"], fname
