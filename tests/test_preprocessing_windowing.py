"""Tests for src.preprocessing.windowing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.labels import INT_TO_LABEL, LABEL_TO_INT, LABEL_VOCABULARY
from src.preprocessing.schema import (
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    ORIGINAL_INDEX_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.preprocessing.windowing import WindowingStats, _fill_kept_rows, build_windows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(path: Path, n_rows: int, label: str = "Benign", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 10, size=n_rows) for col in MODEL_FEATURE_COLUMNS}
    data[ORIGINAL_INDEX_COLUMN] = np.arange(n_rows)
    data[TIMESTAMP_COLUMN] = [f"14/02/2018 10:{i:02d}:00" for i in range(n_rows)]
    data[LABEL_COLUMN] = label
    pd.DataFrame(data).to_csv(path, index=False)


def _kept(n: int, offset: int = 0) -> np.ndarray:
    return np.arange(offset, offset + n, dtype=np.int64)


# ---------------------------------------------------------------------------
# WindowingStats
# ---------------------------------------------------------------------------

class TestWindowingStats:
    def test_to_dict_keys(self):
        stats = WindowingStats(
            window_size=5,
            total_clean_rows=100,
            n_windows=20,
            tail_rows_discarded=0,
            class_distribution={"Benign": 20},
        )
        d = stats.to_dict()
        assert d["window_size"] == 5
        assert d["n_windows"] == 20
        assert "label_source" in d
        assert "last" in d["label_source"].lower()


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------

class TestBuildWindows:
    def test_fill_stops_after_largest_kept_original_index(self, tmp_path, monkeypatch):
        """A debug keep-index near the head must not scan the whole raw file."""
        first_chunk = pd.DataFrame(
            {
                **{col: np.arange(10, dtype=float) for col in MODEL_FEATURE_COLUMNS},
                LABEL_COLUMN: ["Benign"] * 10,
            }
        )

        class _Reader:
            def __init__(self):
                self.calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.calls += 1
                if self.calls == 1:
                    return first_chunk
                raise AssertionError("reader advanced past the largest kept index")

        reader = _Reader()
        monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: reader)
        feature_out = np.empty((1, 78), dtype=np.float32)
        label_out = np.empty(1, dtype=np.int16)

        written = _fill_kept_rows(
            tmp_path / "unused.csv",
            np.array([0], dtype=np.int64),
            chunk_size=10,
            feature_out=feature_out,
            label_out=label_out,
            offset=0,
        )
        assert written == 1
        assert reader.calls == 1

    def test_basic_shape(self, tmp_path):
        n_rows = 20
        T = 5
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, n_rows)

        result = build_windows([csv], [("02-14-2018", _kept(n_rows))], window_size=T)
        assert result.X.shape == (4, T, 78)
        assert result.y.shape == (4,)
        assert result.X.dtype == np.float32
        assert result.y.dtype == np.int16

    def test_disk_backed_path_matches_in_memory_path(self, tmp_path):
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, 23, seed=91)
        kept = [("02-14-2018", _kept(23))]
        expected = build_windows([csv], kept, window_size=5)
        actual = build_windows(
            [csv], kept, window_size=5, work_dir=tmp_path / "work"
        )
        assert isinstance(actual.X, np.memmap)
        assert isinstance(actual.y, np.memmap)
        np.testing.assert_array_equal(actual.X, expected.X)
        np.testing.assert_array_equal(actual.y, expected.y)
        assert (tmp_path / "work" / "unscaled_flow_features.npy").is_file()
        assert (tmp_path / "work" / "flow_labels.npy").is_file()

    def test_tail_rows_discarded(self, tmp_path):
        n_rows = 22   # 22 / 5 = 4 windows, 2 tail rows discarded
        T = 5
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, n_rows)

        result = build_windows([csv], [("02-14-2018", _kept(n_rows))], window_size=T)
        assert result.stats.n_windows == 4
        assert result.stats.tail_rows_discarded == 2

    def test_label_is_last_flow(self, tmp_path):
        """y[i] must equal the label of the (T-th) flow in window i."""
        T = 3
        labels = ["Benign", "Bot", "Benign", "Benign", "Bot", "Benign"]
        n_rows = len(labels)
        rng = np.random.default_rng(1)
        data = {col: rng.uniform(0, 1, n_rows) for col in MODEL_FEATURE_COLUMNS}
        data[ORIGINAL_INDEX_COLUMN] = np.arange(n_rows)
        data[TIMESTAMP_COLUMN] = [f"14/02/2018 10:0{i}:00" for i in range(n_rows)]
        data[LABEL_COLUMN] = labels
        csv = tmp_path / "02-14-2018.csv"
        pd.DataFrame(data).to_csv(csv, index=False)

        result = build_windows([csv], [("02-14-2018", _kept(n_rows))], window_size=T)
        # Window 0: flows 0,1,2 — last label = "Benign" (index 2)
        # Window 1: flows 3,4,5 — last label = "Benign" (index 5)
        assert result.stats.n_windows == 2
        assert INT_TO_LABEL[int(result.y[0])] == "Benign"
        assert INT_TO_LABEL[int(result.y[1])] == "Benign"

    def test_multi_file_concatenation(self, tmp_path):
        """Rows from multiple files are concatenated in capture-date order."""
        for i, stem in enumerate(["f1", "f2"]):
            _make_csv(tmp_path / f"{stem}.csv", 5, seed=i)
        csvs = [tmp_path / "f1.csv", tmp_path / "f2.csv"]
        kept = [("f1", _kept(5)), ("f2", _kept(5))]
        result = build_windows(csvs, kept, window_size=5)
        # 10 total rows / T=5 = 2 windows
        assert result.stats.n_windows == 2
        assert result.stats.total_clean_rows == 10

    def test_empty_file_kept_list(self, tmp_path):
        """A file with zero kept rows contributes nothing."""
        _make_csv(tmp_path / "f1.csv", 5)
        _make_csv(tmp_path / "f2.csv", 5)
        csvs = [tmp_path / "f1.csv", tmp_path / "f2.csv"]
        kept = [("f1", _kept(5)), ("f2", np.empty(0, dtype=np.int64))]
        result = build_windows(csvs, kept, window_size=5)
        assert result.stats.n_windows == 1

    def test_class_distribution(self, tmp_path):
        T = 5
        # 5 Benign rows → 1 window labelled Benign
        _make_csv(tmp_path / "f.csv", 5, label="Benign")
        result = build_windows(
            [tmp_path / "f.csv"], [("f", _kept(5))], window_size=T
        )
        assert result.stats.class_distribution.get("Benign", 0) == 1

    def test_meta_length(self, tmp_path):
        _make_csv(tmp_path / "f.csv", 10)
        result = build_windows(
            [tmp_path / "f.csv"], [("f", _kept(10))], window_size=5
        )
        assert len(result.meta) == result.stats.n_windows

    def test_meta_fields(self, tmp_path):
        _make_csv(tmp_path / "f.csv", 5)
        result = build_windows(
            [tmp_path / "f.csv"], [("f", _kept(5))], window_size=5
        )
        m = result.meta[0]
        assert "window_idx" in m
        assert "label_code" in m
        assert "label_str" in m
        assert m["label_str"] in LABEL_VOCABULARY
