"""Tests for src.preprocessing.cleaning."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

import src.preprocessing.cleaning as cleaning_module

from src.preprocessing.cleaning import (
    CATEGORY_CONFLICTING,
    CATEGORY_EXACT_DUPLICATE,
    CATEGORY_TEMPORAL_DUPLICATE,
    CATEGORY_UNIQUE,
    EXCLUDED_CATEGORIES,
    RETAINED_CATEGORIES,
    AccountingError,
    CleaningResult,
    CleaningStats,
    _fnv1a_uint64,
    _row_hashes,
    build_keep_index,
    classify_rows,
)
from src.preprocessing.schema import (
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    ORIGINAL_INDEX_COLUMN,
    TIMESTAMP_COLUMN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(path: Path, rows: List[dict]) -> None:
    """Write a minimal CIC-IDS2018-shaped CSV to *path*."""
    cols = [ORIGINAL_INDEX_COLUMN] + list(MODEL_FEATURE_COLUMNS) + [TIMESTAMP_COLUMN, LABEL_COLUMN]
    records = []
    for r in rows:
        rec = {c: r.get(c, 0.0) for c in cols}
        records.append(rec)
    df = pd.DataFrame(records, columns=cols)
    df.to_csv(path, index=False)


def _feature_row(seed: int) -> dict:
    """Return a feature dict with deterministic values."""
    rng = np.random.default_rng(seed)
    return {col: float(rng.uniform(0, 100)) for col in MODEL_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Unit tests: hashing
# ---------------------------------------------------------------------------

class TestFnv1aHash:
    def test_empty(self):
        h = _fnv1a_uint64(b"")
        assert isinstance(h, int)
        assert 0 <= h <= 0xFFFF_FFFF_FFFF_FFFF

    def test_deterministic(self):
        assert _fnv1a_uint64(b"hello") == _fnv1a_uint64(b"hello")

    def test_different_inputs_differ(self):
        assert _fnv1a_uint64(b"abc") != _fnv1a_uint64(b"abd")

    def test_known_value(self):
        # FNV-1a 64-bit of b"" = 14695981039346656037
        assert _fnv1a_uint64(b"") == 14695981039346656037


class TestRowHashes:
    def test_shapes(self):
        n = 10
        feats = np.zeros((n, 78), dtype=np.float64)
        ts = np.array([b"ts"] * n, dtype=object)
        labels = np.zeros(n, dtype=np.int16)
        fh, ftlh = _row_hashes(feats, ts, labels)
        assert fh.shape == (n,)
        assert ftlh.shape == (n,)
        assert fh.dtype == np.uint64
        assert ftlh.dtype == np.uint64

    def test_same_features_diff_labels_give_same_feat_hash(self):
        feats = np.ones((2, 78), dtype=np.float64)
        ts = np.array([b"t1", b"t2"], dtype=object)
        labels = np.array([0, 1], dtype=np.int16)
        fh, ftlh = _row_hashes(feats, ts, labels)
        assert fh[0] == fh[1]          # feat_hash same
        assert ftlh[0] != ftlh[1]      # combined differs

    def test_same_row_same_hash(self):
        feats = np.random.default_rng(0).random((3, 78))
        ts = np.array([b"t"] * 3, dtype=object)
        labels = np.zeros(3, dtype=np.int16)
        fh1, ftlh1 = _row_hashes(feats, ts, labels)
        fh2, ftlh2 = _row_hashes(feats, ts, labels)
        np.testing.assert_array_equal(fh1, fh2)
        np.testing.assert_array_equal(ftlh1, ftlh2)


# ---------------------------------------------------------------------------
# Regression tests: forced 64-bit fingerprint collisions
# ---------------------------------------------------------------------------

class TestForcedFingerprintCollisions:
    """A hash narrows the candidate bucket; payload equality decides identity."""

    @staticmethod
    def _force_hashes(monkeypatch, feature_hashes, state_hashes, exact_hashes):
        def forced(_features, _timestamps, _labels):
            return (
                np.asarray(feature_hashes, dtype=np.uint64),
                np.asarray(state_hashes, dtype=np.uint64),
                np.asarray(exact_hashes, dtype=np.uint64),
            )

        monkeypatch.setattr(cleaning_module, "_cascade_hashes", forced)

    def test_same_feature_hash_different_vectors_are_not_merged(self, monkeypatch):
        self._force_hashes(monkeypatch, [7, 7], [11, 12], [21, 22])
        features = np.array([[1.0] * 78, [2.0] * 78], dtype=np.float64)

        categories = classify_rows(features, ["t1", "t2"], ["Benign", "Bot"])

        assert categories == [CATEGORY_UNIQUE, CATEGORY_UNIQUE]

    def test_same_feature_hash_identical_vectors_form_one_conflicting_group(
        self, monkeypatch
    ):
        self._force_hashes(monkeypatch, [7, 7], [11, 12], [21, 22])
        features = np.array([[1.0] * 78, [1.0] * 78], dtype=np.float64)

        categories = classify_rows(features, ["t1", "t2"], ["Benign", "Bot"])

        assert categories == [CATEGORY_CONFLICTING, CATEGORY_CONFLICTING]

    def test_same_exact_hash_different_payloads_are_not_exact_duplicates(
        self, monkeypatch
    ):
        self._force_hashes(monkeypatch, [7, 7], [11, 11], [21, 21])
        features = np.array([[1.0] * 78, [1.0] * 78], dtype=np.float64)

        categories = classify_rows(
            features, ["t1", "t2"], ["Benign", "Benign"]
        )

        assert categories == [CATEGORY_UNIQUE, CATEGORY_TEMPORAL_DUPLICATE]

    def test_same_exact_hash_identical_payload_removes_later_occurrence(
        self, monkeypatch
    ):
        self._force_hashes(monkeypatch, [7, 7], [11, 11], [21, 21])
        features = np.array([[1.0] * 78, [1.0] * 78], dtype=np.float64)

        categories = classify_rows(
            features, ["t1", "t1"], ["Benign", "Benign"]
        )

        assert categories == [CATEGORY_UNIQUE, CATEGORY_EXACT_DUPLICATE]

    def test_same_state_hash_different_states_are_not_temporal_duplicates(
        self, monkeypatch
    ):
        self._force_hashes(monkeypatch, [7, 7], [11, 11], [21, 22])
        features = np.array([[1.0] * 78, [2.0] * 78], dtype=np.float64)

        categories = classify_rows(
            features, ["t1", "t2"], ["Benign", "Benign"]
        )

        assert categories == [CATEGORY_UNIQUE, CATEGORY_UNIQUE]

    def test_same_state_hash_identical_state_flags_later_temporal_occurrence(
        self, monkeypatch
    ):
        self._force_hashes(monkeypatch, [7, 7], [11, 11], [21, 22])
        features = np.array([[1.0] * 78, [1.0] * 78], dtype=np.float64)

        categories = classify_rows(
            features, ["t1", "t2"], ["Benign", "Benign"]
        )

        assert categories == [CATEGORY_UNIQUE, CATEGORY_TEMPORAL_DUPLICATE]


# ---------------------------------------------------------------------------
# Integration tests: build_keep_index
# ---------------------------------------------------------------------------

class TestBuildKeepIndex:
    def test_forced_collisions_use_exact_payloads_in_corpus_pass(
        self, tmp_path, monkeypatch
    ):
        rows = []
        specifications = [
            (1.0, "t1", "Benign"),
            (2.0, "t2", "Bot"),       # L1 hash collision, distinct vector
            (3.0, "t3", "Benign"),
            (3.0, "t4", "Benign"),    # L2 hash collision, true temporal state
            (4.0, "t5", "Benign"),
            (5.0, "t6", "Benign"),    # L3 hash collision, distinct state
        ]
        for i, (value, timestamp, label) in enumerate(specifications):
            rows.append({
                ORIGINAL_INDEX_COLUMN: i,
                TIMESTAMP_COLUMN: timestamp,
                LABEL_COLUMN: label,
                **{column: value for column in MODEL_FEATURE_COLUMNS},
            })

        def forced(features, timestamps, _labels):
            values = features[:, 0].astype(int)
            feature_hash = np.array(
                [{1: 7, 2: 7, 3: 8, 4: 9, 5: 9}[value] for value in values],
                dtype=np.uint64,
            )
            state_hash = np.array(
                [{1: 11, 2: 12, 3: 13, 4: 14, 5: 14}[value] for value in values],
                dtype=np.uint64,
            )
            exact_hash = np.array(
                [23 if value == 3 else 20 + value for value in values],
                dtype=np.uint64,
            )
            return feature_hash, state_hash, exact_hash

        monkeypatch.setattr(cleaning_module, "_cascade_hashes", forced)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=2)

        assert result.stats.conflicting_rows == 0
        assert result.stats.exact_duplicate_rows_removed == 0
        assert result.stats.temporal_duplicate_rows_retained == 1
        assert result.stats.unique_rows == 5
        assert result.stats.clean_rows == 6
        assert result.kept_per_file[0][1].tolist() == list(range(6))

    def test_clean_data_all_kept(self, tmp_path):
        rows = []
        for i in range(10):
            r = _feature_row(i)
            r[ORIGINAL_INDEX_COLUMN] = i
            r[TIMESTAMP_COLUMN] = f"14/02/2018 10:{i:02d}:00"
            r[LABEL_COLUMN] = "Benign"
            rows.append(r)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        assert isinstance(result, CleaningResult)
        assert result.stats.raw_rows == 10
        assert result.stats.usable_rows == 10
        assert result.stats.exact_duplicate_rows_removed == 0
        assert result.stats.conflicting_rows == 0
        assert result.stats.temporal_duplicate_rows_retained == 0
        assert result.stats.unique_rows == 10
        assert result.stats.clean_rows == 10

    def test_exact_duplicates_excluded(self, tmp_path):
        base = _feature_row(42)
        rows = []
        for i in range(3):
            r = dict(base)
            r[ORIGINAL_INDEX_COLUMN] = i
            r[TIMESTAMP_COLUMN] = "14/02/2018 10:00:00"
            r[LABEL_COLUMN] = "Benign"
            rows.append(r)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        # First occurrence kept; 2 exact duplicates excluded.
        assert result.stats.exact_duplicate_rows_removed == 2
        assert result.stats.unique_rows == 1
        assert result.stats.temporal_duplicate_rows_retained == 0
        assert result.stats.clean_rows == 1

    def test_first_occurrence_is_the_one_retained(self, tmp_path):
        """The retained exact duplicate is the lowest Original_Index."""
        base = _feature_row(11)
        rows = []
        for i in range(4):
            r = dict(base)
            r[ORIGINAL_INDEX_COLUMN] = i
            r[TIMESTAMP_COLUMN] = "14/02/2018 10:00:00"
            r[LABEL_COLUMN] = "Benign"
            rows.append(r)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        _, kept = result.kept_per_file[0]
        assert kept.tolist() == [0]

    def test_temporal_duplicates_retained(self, tmp_path):
        """Same features+label, different Timestamps → all kept (level 3)."""
        base = _feature_row(7)
        rows = []
        for i in range(3):
            r = dict(base)
            r[ORIGINAL_INDEX_COLUMN] = i
            r[TIMESTAMP_COLUMN] = f"14/02/2018 10:0{i}:00"
            r[LABEL_COLUMN] = "Bot"
            rows.append(r)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        assert result.stats.exact_duplicate_rows_removed == 0
        # First is unique; the two later timestamps are temporal duplicates and
        # are RETAINED, so the count must be non-zero (the old formula always
        # produced 0 here).
        assert result.stats.unique_rows == 1
        assert result.stats.temporal_duplicate_rows_retained == 2
        assert result.stats.clean_rows == 3

    def test_conflicting_rows_excluded(self, tmp_path):
        """Same feature vector, two different labels → both excluded (level 1)."""
        base = _feature_row(99)
        rows = []
        for lbl, i in [("Benign", 0), ("Bot", 1)]:
            r = dict(base)
            r[ORIGINAL_INDEX_COLUMN] = i
            r[TIMESTAMP_COLUMN] = f"14/02/2018 11:0{i}:00"
            r[LABEL_COLUMN] = lbl
            rows.append(r)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        assert result.stats.conflicting_rows == 2
        assert result.stats.conflicting_groups == 1
        assert result.stats.clean_rows == 0

    def test_multi_file_ordering(self, tmp_path):
        """kept_per_file preserves file order."""
        csvs = []
        for day in range(3):
            rows = [{
                ORIGINAL_INDEX_COLUMN: 0,
                TIMESTAMP_COLUMN: f"1{day+4}/02/2018 09:00:00",
                LABEL_COLUMN: "Benign",
                **{c: float(day) for c in MODEL_FEATURE_COLUMNS},
            }]
            p = tmp_path / f"0{day+1}.csv"
            _make_csv(p, rows)
            csvs.append(p)

        result = build_keep_index(csvs, chunk_size=100)
        assert len(result.kept_per_file) == 3
        stems = [stem for stem, _ in result.kept_per_file]
        assert stems == ["01", "02", "03"]

    def test_first_occurrence_spans_files(self, tmp_path):
        """Across files, the earlier file supplies the retained occurrence."""
        base = _feature_row(5)
        csvs = []
        for day in range(2):
            r = dict(base)
            r[ORIGINAL_INDEX_COLUMN] = 0
            r[TIMESTAMP_COLUMN] = "14/02/2018 10:00:00"
            r[LABEL_COLUMN] = "Benign"
            p = tmp_path / f"day{day}.csv"
            _make_csv(p, [r])
            csvs.append(p)

        result = build_keep_index(csvs, chunk_size=100)
        assert result.stats.exact_duplicate_rows_removed == 1
        assert result.stats.clean_rows == 1
        kept = {stem: arr.tolist() for stem, arr in result.kept_per_file}
        assert kept["day0"] == [0]   # first file keeps it
        assert kept["day1"] == []    # second file's copy is the duplicate

    def test_row_limit(self, tmp_path):
        rows = []
        for i in range(20):
            r = _feature_row(i)
            r[ORIGINAL_INDEX_COLUMN] = i
            r[TIMESTAMP_COLUMN] = "14/02/2018 10:00:00"
            r[LABEL_COLUMN] = "Benign"
            rows.append(r)
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100, row_limit=5)
        assert result.stats.raw_rows <= 5

    def test_unknown_labels_skipped(self, tmp_path):
        rows = [
            {ORIGINAL_INDEX_COLUMN: 0, TIMESTAMP_COLUMN: "t", LABEL_COLUMN: "Benign",
             **{c: 1.0 for c in MODEL_FEATURE_COLUMNS}},
            {ORIGINAL_INDEX_COLUMN: 1, TIMESTAMP_COLUMN: "t", LABEL_COLUMN: "UnknownGarbage",
             **{c: 2.0 for c in MODEL_FEATURE_COLUMNS}},
        ]
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        # Only the Benign row is valid, and the skipped row is accounted for.
        assert result.stats.raw_rows == 2
        assert result.stats.unknown_label_rows == 1
        assert result.stats.usable_rows == 1
        assert result.stats.clean_rows == 1

    def test_embedded_header_rows_counted(self, tmp_path):
        """A repeated header line is removed and counted, not treated as data."""
        rows = [
            {ORIGINAL_INDEX_COLUMN: 0, TIMESTAMP_COLUMN: "t1", LABEL_COLUMN: "Benign",
             **{c: 1.0 for c in MODEL_FEATURE_COLUMNS}},
            # Embedded header: every field equals its own column name.
            {ORIGINAL_INDEX_COLUMN: ORIGINAL_INDEX_COLUMN,
             TIMESTAMP_COLUMN: TIMESTAMP_COLUMN, LABEL_COLUMN: LABEL_COLUMN,
             **{c: c for c in MODEL_FEATURE_COLUMNS}},
            {ORIGINAL_INDEX_COLUMN: 2, TIMESTAMP_COLUMN: "t2", LABEL_COLUMN: "Benign",
             **{c: 2.0 for c in MODEL_FEATURE_COLUMNS}},
        ]
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)
        assert result.stats.raw_rows == 3
        assert result.stats.embedded_header_rows == 1
        assert result.stats.usable_rows == 2
        assert result.stats.clean_rows == 2
        # Original_Index is assigned before header filtering, so the data row
        # after the header keeps source position 2 — not 1.
        _, kept = result.kept_per_file[0]
        assert kept.tolist() == [0, 2]

    def test_cleaning_stats_to_dict(self):
        stats = CleaningStats(
            raw_rows=100,
            embedded_header_rows=0,
            unknown_label_rows=0,
            usable_rows=100,
            conflicting_rows=2,
            conflicting_groups=1,
            exact_duplicate_rows_removed=5,
            temporal_duplicate_rows_retained=3,
            unique_rows=90,
            clean_rows=93,
            source_files=["a.csv"],
        )
        stats.validate()
        d = stats.to_dict()
        assert d["raw_rows"] == 100
        assert d["clean_rows"] == 93
        assert d["policy"]["mutually_exclusive"] is True
        assert d["policy"]["raw_files_modified"] is False
        assert "ambiguous under the feature representation" in \
            d["policy"]["level_1_conflicting"]
        assert "NOT treated as noise" in d["policy"]["level_3_temporal"]
        table = {row["quantity"]: row["count"] for row in d["accounting_table"]}
        assert table["raw rows"] == 100
        assert table["retained rows"] == 93
        assert table["final clean rows"] == 93
        assert d["retained_rows"] == 93
        assert d["policy"]["final_states"] == [
            "CONFLICTING",
            "EXACT_DUPLICATE",
            "RETAINED_NON_DUPLICATE",
        ]
        assert d["policy"]["temporal_is_retained_indicator"] is True


# ---------------------------------------------------------------------------
# §3/§4 — mutual exclusivity and accounting reconciliation
# ---------------------------------------------------------------------------

class TestCascadeMutualExclusivity:
    """Every usable row has one final state plus a retained subtype."""

    def test_categories_are_disjoint_sets(self):
        assert set(EXCLUDED_CATEGORIES) & set(RETAINED_CATEGORIES) == set()
        assert set(EXCLUDED_CATEGORIES) | set(RETAINED_CATEGORIES) == {
            CATEGORY_CONFLICTING,
            CATEGORY_EXACT_DUPLICATE,
            CATEGORY_TEMPORAL_DUPLICATE,
            CATEGORY_UNIQUE,
        }

    def test_one_category_per_row(self):
        """classify_rows returns exactly one valid category per input row."""
        feats = np.array(
            [
                [1.0] * 78,   # conflicting group (two labels)
                [1.0] * 78,
                [2.0] * 78,   # exact duplicate pair
                [2.0] * 78,
                [3.0] * 78,   # temporal duplicate pair
                [3.0] * 78,
                [4.0] * 78,   # unique
            ],
            dtype=np.float64,
        )
        ts = ["t1", "t2", "t1", "t1", "t1", "t2", "t1"]
        labels = ["Benign", "Bot", "Benign", "Benign", "Benign", "Benign", "Benign"]

        cats = classify_rows(feats, ts, labels)
        assert len(cats) == len(labels)
        valid = set(EXCLUDED_CATEGORIES) | set(RETAINED_CATEGORIES)
        assert all(c in valid for c in cats)
        assert cats == [
            CATEGORY_CONFLICTING,
            CATEGORY_CONFLICTING,
            CATEGORY_UNIQUE,
            CATEGORY_EXACT_DUPLICATE,
            CATEGORY_UNIQUE,               # first row with feature value 3.0
            CATEGORY_TEMPORAL_DUPLICATE,   # same feats/label, later timestamp
            CATEGORY_UNIQUE,
        ]

    def test_conflicting_wins_over_exact(self):
        """A row that is BOTH conflicting and an exact duplicate is conflicting.

        This is the precedence bug the old pass 2 had: it tested the exact key
        first, so such rows were counted as exact duplicates and the
        conflicting total was understated.
        """
        feats = np.array([[7.0] * 78] * 3, dtype=np.float64)
        ts = ["t1", "t1", "t9"]
        # Same feature vector with two labels → the whole group conflicts.
        labels = ["Benign", "Benign", "Bot"]

        cats = classify_rows(feats, ts, labels)
        assert cats == [CATEGORY_CONFLICTING] * 3
        assert CATEGORY_EXACT_DUPLICATE not in cats

    def test_exact_wins_over_temporal(self):
        """A repeated identical row is exact, never temporal."""
        feats = np.array([[8.0] * 78] * 2, dtype=np.float64)
        cats = classify_rows(feats, ["t1", "t1"], ["Bot", "Bot"])
        assert cats == [CATEGORY_UNIQUE, CATEGORY_EXACT_DUPLICATE]
        assert CATEGORY_TEMPORAL_DUPLICATE not in cats

    def test_group_of_k_yields_one_original(self):
        """In a group of k identical rows, 1 is retained and k-1 are duplicates."""
        for k in (2, 3, 5, 8):
            feats = np.array([[9.0] * 78] * k, dtype=np.float64)
            cats = classify_rows(feats, ["t"] * k, ["Benign"] * k)
            assert cats.count(CATEGORY_UNIQUE) == 1
            assert cats.count(CATEGORY_EXACT_DUPLICATE) == k - 1

    def test_classify_rows_matches_corpus_pass(self, tmp_path):
        """classify_rows and build_keep_index agree on the same input."""
        specs = [
            ([1.0] * 78, "t1", "Benign"),
            ([1.0] * 78, "t2", "Bot"),       # conflicting with the row above
            ([2.0] * 78, "t1", "Benign"),
            ([2.0] * 78, "t1", "Benign"),    # exact duplicate
            ([2.0] * 78, "t5", "Benign"),    # temporal duplicate
            ([3.0] * 78, "t1", "DoS attacks-Hulk"),
        ]
        rows = []
        for i, (vals, ts, lbl) in enumerate(specs):
            rows.append({
                ORIGINAL_INDEX_COLUMN: i,
                TIMESTAMP_COLUMN: ts,
                LABEL_COLUMN: lbl,
                **{c: v for c, v in zip(MODEL_FEATURE_COLUMNS, vals)},
            })
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)

        result = build_keep_index([csv], chunk_size=100)

        feats = np.array([s[0] for s in specs], dtype=np.float64)
        cats = classify_rows(feats, [s[1] for s in specs], [s[2] for s in specs])

        assert result.stats.conflicting_rows == cats.count(CATEGORY_CONFLICTING)
        assert result.stats.exact_duplicate_rows_removed == \
            cats.count(CATEGORY_EXACT_DUPLICATE)
        assert result.stats.temporal_duplicate_rows_retained == \
            cats.count(CATEGORY_TEMPORAL_DUPLICATE)
        assert result.stats.unique_rows == cats.count(CATEGORY_UNIQUE)

        # And the kept rows are exactly the rows in the retained categories.
        expected_kept = [
            i for i, c in enumerate(cats) if c in RETAINED_CATEGORIES
        ]
        _, kept = result.kept_per_file[0]
        assert kept.tolist() == expected_kept


class TestAccountingReconciliation:
    """The three final states and retained subtypes reconcile exactly."""

    def _mixed_corpus(self, tmp_path):
        specs = []
        # 6 unique rows
        for i in range(6):
            specs.append(([float(i + 10)] * 78, f"t{i}", "Benign"))
        # 1 exact-duplicate group of 3 (1 unique + 2 exact)
        for _ in range(3):
            specs.append(([100.0] * 78, "tx", "Bot"))
        # 1 temporal group of 4 (1 unique + 3 temporal)
        for i in range(4):
            specs.append(([200.0] * 78, f"ty{i}", "DoS attacks-Hulk"))
        # 1 conflicting group of 2
        specs.append(([300.0] * 78, "tz", "Benign"))
        specs.append(([300.0] * 78, "tz", "SSH-Bruteforce"))

        rows = []
        for i, (vals, ts, lbl) in enumerate(specs):
            rows.append({
                ORIGINAL_INDEX_COLUMN: i,
                TIMESTAMP_COLUMN: ts,
                LABEL_COLUMN: lbl,
                **{c: v for c, v in zip(MODEL_FEATURE_COLUMNS, vals)},
            })
        csv = tmp_path / "02-14-2018.csv"
        _make_csv(csv, rows)
        return csv, specs

    def test_partition_identity(self, tmp_path):
        csv, specs = self._mixed_corpus(tmp_path)
        s = build_keep_index([csv], chunk_size=100).stats

        assert s.raw_rows == len(specs)
        assert s.usable_rows == len(specs)
        assert (
            s.conflicting_rows
            + s.exact_duplicate_rows_removed
            + s.retained_rows
        ) == s.usable_rows
        assert s.retained_rows == (
            s.temporal_duplicate_rows_retained + s.unique_rows
        )

    def test_clean_identity(self, tmp_path):
        csv, _ = self._mixed_corpus(tmp_path)
        s = build_keep_index([csv], chunk_size=100).stats

        assert s.clean_rows == s.temporal_duplicate_rows_retained + s.unique_rows
        assert s.clean_rows == (
            s.usable_rows - s.conflicting_rows - s.exact_duplicate_rows_removed
        )

    def test_expected_counts(self, tmp_path):
        csv, _ = self._mixed_corpus(tmp_path)
        s = build_keep_index([csv], chunk_size=100).stats

        assert s.conflicting_rows == 2
        assert s.conflicting_groups == 1
        assert s.exact_duplicate_rows_removed == 2
        assert s.temporal_duplicate_rows_retained == 3
        assert s.unique_rows == 8   # 6 benign + 1 bot + 1 hulk
        assert s.clean_rows == 11

    def test_temporal_is_not_a_subtraction(self, tmp_path):
        """Temporal duplicates are a subset of the clean rows.

        The old implementation computed
        ``temporal = total - exact - conflicting - clean``, which is
        identically 0 because ``clean == total - exact - conflicting``.  Guard
        against that formula coming back.
        """
        csv, _ = self._mixed_corpus(tmp_path)
        s = build_keep_index([csv], chunk_size=100).stats

        assert s.temporal_duplicate_rows_retained > 0
        assert s.temporal_duplicate_rows_retained <= s.clean_rows
        broken = (
            s.usable_rows
            - s.exact_duplicate_rows_removed
            - s.conflicting_rows
            - s.clean_rows
        )
        assert broken == 0
        assert s.temporal_duplicate_rows_retained != broken

    def test_chunk_size_does_not_change_accounting(self, tmp_path):
        csv, _ = self._mixed_corpus(tmp_path)
        a = build_keep_index([csv], chunk_size=100).stats.to_dict()
        b = build_keep_index([csv], chunk_size=3).stats.to_dict()
        assert a == b

    def test_repeated_runs_are_identical(self, tmp_path):
        """Determinism: the same input yields byte-identical accounting."""
        csv, _ = self._mixed_corpus(tmp_path)
        r1 = build_keep_index([csv], chunk_size=100)
        r2 = build_keep_index([csv], chunk_size=100)
        assert r1.stats.to_dict() == r2.stats.to_dict()
        for (s1, k1), (s2, k2) in zip(r1.kept_per_file, r2.kept_per_file):
            assert s1 == s2
            np.testing.assert_array_equal(k1, k2)

    def test_validate_rejects_inconsistent_stats(self):
        bad = CleaningStats(
            raw_rows=100, usable_rows=100,
            conflicting_rows=1, exact_duplicate_rows_removed=1,
            temporal_duplicate_rows_retained=1, unique_rows=1,
            clean_rows=2,
        )
        with pytest.raises(AccountingError):
            bad.validate()

    def test_validate_rejects_broken_clean_total(self):
        bad = CleaningStats(
            raw_rows=10, usable_rows=10,
            conflicting_rows=2, exact_duplicate_rows_removed=3,
            temporal_duplicate_rows_retained=1, unique_rows=4,
            clean_rows=99,
        )
        with pytest.raises(AccountingError):
            bad.validate()
