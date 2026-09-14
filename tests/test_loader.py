"""Tests for the raw CIC-IDS2018 day-CSV loader.

The primary tests run against the **real** file ``data/raw/csv/03-01-2018.csv``
and assert the exact values that were measured from it during the dataset
inspection phase (see ``results/inspection/csv_scan_03-01-2018.json`` and
``csv_extra_03-01-2018.json``). If the file on disk ever stops matching those
measurements the suite fails loudly instead of quietly accepting new numbers.

They are skipped — not failed — when the dataset is not present, so the suite
still runs on a machine without the ~108 MB file.

A handful of tests use tiny **synthetic** CSVs written into pytest's ``tmp_path``
to exercise failure paths the real file cannot reach (a mismatched header, a
``-inf`` value, an unparseable timestamp). Those files hold obviously fake values,
never touch ``data/raw``, and are never used to report a dataset statistic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ConfigError
from src.preprocessing import schema
from src.preprocessing.loader import (
    DataQualityError,
    LoadedDay,
    RowCountMismatchError,
    SchemaMismatchError,
    count_physical_lines,
    load_day_csv,
    load_from_config,
    read_header,
    resolve_primary_csv,
    verify_header,
)

# --------------------------------------------------------------------------- #
# Values measured from data/raw/csv/03-01-2018.csv during dataset inspection.
# Every number below was produced by a full read of the file, cross-checked by a
# second independent scan. None of them is an estimate.
# --------------------------------------------------------------------------- #
EXPECTED_RAW_ROWS = 331_125
EXPECTED_EMBEDDED_HEADER_ROWS = 25
EXPECTED_CLEAN_ROWS = 331_100

# The exact source positions of the 25 repeated header rows: 12 scattered
# single rows, then a contiguous run of 13 at the very end of the file. So the
# last surviving data row sits at position 331,111, not 331,124.
EXPECTED_HEADER_POSITIONS: List[int] = [
    413, 19_761, 19_906, 39_019, 60_809, 76_528, 81_059, 85_448, 89_953,
    91_404, 92_657, 95_060,
] + list(range(331_112, 331_125))
EXPECTED_LAST_DATA_POSITION = 331_111

EXPECTED_LABEL_DISTRIBUTION: Dict[str, int] = {
    "Benign": 238_037,
    "Infilteration": 93_063,
}

EXPECTED_NAN_BEFORE: Dict[str, int] = {"Flow Byts/s": 1_834}
EXPECTED_POSINF_BEFORE: Dict[str, int] = {"Flow Byts/s": 1_085, "Flow Pkts/s": 2_919}
EXPECTED_SENTINEL_NEGATIVES: Dict[str, int] = {
    "Init Fwd Win Byts": 118_201,
    "Init Bwd Win Byts": 159_458,
}

# Duplicates. The inspection pass measured 97 exact duplicates over all 331,125
# raw rows; 24 of those are the 25 byte-identical embedded header rows, so 73
# remain among the 331,100 real rows. Confirmed independently on both a raw-string
# and a cleaned-float64 basis.
EXPECTED_DUPLICATES_EXACT = 73
EXPECTED_DUPLICATES_TEMPORAL = 40_144
EXPECTED_CONFLICTING_GROUPS = 8_227
EXPECTED_CONFLICTING_ROWS = 44_625
EXPECTED_DISTINCT_FEATURE_VECTORS = 282_656

EXPECTED_TIMESTAMP_MIN = "2018-03-01T01:00:00"
EXPECTED_TIMESTAMP_MAX = "2018-03-01T12:59:59"
EXPECTED_DISTINCT_TIMESTAMPS = 32_068
EXPECTED_DATE_HISTOGRAM = {"01/03/2018": EXPECTED_CLEAN_ROWS}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Fixtures for the real file
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def raw_csv_path(config: Config) -> Path:
    path = resolve_primary_csv(config)
    if not path.is_file():
        pytest.skip(f"primary day CSV not present: {path}")
    return path


@pytest.fixture(scope="session")
def fingerprint_before(raw_csv_path: Path) -> Dict[str, object]:
    """Size / mtime / md5 captured *before* the loader ever opens the file."""
    stat = raw_csv_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "md5": _md5(raw_csv_path),
    }


@pytest.fixture(scope="session")
def day(raw_csv_path: Path, fingerprint_before: Dict[str, object]) -> LoadedDay:
    """The real file, loaded once for the whole session.

    Depends on ``fingerprint_before`` purely for ordering: the checksum must be
    taken before the load so the "file untouched" test is meaningful.
    """
    return load_day_csv(
        raw_csv_path,
        expected_raw_rows=EXPECTED_RAW_ROWS,
        expected_embedded_header_rows=EXPECTED_EMBEDDED_HEADER_ROWS,
    )


# --------------------------------------------------------------------------- #
# Synthetic CSV helpers (failure paths only - never real data)
# --------------------------------------------------------------------------- #
def _synthetic_row(**overrides: str) -> List[str]:
    """One schema-shaped row of obviously fake values."""
    values = {column: "0" for column in schema.CSV_COLUMNS}
    values[schema.TIMESTAMP_COLUMN] = "01/03/2018 01:00:00"
    values[schema.LABEL_COLUMN] = "SyntheticClass"
    values.update(overrides)
    return [values[column] for column in schema.CSV_COLUMNS]


def _write_csv(path: Path, rows: List[List[str]], header: List[str] | None = None) -> Path:
    names = header if header is not None else list(schema.CSV_COLUMNS)
    lines = [",".join(names)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =========================================================================== #
# 1. the file loads
# =========================================================================== #
def test_primary_csv_loads(day: LoadedDay) -> None:
    assert isinstance(day, LoadedDay)
    assert day.source_file == "03-01-2018.csv"
    assert day.n_rows > 0
    assert day.shape == (EXPECTED_CLEAN_ROWS, len(schema.MODEL_FEATURE_COLUMNS))


def test_header_matches_confirmed_schema(raw_csv_path: Path) -> None:
    names, md5 = read_header(raw_csv_path)
    assert names == list(schema.CSV_COLUMNS)
    assert len(names) == 80
    assert md5 == schema.HEADER_MD5
    assert verify_header(raw_csv_path) == schema.HEADER_MD5


# =========================================================================== #
# 2. exactly 331,125 raw rows are detected
# =========================================================================== #
def test_raw_row_count(day: LoadedDay) -> None:
    assert day.stats.cleaning.n_raw_rows == EXPECTED_RAW_ROWS


def test_raw_row_count_agrees_with_physical_line_count(
    day: LoadedDay, raw_csv_path: Path
) -> None:
    clean = day.stats.cleaning
    assert clean.line_count_verified is True
    assert clean.n_physical_lines == EXPECTED_RAW_ROWS + 1
    # independent recount, proving no field holds an embedded newline
    assert count_physical_lines(raw_csv_path) == EXPECTED_RAW_ROWS + 1


def test_wrong_expected_raw_rows_fails_clearly(raw_csv_path: Path) -> None:
    with pytest.raises(RowCountMismatchError, match="raw data rows"):
        load_day_csv(
            raw_csv_path,
            expected_raw_rows=EXPECTED_RAW_ROWS + 1,
            compute_duplicate_stats=False,
            verify_raw_line_count=False,
        )


# =========================================================================== #
# 3. exactly 25 embedded header rows are removed
# =========================================================================== #
def test_embedded_header_rows_removed(day: LoadedDay) -> None:
    clean = day.stats.cleaning
    assert clean.n_embedded_header_rows == EXPECTED_EMBEDDED_HEADER_ROWS
    assert len(clean.embedded_header_original_indices) == EXPECTED_EMBEDDED_HEADER_ROWS
    assert len(set(clean.embedded_header_original_indices)) == (
        EXPECTED_EMBEDDED_HEADER_ROWS
    )
    # every removed position is a real row position in the source file
    assert all(0 <= i < EXPECTED_RAW_ROWS for i in clean.embedded_header_original_indices)
    # and none of them survived into the output
    removed = set(clean.embedded_header_original_indices)
    assert removed.isdisjoint(set(day.original_index.to_numpy().tolist()))


def test_wrong_expected_header_rows_fails_clearly(raw_csv_path: Path) -> None:
    with pytest.raises(RowCountMismatchError, match="embedded header"):
        load_day_csv(
            raw_csv_path,
            expected_embedded_header_rows=EXPECTED_EMBEDDED_HEADER_ROWS + 1,
            compute_duplicate_stats=False,
            verify_raw_line_count=False,
        )


def test_only_embedded_header_rows_are_removed(tmp_path: Path) -> None:
    """A header row in the middle of the data is dropped; nothing else is."""
    rows = [
        _synthetic_row(**{schema.LABEL_COLUMN: "SyntheticA"}),
        _synthetic_row(**{schema.LABEL_COLUMN: "SyntheticB"}),
        list(schema.CSV_COLUMNS),  # a repeated header line, as in the real files
        _synthetic_row(**{schema.LABEL_COLUMN: "SyntheticC"}),
    ]
    path = _write_csv(tmp_path / "synthetic.csv", rows)

    loaded = load_day_csv(path)
    assert loaded.stats.cleaning.n_raw_rows == 4
    assert loaded.stats.cleaning.n_embedded_header_rows == 1
    assert loaded.stats.cleaning.embedded_header_original_indices == [2]
    assert loaded.n_rows == 3
    assert loaded.labels.tolist() == ["SyntheticA", "SyntheticB", "SyntheticC"]
    # the gap left by the removed row is visible in the original positions
    assert loaded.original_index.tolist() == [0, 1, 3]


# =========================================================================== #
# 4. exactly 331,100 rows remain
# =========================================================================== #
def test_clean_row_count(day: LoadedDay) -> None:
    clean = day.stats.cleaning
    assert clean.n_rows_after_header_removal == EXPECTED_CLEAN_ROWS
    assert day.n_rows == EXPECTED_CLEAN_ROWS
    assert EXPECTED_RAW_ROWS - EXPECTED_EMBEDDED_HEADER_ROWS == EXPECTED_CLEAN_ROWS
    assert len(day.labels) == EXPECTED_CLEAN_ROWS
    assert len(day.timestamps) == EXPECTED_CLEAN_ROWS
    assert len(day.original_index) == EXPECTED_CLEAN_ROWS


# =========================================================================== #
# 5. timestamp parsing succeeds for the surviving rows
# =========================================================================== #
def test_timestamps_all_parse(day: LoadedDay) -> None:
    stats = day.stats.timestamps
    assert stats.format == "%d/%m/%Y %H:%M:%S"
    assert stats.n_unparsed == 0
    assert stats.n_parsed == EXPECTED_CLEAN_ROWS
    assert pd.api.types.is_datetime64_any_dtype(day.timestamps)
    assert day.timestamps.notna().all()


def test_timestamp_range_and_distribution(day: LoadedDay) -> None:
    stats = day.stats.timestamps
    assert stats.min_parsed == EXPECTED_TIMESTAMP_MIN
    assert stats.max_parsed == EXPECTED_TIMESTAMP_MAX
    assert stats.n_distinct_raw == EXPECTED_DISTINCT_TIMESTAMPS
    assert stats.date_histogram == EXPECTED_DATE_HISTOGRAM
    # measured: the hour field only ever holds 01-12 and no AM/PM marker exists,
    # so the wall-clock hour is ambiguous. This stage must not "fix" that.
    assert stats.n_rows_with_am_pm_marker == 0
    assert all(1 <= int(hour) <= 12 for hour in stats.hour_histogram)
    assert sum(stats.hour_histogram.values()) == EXPECTED_CLEAN_ROWS


def test_raw_timestamp_strings_are_unmodified(day: LoadedDay, raw_csv_path: Path) -> None:
    """The raw Timestamp text is preserved verbatim, not reformatted."""
    first_rows = pd.read_csv(
        raw_csv_path, usecols=[schema.TIMESTAMP_COLUMN], dtype=str, nrows=5, na_filter=False
    )
    assert day.timestamps_raw.head(5).tolist() == (
        first_rows[schema.TIMESTAMP_COLUMN].tolist()
    )
    assert day.timestamps_raw.dtype == object


def test_timestamp_is_not_the_ordering_key(day: LoadedDay) -> None:
    """Rows keep source order; the timestamp column is NOT sorted."""
    assert not day.timestamps.is_monotonic_increasing
    assert day.original_index.is_monotonic_increasing


def test_unparseable_timestamp_fails_clearly(tmp_path: Path) -> None:
    rows = [_synthetic_row(**{schema.TIMESTAMP_COLUMN: "2018-03-01 01:00:00"})]
    path = _write_csv(tmp_path / "bad_timestamp.csv", rows)
    with pytest.raises(DataQualityError, match="Timestamp"):
        load_day_csv(path)
    # ... and can be inspected instead of raising, when asked explicitly
    loaded = load_day_csv(path, require_all_timestamps_parse=False)
    assert loaded.stats.timestamps.n_unparsed == 1


# =========================================================================== #
# 6. the Label column exists and is kept separate from the features
# =========================================================================== #
def test_label_series_present_and_separate(day: LoadedDay) -> None:
    assert isinstance(day.labels, pd.Series)
    assert day.labels.name == schema.LABEL_COLUMN
    assert schema.LABEL_COLUMN not in day.features.columns
    assert schema.TIMESTAMP_COLUMN not in day.features.columns
    assert schema.LABEL_COLUMN not in day.feature_columns
    assert schema.TIMESTAMP_COLUMN not in day.feature_columns


# =========================================================================== #
# 7. the feature columns are numeric
# =========================================================================== #
def test_feature_columns_are_numeric(day: LoadedDay) -> None:
    assert day.features.shape[1] == 78
    for column, dtype in day.features.dtypes.items():
        assert pd.api.types.is_numeric_dtype(dtype), f"{column} is {dtype}"
        assert dtype == np.float64, f"{column} is {dtype}"


def test_feature_set_decomposition(day: LoadedDay) -> None:
    """80 CSV columns = 76 flow features + Dst Port + Protocol + Timestamp + Label."""
    assert len(schema.CSV_COLUMNS) == 80
    assert len(schema.FLOW_FEATURE_COLUMNS) == 76
    assert len(schema.IDENTIFIER_FEATURE_COLUMNS) == 2
    assert len(schema.MODEL_FEATURE_COLUMNS) == 78
    assert list(day.features.columns) == list(schema.MODEL_FEATURE_COLUMNS)
    assert list(day.feature_columns) == list(schema.MODEL_FEATURE_COLUMNS)
    # the 78 model features are the 76 flow features plus the 2 identifiers
    assert set(schema.MODEL_FEATURE_COLUMNS) == set(schema.FLOW_FEATURE_COLUMNS) | set(
        schema.IDENTIFIER_FEATURE_COLUMNS
    )
    assert day.stats.n_flow_features == 76
    assert day.stats.n_model_features == 78
    # order is the original CSV order, not alphabetical or regrouped
    csv_order = [c for c in schema.CSV_COLUMNS if c in set(schema.MODEL_FEATURE_COLUMNS)]
    assert list(day.features.columns) == csv_order


# =========================================================================== #
# 8. no NaN or +/-inf remains in the cleaned features
# =========================================================================== #
def test_features_are_finite(day: LoadedDay) -> None:
    clean = day.stats.cleaning
    assert clean.total_nan_after == 0
    assert clean.total_infinities_after == 0
    for column in day.features.columns:
        values = day.features[column].to_numpy(copy=False)
        assert np.isfinite(values).all(), f"{column} holds NaN/inf"


def test_nonfinite_values_were_counted_before_replacement(day: LoadedDay) -> None:
    clean = day.stats.cleaning
    assert clean.nan_counts_before == EXPECTED_NAN_BEFORE
    assert clean.posinf_counts_before == EXPECTED_POSINF_BEFORE
    assert clean.neginf_counts_before == {}
    assert clean.total_nan_before == sum(EXPECTED_NAN_BEFORE.values())
    assert clean.total_infinities_before == sum(EXPECTED_POSINF_BEFORE.values())
    # both affected columns are the known rate columns
    affected = set(clean.nan_counts_before) | set(clean.posinf_counts_before)
    assert affected == set(schema.RATE_COLUMNS)


def test_infinity_substitutions_are_recorded(day: LoadedDay) -> None:
    clean = day.stats.cleaning
    assert clean.infinity_policy == "column_finite_extremum"
    assert set(clean.infinity_substitutions) == set(schema.RATE_COLUMNS)
    for column, substitutes in clean.infinity_substitutions.items():
        assert "posinf" in substitutes
        assert np.isfinite(substitutes["posinf"])
        # +inf became the column's own finite maximum, so nothing exceeds it
        assert day.features[column].max() == pytest.approx(substitutes["posinf"])


def test_negative_infinity_uses_the_column_minimum(tmp_path: Path) -> None:
    column = "Flow Byts/s"
    rows = [
        _synthetic_row(**{column: "-Infinity"}),
        _synthetic_row(**{column: "-5"}),
        _synthetic_row(**{column: "10"}),
    ]
    path = _write_csv(tmp_path / "neg_inf.csv", rows)

    loaded = load_day_csv(path)
    assert loaded.stats.cleaning.neginf_counts_before == {column: 1}
    assert loaded.stats.cleaning.infinity_substitutions[column]["neginf"] == -5.0
    assert loaded.features[column].tolist() == [-5.0, -5.0, 10.0]


def test_constant_infinity_policy(tmp_path: Path) -> None:
    column = "Flow Pkts/s"
    rows = [
        _synthetic_row(**{column: "Infinity"}),
        _synthetic_row(**{column: "NaN"}),
        _synthetic_row(**{column: "7"}),
    ]
    path = _write_csv(tmp_path / "const_policy.csv", rows)

    loaded = load_day_csv(
        path, infinity_policy="constant", infinity_replacement=-999.0, nan_replacement=0.0
    )
    assert loaded.features[column].tolist() == [-999.0, 0.0, 7.0]
    assert loaded.stats.cleaning.infinity_substitutions[column] == {"posinf": -999.0}


def test_unknown_infinity_policy_is_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "policy.csv", [_synthetic_row()])
    with pytest.raises(ValueError, match="infinity_policy"):
        load_day_csv(path, infinity_policy="nonsense")


# =========================================================================== #
# 9. Original_Index is preserved and unique
# =========================================================================== #
def test_original_index_preserved_and_unique(day: LoadedDay) -> None:
    original = day.original_index
    assert original.name == schema.ORIGINAL_INDEX_COLUMN
    assert original.dtype == np.int64
    assert len(original) == EXPECTED_CLEAN_ROWS
    assert original.is_unique
    assert original.is_monotonic_increasing
    assert original.min() == 0
    # the file's last 13 data rows are repeated header rows, so the highest
    # surviving source position is 331,111 rather than 331,124
    assert original.max() == EXPECTED_LAST_DATA_POSITION
    # exactly the source positions minus the removed header rows
    removed = set(day.stats.cleaning.embedded_header_original_indices)
    expected = np.setdiff1d(np.arange(EXPECTED_RAW_ROWS), np.array(sorted(removed)))
    assert np.array_equal(original.to_numpy(), expected)


def test_embedded_header_positions_are_exactly_the_measured_ones(day: LoadedDay) -> None:
    """12 scattered header rows plus a run of 13 at the end of the file."""
    positions = day.stats.cleaning.embedded_header_original_indices
    assert positions == EXPECTED_HEADER_POSITIONS
    assert positions == sorted(positions)
    assert positions[-1] == EXPECTED_RAW_ROWS - 1
    trailing = [p for p in positions if p > EXPECTED_LAST_DATA_POSITION]
    assert trailing == list(range(EXPECTED_LAST_DATA_POSITION + 1, EXPECTED_RAW_ROWS))
    assert len(trailing) == 13
    assert len(positions) - len(trailing) == 12


def test_returned_objects_share_one_index(day: LoadedDay) -> None:
    index = day.features.index
    assert isinstance(index, pd.RangeIndex)
    for series in (day.labels, day.timestamps, day.timestamps_raw, day.original_index):
        assert series.index.equals(index)


def test_row_content_matches_its_original_position(day: LoadedDay, raw_csv_path: Path) -> None:
    """Spot-check that Original_Index really points at the source row."""
    probes = [0, 1, 500, 200_000, EXPECTED_CLEAN_ROWS - 1]
    for position in probes:
        source_row = int(day.original_index.iloc[position])
        raw = pd.read_csv(
            raw_csv_path,
            skiprows=range(1, source_row + 1),
            nrows=1,
            dtype=str,
            na_filter=False,
        )
        assert raw[schema.LABEL_COLUMN].iloc[0] == day.labels.iloc[position]
        assert raw[schema.TIMESTAMP_COLUMN].iloc[0] == day.timestamps_raw.iloc[position]
        assert float(raw["Flow Duration"].iloc[0]) == pytest.approx(
            day.features["Flow Duration"].iloc[position]
        )


# =========================================================================== #
# 10. no row still carries the embedded-header label
# =========================================================================== #
def test_no_embedded_header_label_remains(day: LoadedDay) -> None:
    assert not (day.labels == schema.EMBEDDED_HEADER_LABEL).any()
    assert schema.EMBEDDED_HEADER_LABEL not in set(day.labels.unique())


# =========================================================================== #
# 11. the original string labels are intact and unencoded
# =========================================================================== #
def test_labels_are_original_strings(day: LoadedDay) -> None:
    assert day.labels.dtype == object
    assert all(isinstance(value, str) for value in day.labels.unique())
    assert not day.labels.isna().any()
    assert set(day.labels.unique()) == set(EXPECTED_LABEL_DISTRIBUTION)


def test_label_distribution_matches_measurement(day: LoadedDay) -> None:
    assert day.label_distribution == EXPECTED_LABEL_DISTRIBUTION
    assert day.stats.label_distribution == EXPECTED_LABEL_DISTRIBUTION
    assert day.stats.n_labels == len(EXPECTED_LABEL_DISTRIBUTION)
    assert sum(EXPECTED_LABEL_DISTRIBUTION.values()) == EXPECTED_CLEAN_ROWS


def test_labels_are_not_encoded(day: LoadedDay) -> None:
    """No LabelEncoder is fitted at this stage - strings stay strings."""
    assert not pd.api.types.is_numeric_dtype(day.labels)
    assert not isinstance(day.labels.dtype, pd.CategoricalDtype)


# =========================================================================== #
# 12. the loader does not modify the raw CSV
# =========================================================================== #
def test_raw_csv_untouched(
    day: LoadedDay, raw_csv_path: Path, fingerprint_before: Dict[str, object]
) -> None:
    stat = raw_csv_path.stat()
    assert stat.st_size == fingerprint_before["size"]
    assert stat.st_mtime_ns == fingerprint_before["mtime_ns"]
    assert _md5(raw_csv_path) == fingerprint_before["md5"]


def test_no_files_written_into_the_raw_data_dir(config: Config, day: LoadedDay) -> None:
    """Nothing the loader produces lands in data/raw."""
    raw_dir = config.paths.raw_data_dir
    assert day.source_path.startswith(str(raw_dir))
    # the loader returns in-memory objects only; it has no output path at all
    assert not hasattr(day, "output_path")


# =========================================================================== #
# 13. duplicate statistics are reported (and nothing is removed)
# =========================================================================== #
def test_duplicate_statistics_reported(day: LoadedDay) -> None:
    duplicates = day.stats.duplicates
    assert duplicates is not None
    assert duplicates.n_rows == EXPECTED_CLEAN_ROWS
    assert duplicates.n_distinct_feature_vectors == EXPECTED_DISTINCT_FEATURE_VECTORS
    # A. exact: same features + same timestamp + same label
    assert duplicates.n_exact_duplicate_rows == EXPECTED_DUPLICATES_EXACT
    # B. temporal: same features + same label, timestamp differs
    assert duplicates.n_temporal_duplicate_rows == EXPECTED_DUPLICATES_TEMPORAL
    assert duplicates.n_feature_label_duplicate_rows == (
        EXPECTED_DUPLICATES_EXACT + EXPECTED_DUPLICATES_TEMPORAL
    )
    # C. same feature vector, different labels
    assert duplicates.n_conflicting_feature_groups == EXPECTED_CONFLICTING_GROUPS
    assert duplicates.n_conflicting_rows == EXPECTED_CONFLICTING_ROWS
    assert sum(duplicates.conflicting_label_pairs.values()) == EXPECTED_CONFLICTING_ROWS


def test_duplicates_are_measured_not_removed(day: LoadedDay) -> None:
    """The row count is unaffected by the duplicate measurement."""
    assert day.n_rows == EXPECTED_CLEAN_ROWS
    assert day.n_rows > day.stats.duplicates.n_distinct_feature_vectors
    # the duplicate rows are still physically present
    assert day.features.duplicated().sum() > 0


def test_duplicate_stats_can_be_skipped(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "no_dupes.csv", [_synthetic_row(), _synthetic_row()])
    loaded = load_day_csv(path, compute_duplicate_stats=False)
    assert loaded.stats.duplicates is None
    assert loaded.n_rows == 2


# =========================================================================== #
# Negative sentinels: measured and preserved, never clipped
# =========================================================================== #
def test_sentinel_negatives_preserved(day: LoadedDay) -> None:
    clean = day.stats.cleaning
    assert clean.negative_value_counts == EXPECTED_SENTINEL_NEGATIVES
    assert clean.sentinel_negative_counts == EXPECTED_SENTINEL_NEGATIVES
    for column, count in EXPECTED_SENTINEL_NEGATIVES.items():
        values = day.features[column].to_numpy(copy=False)
        assert int((values < 0).sum()) == count
        # the only negative value present is the -1 "not observed" sentinel
        assert int((values == schema.NOT_OBSERVED_SENTINEL).sum()) == count
        assert values.min() == float(schema.NOT_OBSERVED_SENTINEL)


def test_no_unexpected_negative_columns(day: LoadedDay) -> None:
    """Only the two known sentinel columns hold negatives in this file."""
    assert day.stats.cleaning.non_sentinel_negative_columns == {}
    assert set(day.stats.cleaning.negative_value_counts) == set(
        schema.SENTINEL_NEGATIVE_COLUMNS
    )


def test_sentinel_negatives_survive_cleaning(tmp_path: Path) -> None:
    column = "Init Fwd Win Byts"
    rows = [
        _synthetic_row(**{column: "-1"}),
        _synthetic_row(**{column: "8192"}),
        # a NaN elsewhere forces the cleaning path to run
        _synthetic_row(**{column: "-1", "Flow Byts/s": "NaN"}),
    ]
    path = _write_csv(tmp_path / "sentinels.csv", rows)

    loaded = load_day_csv(path)
    assert loaded.features[column].tolist() == [-1.0, 8192.0, -1.0]
    assert loaded.stats.cleaning.sentinel_negative_counts == {column: 2}


# =========================================================================== #
# Schema validation failures
# =========================================================================== #
def test_renamed_column_fails_clearly(tmp_path: Path) -> None:
    header = list(schema.CSV_COLUMNS)
    header[4] = "Total Fwd Packets"  # the CIC-IDS2017 spelling
    path = _write_csv(tmp_path / "renamed.csv", [_synthetic_row()], header=header)
    with pytest.raises(SchemaMismatchError) as excinfo:
        load_day_csv(path)
    assert "Tot Fwd Pkts" in str(excinfo.value)
    assert "Total Fwd Packets" in str(excinfo.value)


def test_reordered_columns_fail_clearly(tmp_path: Path) -> None:
    header = list(schema.CSV_COLUMNS)
    header[0], header[1] = header[1], header[0]
    path = _write_csv(tmp_path / "reordered.csv", [_synthetic_row()], header=header)
    with pytest.raises(SchemaMismatchError, match="different order"):
        load_day_csv(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_day_csv(tmp_path / "does_not_exist.csv")


def test_schema_module_is_self_consistent() -> None:
    assert len(set(schema.CSV_COLUMNS)) == 80
    assert schema.TIMESTAMP_COLUMN in schema.CSV_COLUMNS
    assert schema.LABEL_COLUMN in schema.CSV_COLUMNS
    assert set(schema.NON_FEATURE_COLUMNS).isdisjoint(set(schema.MODEL_FEATURE_COLUMNS))
    assert set(schema.FLOW_FEATURE_COLUMNS).isdisjoint(
        set(schema.IDENTIFIER_FEATURE_COLUMNS)
    )
    for column in schema.RATE_COLUMNS + schema.SENTINEL_NEGATIVE_COLUMNS:
        assert column in schema.MODEL_FEATURE_COLUMNS
    assert "76" in schema.describe()
    assert "78" in schema.describe()


# =========================================================================== #
# Config wiring and reporting
# =========================================================================== #
def test_loader_config_defaults(config: Config) -> None:
    assert config.loader.primary_csv == "csv/03-01-2018.csv"
    assert config.loader.infinity_policy == "column_finite_extremum"
    assert config.loader.nan_replacement == 0.0
    assert config.loader.compute_duplicate_stats is True
    assert config.loader.verify_raw_line_count is True


def test_loader_config_rejects_bad_values() -> None:
    with pytest.raises(ConfigError, match="infinity_policy"):
        Config.from_dict({"loader": {"infinity_policy": "clip"}}).validate()
    with pytest.raises(ConfigError, match="primary_csv"):
        Config.from_dict({"loader": {"primary_csv": ""}}).validate()
    with pytest.raises(ConfigError, match="nan_replacement"):
        Config.from_dict({"loader": {"nan_replacement": "zero"}}).validate()


def test_resolve_primary_csv_is_inside_the_raw_dir(config: Config) -> None:
    path = resolve_primary_csv(config)
    assert path.is_absolute()
    assert str(path).startswith(str(config.paths.raw_data_dir))


def test_debug_row_limit_truncates(config: Config, raw_csv_path: Path) -> None:
    cfg = Config.load()
    cfg.debug.enabled = True
    cfg.debug.row_limit = 5_000
    cfg.loader.compute_duplicate_stats = False
    loaded = load_from_config(cfg, path=raw_csv_path)
    assert loaded.stats.cleaning.row_limit == 5_000
    assert loaded.stats.cleaning.truncated is True
    assert loaded.stats.cleaning.n_raw_rows == 5_000
    assert loaded.n_rows == 5_000 - loaded.stats.cleaning.n_embedded_header_rows
    assert loaded.original_index.max() < 5_000


def test_metadata_is_json_serialisable(day: LoadedDay) -> None:
    payload = day.to_metadata_dict()
    text = json.dumps(payload, indent=2)
    restored = json.loads(text)
    assert restored["shape"]["rows"] == EXPECTED_CLEAN_ROWS
    assert restored["shape"]["features"] == 78
    assert len(restored["feature_columns"]) == 78
    assert restored["stats"]["cleaning"]["n_raw_rows"] == EXPECTED_RAW_ROWS
    assert restored["stats"]["labels"]["distribution"] == EXPECTED_LABEL_DISTRIBUTION
    # the record stays small: no feature values are ever serialised
    assert len(text) < 100_000


def test_summary_reports_the_measured_numbers(day: LoadedDay) -> None:
    text = day.summary()
    assert "331,125" in text
    assert "331,100" in text
    assert "03-01-2018.csv" in text
    assert "Infilteration" in text


def test_validate_catches_a_corrupted_result(day: LoadedDay) -> None:
    """The integrity check is real, not decorative."""
    broken = LoadedDay(
        features=day.features.head(10).copy(),
        labels=day.labels.head(10),
        timestamps=day.timestamps.head(10),
        timestamps_raw=day.timestamps_raw.head(10),
        original_index=day.original_index.head(10),
        source_file=day.source_file,
        source_path=day.source_path,
        feature_columns=list(day.feature_columns),
        stats=day.stats,
    )
    broken.validate()  # the slice itself is consistent

    broken.features.iloc[0, 0] = np.inf
    with pytest.raises(DataQualityError, match="NaN/inf"):
        broken.validate()

    broken.features.iloc[0, 0] = 0.0
    broken.original_index = broken.original_index.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataQualityError, match="not increasing"):
        broken.validate()
