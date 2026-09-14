"""Tests for the Stage B multi-day corpus profiler.

Two clearly separated groups, so a synthetic fixture can never be mistaken for
a measured dataset fact:

**Synthetic** (the majority). Tiny schema-shaped CSVs written into pytest's
``tmp_path`` with obviously fake values (``SyntheticA``, ``FakeAttack``). They
exercise the edge cases the real files cannot reach on demand: a shuffled
filesystem order, a file whose name carries no date, an unknown header, a
``-inf`` value, a hand-built label conflict. They never touch ``data/raw`` and
no number from them is ever reported as a dataset statistic.

**Real data** (skipped, not failed, when the corpus is absent). Discovery and
header inspection over the whole corpus, which read at most one line per file,
plus one full profile of the smallest file ``03-01-2018.csv`` asserted against
the values measured during the inspection phase and reproduced independently by
the Stage A loader. If the profiler and the loader ever disagree about that
file, this suite fails rather than quietly accepting a new number.

Every test also upholds the Stage B contract: the profiler measures only. The
``test_raw_*`` tests fingerprint the corpus before and after and prove nothing
under ``data/raw`` was modified.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.preprocessing import profile_artifacts, schema
from src.preprocessing.loader import RowCountMismatchError
from src.preprocessing.profiling import (
    CorpusNotFoundError,
    CorpusProfile,
    DiscoveredFile,
    FileProfile,
    analyse_split_coverage,
    combine_hashes,
    discover_csv_files,
    inspect_header,
    kth_kept_positions,
    measure_duplicates,
    parse_capture_date,
    profile_corpus,
    scout_file,
    sequence_files,
)

# --------------------------------------------------------------------------- #
# Values measured from data/raw/csv/03-01-2018.csv during dataset inspection and
# reproduced independently by the Stage A loader (see tests/test_loader.py).
# None of them is an estimate; the profiler must reproduce all of them exactly.
# --------------------------------------------------------------------------- #
REAL_FILE = "03-01-2018.csv"
REAL_RAW_ROWS = 331_125
REAL_EMBEDDED_HEADERS = 25
REAL_USABLE_ROWS = 331_100
REAL_LAST_ORIGINAL_INDEX = 331_111
REAL_LABELS: Dict[str, int] = {"Benign": 238_037, "Infilteration": 93_063}
REAL_NAN = 1_834
REAL_POSINF = 4_004
REAL_NEGINF = 0
REAL_NEGATIVES: Dict[str, int] = {
    "Init Fwd Win Byts": 118_201,
    "Init Bwd Win Byts": 159_458,
}
REAL_EXACT_DUPLICATES = 73
REAL_TEMPORAL_DUPLICATES = 40_144
REAL_CONFLICTING_GROUPS = 8_227
REAL_CONFLICTING_ROWS = 44_625
REAL_DISTINCT_FEATURE_VECTORS = 282_656
REAL_DISTINCT_TIMESTAMPS = 32_068
REAL_TIMESTAMP_MIN = "2018-03-01T01:00:00"
REAL_TIMESTAMP_MAX = "2018-03-01T12:59:59"

#: Capture dates the corpus is expected to cover, derived from the CIC-IDS2018
#: capture schedule rather than from a hard-coded filename list. Discovery still
#: globs the directory; this only says which dates must turn up.
EXPECTED_CAPTURE_DATES = [
    date(2018, 2, 14),
    date(2018, 2, 15),
    date(2018, 2, 16),
    date(2018, 2, 20),
    date(2018, 2, 21),
    date(2018, 2, 22),
    date(2018, 2, 23),
    date(2018, 2, 28),
    date(2018, 3, 1),
    date(2018, 3, 2),
]


# --------------------------------------------------------------------------- #
# Synthetic corpus helpers - fake values only, always under tmp_path
# --------------------------------------------------------------------------- #
def _row(**overrides: str) -> List[str]:
    """One schema-shaped row of obviously fake values."""
    values = {column: "0" for column in schema.CSV_COLUMNS}
    values[schema.TIMESTAMP_COLUMN] = "01/03/2018 01:00:00"
    values[schema.LABEL_COLUMN] = "SyntheticA"
    values.update(overrides)
    return [values[column] for column in schema.CSV_COLUMNS]


def _write_csv(
    path: Path,
    rows: List[List[str]],
    *,
    header: Optional[List[str]] = None,
    newline: str = "\n",
) -> Path:
    names = header if header is not None else list(schema.CSV_COLUMNS)
    lines = [",".join(names)] + [",".join(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((newline.join(lines) + newline).encode("utf-8"))
    return path


def _fingerprint(paths: List[Path]) -> Dict[str, str]:
    prints: Dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        stat = path.stat()
        prints[path.name] = f"{stat.st_size}:{stat.st_mtime_ns}:{digest.hexdigest()}"
    return prints


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    """A three-day synthetic corpus with the edge cases wired in.

    * ``02-14-2018.csv`` - 4 rows, one embedded header line in the middle.
    * ``02-15-2018.csv`` - 3 rows, one NaN and one -inf, all in a rate column.
    * ``nested/03-02-2018.csv`` - 2 rows in a subdirectory, CRLF line endings,
      carrying a class that exists on no earlier date.
    """
    root = tmp_path / "csv"
    _write_csv(
        root / "02-14-2018.csv",
        [
            _row(),
            _row(**{schema.LABEL_COLUMN: "FakeAttack"}),
            list(schema.CSV_COLUMNS),  # a repeated header line, as in the real files
            _row(**{schema.TIMESTAMP_COLUMN: "14/02/2018 03:00:00"}),
        ],
    )
    _write_csv(
        root / "02-15-2018.csv",
        [
            _row(**{"Flow Byts/s": "NaN"}),
            _row(**{"Flow Pkts/s": "-Infinity"}),
            _row(**{schema.LABEL_COLUMN: "FakeAttack", "Init Fwd Win Byts": "-1"}),
        ],
    )
    _write_csv(
        root / "nested" / "03-02-2018.csv",
        [
            _row(**{schema.LABEL_COLUMN: "LateOnlyClass"}),
            _row(),
        ],
        newline="\r\n",
    )
    return root


@pytest.fixture
def synthetic_profile(synthetic_corpus: Path) -> CorpusProfile:
    return profile_corpus(
        synthetic_corpus, chunk_size=2, compare_parquet=False, max_examples=8
    )


# --------------------------------------------------------------------------- #
# Fixtures for the real corpus
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def real_csv_dir(config: Config) -> Path:
    path = config.paths.raw_data_dir / config.profiling.csv_subdir
    if not path.is_dir():
        pytest.skip(f"raw CSV corpus not present: {path}")
    return path


@pytest.fixture(scope="session")
def real_discovered(real_csv_dir: Path) -> List[DiscoveredFile]:
    return discover_csv_files(real_csv_dir)


@pytest.fixture(scope="session")
def real_fingerprint_before(real_discovered: List[DiscoveredFile]) -> Dict[str, str]:
    """Fingerprints taken before the profiler ever opens a corpus file."""
    return _fingerprint([item.path for item in real_discovered])


@pytest.fixture(scope="session")
def real_smallest_profile(
    real_csv_dir: Path, real_fingerprint_before: Dict[str, str]
) -> CorpusProfile:
    """One full profile of the smallest real file (~108 MB, a few seconds).

    Depends on ``real_fingerprint_before`` purely for ordering, so the
    "raw files untouched" test is meaningful.
    """
    if not (real_csv_dir / REAL_FILE).is_file():
        pytest.skip(f"{REAL_FILE} not present under {real_csv_dir}")
    return profile_corpus(
        real_csv_dir, only_files=[REAL_FILE], compare_parquet=False, chunk_size=100_000
    )


# =========================================================================== #
# 1. every expected CSV file is discovered
# =========================================================================== #
def test_discovery_finds_every_synthetic_file(synthetic_corpus: Path) -> None:
    found = discover_csv_files(synthetic_corpus)
    assert [item.source_file for item in found] == [
        "02-14-2018.csv",
        "02-15-2018.csv",
        "03-02-2018.csv",
    ]
    # the third file lives in a subdirectory: discovery must recurse
    assert found[2].path.parent.name == "nested"


def test_discovery_is_by_glob_not_by_hard_coded_name(tmp_path: Path) -> None:
    """A file whose name is nothing like the real ones is still discovered."""
    root = tmp_path / "csv"
    _write_csv(root / "some-unexpected-name.csv", [_row()])
    found = discover_csv_files(root)
    assert [item.source_file for item in found] == ["some-unexpected-name.csv"]
    assert found[0].capture_date is None
    assert found[0].capture_date_source == "unknown"


def test_discovery_reports_size_without_reading(synthetic_corpus: Path) -> None:
    for item in discover_csv_files(synthetic_corpus):
        assert item.size_bytes == item.path.stat().st_size
        assert item.size_bytes > 0


def test_discovery_ignores_non_matching_extensions(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(root / "02-14-2018.csv", [_row()])
    (root / "notes.txt").write_text("not a corpus file\n", encoding="utf-8")
    (root / "02-14-2018.parquet").write_bytes(b"PAR1")
    assert [item.source_file for item in discover_csv_files(root)] == ["02-14-2018.csv"]


def test_missing_corpus_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotFoundError, match="not found"):
        discover_csv_files(tmp_path / "does-not-exist")


def test_empty_corpus_dir_raises(empty_data_dir: Path) -> None:
    with pytest.raises(CorpusNotFoundError, match="No CSV files"):
        discover_csv_files(empty_data_dir)


def test_unknown_requested_file_raises(synthetic_corpus: Path) -> None:
    with pytest.raises(CorpusNotFoundError, match="not found"):
        profile_corpus(
            synthetic_corpus, only_files=["99-99-2018.csv"], compare_parquet=False
        )


@pytest.mark.slow
def test_real_corpus_discovers_every_expected_date(
    real_discovered: List[DiscoveredFile],
) -> None:
    dates = [item.capture_date for item in real_discovered]
    assert dates == EXPECTED_CAPTURE_DATES, (
        "the corpus no longer covers the expected CIC-IDS2018 capture dates"
    )
    assert len(real_discovered) == len(EXPECTED_CAPTURE_DATES)
    assert all(item.capture_date_source == "filename" for item in real_discovered)
    assert len({item.path for item in real_discovered}) == len(real_discovered)


# =========================================================================== #
# 2. capture-date ordering is deterministic and independent of the filesystem
# =========================================================================== #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("02-14-2018.csv", date(2018, 2, 14)),
        ("03-02-2018.csv", date(2018, 3, 2)),
        ("a/b/02-28-2018.csv", date(2018, 2, 28)),
        ("prefix_12-31-2017_suffix.csv", date(2017, 12, 31)),
        ("no-date-here.csv", None),
        ("13-45-2018.csv", None),  # month 13 / day 45: rejected, not guessed
        ("1234-56-78901.csv", None),  # digits, but not an MM-DD-YYYY group
    ],
)
def test_capture_date_comes_from_the_name(name: str, expected: Optional[date]) -> None:
    assert parse_capture_date(Path(name).name) == expected


def test_sequence_is_stable_under_any_input_order() -> None:
    files = [
        DiscoveredFile("03-02-2018.csv", Path("c"), 1, date(2018, 3, 2), "filename"),
        DiscoveredFile("02-14-2018.csv", Path("a"), 1, date(2018, 2, 14), "filename"),
        DiscoveredFile("02-28-2018.csv", Path("b"), 1, date(2018, 2, 28), "filename"),
        DiscoveredFile("undated.csv", Path("d"), 1, None, "unknown"),
    ]
    expected = ["02-14-2018.csv", "02-28-2018.csv", "03-02-2018.csv", "undated.csv"]
    rng = random.Random(0)
    for _ in range(20):
        shuffled = files[:]
        rng.shuffle(shuffled)
        assert [item.source_file for item in sequence_files(shuffled)] == expected


def test_same_date_ties_break_on_name_not_arrival_order() -> None:
    same_day = date(2018, 2, 14)
    files = [
        DiscoveredFile("b.csv", Path("b"), 1, same_day, "filename"),
        DiscoveredFile("a.csv", Path("a"), 1, same_day, "filename"),
    ]
    assert [item.source_file for item in sequence_files(files)] == ["a.csv", "b.csv"]
    assert [item.source_file for item in sequence_files(files[::-1])] == [
        "a.csv",
        "b.csv",
    ]


def test_filesystem_order_does_not_affect_the_sequence(
    synthetic_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reverse what ``rglob`` yields; the profiled sequence must not change."""
    baseline = [item.source_file for item in discover_csv_files(synthetic_corpus)]

    original_rglob = Path.rglob

    def reversed_rglob(self: Path, pattern: str):  # type: ignore[no-untyped-def]
        return iter(list(original_rglob(self, pattern))[::-1])

    monkeypatch.setattr(Path, "rglob", reversed_rglob)
    assert [item.source_file for item in discover_csv_files(synthetic_corpus)] == baseline


def test_profile_sequence_follows_capture_date(synthetic_profile: CorpusProfile) -> None:
    # Sequence entries are "<capture date>/<file name>", so the ordering key is
    # visible in the artifact itself.
    assert synthetic_profile.sequence == [
        "2018-02-14/02-14-2018.csv",
        "2018-02-15/02-15-2018.csv",
        "2018-03-02/03-02-2018.csv",
    ]
    assert synthetic_profile.dates == ["2018-02-14", "2018-02-15", "2018-03-02"]
    assert "capture-order proxy" in synthetic_profile.ordering_contract
    # the contract names the two ordering keys and nothing else
    assert "capture date" in synthetic_profile.ordering_contract
    assert schema.ORIGINAL_INDEX_COLUMN in synthetic_profile.ordering_contract


def test_two_runs_over_the_same_synthetic_corpus_agree(synthetic_corpus: Path) -> None:
    first = profile_corpus(synthetic_corpus, chunk_size=2, compare_parquet=False)
    second = profile_corpus(synthetic_corpus, chunk_size=3, compare_parquet=False)
    assert profile_artifacts.measurement_digest(
        first
    ) == profile_artifacts.measurement_digest(second)


# =========================================================================== #
# 3. source file + original index uniquely identifies a row
# =========================================================================== #
def test_kth_kept_positions_matches_a_brute_force_mask() -> None:
    rng = np.random.default_rng(7)
    for n_rows, n_excluded in ((50, 0), (50, 1), (200, 17), (1000, 250)):
        excluded = np.sort(
            rng.choice(n_rows, size=n_excluded, replace=False).astype(np.int64)
        )
        mask = np.ones(n_rows, dtype=bool)
        mask[excluded] = False
        expected = np.flatnonzero(mask)
        got = kth_kept_positions(np.arange(expected.size, dtype=np.int64), excluded)
        assert np.array_equal(got, expected)


def test_kth_kept_positions_handles_a_trailing_run() -> None:
    """The real 03-01 file ends in a run of 13 header rows; that shape matters."""
    excluded = np.array([2] + list(range(7, 10)), dtype=np.int64)
    assert kth_kept_positions(np.arange(6, dtype=np.int64), excluded).tolist() == [
        0,
        1,
        3,
        4,
        5,
        6,
    ]


def test_row_identity_is_unique_and_skips_embedded_headers(
    synthetic_profile: CorpusProfile,
) -> None:
    day = synthetic_profile.by_name("02-14-2018.csv")
    assert day is not None
    # 4 raw rows, 1 of them an embedded header at position 2
    assert day.intake.n_raw_rows == 4
    assert day.intake.embedded_header_positions == [2]
    assert day.intake.n_usable_rows == 3
    assert day.intake.first_original_index == 0
    assert day.intake.last_original_index == 3


def test_original_index_uniquely_identifies_a_row(tmp_path: Path) -> None:
    """Rebuild (source_file, original_index) for every row and check uniqueness."""
    root = tmp_path / "csv"
    _write_csv(
        root / "02-14-2018.csv",
        [_row() if i not in (1, 4) else list(schema.CSV_COLUMNS) for i in range(6)],
    )
    _write_csv(root / "02-15-2018.csv", [_row() for _ in range(3)])
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)

    identities: List[tuple] = []
    for item in profile.files:
        excluded = np.asarray(item.intake.embedded_header_positions, dtype=np.int64)
        ranks = np.arange(item.intake.n_usable_rows, dtype=np.int64)
        for position in kth_kept_positions(ranks, excluded).tolist():
            identities.append((item.source_file, position))
            assert position not in set(excluded.tolist())
    assert len(identities) == len(set(identities)) == profile.total_usable_rows == 7


def test_row_counts_are_cross_checked_against_physical_lines(tmp_path: Path) -> None:
    """A file with an embedded newline is refused, not silently mis-indexed."""
    path = tmp_path / "02-14-2018.csv"
    _write_csv(path, [_row()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('"a\nb"' + "," * (len(schema.CSV_COLUMNS) - 1) + "\n")
    with pytest.raises(RowCountMismatchError, match="physical lines"):
        scout_file(path)


# =========================================================================== #
# 4. embedded header rows are detected, counted and never written back
# =========================================================================== #
def test_embedded_header_positions_are_exact(tmp_path: Path) -> None:
    path = tmp_path / "02-14-2018.csv"
    header_positions = [0, 3, 4, 9]
    rows = [
        list(schema.CSV_COLUMNS) if i in header_positions else _row()
        for i in range(10)
    ]
    _write_csv(path, rows)
    scout = scout_file(path)
    assert scout.n_raw_rows == 10
    assert scout.embedded_header_positions == header_positions
    assert scout.n_usable_rows == 6
    assert schema.EMBEDDED_HEADER_LABEL not in scout.label_counts
    assert scout.label_counts == {"SyntheticA": 6}


def test_embedded_header_rows_are_excluded_not_deleted(tmp_path: Path) -> None:
    path = tmp_path / "02-14-2018.csv"
    rows = [_row(), list(schema.CSV_COLUMNS), _row()]
    _write_csv(path, rows)
    before = _fingerprint([path])
    profile = profile_corpus(path.parent, chunk_size=1, compare_parquet=False)
    day = profile.files[0]
    assert day.intake.n_embedded_header_rows == 1
    assert day.intake.n_usable_rows == 2
    assert day.intake.n_rows_profiled == 2
    # the raw file still holds all three data rows
    assert _fingerprint([path]) == before
    assert path.read_text(encoding="utf-8").count("\n") == 4


def test_a_file_of_only_header_rows_yields_no_usable_rows(tmp_path: Path) -> None:
    path = tmp_path / "02-14-2018.csv"
    _write_csv(path, [list(schema.CSV_COLUMNS) for _ in range(3)])
    profile = profile_corpus(path.parent, chunk_size=2, compare_parquet=False)
    assert profile.total_raw_rows == 3
    assert profile.total_embedded_header_rows == 3
    assert profile.total_usable_rows == 0
    assert profile.label_counts == {}


@pytest.mark.slow
def test_real_file_embedded_headers_match_the_inspection(
    real_smallest_profile: CorpusProfile,
) -> None:
    day = real_smallest_profile.by_name(REAL_FILE)
    assert day is not None
    assert day.intake.n_raw_rows == REAL_RAW_ROWS
    assert day.intake.n_physical_lines == REAL_RAW_ROWS + 1
    assert day.intake.line_count_verified is True
    assert day.intake.n_embedded_header_rows == REAL_EMBEDDED_HEADERS
    assert day.intake.n_usable_rows == REAL_USABLE_ROWS
    assert day.intake.n_rows_profiled == REAL_USABLE_ROWS
    # the file's last 13 data rows are header lines, so the last usable row is
    # not at n_raw_rows - 1
    assert day.intake.last_original_index == REAL_LAST_ORIGINAL_INDEX
    assert day.intake.at_excel_row_cap is False


# =========================================================================== #
# 5. schema detection and validation
# =========================================================================== #
def test_standard_header_is_detected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "02-14-2018.csv", [_row()])
    info = inspect_header(path)
    assert info.n_columns == 80
    assert info.variant.variant == schema.VARIANT_STANDARD
    assert info.variant.usable is True
    assert info.md5_stripped == schema.HEADER_MD5
    assert info.line_ending == "LF"
    assert info.variant.extra_columns == []
    assert info.variant.missing_columns == []


def test_extended_84_column_header_is_detected_and_reported(tmp_path: Path) -> None:
    """The 4 extra identifier columns are reported, never silently discarded."""
    extended = list(schema.CSV_COLUMNS_EXTENDED)
    row = ["synthetic-flow-id", "10.0.0.1", "1234", "10.0.0.2"] + _row()
    path = _write_csv(
        tmp_path / "02-20-2018.csv", [row], header=extended, newline="\r\n"
    )
    info = inspect_header(path)
    assert info.n_columns == 84
    assert info.variant.variant == schema.VARIANT_EXTENDED
    assert info.variant.usable is True
    assert info.variant.extra_columns == list(schema.EXTENDED_IDENTIFIER_COLUMNS)
    assert info.variant.missing_columns == []
    assert info.md5_stripped == schema.HEADER_MD5_EXTENDED
    assert info.line_ending == "CRLF"
    assert any("84-column" in note for note in info.variant.notes)


def test_extended_file_is_read_on_the_78_model_features(tmp_path: Path) -> None:
    extended = list(schema.CSV_COLUMNS_EXTENDED)
    rows = [
        ["fid-1", "10.0.0.1", "1", "10.0.0.2"] + _row(),
        ["fid-2", "10.0.0.3", "2", "10.0.0.4"] + _row(**{schema.LABEL_COLUMN: "FakeAttack"}),
    ]
    root = tmp_path / "csv"
    _write_csv(root / "02-20-2018.csv", rows, header=extended)
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)
    day = profile.files[0]
    assert day.usable is True
    assert day.error is None
    assert day.intake.n_usable_rows == 2
    assert day.label_counts == {"FakeAttack": 1, "SyntheticA": 1}
    assert profile.schema_variants == {schema.VARIANT_EXTENDED: ["02-20-2018.csv"]}


def test_unknown_header_is_reported_and_the_run_continues(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(root / "02-14-2018.csv", [_row()])
    _write_csv(
        root / "02-15-2018.csv",
        [["1", "2", "3"]],
        header=["not_a_cic_column", "another", "third"],
    )
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    assert len(profile.files) == 2
    good = profile.by_name("02-14-2018.csv")
    bad = profile.by_name("02-15-2018.csv")
    assert good is not None and good.usable is True
    assert bad is not None and bad.usable is False
    assert bad.error is not None and "schema" in bad.error
    assert bad.header.variant.missing_columns  # the difference is reported
    assert any("02-15-2018.csv" in warning for warning in profile.warnings)
    # the good file was still measured
    assert profile.total_usable_rows == 1


def test_schema_variants_are_grouped_per_file(synthetic_profile: CorpusProfile) -> None:
    assert synthetic_profile.schema_variants == {
        schema.VARIANT_STANDARD: ["02-14-2018.csv", "02-15-2018.csv", "03-02-2018.csv"]
    }


def test_model_feature_count_is_the_documented_78() -> None:
    assert len(schema.CSV_COLUMNS) == 80
    assert len(schema.FLOW_FEATURE_COLUMNS) == 76
    assert len(schema.MODEL_FEATURE_COLUMNS) == 78
    assert set(schema.MODEL_FEATURE_COLUMNS) == set(schema.FLOW_FEATURE_COLUMNS) | {
        "Dst Port",
        "Protocol",
    }
    assert schema.TIMESTAMP_COLUMN not in schema.MODEL_FEATURE_COLUMNS
    assert schema.LABEL_COLUMN not in schema.MODEL_FEATURE_COLUMNS


@pytest.mark.slow
def test_real_headers_are_all_usable(real_discovered: List[DiscoveredFile]) -> None:
    """Reads one line per file, so this is cheap even over 6.4 GiB."""
    variants: Dict[str, List[str]] = {}
    for item in real_discovered:
        info = inspect_header(item.path)
        assert info.variant.usable is True, f"{item.source_file}: {info.variant.notes}"
        assert info.variant.missing_columns == []
        variants.setdefault(info.variant.variant, []).append(item.source_file)
    assert sorted(variants) == [schema.VARIANT_EXTENDED, schema.VARIANT_STANDARD]
    assert variants[schema.VARIANT_EXTENDED] == ["02-20-2018.csv"]
    assert len(variants[schema.VARIANT_STANDARD]) == 9


# =========================================================================== #
# 6. label collection - original spelling preserved, nothing encoded
# =========================================================================== #
def test_global_label_vocabulary_is_the_union_of_the_files(
    synthetic_profile: CorpusProfile,
) -> None:
    assert synthetic_profile.label_vocabulary == [
        "FakeAttack",
        "LateOnlyClass",
        "SyntheticA",
    ]
    assert synthetic_profile.label_counts == {
        "SyntheticA": 5,
        "FakeAttack": 2,
        "LateOnlyClass": 1,
    }
    assert sum(synthetic_profile.label_counts.values()) == (
        synthetic_profile.total_usable_rows
    )


def test_per_file_label_counts_sum_to_the_global_counts(
    synthetic_profile: CorpusProfile,
) -> None:
    totals: Dict[str, int] = {}
    for item in synthetic_profile.files:
        assert sum(item.label_counts.values()) == item.intake.n_rows_profiled
        for label, count in item.label_counts.items():
            totals[label] = totals.get(label, 0) + count
    assert totals == synthetic_profile.label_counts


def test_label_spelling_is_preserved_exactly(tmp_path: Path) -> None:
    """Casing, spaces and hyphens survive untouched; nothing is normalised."""
    root = tmp_path / "csv"
    odd = ["DoS attacks-Hulk", "dos attacks-hulk", "Infilteration", " Padded "]
    _write_csv(
        root / "02-14-2018.csv",
        [_row(**{schema.LABEL_COLUMN: name}) for name in odd],
    )
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    assert sorted(profile.label_counts) == sorted(odd)
    assert set(profile.label_vocabulary) == set(odd)


def test_no_label_encoding_artifact_is_produced(
    synthetic_profile: CorpusProfile, tmp_path: Path
) -> None:
    """Stage B must not persist a class mapping (no LabelEncoder yet)."""
    payload = synthetic_profile.to_dict()
    assert "label_mapping" not in json.dumps(payload)
    out = tmp_path / "artifacts"
    written = profile_artifacts.write_artifacts(
        synthetic_profile, out, config={}, config_summary="", chunk_size=2
    )
    assert set(written) == set(profile_artifacts.ARTIFACT_NAMES)
    names = {path.name for path in out.iterdir()}
    forbidden = {
        "label_mapping.json",
        "label_encoder.json",
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "X_test.npy",
    }
    assert names & forbidden == set()


def test_class_timeline_records_first_and_last_date(
    synthetic_profile: CorpusProfile,
) -> None:
    timeline = synthetic_profile.timeline
    assert timeline.dates == ["2018-02-14", "2018-02-15", "2018-03-02"]
    assert timeline.first_date["SyntheticA"] == "2018-02-14"
    assert timeline.last_date["SyntheticA"] == "2018-03-02"
    assert timeline.first_date["LateOnlyClass"] == "2018-03-02"
    assert timeline.n_dates_present["LateOnlyClass"] == 1
    assert "LateOnlyClass" in timeline.single_date_classes
    assert "LateOnlyClass" in timeline.late_only_classes
    assert "SyntheticA" not in timeline.single_date_classes


def test_split_coverage_flags_a_class_absent_from_training() -> None:
    """A class confined to the final rows is reported, not fixed."""
    vocabulary = ["Benign", "LateOnlyClass"]
    codes = np.zeros(100, dtype=np.int16)
    codes[95:] = 1
    profiles = [FileProfile(file=DiscoveredFile("synthetic.csv", Path("x"), 1))]
    profiles[0].intake.n_rows_profiled = 100
    coverage = analyse_split_coverage(codes, vocabulary, profiles, (0.70, 0.10, 0.20))
    assert coverage.train_end == 70
    assert coverage.val_end == 80
    assert coverage.absent_from_train == ["LateOnlyClass"]
    assert coverage.absent_from_val == ["LateOnlyClass"]
    assert coverage.absent_from_test == []
    assert coverage.train_counts == {"Benign": 70}
    assert coverage.test_counts == {"Benign": 15, "LateOnlyClass": 5}
    assert coverage.exact is True
    assert "does not change the split strategy" in coverage.to_dict()["note"]


def test_split_coverage_flags_a_starved_class() -> None:
    vocabulary = ["Benign", "RareClass"]
    codes = np.zeros(1000, dtype=np.int16)
    codes[:5] = 1
    profiles = [FileProfile(file=DiscoveredFile("synthetic.csv", Path("x"), 1))]
    profiles[0].intake.n_rows_profiled = 1000
    coverage = analyse_split_coverage(
        codes, vocabulary, profiles, (0.70, 0.10, 0.20), starved_threshold=100
    )
    assert coverage.starved_in_train == {"RareClass": 5}
    assert coverage.train_only == ["RareClass"]


def test_split_ratios_come_from_the_configuration(synthetic_corpus: Path) -> None:
    profile = profile_corpus(
        synthetic_corpus,
        chunk_size=2,
        compare_parquet=False,
        split_ratios=(0.5, 0.25, 0.25),
    )
    assert profile.split_coverage.ratios == (0.5, 0.25, 0.25)
    assert profile.split_coverage.train_end == 4  # 8 usable rows * 0.5


@pytest.mark.slow
def test_real_file_labels_match_the_inspection(
    real_smallest_profile: CorpusProfile,
) -> None:
    day = real_smallest_profile.by_name(REAL_FILE)
    assert day is not None
    assert day.label_counts == REAL_LABELS
    assert real_smallest_profile.label_counts == REAL_LABELS
    assert sum(REAL_LABELS.values()) == REAL_USABLE_ROWS
    # the dataset's own misspelling is preserved
    assert "Infilteration" in real_smallest_profile.label_vocabulary
    assert "Infiltration" not in real_smallest_profile.label_vocabulary


@pytest.mark.slow
def test_heartbleed_is_reported_absent_not_fabricated(
    real_smallest_profile: CorpusProfile,
) -> None:
    status = profile_artifacts.heartbleed_status(real_smallest_profile)
    assert status["present"] is False
    assert status["matched_labels"] == []
    assert status["rows"] == 0
    assert status["n_files_searched"] == 1
    assert "ABSENT" in status["statement"]


# =========================================================================== #
# 7. non-finite values are measured, never silently altered
# =========================================================================== #
def test_non_finite_values_are_counted_per_column(synthetic_profile: CorpusProfile) -> None:
    day = synthetic_profile.by_name("02-15-2018.csv")
    assert day is not None
    assert day.non_finite.nan_counts == {"Flow Byts/s": 1}
    assert day.non_finite.neginf_counts == {"Flow Pkts/s": 1}
    assert day.non_finite.posinf_counts == {}
    assert day.non_finite.total_nan == 1
    assert day.non_finite.total_neginf == 1
    assert day.non_finite.unexpected_columns == []


def test_non_finite_outside_the_rate_columns_is_flagged(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(root / "02-14-2018.csv", [_row(**{"Flow Duration": "Infinity"})])
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)
    day = profile.files[0]
    assert day.non_finite.posinf_counts == {"Flow Duration": 1}
    assert day.non_finite.unexpected_columns == ["Flow Duration"]


def test_stage_b_replaces_no_non_finite_value(tmp_path: Path) -> None:
    """The counts survive the run and the source bytes are unchanged."""
    root = tmp_path / "csv"
    path = _write_csv(
        root / "02-14-2018.csv",
        [
            _row(**{"Flow Byts/s": "NaN"}),
            _row(**{"Flow Byts/s": "Infinity"}),
            _row(**{"Flow Pkts/s": "-Infinity"}),
        ],
    )
    before = path.read_bytes()
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)
    day = profile.files[0]
    assert (
        day.non_finite.total_nan,
        day.non_finite.total_posinf,
        day.non_finite.total_neginf,
    ) == (1, 1, 1)
    assert path.read_bytes() == before
    assert "TRAIN split alone" in day.non_finite.policy_note


def test_negative_sentinels_are_measured_not_clipped(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(
        root / "02-14-2018.csv",
        [
            _row(**{"Init Fwd Win Byts": "-1", "Init Bwd Win Byts": "-1"}),
            _row(**{"Init Fwd Win Byts": "-1"}),
            _row(**{"Init Fwd Win Byts": "8192"}),
        ],
    )
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    day = profile.files[0]
    assert day.negatives.negative_counts == {
        "Init Fwd Win Byts": 2,
        "Init Bwd Win Byts": 1,
    }
    assert day.negatives.sentinel_minus_one_counts == {
        "Init Fwd Win Byts": 2,
        "Init Bwd Win Byts": 1,
    }
    assert sorted(day.negatives.only_minus_one_columns) == [
        "Init Bwd Win Byts",
        "Init Fwd Win Byts",
    ]
    assert day.negatives.non_sentinel_columns == {}
    assert schema.NOT_OBSERVED_SENTINEL == -1
    assert "preserved" in day.negatives.to_dict()["policy_note"]


def test_negatives_outside_the_sentinel_columns_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(root / "02-14-2018.csv", [_row(**{"Flow Duration": "-5"})])
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)
    day = profile.files[0]
    assert day.negatives.negative_counts == {"Flow Duration": 1}
    assert day.negatives.non_sentinel_columns == {"Flow Duration": 1}
    assert day.negatives.only_minus_one_columns == []


@pytest.mark.slow
def test_real_file_non_finite_counts_match_the_inspection(
    real_smallest_profile: CorpusProfile,
) -> None:
    day = real_smallest_profile.by_name(REAL_FILE)
    assert day is not None
    assert day.non_finite.total_nan == REAL_NAN
    assert day.non_finite.total_posinf == REAL_POSINF
    assert day.non_finite.total_neginf == REAL_NEGINF
    assert set(day.non_finite.nan_counts) | set(day.non_finite.posinf_counts) <= set(
        schema.RATE_COLUMNS
    )
    assert day.non_finite.unexpected_columns == []
    assert day.negatives.negative_counts == REAL_NEGATIVES
    assert day.negatives.non_sentinel_columns == {}


# =========================================================================== #
# 8. duplicate categories are measured, nothing is removed
# =========================================================================== #
def _hashes(features: List[int], labels: List[int], stamps: List[int]):
    feature_hash = np.asarray(features, dtype=np.uint64)
    label_hash = np.asarray(labels, dtype=np.uint64)
    stamp_hash = np.asarray(stamps, dtype=np.uint64)
    feature_label = combine_hashes(feature_hash, label_hash)
    return feature_hash, feature_label, combine_hashes(feature_label, stamp_hash)


def test_duplicate_categories_are_separated() -> None:
    #     rows 0,1 -> identical features, label and timestamp  => A exact
    #     rows 2,3 -> identical features and label, timestamps differ => B temporal
    #     rows 4,5 -> identical features, different labels => C conflicting
    features = [10, 10, 20, 20, 30, 30, 40]
    labels = [0, 0, 0, 0, 0, 1, 1]
    stamps = [1, 1, 1, 2, 1, 1, 1]
    feature_hash, feature_label, full = _hashes(features, labels, stamps)
    profile = measure_duplicates(
        feature_hash,
        feature_label,
        full,
        np.asarray(labels, dtype=np.int16),
        ["Benign", "FakeAttack"],
        basis="synthetic",
    )
    assert profile.n_rows == 7
    assert profile.n_distinct_feature_vectors == 4
    assert profile.n_exact == 1
    assert profile.n_temporal == 1
    assert profile.n_feature_label_duplicates == 2
    assert profile.n_conflicting_groups == 1
    assert profile.n_conflicting_rows == 2
    assert profile.conflicting_label_combinations == {
        "Benign + FakeAttack": {"groups": 1, "rows": 2}
    }
    assert profile.n_conflicting_combinations == 1
    assert "deletes nothing" in profile.note


def test_no_duplicates_is_reported_as_zero() -> None:
    feature_hash, feature_label, full = _hashes([1, 2, 3], [0, 0, 1], [1, 2, 3])
    profile = measure_duplicates(
        feature_hash,
        feature_label,
        full,
        np.asarray([0, 0, 1], dtype=np.int16),
        ["Benign", "FakeAttack"],
        basis="synthetic",
    )
    assert (profile.n_exact, profile.n_temporal, profile.n_conflicting_groups) == (0, 0, 0)
    assert profile.n_distinct_feature_vectors == 3
    assert profile.conflicting_label_combinations == {}


def test_duplicate_measurement_on_an_empty_input() -> None:
    empty_u64 = np.zeros(0, dtype=np.uint64)
    profile = measure_duplicates(
        empty_u64,
        empty_u64,
        empty_u64,
        np.zeros(0, dtype=np.int16),
        ["Benign"],
        basis="synthetic",
    )
    assert profile.n_rows == 0
    assert profile.n_distinct_feature_vectors == 0


def test_three_way_label_conflict_is_enumerated() -> None:
    features = [1, 1, 1]
    labels = [0, 1, 2]
    feature_hash, feature_label, full = _hashes(features, labels, [1, 2, 3])
    profile = measure_duplicates(
        feature_hash,
        feature_label,
        full,
        np.asarray(labels, dtype=np.int16),
        ["A", "B", "C"],
        basis="synthetic",
    )
    assert profile.n_conflicting_groups == 1
    assert profile.n_conflicting_rows == 3
    assert profile.conflicting_label_combinations == {
        "A + B + C": {"groups": 1, "rows": 3}
    }


def test_duplicates_are_measured_per_file_and_corpus_wide(tmp_path: Path) -> None:
    """The same row in two files is a corpus duplicate, not a per-file one."""
    root = tmp_path / "csv"
    shared = _row(**{"Flow Duration": "12345"})
    _write_csv(root / "02-14-2018.csv", [shared, _row(**{"Flow Duration": "1"})])
    _write_csv(root / "02-15-2018.csv", [shared])
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)

    for item in profile.files:
        assert item.duplicates is not None
        assert item.duplicates.n_exact == 0
    assert profile.global_duplicates is not None
    assert profile.global_duplicates.n_rows == 3
    assert profile.global_duplicates.n_distinct_feature_vectors == 2
    assert profile.global_duplicates.n_exact == 1


def test_duplicate_measurement_can_be_skipped(synthetic_corpus: Path) -> None:
    profile = profile_corpus(
        synthetic_corpus, chunk_size=2, compare_parquet=False, compute_duplicates=False
    )
    assert all(item.duplicates is None for item in profile.files)
    assert profile.global_duplicates is None
    # the rest of the measurement is unaffected
    assert profile.total_usable_rows == 8


def test_no_row_is_removed_when_duplicates_are_found(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    identical = _row()
    path = _write_csv(root / "02-14-2018.csv", [identical] * 5)
    before = path.read_bytes()
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    day = profile.files[0]
    assert day.duplicates is not None
    assert day.duplicates.n_exact == 4
    # all five rows were still measured, and the file is untouched
    assert day.intake.n_rows_profiled == 5
    assert profile.total_usable_rows == 5
    assert path.read_bytes() == before


@pytest.mark.slow
def test_real_file_duplicate_counts_match_the_inspection(
    real_smallest_profile: CorpusProfile,
) -> None:
    day = real_smallest_profile.by_name(REAL_FILE)
    assert day is not None and day.duplicates is not None
    duplicates = day.duplicates
    assert duplicates.n_rows == REAL_USABLE_ROWS
    assert duplicates.n_distinct_feature_vectors == REAL_DISTINCT_FEATURE_VECTORS
    assert duplicates.n_exact == REAL_EXACT_DUPLICATES
    assert duplicates.n_temporal == REAL_TEMPORAL_DUPLICATES
    assert duplicates.n_conflicting_groups == REAL_CONFLICTING_GROUPS
    assert duplicates.n_conflicting_rows == REAL_CONFLICTING_ROWS
    assert list(duplicates.conflicting_label_combinations) == ["Benign + Infilteration"]


# =========================================================================== #
# 9. timestamps are diagnostic only
# =========================================================================== #
def test_timestamps_are_parsed_and_tie_groups_measured(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    stamps = [
        "14/02/2018 01:00:00",
        "14/02/2018 01:00:00",
        "14/02/2018 01:00:00",
        "14/02/2018 02:30:15",
    ]
    _write_csv(
        root / "02-14-2018.csv",
        [_row(**{schema.TIMESTAMP_COLUMN: value}) for value in stamps],
    )
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    stamp = profile.files[0].timestamps
    assert stamp.n_parsed == 4
    assert stamp.n_unparsed == 0
    assert stamp.n_distinct == 2
    assert stamp.min_parsed == "2018-02-14T01:00:00"
    assert stamp.max_parsed == "2018-02-14T02:30:15"
    assert stamp.tie_group_max == 3
    assert stamp.n_rows_with_am_pm_marker == 0
    assert stamp.hour_histogram == {"01": 3, "02": 1}
    assert stamp.format == schema.TIMESTAMP_FORMAT


def test_unparseable_timestamp_is_counted_not_repaired(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(
        root / "02-14-2018.csv",
        [
            _row(**{schema.TIMESTAMP_COLUMN: "14/02/2018 01:00:00"}),
            _row(**{schema.TIMESTAMP_COLUMN: "not-a-timestamp"}),
        ],
    )
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)
    stamp = profile.files[0].timestamps
    assert stamp.n_parsed == 1
    assert stamp.n_unparsed == 1
    assert stamp.unparsed_examples == ["not-a-timestamp"]
    # the row itself is still profiled - nothing is dropped for a bad timestamp
    assert profile.total_usable_rows == 2


def test_am_pm_marker_is_detected_not_applied(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write_csv(
        root / "02-14-2018.csv",
        [_row(**{schema.TIMESTAMP_COLUMN: "14/02/2018 01:00:00 PM"})],
    )
    profile = profile_corpus(root, chunk_size=1, compare_parquet=False)
    stamp = profile.files[0].timestamps
    assert stamp.n_rows_with_am_pm_marker == 1
    assert "Diagnostic only" in stamp.to_dict()["ordering_note"]


def test_decreasing_timestamps_do_not_reorder_rows(tmp_path: Path) -> None:
    """Rows keep file order even when the timestamp column runs backwards."""
    root = tmp_path / "csv"
    stamps = ["14/02/2018 05:00:00", "14/02/2018 03:00:00", "14/02/2018 04:00:00"]
    _write_csv(
        root / "02-14-2018.csv",
        [_row(**{schema.TIMESTAMP_COLUMN: value}) for value in stamps],
    )
    profile = profile_corpus(root, chunk_size=3, compare_parquet=False)
    stamp = profile.files[0].timestamps
    assert stamp.n_decreasing_steps == 1
    assert stamp.monotonic_in_file_order is False
    assert profile.files[0].intake.first_original_index == 0
    assert profile.files[0].intake.last_original_index == 2


@pytest.mark.slow
def test_real_file_timestamps_match_the_inspection(
    real_smallest_profile: CorpusProfile,
) -> None:
    day = real_smallest_profile.by_name(REAL_FILE)
    assert day is not None
    stamp = day.timestamps
    assert stamp.n_parsed == REAL_USABLE_ROWS
    assert stamp.n_unparsed == 0
    assert stamp.n_distinct == REAL_DISTINCT_TIMESTAMPS
    assert stamp.min_parsed == REAL_TIMESTAMP_MIN
    assert stamp.max_parsed == REAL_TIMESTAMP_MAX
    # 12-hour clock with no AM/PM marker anywhere: the ambiguity is real and is
    # deliberately not reconstructed
    assert stamp.n_rows_with_am_pm_marker == 0
    assert all(1 <= int(hour) <= 12 for hour in stamp.hour_histogram)
    assert stamp.tie_group_max > 1


def test_off_capture_date_rows_are_counted_and_explained_in_the_report(
    tmp_path: Path,
) -> None:
    """A stray pre-2018 timestamp is counted, kept, and named in the report.

    The real corpus has 14 such rows (02-14 and 02-22), which is what drags
    those two files' reported minimum back to 1970. Left unexplained the report
    reads as if the capture spans 48 years, so the count has to reach the prose.
    """
    root = tmp_path / "csv"
    _write_csv(
        root / "03-01-2018.csv",
        [
            _row(Timestamp="01/03/2018 01:00:00"),
            _row(Timestamp="05/01/1970 03:01:17"),
            _row(Timestamp="01/03/2018 01:00:02"),
        ],
    )
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    stamp = profile.files[0].timestamps

    # measured, with the raw string kept verbatim
    assert stamp.n_rows_off_capture_date == 1
    assert stamp.off_capture_date_examples == ["05/01/1970 03:01:17"]
    # counted as parsed, not as a failure, and not dropped from the row accounting
    assert stamp.n_unparsed == 0
    assert stamp.n_parsed == 3
    assert profile.files[0].intake.n_rows_profiled == 3
    assert stamp.min_parsed == "1970-01-05T03:01:17"

    out = tmp_path / "profile_out"
    profile_artifacts.write_artifacts(
        profile, out, config={}, config_summary="", chunk_size=2
    )
    report = (out / profile_artifacts.REPORT_NAME).read_text("utf-8")
    assert "1 row(s) carry a timestamp outside their file's capture date" in report
    assert "05/01/1970 03:01:17" in report


def test_report_omits_the_off_capture_date_note_when_every_row_is_on_date(
    tmp_path: Path,
) -> None:
    root = tmp_path / "csv"
    _write_csv(root / "03-01-2018.csv", [_row(), _row()])
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    assert profile.files[0].timestamps.n_rows_off_capture_date == 0
    out = tmp_path / "profile_out"
    profile_artifacts.write_artifacts(
        profile, out, config={}, config_summary="", chunk_size=2
    )
    report = (out / profile_artifacts.REPORT_NAME).read_text("utf-8")
    assert "outside their file's capture date" not in report


# =========================================================================== #
# 10. truncation, Parquet and the artifacts
# =========================================================================== #
def test_excel_row_cap_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4-row file is flagged when the cap is lowered to 4, proving the check."""
    monkeypatch.setattr(schema, "EXCEL_ROW_CAP", 4)
    root = tmp_path / "csv"
    _write_csv(root / "02-14-2018.csv", [_row() for _ in range(4)])
    _write_csv(root / "02-15-2018.csv", [_row() for _ in range(3)])
    profile = profile_corpus(root, chunk_size=2, compare_parquet=False)
    assert profile.truncated_files == ["02-14-2018.csv"]
    assert profile.by_name("02-14-2018.csv").intake.at_excel_row_cap is True  # type: ignore[union-attr]
    assert profile.by_name("02-15-2018.csv").intake.at_excel_row_cap is False  # type: ignore[union-attr]


def test_artifacts_are_written_and_hold_no_row_data(
    synthetic_profile: CorpusProfile, tmp_path: Path
) -> None:
    out = tmp_path / "profile_out"
    written = profile_artifacts.write_artifacts(
        synthetic_profile,
        out,
        config={"profiling": {"chunk_size": 2}},
        config_summary="synthetic",
        chunk_size=2,
    )
    assert set(written) == set(profile_artifacts.ARTIFACT_NAMES)
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
    payload = json.loads((out / "global_dataset_summary.json").read_text("utf-8"))
    assert payload["totals"]["usable_rows"] == synthetic_profile.total_usable_rows
    assert payload["labels"]["vocabulary"] == synthetic_profile.label_vocabulary
    report = (out / profile_artifacts.REPORT_NAME).read_text("utf-8")
    assert "measures only" in report
    # every artifact is small: none of them is a dataset copy
    assert all(path.stat().st_size < 1 << 20 for path in written.values())


def test_class_distribution_csv_rows_sum_to_the_totals(
    synthetic_profile: CorpusProfile, tmp_path: Path
) -> None:
    out = tmp_path / "profile_out"
    profile_artifacts.write_artifacts(
        synthetic_profile, out, config={}, config_summary="", chunk_size=2
    )
    frame = pd.read_csv(out / "class_distribution.csv")
    assert frame["total_rows"].sum() == synthetic_profile.total_usable_rows
    assert sorted(frame["label"]) == sorted(synthetic_profile.label_vocabulary)
    by_day = pd.read_csv(out / "class_by_day.csv")
    assert by_day["count"].sum() == synthetic_profile.total_usable_rows


def test_measurement_digest_ignores_wall_clock_timings(
    synthetic_profile: CorpusProfile,
) -> None:
    baseline = profile_artifacts.measurement_digest(synthetic_profile)
    synthetic_profile.elapsed_seconds += 42.0
    synthetic_profile.files[0].profile_seconds += 7.0
    assert profile_artifacts.measurement_digest(synthetic_profile) == baseline


def test_measurement_digest_changes_with_a_measurement(
    synthetic_profile: CorpusProfile,
) -> None:
    baseline = profile_artifacts.measurement_digest(synthetic_profile)
    synthetic_profile.total_usable_rows += 1
    assert profile_artifacts.measurement_digest(synthetic_profile) != baseline


def test_strip_volatile_removes_only_timing_keys() -> None:
    payload = {
        "generated_at_utc": "now",
        "elapsed_seconds": 1.0,
        "rows": 5,
        "nested": {"profile_seconds": 2.0, "keep": True},
        "list": [{"run": 1, "value": 3}],
    }
    assert profile_artifacts.strip_volatile(payload) == {
        "rows": 5,
        "nested": {"keep": True},
        "list": [{"value": 3}],
    }


# =========================================================================== #
# 11. the raw corpus is never modified
# =========================================================================== #
def test_synthetic_corpus_is_byte_identical_after_profiling(
    synthetic_corpus: Path,
) -> None:
    paths = sorted(synthetic_corpus.rglob("*.csv"))
    before = _fingerprint(paths)
    listing_before = sorted(str(p.relative_to(synthetic_corpus)) for p in paths)
    profile_corpus(synthetic_corpus, chunk_size=2, compare_parquet=False)
    after_paths = sorted(synthetic_corpus.rglob("*.csv"))
    assert _fingerprint(after_paths) == before
    assert sorted(
        str(p.relative_to(synthetic_corpus)) for p in after_paths
    ) == listing_before


def test_profiling_writes_nothing_into_the_corpus_directory(
    synthetic_corpus: Path,
) -> None:
    before = sorted(str(p) for p in synthetic_corpus.rglob("*"))
    profile_corpus(synthetic_corpus, chunk_size=2, compare_parquet=False)
    assert sorted(str(p) for p in synthetic_corpus.rglob("*")) == before


@pytest.mark.slow
def test_real_corpus_is_untouched_by_the_profiler(
    real_smallest_profile: CorpusProfile,
    real_discovered: List[DiscoveredFile],
    real_fingerprint_before: Dict[str, str],
) -> None:
    """Size, mtime and sha256 of every raw file, before and after profiling."""
    assert real_smallest_profile.total_usable_rows == REAL_USABLE_ROWS
    after = _fingerprint([item.path for item in real_discovered])
    assert after == real_fingerprint_before


#: Directory names under which Stage C legitimately writes split arrays.
#: Stage B must not write arrays anywhere; Stage C writes them only here.
_STAGE_C_EXPERIMENT_DIRS = ("exp1", "exp2", "exp3")


@pytest.mark.slow
def test_no_processed_copy_of_the_corpus_exists(config: Config) -> None:
    """Stage B must not have created a cleaned dataset copy or split arrays.

    Stage C does produce split arrays, under ``<output>/exp1|exp2|exp3/``.  The
    guard therefore skips those directories and still asserts that no split
    array appears anywhere else — which is what would indicate Stage B (or any
    profiling code) materialising a copy of the corpus.
    """
    forbidden = ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test")
    for directory in (config.paths.processed_data_dir, config.paths.results_dir):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.parent.name in _STAGE_C_EXPERIMENT_DIRS:
                continue  # Stage C split output, not a Stage B artefact
            assert not path.name.startswith(forbidden), f"unexpected artifact: {path}"


# =========================================================================== #
# 12. the written artifacts of the real run are internally consistent
# =========================================================================== #
@pytest.fixture(scope="session")
def real_artifacts(config: Config) -> Dict[str, object]:
    directory = config.paths.results_dir / config.profiling.output_subdir
    path = directory / "global_dataset_summary.json"
    if not path.is_file():
        pytest.skip(f"Stage B artifacts not generated yet: {path}")
    return {
        "dir": directory,
        "summary": json.loads(path.read_text(encoding="utf-8")),
        "per_file": json.loads(
            (directory / "per_file_summary.json").read_text(encoding="utf-8")
        ),
    }


@pytest.mark.slow
def test_real_artifact_totals_are_internally_consistent(
    real_artifacts: Dict[str, object],
) -> None:
    summary = real_artifacts["summary"]  # type: ignore[index]
    per_file = real_artifacts["per_file"]["files"]  # type: ignore[index]
    totals = summary["totals"]

    assert sum(item["intake"]["n_raw_rows"] for item in per_file) == totals["raw_rows"]
    assert (
        sum(item["intake"]["n_embedded_header_rows"] for item in per_file)
        == totals["embedded_header_rows"]
    )
    assert (
        sum(item["intake"]["n_usable_rows"] for item in per_file)
        == totals["usable_rows"]
    )
    assert totals["raw_rows"] - totals["embedded_header_rows"] == totals["usable_rows"]
    assert sum(summary["labels"]["counts"].values()) == totals["rows_profiled"]
    assert sorted(summary["labels"]["vocabulary"]) == sorted(summary["labels"]["counts"])
    assert summary["sequence"] == [
        f"{item['capture_date']}/{item['source_file']}" for item in per_file
    ]


@pytest.mark.slow
def test_real_artifact_sequence_is_ordered_by_capture_date(
    real_artifacts: Dict[str, object],
) -> None:
    per_file = real_artifacts["per_file"]["files"]  # type: ignore[index]
    dates = [item["capture_date"] for item in per_file]
    assert dates == sorted(dates)
    assert all(item["capture_date_source"] == "filename" for item in per_file)


@pytest.mark.slow
def test_real_artifacts_hold_no_label_encoding(real_artifacts: Dict[str, object]) -> None:
    directory = real_artifacts["dir"]  # type: ignore[index]
    names = {path.name for path in directory.iterdir()}  # type: ignore[union-attr]
    assert "label_mapping.json" not in names
    assert "label_encoder.json" not in names
    assert not any(name.endswith((".npy", ".npz", ".pt")) for name in names)
