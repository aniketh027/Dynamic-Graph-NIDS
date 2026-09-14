"""Tests for the T=5 window continuity analysis.

Synthetic fixtures use hand-built CSVs in ``tmp_path`` with obviously fake IPs
and labels, so no synthetic number can be mistaken for a measurement. Tests that
read the real corpus are marked ``slow`` and skip when the dataset is absent.

The analysis is measure-only, so the tests check two things above all: that each
metric matches an independently hand-computed value, and that nothing under
``data/raw`` is touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from src.analysis import continuity_artifacts
from src.analysis.window_continuity import (
    READ_COLUMNS,
    WINDOW_SIZE,
    AnalysisError,
    ContinuityResult,
    _majority_is_fifth,
    _row_wise_unique_counts,
    analyze_file,
)
from src.preprocessing import schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CORPUS = PROJECT_ROOT / "data" / "raw" / "csv"
REAL_FILE = "02-20-2018.csv"

# Measured in Stage B and re-verified here, not assumed.
REAL_RAW_ROWS = 7_948_748
REAL_EMBEDDED_HEADERS = 0
REAL_USABLE_ROWS = 7_948_748


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
def _flow(
    *,
    src: str,
    dst: str,
    label: str = "SyntheticBenign",
    src_port: str = "1000",
    dst_port: str = "80",
    protocol: str = "6",
    timestamp: str = "20/02/2018 01:00:00",
) -> Dict[str, str]:
    return {
        "Src IP": src,
        "Dst IP": dst,
        "Src Port": src_port,
        "Dst Port": dst_port,
        "Protocol": protocol,
        schema.TIMESTAMP_COLUMN: timestamp,
        schema.LABEL_COLUMN: label,
    }


def _write_extended_csv(path: Path, flows: Sequence[Dict[str, str]]) -> Path:
    """Write a file with the real 84-column extended header.

    The analysis reads by column name, so the 76 unused feature columns are
    filled with zeros - this proves the reader picks the right 8 of 84.
    """
    names = list(schema.CSV_COLUMNS_EXTENDED)
    lines = [",".join(names)]
    for flow in flows:
        values = {column: "0" for column in names}
        values["Flow ID"] = f"{flow['Src IP']}-{flow['Dst IP']}"
        values.update(flow)
        lines.append(",".join(values[column] for column in names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}:{digest.hexdigest()}"


# =========================================================================== #
# 1. the vectorised primitives match a brute-force implementation
# =========================================================================== #
def test_row_wise_unique_counts_matches_python_sets() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(500, WINDOW_SIZE))
    measured = _row_wise_unique_counts(codes)
    expected = np.array([len(set(row.tolist())) for row in codes])
    assert np.array_equal(measured, expected)


def test_row_wise_unique_counts_spans_the_full_range() -> None:
    codes = np.array(
        [
            [7, 7, 7, 7, 7],   # 1 distinct
            [7, 7, 7, 7, 8],   # 2
            [1, 2, 3, 1, 2],   # 3
            [1, 2, 3, 4, 1],   # 4
            [1, 2, 3, 4, 5],   # 5
        ]
    )
    assert _row_wise_unique_counts(codes).tolist() == [1, 2, 3, 4, 5]


def test_majority_is_fifth_matches_a_brute_force_tiebreak() -> None:
    """Cross-check against an explicit smallest-code-wins implementation."""
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 3, size=(2000, WINDOW_SIZE))

    def brute(row: np.ndarray) -> bool:
        counts: Dict[int, int] = {}
        for value in row.tolist():
            counts[value] = counts.get(value, 0) + 1
        best = max(counts, key=lambda value: (counts[value], -value))
        return bool(row[-1] == best)

    expected = np.array([brute(row) for row in codes])
    assert np.array_equal(_majority_is_fifth(codes), expected)


def test_majority_is_fifth_breaks_ties_toward_the_smallest_code() -> None:
    # Five distinct values: every count is 1, so the smallest code wins and the
    # fifth element is the majority only when it is that smallest code.
    assert _majority_is_fifth(np.array([[1, 2, 3, 4, 5]])).tolist() == [False]
    assert _majority_is_fifth(np.array([[2, 3, 4, 5, 1]])).tolist() == [True]
    # A genuine majority beats the tiebreak regardless of position.
    assert _majority_is_fifth(np.array([[9, 9, 9, 1, 9]])).tolist() == [True]
    assert _majority_is_fifth(np.array([[9, 9, 9, 9, 1]])).tolist() == [False]


# =========================================================================== #
# 2. hand-computed windows: every metric verified by inspection
# =========================================================================== #
@pytest.fixture()
def handmade(tmp_path: Path) -> ContinuityResult:
    """Three windows whose every statistic is known by construction.

    w0: all five flows A->B, one label            -> 1 pair, 4 adjacent, pure
    w1: five distinct pairs, one label            -> 5 pairs, 0 adjacent, pure
    w2: A->B, A->B, C->D, D->E, A->B, two labels  -> 3 pairs, 1 adjacent, mixed
    """
    flows: List[Dict[str, str]] = []
    flows += [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]
    flows += [
        _flow(src=f"10.0.1.{i}", dst=f"10.0.2.{i}") for i in range(1, 6)
    ]
    flows += [
        _flow(src="10.0.0.1", dst="10.0.0.2"),
        _flow(src="10.0.0.1", dst="10.0.0.2"),
        _flow(src="10.0.3.1", dst="10.0.3.2"),
        _flow(src="10.0.3.2", dst="10.0.3.3"),
        _flow(src="10.0.0.1", dst="10.0.0.2", label="SyntheticAttack"),
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    return analyze_file(path, seed=42, chunk_size=4, with_random_baseline=False)


def test_hand_computed_window_statistics(handmade: ContinuityResult) -> None:
    frame = handmade.per_window
    assert frame is not None
    assert handmade.n_windows == 3
    assert handmade.n_rows_analyzed == 15
    assert handmade.n_trailing_rows_discarded == 0

    # w2 holds A->B three times (positions 0, 1, 4) plus C->D and D->E: 3 pairs.
    assert frame["n_unique_pairs"].tolist() == [1, 5, 3]
    assert frame["n_unique_src_ip"].tolist() == [1, 5, 3]
    assert frame["n_unique_dst_ip"].tolist() == [1, 5, 3]
    assert frame["n_adjacent_same_pair"].tolist() == [4, 0, 1]
    assert frame["all_same_pair"].tolist() == [True, False, False]
    assert frame["any_adjacent_same_pair"].tolist() == [True, False, True]
    assert frame["n_unique_labels"].tolist() == [1, 1, 2]
    assert frame["label_pure"].tolist() == [True, True, False]
    # every flow here carries the same ports and protocol, so the 5-tuple count
    # tracks the pair count exactly
    assert frame["n_unique_five_tuples"].tolist() == [1, 5, 3]


def test_hand_computed_aggregate_percentages(handmade: ContinuityResult) -> None:
    payload = handmade.to_dict()["consecutive"]
    endpoint = payload["endpoint_continuity"]
    label = payload["label_continuity"]

    assert endpoint["A_pct_windows_all_five_same_src_dst_pair"] == pytest.approx(100 / 3)
    assert endpoint["B_pct_windows_any_adjacent_same_src_dst_pair"] == pytest.approx(
        200 / 3
    )
    # pair counts are 1, 5, 3 -> mean 3.0, median 3.0
    assert endpoint["C_mean_unique_pairs_per_window"] == pytest.approx(3.0)
    assert endpoint["D_median_unique_pairs_per_window"] == 3.0
    assert endpoint["pct_windows_fully_heterogeneous_five_distinct_pairs"] == (
        pytest.approx(100 / 3)
    )
    assert endpoint["E_pct_windows_by_unique_pair_count"]["1"] == pytest.approx(100 / 3)
    assert endpoint["E_pct_windows_by_unique_pair_count"]["3"] == pytest.approx(100 / 3)
    assert endpoint["E_pct_windows_by_unique_pair_count"]["5"] == pytest.approx(100 / 3)
    assert endpoint["E_pct_windows_by_unique_pair_count"]["2"] == 0.0
    assert endpoint["E_pct_windows_by_unique_pair_count"]["4"] == 0.0
    # 5 of 12 adjacent slots share a pair (4 in w0, 0 in w1, 1 in w2)
    assert endpoint["counts"]["adjacent_flow_pairs_sharing_src_dst"] == 5
    assert endpoint["counts"]["adjacent_flow_pairs_total"] == 12
    assert label["F_pct_windows_label_pure"] == pytest.approx(200 / 3)
    assert label["n_windows_by_unique_label_count"] == {"1": 2, "2": 1}


def test_window_percentages_are_internally_consistent(handmade: ContinuityResult) -> None:
    payload = handmade.to_dict()["consecutive"]
    pair_pcts = payload["endpoint_continuity"]["E_pct_windows_by_unique_pair_count"]
    assert sum(pair_pcts.values()) == pytest.approx(100.0)
    label_pcts = payload["label_continuity"]["pct_windows_by_unique_label_count"]
    assert sum(label_pcts.values()) == pytest.approx(100.0)
    label_block = payload["label_continuity"]
    assert (
        label_block["F_pct_windows_label_pure"]
        + label_block["pct_windows_mixed_label"]
    ) == pytest.approx(100.0)
    assert (
        label_block["pct_windows_fifth_label_eq_majority"]
        + label_block["G_pct_windows_fifth_label_ne_majority"]
    ) == pytest.approx(100.0)
    counts = label_block["counts"]
    assert counts["fifth_eq_majority"] + counts["fifth_ne_majority"] == handmade.n_windows


# =========================================================================== #
# 3. ordering, windowing and row identity
# =========================================================================== #
def test_windows_are_non_overlapping_and_in_capture_order(tmp_path: Path) -> None:
    flows = [_flow(src=f"10.0.0.{i}", dst="10.1.1.1") for i in range(20)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, chunk_size=7, with_random_baseline=False)
    frame = result.per_window
    assert frame is not None
    assert result.n_windows == 4
    # window i covers original indices [5i, 5i+4]: contiguous, ordered, no overlap
    assert frame["first_original_index"].tolist() == [0, 5, 10, 15]
    assert frame["last_original_index"].tolist() == [4, 9, 14, 19]


def test_trailing_rows_are_discarded_and_counted(tmp_path: Path) -> None:
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(13)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, chunk_size=5, with_random_baseline=False)
    assert result.n_rows_analyzed == 13
    assert result.n_windows == 2
    assert result.n_trailing_rows_discarded == 3


def test_chunk_size_never_splits_a_window(tmp_path: Path) -> None:
    """Chunk boundaries must not change a single measurement."""
    rng = np.random.default_rng(3)
    flows = [
        _flow(
            src=f"10.0.0.{rng.integers(1, 4)}",
            dst=f"10.1.1.{rng.integers(1, 4)}",
            label="SyntheticAttack" if rng.random() < 0.3 else "SyntheticBenign",
        )
        for _ in range(200)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    digests = set()
    for chunk_size in (3, 4, 5, 7, 64, 1000):
        result = analyze_file(path, chunk_size=chunk_size, seed=42)
        digests.add(continuity_artifacts.measurement_digest(result))
    assert len(digests) == 1, "chunk size changed a measurement"


def test_embedded_header_rows_are_excluded_but_indices_stay_original(
    tmp_path: Path,
) -> None:
    """A header line mid-file shifts membership, not the recorded row identity."""
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(10)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    # Splice a literal header line in as data row 3.
    lines = path.read_text("utf-8").splitlines()
    lines.insert(4, ",".join(schema.CSV_COLUMNS_EXTENDED))
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    result = analyze_file(path, chunk_size=4, with_random_baseline=False)
    assert result.n_raw_rows == 11
    assert result.n_embedded_header_rows == 1
    assert result.n_usable_rows == 10
    assert result.n_rows_analyzed == 10
    assert result.n_windows == 2
    frame = result.per_window
    assert frame is not None
    # The excluded row sat at position 3, so the first window spans original
    # indices 0,1,2,4,5 - its last index is 5, not 4.
    assert frame["first_original_index"].tolist() == [0, 6]
    assert frame["last_original_index"].tolist() == [5, 10]


def test_timestamp_is_not_used_for_ordering(tmp_path: Path) -> None:
    """Descending timestamps must not reorder rows or change window membership."""
    flows = [
        _flow(
            src=f"10.0.0.{i}",
            dst="10.1.1.1",
            timestamp=f"20/02/2018 10:{59 - i:02d}:00",
        )
        for i in range(10)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, chunk_size=4, with_random_baseline=False)
    frame = result.per_window
    assert frame is not None
    assert frame["first_original_index"].tolist() == [0, 5]
    assert frame["last_original_index"].tolist() == [4, 9]
    # Span is |max - min| so it stays positive despite the descending column.
    assert (frame["span_seconds"] >= 0).all()
    assert frame["span_seconds"].tolist() == [240.0, 240.0]


# =========================================================================== #
# 4. timestamp diagnostics
# =========================================================================== #
def test_same_second_windows_are_detected(tmp_path: Path) -> None:
    same = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]
    spread = [
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp=f"20/02/2018 01:00:{i:02d}")
        for i in range(5)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", same + spread)
    result = analyze_file(path, chunk_size=5, with_random_baseline=False)
    frame = result.per_window
    assert frame is not None
    assert frame["span_seconds"].tolist() == [0.0, 4.0]
    assert frame["same_second"].tolist() == [True, False]
    time_stats = result.to_dict()["consecutive"]["timestamp_diagnostics"]
    assert time_stats["pct_windows_entirely_within_one_second"] == 50.0
    assert time_stats["max_span_seconds"] == 4.0


def test_unparseable_timestamp_excludes_only_that_window_span(tmp_path: Path) -> None:
    good = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]
    bad = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]
    bad[2][schema.TIMESTAMP_COLUMN] = "not-a-timestamp"
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", good + bad)
    result = analyze_file(path, chunk_size=5, with_random_baseline=False)

    assert result.n_unparsed_timestamps == 1
    # the row still counts everywhere else
    assert result.n_rows_analyzed == 10
    assert result.n_windows == 2
    frame = result.per_window
    assert frame is not None
    assert bool(np.isnan(frame["span_seconds"].iloc[1]))
    assert frame["same_second"].tolist() == [True, False]
    # span statistics come from the one valid window only
    time_stats = result.to_dict()["consecutive"]["timestamp_diagnostics"]
    assert time_stats["max_span_seconds"] == 0.0
    assert any("did not parse" in note for note in result.notes)


def test_nat_becomes_nan_not_the_int64_sentinel(tmp_path: Path) -> None:
    """Regression: NaT cast through datetime64 -> float64 yields ~9.22e18.

    That sentinel is finite, so it would pass the validity check and produce an
    absurd span (~292 billion years) instead of excluding the window.
    """
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]
    flows[0][schema.TIMESTAMP_COLUMN] = "garbage"
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, chunk_size=5, with_random_baseline=False)
    frame = result.per_window
    assert frame is not None
    assert bool(np.isnan(frame["span_seconds"].iloc[0]))
    time_stats = result.to_dict()["consecutive"]["timestamp_diagnostics"]
    assert time_stats["n_windows_with_a_valid_span"] == 0
    assert time_stats["max_span_seconds"] == 0.0
    assert time_stats["pct_windows_entirely_within_one_second"] == 0.0


def test_span_buckets_are_cumulative_and_relative_to_valid_windows(
    tmp_path: Path,
) -> None:
    """Four windows with spans 0 s, 5 s, 120 s and one invalid."""
    flows: List[Dict[str, str]] = []
    flows += [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]      # 0 s
    flows += [
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp=f"20/02/2018 01:00:{i:02d}")
        for i in (0, 1, 2, 3, 5)
    ]                                                                        # 5 s
    flows += [
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp=f"20/02/2018 01:0{i}:00")
        for i in (0, 0, 1, 1, 2)
    ]                                                                        # 120 s
    invalid = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(5)]
    invalid[3][schema.TIMESTAMP_COLUMN] = "not-a-time"
    flows += invalid
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, chunk_size=5, with_random_baseline=False)

    time_stats = result.to_dict()["consecutive"]["timestamp_diagnostics"]
    assert result.n_windows == 4
    assert time_stats["n_windows_with_a_valid_span"] == 3
    counts = time_stats["n_windows_with_span_at_most"]
    assert counts["0"] == 1        # only the same-second window
    assert counts["10"] == 2       # + the 5 s window
    assert counts["300"] == 3      # + the 120 s window
    assert counts["3600"] == 3     # the invalid window is never counted
    # percentages are over span-valid windows, not all windows
    assert time_stats["pct_windows_with_span_at_most"]["0"] == pytest.approx(100 / 3)
    assert time_stats["pct_windows_with_span_at_most"]["300"] == 100.0
    # but the same-second percentage is over all windows, as documented
    assert time_stats["pct_windows_entirely_within_one_second"] == 25.0


def test_am_pm_ambiguity_is_not_reconstructed(tmp_path: Path) -> None:
    """A 12-hour clock is parsed as written; no AM/PM is inferred."""
    flows = [
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp="20/02/2018 01:00:00"),
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp="20/02/2018 12:00:00"),
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp="20/02/2018 01:00:00"),
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp="20/02/2018 01:00:00"),
        _flow(src="10.0.0.1", dst="10.0.0.2", timestamp="20/02/2018 01:00:00"),
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, chunk_size=5, with_random_baseline=False)
    frame = result.per_window
    assert frame is not None
    # 01:00 -> 12:00 taken literally is 11h; a 13:00 reinterpretation would give 1h
    assert frame["span_seconds"].iloc[0] == 11 * 3600.0
    assert result.n_unparsed_timestamps == 0


# =========================================================================== #
# 5. the random baseline
# =========================================================================== #
def test_random_baseline_is_reproducible_for_a_fixed_seed(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    flows = [
        _flow(
            src=f"10.0.0.{rng.integers(1, 6)}",
            dst=f"10.1.1.{rng.integers(1, 6)}",
            label="SyntheticAttack" if rng.random() < 0.4 else "SyntheticBenign",
        )
        for _ in range(300)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    first = analyze_file(path, seed=42)
    second = analyze_file(path, seed=42)
    assert continuity_artifacts.measurement_digest(
        first
    ) == continuity_artifacts.measurement_digest(second)
    assert first.random_baseline is not None
    assert second.random_baseline is not None
    assert first.random_baseline.to_dict() == second.random_baseline.to_dict()


def test_a_different_seed_changes_only_the_baseline(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    flows = [
        _flow(
            src=f"10.0.0.{rng.integers(1, 8)}",
            dst=f"10.1.1.{rng.integers(1, 8)}",
        )
        for _ in range(400)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    left = analyze_file(path, seed=1)
    right = analyze_file(path, seed=2)
    # the consecutive arm is seed-independent by construction
    assert left.consecutive.to_dict() == right.consecutive.to_dict()
    assert left.random_baseline is not None and right.random_baseline is not None
    assert left.random_baseline.to_dict() != right.random_baseline.to_dict()


def test_baseline_preserves_row_content_and_window_count(tmp_path: Path) -> None:
    """The baseline regroups the same rows: totals must be identical."""
    rng = np.random.default_rng(7)
    flows = [
        _flow(
            src=f"10.0.0.{rng.integers(1, 5)}",
            dst=f"10.1.1.{rng.integers(1, 5)}",
            label="SyntheticAttack" if rng.random() < 0.5 else "SyntheticBenign",
        )
        for _ in range(500)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, seed=42)
    assert result.random_baseline is not None
    assert result.random_baseline.n_windows == result.consecutive.n_windows


def test_baseline_can_be_switched_off(tmp_path: Path) -> None:
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(10)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, with_random_baseline=False)
    assert result.random_baseline is None
    assert result.comparison() == {}
    assert "random_baseline" not in result.to_dict()


def test_consecutive_grouping_beats_random_when_flows_are_grouped(
    tmp_path: Path,
) -> None:
    """A file with genuine locality must show it against the baseline.

    Ten flows per conversation, laid out consecutively: the consecutive arm
    should be nearly all-same-pair while the shuffled arm should not.
    """
    flows: List[Dict[str, str]] = []
    for conversation in range(60):
        flows += [
            _flow(src=f"10.0.{conversation}.1", dst=f"10.9.{conversation}.2")
            for _ in range(10)
        ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, seed=42)
    assert result.random_baseline is not None
    consecutive = result.consecutive.to_dict()["endpoint_continuity"]
    random_arm = result.random_baseline.to_dict()["endpoint_continuity"]
    assert consecutive["A_pct_windows_all_five_same_src_dst_pair"] == 100.0
    assert random_arm["A_pct_windows_all_five_same_src_dst_pair"] < 5.0
    assert consecutive["C_mean_unique_pairs_per_window"] == 1.0
    assert random_arm["C_mean_unique_pairs_per_window"] > 4.0


def test_high_adjacency_can_arise_from_low_pair_cardinality_alone(
    tmp_path: Path,
) -> None:
    """Metric B is chance-inflated when few distinct pairs exist.

    Five pairs cycled so that *no* two adjacent flows ever share a pair: the
    consecutive arm must score B = 0% while a random regrouping still scores
    well above zero purely by collision. This is why B must be read against the
    baseline rather than on its own.
    """
    pairs = [(f"10.0.0.{i}", f"10.1.1.{i}") for i in range(5)]
    flows = [
        _flow(src=pairs[i % 5][0], dst=pairs[i % 5][1]) for i in range(1000)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, seed=42)
    assert result.random_baseline is not None
    consecutive = result.consecutive.to_dict()["endpoint_continuity"]
    random_arm = result.random_baseline.to_dict()["endpoint_continuity"]
    assert consecutive["B_pct_windows_any_adjacent_same_src_dst_pair"] == 0.0
    assert random_arm["B_pct_windows_any_adjacent_same_src_dst_pair"] > 20.0


# =========================================================================== #
# 6. guard rails: schema requirements and read-only behaviour
# =========================================================================== #
def test_a_file_without_identifier_columns_is_refused(tmp_path: Path) -> None:
    """The 80-column layout cannot support this analysis and must say so."""
    names = list(schema.CSV_COLUMNS)
    values = {column: "0" for column in names}
    values[schema.TIMESTAMP_COLUMN] = "01/03/2018 01:00:00"
    values[schema.LABEL_COLUMN] = "SyntheticBenign"
    path = tmp_path / "03-01-2018.csv"
    path.write_bytes(
        (
            ",".join(names)
            + "\n"
            + "\n".join(",".join(values[c] for c in names) for _ in range(10))
            + "\n"
        ).encode("utf-8")
    )
    with pytest.raises(AnalysisError, match="identifier column"):
        analyze_file(path)


def test_missing_file_raises_analysis_error(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="file not found"):
        analyze_file(tmp_path / "does-not-exist.csv")


def test_only_the_needed_columns_are_read(tmp_path: Path, monkeypatch) -> None:
    """Guards the 8-of-84 claim in the report."""
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(10)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    seen: List[Optional[Sequence[str]]] = []
    original = pd.read_csv

    def spy(*args, **kwargs):
        seen.append(kwargs.get("usecols"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    analyze_file(path, with_random_baseline=False)
    # the data pass reads exactly READ_COLUMNS; the scout pass reads only Label
    assert set(READ_COLUMNS) in [set(item) for item in seen if item]
    assert all(set(item) <= set(READ_COLUMNS) for item in seen if item)


def test_analysis_does_not_modify_the_source_file(tmp_path: Path) -> None:
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(50)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    before = _fingerprint(path)
    analyze_file(path, seed=42)
    assert _fingerprint(path) == before


def test_analysis_writes_nothing_into_the_source_directory(tmp_path: Path) -> None:
    corpus = tmp_path / "csv"
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(50)]
    path = _write_extended_csv(corpus / "02-20-2018.csv", flows)
    before = sorted(item.name for item in corpus.iterdir())
    result = analyze_file(path, seed=42)
    continuity_artifacts.write_artifacts(result, tmp_path / "out")
    assert sorted(item.name for item in corpus.iterdir()) == before


def test_row_limit_is_recorded_as_a_truncated_run(tmp_path: Path) -> None:
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(100)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, row_limit=20, with_random_baseline=False)
    assert result.n_rows_analyzed == 20
    assert result.n_windows == 4
    assert any("truncated debug run" in note for note in result.notes)


# =========================================================================== #
# 7. artifacts
# =========================================================================== #
@pytest.fixture()
def written_artifacts(tmp_path: Path) -> Dict[str, Path]:
    rng = np.random.default_rng(9)
    flows = [
        _flow(
            src=f"10.0.0.{rng.integers(1, 4)}",
            dst=f"10.1.1.{rng.integers(1, 4)}",
            label="SyntheticAttack" if rng.random() < 0.3 else "SyntheticBenign",
        )
        for _ in range(250)
    ]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, seed=42)
    return continuity_artifacts.write_artifacts(result, tmp_path / "out")


def test_all_three_artifacts_are_written(written_artifacts: Dict[str, Path]) -> None:
    assert set(written_artifacts) == set(continuity_artifacts.ARTIFACT_NAMES)
    for path in written_artifacts.values():
        assert path.is_file() and path.stat().st_size > 0


def test_summary_json_carries_provenance_and_definitions(
    written_artifacts: Dict[str, Path],
) -> None:
    payload = json.loads(
        written_artifacts[continuity_artifacts.SUMMARY_NAME].read_text("utf-8")
    )
    dataset = payload["dataset"]
    for key in (
        "source_file",
        "n_raw_data_rows",
        "n_usable_rows",
        "n_rows_analyzed",
        "n_windows",
        "n_trailing_rows_discarded",
    ):
        assert key in dataset, key
    assert payload["parameters"]["seed"] == 42
    assert payload["parameters"]["window_size"] == WINDOW_SIZE
    assert "Original_Index" in payload["parameters"]["ordering_rule"]
    # every metric letter is defined
    for letter in "ABCDEFG":
        assert letter in payload["definitions"]
    assert payload["measurement_digest_sha256"]


def test_per_window_csv_has_one_row_per_window(
    written_artifacts: Dict[str, Path],
) -> None:
    payload = json.loads(
        written_artifacts[continuity_artifacts.SUMMARY_NAME].read_text("utf-8")
    )
    frame = pd.read_csv(written_artifacts[continuity_artifacts.WINDOW_CSV_NAME])
    assert len(frame) == payload["dataset"]["n_windows"]
    assert frame["window_id"].tolist() == list(range(len(frame)))
    assert frame["n_unique_pairs"].between(1, WINDOW_SIZE).all()
    assert frame["n_adjacent_same_pair"].between(0, WINDOW_SIZE - 1).all()
    # the CSV and the JSON must agree
    endpoint = payload["consecutive"]["endpoint_continuity"]
    assert int(frame["all_same_pair"].sum()) == endpoint["counts"]["all_same_pair"]
    assert int(frame["label_pure"].sum()) == (
        payload["consecutive"]["label_continuity"]["counts"]["label_pure"]
    )


def test_report_states_the_analysis_only_scope(
    written_artifacts: Dict[str, Path],
) -> None:
    report = written_artifacts[continuity_artifacts.REPORT_NAME].read_text("utf-8")
    assert "Analysis only" in report
    assert "does not choose an architecture" in report
    assert "diagnostic only" in report
    # the headline metrics all reach the report
    for letter in ("**A**", "**B**", "**C**", "**D**", "**F**", "**G**"):
        assert letter in report


def test_report_answers_all_eight_interpretation_questions(
    written_artifacts: Dict[str, Path],
) -> None:
    report = written_artifacts[continuity_artifacts.REPORT_NAME].read_text("utf-8")
    assert "Evidence for the interpretation questions" in report
    for number in range(1, 9):
        assert f"**{number}." in report
    # the evidence section must not smuggle in an architecture decision
    assert "it does not select an architecture" in report
    assert "No architecture conclusion is drawn here" in report
    lowered = report.lower()
    assert "therefore use ip-pair windows" not in lowered
    # numbered sections stay in order and do not collide with the evidence one
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.",
                    "## 7.", "## 8."):
        assert report.count(heading) == 1


def test_evidence_section_renumbers_when_the_baseline_is_off(tmp_path: Path) -> None:
    flows = [_flow(src=f"10.0.0.{i % 7}", dst="10.0.0.9") for i in range(50)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, with_random_baseline=False)
    report = continuity_artifacts.build_report(result, digest="0" * 64)
    assert "## 6. Evidence for the interpretation questions" in report
    assert "## 7. Exact definitions" in report
    assert "Random-grouping baseline" not in report
    for number in range(1, 9):
        assert f"**{number}." in report


def test_digest_ignores_timings_but_tracks_measurements(tmp_path: Path) -> None:
    flows = [_flow(src="10.0.0.1", dst="10.0.0.2") for _ in range(50)]
    path = _write_extended_csv(tmp_path / "02-20-2018.csv", flows)
    result = analyze_file(path, seed=42)
    baseline = continuity_artifacts.measurement_digest(result)

    result.elapsed_seconds += 99.0
    result.generated_at_utc = "1999-01-01T00:00:00+00:00"
    assert continuity_artifacts.measurement_digest(result) == baseline

    result.consecutive.n_all_same_pair += 1
    assert continuity_artifacts.measurement_digest(result) != baseline


def test_artifacts_are_not_dataset_copies(written_artifacts: Dict[str, Path]) -> None:
    """The per-window CSV holds statistics, never flow features or identifiers."""
    frame = pd.read_csv(written_artifacts[continuity_artifacts.WINDOW_CSV_NAME])
    forbidden = {"Src IP", "Dst IP", "Src Port", "Dst Port", "Label", "Timestamp"}
    assert not forbidden & set(frame.columns)
    for column in schema.FLOW_FEATURE_COLUMNS:
        assert column not in frame.columns


# =========================================================================== #
# 8. the real corpus (skips when the dataset is absent)
# =========================================================================== #
@pytest.mark.slow
def test_real_file_row_accounting_matches_stage_b() -> None:
    path = REAL_CORPUS / REAL_FILE
    if not path.is_file():
        pytest.skip(f"real corpus file not present: {path}")
    result = analyze_file(path, row_limit=100_000, with_random_baseline=False)
    # row accounting comes from the full scout pass, so it is the whole file
    assert result.n_raw_rows == REAL_RAW_ROWS
    assert result.n_embedded_header_rows == REAL_EMBEDDED_HEADERS
    assert result.n_usable_rows == REAL_USABLE_ROWS
    assert result.schema_variant == schema.VARIANT_EXTENDED
    assert result.n_columns_in_file == 84


@pytest.mark.slow
def test_real_file_is_not_modified_by_the_analysis() -> None:
    path = REAL_CORPUS / REAL_FILE
    if not path.is_file():
        pytest.skip(f"real corpus file not present: {path}")
    before = _fingerprint(path)
    analyze_file(path, row_limit=100_000, seed=42)
    assert _fingerprint(path) == before


@pytest.mark.slow
def test_published_artifacts_are_internally_consistent() -> None:
    """Cross-check the committed artifacts against each other."""
    out = PROJECT_ROOT / "results" / "temporal_analysis"
    summary = out / continuity_artifacts.SUMMARY_NAME
    windows = out / continuity_artifacts.WINDOW_CSV_NAME
    if not summary.is_file() or not windows.is_file():
        pytest.skip("run scripts/analyze_window_continuity.py first")

    payload = json.loads(summary.read_text("utf-8"))
    dataset = payload["dataset"]
    assert (
        dataset["n_rows_analyzed"]
        == dataset["n_windows"] * WINDOW_SIZE + dataset["n_trailing_rows_discarded"]
    )
    assert (
        dataset["n_usable_rows"]
        == dataset["n_raw_data_rows"] - dataset["n_embedded_header_rows_excluded"]
    )

    frame = pd.read_csv(windows)
    assert len(frame) == dataset["n_windows"]
    endpoint = payload["consecutive"]["endpoint_continuity"]
    assert int(frame["all_same_pair"].sum()) == endpoint["counts"]["all_same_pair"]
    assert int(frame["any_adjacent_same_pair"].sum()) == (
        endpoint["counts"]["any_adjacent_same_pair"]
    )
    assert float(frame["n_unique_pairs"].mean()) == pytest.approx(
        endpoint["C_mean_unique_pairs_per_window"], rel=1e-9
    )
    assert float(frame["n_unique_pairs"].median()) == (
        endpoint["D_median_unique_pairs_per_window"]
    )
    label = payload["consecutive"]["label_continuity"]
    assert int(frame["label_pure"].sum()) == label["counts"]["label_pure"]
    # The CSV stores flags as 0/1 int8, so compare against 0 rather than using
    # `~`, which on an integer column is a bitwise negation (~1 == -2).
    assert int((frame["fifth_is_majority"] == 0).sum()) == (
        label["counts"]["fifth_ne_majority"]
    )
    # Flag columns really are 0/1 and read back as a boolean without surprises.
    for column in ("all_same_pair", "any_adjacent_same_pair", "label_pure",
                   "fifth_is_majority", "same_second"):
        assert set(frame[column].unique()) <= {0, 1}
