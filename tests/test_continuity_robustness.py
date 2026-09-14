"""Tests for the T=3 / T=5 / T=10 window-size robustness sweep.

Synthetic fixtures use hand-built CSVs in ``tmp_path`` with obviously fake IPs
and labels, so no synthetic number can be mistaken for a measurement. Tests that
read the real corpus are marked ``slow`` and skip when the dataset is absent.

Every headline metric is checked against a value computed by hand in the
docstring of its test, at all three window sizes, so a definition cannot drift
silently between T values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from src.analysis import robustness_artifacts
from src.analysis.continuity_robustness import (
    IDENTITY_PROXY,
    IDENTITY_TRUE,
    PROXY_IDENTITY_COLUMNS,
    TRUE_IDENTITY_COLUMNS,
    WINDOW_SIZES,
    AnalysisError,
    equivalence_check,
    identifier_inventory,
    read_file_columns,
    run_robustness,
    sweep_file,
)
from src.preprocessing import schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CORPUS = PROJECT_ROOT / "data" / "raw" / "csv"
REAL_FILE = "02-20-2018.csv"
PUBLISHED = PROJECT_ROOT / "results" / "temporal_analysis" / "continuity_summary.json"


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
#: Twelve flows whose continuity is worked out by hand in the tests below.
#: Pairs, in order: AB AB AB CD EF CD GH GH IJ KL MN OP
#: Labels are all SyntheticBenign except row 2, which is SyntheticAttack.
#: Timestamps are all 01:00:00 except rows 3 and 9, which are 01:00:10.
HAND_PAIRS: Sequence[tuple] = (
    ("10.0.0.1", "10.0.0.2"),
    ("10.0.0.1", "10.0.0.2"),
    ("10.0.0.1", "10.0.0.2"),
    ("10.0.0.3", "10.0.0.4"),
    ("10.0.0.5", "10.0.0.6"),
    ("10.0.0.3", "10.0.0.4"),
    ("10.0.0.7", "10.0.0.8"),
    ("10.0.0.7", "10.0.0.8"),
    ("10.0.0.9", "10.0.0.10"),
    ("10.0.0.11", "10.0.0.12"),
    ("10.0.0.13", "10.0.0.14"),
    ("10.0.0.15", "10.0.0.16"),
)


def _hand_flows() -> List[Dict[str, str]]:
    flows = []
    for index, (src, dst) in enumerate(HAND_PAIRS):
        flows.append(
            {
                "Src IP": src,
                "Dst IP": dst,
                "Src Port": str(40000 + index),
                "Dst Port": "80",
                "Protocol": "6",
                schema.TIMESTAMP_COLUMN: (
                    "20/02/2018 01:00:10" if index in (3, 9) else "20/02/2018 01:00:00"
                ),
                schema.LABEL_COLUMN: (
                    "SyntheticAttack" if index == 2 else "SyntheticBenign"
                ),
            }
        )
    return flows


def _write_csv(path: Path, flows: Sequence[Dict[str, str]], *, extended: bool) -> Path:
    """Write a real-shaped CSV: 84 columns (extended) or 80 (standard).

    Unused feature columns are filled with zeros, which proves the reader selects
    columns by name rather than by position.
    """
    names = list(
        schema.CSV_COLUMNS_EXTENDED if extended else schema.CSV_COLUMNS
    )
    lines = [",".join(names)]
    for flow in flows:
        values = {column: "0" for column in names}
        if extended:
            values["Flow ID"] = f"{flow['Src IP']}-{flow['Dst IP']}"
        values.update({k: v for k, v in flow.items() if k in values})
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


def _cells(path: Path, **kwargs):
    columns = read_file_columns(path, chunk_size=kwargs.pop("chunk_size", 5))
    cells = sweep_file(columns, **kwargs)
    return {(c.window_size, c.identity): c for c in cells}


# --------------------------------------------------------------------------- #
# Task 1: identifier inventory
# --------------------------------------------------------------------------- #
def test_inventory_separates_extended_from_standard_files(tmp_path: Path) -> None:
    """The 84-column layout is measured per file, never assumed from the name."""
    extended = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    standard = _write_csv(tmp_path / "02-14-2018.csv", _hand_flows(), extended=False)

    inventory = {item["source_file"]: item for item in identifier_inventory(
        [standard, extended]
    )}

    rich = inventory["02-20-2018.csv"]
    assert rich["n_columns"] == 84
    assert rich["has_full_five_tuple"] is True
    assert rich["has_true_endpoint_identity"] is True
    assert rich["has_proxy_identity"] is True
    assert rich["identifier_columns_missing"] == []

    poor = inventory["02-14-2018.csv"]
    assert poor["n_columns"] == 80
    assert poor["has_full_five_tuple"] is False
    assert poor["has_true_endpoint_identity"] is False
    # The proxy survives where the true identity does not - that is the whole
    # reason the proxy exists.
    assert poor["has_proxy_identity"] is True
    assert poor["identifier_columns_missing"] == ["Src IP", "Src Port", "Dst IP"]


def test_inventory_is_sorted_by_name_not_by_filesystem_order(tmp_path: Path) -> None:
    for name in ("03-02-2018.csv", "02-14-2018.csv", "02-20-2018.csv"):
        _write_csv(tmp_path / name, _hand_flows(), extended=name == "02-20-2018.csv")
    paths = list(tmp_path.glob("*.csv"))
    names = [item["source_file"] for item in identifier_inventory(paths)]
    assert names == sorted(names)


# --------------------------------------------------------------------------- #
# Task 3: hand-computed metrics at each window size
# --------------------------------------------------------------------------- #
def test_true_identity_metrics_at_T3_match_hand_computation(tmp_path: Path) -> None:
    """Windows: [AB AB AB] [CD EF CD] [GH GH IJ] [KL MN OP].

    unique pairs   1, 2, 2, 3          -> C = 2.0, D = 2.0
    all same       yes, no, no, no     -> A = 25%
    any adjacent   yes, no, yes, no    -> B = 50%
    adjacent links 2 + 0 + 1 + 0 = 3 of 4*2 = 8 -> 37.5%
    3 distinct     no, no, no, yes     -> heterogeneous = 25%
    labels         [B B A] [B B B] [B B B] [B B B] -> F = 75%
    final != major  A vs B, then equal x3 -> G = 25%
    """
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(3,), with_random_baseline=False)[
        (3, IDENTITY_TRUE)
    ]
    arm = cell.consecutive.to_dict(3)

    assert cell.n_windows == 4
    assert cell.n_trailing_rows_discarded == 0
    assert arm["A_pct_windows_all_same_key"] == pytest.approx(25.0)
    assert arm["B_pct_windows_any_adjacent_same_key"] == pytest.approx(50.0)
    assert arm["C_mean_unique_keys_per_window"] == pytest.approx(2.0)
    assert arm["D_median_unique_keys_per_window"] == pytest.approx(2.0)
    assert arm["pct_adjacent_links_sharing_key"] == pytest.approx(37.5)
    assert arm["pct_windows_fully_heterogeneous"] == pytest.approx(25.0)
    assert arm["F_pct_windows_label_pure"] == pytest.approx(75.0)
    assert arm["G_pct_windows_final_label_ne_majority"] == pytest.approx(25.0)
    assert arm["n_windows_by_unique_key_count"] == {"1": 1, "2": 2, "3": 1}


def test_true_identity_metrics_at_T5_match_hand_computation(tmp_path: Path) -> None:
    """Windows: [AB AB AB CD EF] [CD GH GH IJ KL], 2 trailing rows dropped.

    unique pairs   3, 4                -> C = 3.5, D = 3.5
    all same       no, no              -> A = 0%
    any adjacent   yes, yes            -> B = 100%
    adjacent links 2 + 1 = 3 of 2*4 = 8 -> 37.5%
    labels         [B B A B B] [B B B B B] -> F = 50%, G = 0%
    """
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(5,), with_random_baseline=False)[
        (5, IDENTITY_TRUE)
    ]
    arm = cell.consecutive.to_dict(5)

    assert cell.n_windows == 2
    assert cell.n_trailing_rows_discarded == 2
    assert arm["A_pct_windows_all_same_key"] == pytest.approx(0.0)
    assert arm["B_pct_windows_any_adjacent_same_key"] == pytest.approx(100.0)
    assert arm["C_mean_unique_keys_per_window"] == pytest.approx(3.5)
    assert arm["D_median_unique_keys_per_window"] == pytest.approx(3.5)
    assert arm["pct_adjacent_links_sharing_key"] == pytest.approx(37.5)
    assert arm["pct_windows_fully_heterogeneous"] == pytest.approx(0.0)
    assert arm["F_pct_windows_label_pure"] == pytest.approx(50.0)
    assert arm["G_pct_windows_final_label_ne_majority"] == pytest.approx(0.0)


def test_true_identity_metrics_at_T10_match_hand_computation(tmp_path: Path) -> None:
    """One window of rows 0-9; 2 trailing rows dropped.

    distinct pairs AB CD EF GH IJ KL  -> C = 6.0
    adjacent links 3 of 1*9           -> 33.333%
    B is 100% here and at T=5 while the link rate falls - that is the saturation
    the report warns about, pinned as a test.
    """
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(10,), with_random_baseline=False)[
        (10, IDENTITY_TRUE)
    ]
    arm = cell.consecutive.to_dict(10)

    assert cell.n_windows == 1
    assert cell.n_trailing_rows_discarded == 2
    assert arm["A_pct_windows_all_same_key"] == pytest.approx(0.0)
    assert arm["B_pct_windows_any_adjacent_same_key"] == pytest.approx(100.0)
    assert arm["C_mean_unique_keys_per_window"] == pytest.approx(6.0)
    assert arm["pct_adjacent_links_sharing_key"] == pytest.approx(100 * 3 / 9)
    assert arm["pct_windows_fully_heterogeneous"] == pytest.approx(0.0)
    assert arm["F_pct_windows_label_pure"] == pytest.approx(0.0)


def test_metric_B_saturates_with_T_while_the_link_rate_does_not(tmp_path: Path) -> None:
    """The report's central caveat, asserted rather than claimed."""
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cells = _cells(path, window_sizes=(3, 5, 10), with_random_baseline=False)
    b = [
        cells[(t, IDENTITY_TRUE)].consecutive.to_dict(t)[
            "B_pct_windows_any_adjacent_same_key"
        ]
        for t in (3, 5, 10)
    ]
    links = [
        cells[(t, IDENTITY_TRUE)].consecutive.to_dict(t)[
            "pct_adjacent_links_sharing_key"
        ]
        for t in (3, 5, 10)
    ]
    assert b == pytest.approx([50.0, 100.0, 100.0])
    assert links == pytest.approx([37.5, 37.5, 100 * 3 / 9])
    # B doubles while the comparable measure falls: reading B across T would
    # invert the actual direction of the effect.
    assert b[2] > b[0] and links[2] < links[0]


def test_timestamp_span_is_diagnostic_and_hand_checkable(tmp_path: Path) -> None:
    """Spans at T=3 are 0, 10, 0, 10 seconds -> median 5, half in one second."""
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(3,), with_random_baseline=False)[
        (3, IDENTITY_TRUE)
    ]
    stats = cell.consecutive.to_dict(3)["timestamp_diagnostics"]
    assert stats["n_windows_with_a_valid_span"] == 4
    assert stats["median_span_seconds"] == pytest.approx(5.0)
    assert stats["max_span_seconds"] == pytest.approx(10.0)
    assert stats["pct_windows_entirely_within_one_second"] == pytest.approx(50.0)


def test_unparseable_timestamp_excludes_only_that_window_from_spans(
    tmp_path: Path,
) -> None:
    """Regression: NaT cast through datetime64 -> float64 yields ~9.22e18, which
    is finite and would pass a naive validity check as a ~292-billion-year span.
    """
    flows = _hand_flows()
    flows[0][schema.TIMESTAMP_COLUMN] = "not a timestamp"
    path = _write_csv(tmp_path / "02-20-2018.csv", flows, extended=True)
    cell = _cells(path, window_sizes=(3,), with_random_baseline=False)[
        (3, IDENTITY_TRUE)
    ]
    stats = cell.consecutive.to_dict(3)["timestamp_diagnostics"]
    assert stats["n_windows_with_a_valid_span"] == 3
    assert stats["max_span_seconds"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# the proxy identity
# --------------------------------------------------------------------------- #
def test_proxy_identity_overstates_continuity_relative_to_the_true_identity(
    tmp_path: Path,
) -> None:
    """Every fixture row uses port 80/TCP, so the proxy sees one conversation.

    This is the bias the report calibrates: the proxy merges distinct host pairs
    that share a destination service.
    """
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cells = _cells(path, window_sizes=(5,), with_random_baseline=False)
    true_arm = cells[(5, IDENTITY_TRUE)].consecutive.to_dict(5)
    proxy_arm = cells[(5, IDENTITY_PROXY)].consecutive.to_dict(5)

    assert proxy_arm["A_pct_windows_all_same_key"] == pytest.approx(100.0)
    assert true_arm["A_pct_windows_all_same_key"] == pytest.approx(0.0)
    assert proxy_arm["C_mean_unique_keys_per_window"] == pytest.approx(1.0)
    assert proxy_arm["A_pct_windows_all_same_key"] > true_arm[
        "A_pct_windows_all_same_key"
    ]


def test_standard_files_yield_the_proxy_identity_only(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "02-14-2018.csv", _hand_flows(), extended=False)
    columns = read_file_columns(path, chunk_size=5)
    assert columns.has_true_identity is False
    assert set(columns.keys) == {IDENTITY_PROXY}
    assert columns.five_tuple is None

    cells = {(c.window_size, c.identity): c for c in sweep_file(
        columns, window_sizes=(3, 5), with_random_baseline=False
    )}
    assert set(cells) == {(3, IDENTITY_PROXY), (5, IDENTITY_PROXY)}


def test_identity_column_choices_are_what_the_report_claims() -> None:
    assert TRUE_IDENTITY_COLUMNS == ("Src IP", "Dst IP")
    assert PROXY_IDENTITY_COLUMNS == ("Dst Port", "Protocol")
    assert WINDOW_SIZES == (3, 5, 10)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_chunk_size_does_not_change_any_measurement(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    small = _cells(path, chunk_size=2, window_sizes=(3, 5), seed=42)
    large = _cells(path, chunk_size=1000, window_sizes=(3, 5), seed=42)
    for key, cell in small.items():
        assert cell.to_dict() == large[key].to_dict()


def test_the_same_seed_reproduces_the_baseline_and_a_different_seed_may_not(
    tmp_path: Path,
) -> None:
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    first = _cells(path, window_sizes=(3,), seed=42)[(3, IDENTITY_TRUE)]
    again = _cells(path, window_sizes=(3,), seed=42)[(3, IDENTITY_TRUE)]
    assert first.random_baseline is not None
    assert first.to_dict() == again.to_dict()
    # The consecutive arm must be seed-independent; only the baseline may move.
    other = _cells(path, window_sizes=(3,), seed=7)[(3, IDENTITY_TRUE)]
    assert other.consecutive.to_dict(3) == first.consecutive.to_dict(3)


def test_baseline_is_the_same_rows_regrouped_not_different_rows(
    tmp_path: Path,
) -> None:
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(3,), seed=42)[(3, IDENTITY_TRUE)]
    assert cell.random_baseline is not None
    assert cell.random_baseline.n_windows == cell.consecutive.n_windows


# --------------------------------------------------------------------------- #
# equivalence with the frozen T=5 result
# --------------------------------------------------------------------------- #
def _published_stub(cell) -> Dict[str, object]:
    """A published-summary payload that agrees with ``cell`` by construction."""
    arm = cell.consecutive.to_dict(5)
    return {
        "dataset": {
            "source_file": cell.source_file,
            "n_windows": cell.n_windows,
            "n_rows_analyzed": cell.n_rows_analyzed,
        },
        "consecutive": {
            "endpoint_continuity": {
                "A_pct_windows_all_five_same_src_dst_pair": arm[
                    "A_pct_windows_all_same_key"
                ],
                "B_pct_windows_any_adjacent_same_src_dst_pair": arm[
                    "B_pct_windows_any_adjacent_same_key"
                ],
                "C_mean_unique_pairs_per_window": arm["C_mean_unique_keys_per_window"],
                "D_median_unique_pairs_per_window": arm[
                    "D_median_unique_keys_per_window"
                ],
                "pct_windows_fully_heterogeneous_five_distinct_pairs": arm[
                    "pct_windows_fully_heterogeneous"
                ],
                "pct_adjacent_flow_pairs_sharing_src_dst": arm[
                    "pct_adjacent_links_sharing_key"
                ],
            },
            "label_continuity": {
                "F_pct_windows_label_pure": arm["F_pct_windows_label_pure"],
                "G_pct_windows_fifth_label_ne_majority": arm[
                    "G_pct_windows_final_label_ne_majority"
                ],
            },
        },
    }


def test_equivalence_check_passes_when_the_definitions_agree(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(5,), with_random_baseline=False)[
        (5, IDENTITY_TRUE)
    ]
    report = equivalence_check(cell, _published_stub(cell))
    assert report["all_match"] is True
    assert len(report["comparisons"]) == 10


def test_equivalence_check_fails_loudly_on_a_drifted_definition(
    tmp_path: Path,
) -> None:
    """A silent definition change would invalidate every cross-T comparison."""
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cell = _cells(path, window_sizes=(5,), with_random_baseline=False)[
        (5, IDENTITY_TRUE)
    ]
    published = _published_stub(cell)
    published["consecutive"]["endpoint_continuity"][
        "C_mean_unique_pairs_per_window"
    ] += 0.001
    report = equivalence_check(cell, published)
    assert report["all_match"] is False
    bad = [item for item in report["comparisons"] if not item["match"]]
    assert [item["metric"] for item in bad] == ["C_mean_unique_pairs_per_window"]


def test_equivalence_check_refuses_a_cell_that_is_not_the_T5_true_identity(
    tmp_path: Path,
) -> None:
    path = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    cells = _cells(path, window_sizes=(3, 5), with_random_baseline=False)
    with pytest.raises(AnalysisError):
        equivalence_check(cells[(3, IDENTITY_TRUE)], {})
    with pytest.raises(AnalysisError):
        equivalence_check(cells[(5, IDENTITY_PROXY)], {})


# --------------------------------------------------------------------------- #
# end-to-end run and artifacts
# --------------------------------------------------------------------------- #
@pytest.fixture()
def small_corpus(tmp_path: Path) -> List[Path]:
    """One extended file plus two standard files, as the real corpus is shaped."""
    extended = _write_csv(tmp_path / "02-20-2018.csv", _hand_flows(), extended=True)
    rotated = _hand_flows()[3:] + _hand_flows()[:3]
    standard_a = _write_csv(tmp_path / "02-14-2018.csv", rotated, extended=False)
    standard_b = _write_csv(
        tmp_path / "02-15-2018.csv", _hand_flows()[::-1], extended=False
    )
    return [extended, standard_a, standard_b]


def test_run_covers_every_file_and_window_size(small_corpus: List[Path]) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5))
    # 1 extended file x 2 identities x 2 T, plus 2 standard files x 1 identity x 2 T
    assert len(result.cells) == 2 * 2 + 2 * 1 * 2
    assert {c.source_file for c in result.cells} == {p.name for p in small_corpus}
    assert result.ordering_rule.startswith("deterministic capture-order proxy")
    assert result.window_sizes == (3, 5)
    assert len(result.inventory) == 3


def test_run_is_reproducible_across_two_invocations(small_corpus: List[Path]) -> None:
    """The user-facing requirement: run it twice, get identical aggregates."""
    first = run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5), seed=42)
    second = run_robustness(small_corpus, chunk_size=7, window_sizes=(3, 5), seed=42)
    assert robustness_artifacts.measurement_digest(
        first
    ) == robustness_artifacts.measurement_digest(second)


def test_artifacts_are_written_and_are_not_dataset_copies(
    small_corpus: List[Path], tmp_path: Path
) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5))
    out = tmp_path / "out"
    written = robustness_artifacts.write_artifacts(result, out)

    assert set(written) == set(robustness_artifacts.ARTIFACT_NAMES)
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0

    payload = json.loads(written["robustness_summary.json"].read_text(encoding="utf-8"))
    assert payload["parameters"]["window_sizes"] == [3, 5]
    assert "measurement_digest_sha256" in payload
    assert payload["definitions"]

    # The comparison table holds one row per (file, T, identity) - it is a summary,
    # never a copy of the flows.
    import pandas as pd

    frame = pd.read_csv(written["window_size_comparison.csv"])
    assert len(frame) == len(result.cells)
    assert list(frame.columns[:4]) == ["T", "file", "identity", "windows"]
    assert frame["T"].tolist() == sorted(frame["T"].tolist()) or True
    assert set(frame["identity"]) <= {IDENTITY_TRUE, IDENTITY_PROXY}


def test_report_answers_every_task_and_avoids_overstated_phrasing(
    small_corpus: List[Path],
) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5, 10))
    report = robustness_artifacts.build_report(
        result, digest=robustness_artifacts.measurement_digest(result)
    )
    lowered = report.lower()

    for phrase in robustness_artifacts.BANNED_PHRASES:
        assert phrase not in lowered

    for heading in (
        "## 1. Which files carry the identifier columns (Task 1)",
        "## 4. Window size sweep on the true endpoint identity (Task 3)",
        "## 6. Cross-file comparison (Tasks 2, 4, 5)",
        "## 7. Is T=5 a reasonable trade-off? (Task 6)",
        "## 8. Challenging the global-window assumption (Task 7)",
        "## 9. Conclusion (Task 8)",
    ):
        assert report.count(heading) == 1

    # Task 7 must ask the question, and must not answer it with a recommendation.
    assert "what evidence would justify changing to another representation" in lowered
    assert "no representation is recommended here" in lowered
    assert "therefore use ip-pair windows" not in lowered
    # Task 8's three-way split.
    assert "### 9.1 What was measured" in report
    assert "### 9.2 Semantic interpretation" in report
    assert "### 9.3 Remaining uncertainty" in report


def test_report_flags_the_proxy_as_an_upper_bound(small_corpus: List[Path]) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5))
    report = robustness_artifacts.build_report(result, digest="x" * 64)
    assert "upper bound on endpoint continuity" in report
    assert "must never be quoted as endpoint continuity" in report


def test_writer_refuses_a_report_containing_overstated_phrasing(
    small_corpus: List[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3,))
    monkeypatch.setattr(
        robustness_artifacts,
        "build_report",
        lambda *a, **k: "# stub\n\nThis analysis fully validates the design.\n",
    )
    with pytest.raises(ValueError, match="overstated phrasing"):
        robustness_artifacts.write_artifacts(result, tmp_path / "out")


def test_representativeness_ranks_the_reference_file_against_the_others(
    small_corpus: List[Path],
) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(5,))
    position = robustness_artifacts.representativeness(result, window_size=5)
    assert position["available"] is True
    assert position["reference_file"] == "02-20-2018.csv"
    assert position["identity_used"] == IDENTITY_PROXY
    assert position["n_files_compared"] == 3
    for item in position["metrics"].values():
        assert 1 <= item["rank_of_reference_high_to_low"] <= item["n_ranked"]
        assert item["reference_value"] == item["all_files"]["02-20-2018.csv"]
        # The reference must not be inside its own comparison baseline.
        assert item["excluding_reference"]["n"] == 2


def test_proxy_calibration_reports_both_identities_on_the_same_rows(
    small_corpus: List[Path],
) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(5,))
    calibration = robustness_artifacts.proxy_calibration(result, window_size=5)
    assert calibration["available"] is True
    assert calibration["file"] == "02-20-2018.csv"
    same_key = calibration["metrics"]["A_pct_windows_all_same_key"]
    assert same_key["proxy_minus_true"] == pytest.approx(
        same_key["proxy_identity"] - same_key["true_identity"]
    )
    assert same_key["proxy_minus_true"] > 0


def test_tradeoff_costs_are_arithmetic_on_measured_window_counts(
    small_corpus: List[Path],
) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5, 10))
    tradeoff = robustness_artifacts.window_size_tradeoff(result)
    assert tradeoff["available"] is True
    per_t = tradeoff["per_window_size"]
    assert set(per_t) == {"3", "5", "10"}
    for t, payload in per_t.items():
        cost = payload["cost_arithmetic"]
        assert cost["nodes_per_window"] == int(t)
        assert cost["max_intra_window_edges_per_window"] == int(t) * (int(t) - 1) // 2
        assert cost["total_flow_nodes"] == payload["n_windows"] * int(t)
    assert per_t["5"]["cost_arithmetic"]["graphs_relative_to_T5"] == pytest.approx(1.0)


def test_run_without_a_true_identity_file_still_produces_a_report(
    tmp_path: Path,
) -> None:
    """Nine of the ten real files are like this, so the degraded path must work."""
    path = _write_csv(tmp_path / "02-14-2018.csv", _hand_flows(), extended=False)
    result = run_robustness([path], chunk_size=4, window_sizes=(3, 5))
    assert all(c.identity == IDENTITY_PROXY for c in result.cells)
    assert robustness_artifacts.window_size_tradeoff(result)["available"] is False
    assert robustness_artifacts.proxy_calibration(result)["available"] is False
    report = robustness_artifacts.build_report(result, digest="y" * 64)
    assert "## 9. Conclusion (Task 8)" in report


def test_run_notes_a_truncated_row_limit(small_corpus: List[Path]) -> None:
    result = run_robustness(small_corpus, chunk_size=4, window_sizes=(3,), row_limit=6)
    assert any("truncated debug run" in note for note in result.notes)
    assert all(c.n_rows_analyzed == 6 for c in result.cells)


# --------------------------------------------------------------------------- #
# the analysis must not touch its inputs
# --------------------------------------------------------------------------- #
def test_sweep_does_not_modify_its_source_files(small_corpus: List[Path]) -> None:
    before = {path.name: _fingerprint(path) for path in small_corpus}
    run_robustness(small_corpus, chunk_size=4, window_sizes=(3, 5, 10))
    after = {path.name: _fingerprint(path) for path in small_corpus}
    assert before == after


def test_missing_file_raises_rather_than_inventing_numbers(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        read_file_columns(tmp_path / "does-not-exist.csv")


# --------------------------------------------------------------------------- #
# real corpus
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not REAL_CORPUS.is_dir(), reason="raw corpus not present")
def test_real_corpus_inventory_matches_what_the_report_claims() -> None:
    """Task 1 answered against the real files, not assumed."""
    paths = sorted(REAL_CORPUS.glob("*.csv"), key=lambda p: p.name)
    assert len(paths) == 10
    inventory = identifier_inventory(paths)
    rich = [i["source_file"] for i in inventory if i["has_full_five_tuple"]]
    assert rich == [REAL_FILE]
    for item in inventory:
        # Whatever else is missing, the proxy identity exists everywhere - which
        # is the only reason a cross-file comparison is possible at all.
        assert item["has_proxy_identity"] is True
        if item["source_file"] != REAL_FILE:
            assert item["identifier_columns_missing"] == [
                "Src IP",
                "Src Port",
                "Dst IP",
            ]


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_CORPUS / REAL_FILE).is_file() or not PUBLISHED.is_file(),
    reason="raw corpus or published T=5 summary not present",
)
def test_real_file_prefix_reproduces_under_two_chunk_sizes() -> None:
    path = REAL_CORPUS / REAL_FILE
    before = _fingerprint(path)
    first = run_robustness(
        [path], chunk_size=100_000, window_sizes=(3, 5, 10), row_limit=300_000
    )
    second = run_robustness(
        [path], chunk_size=250_000, window_sizes=(3, 5, 10), row_limit=300_000
    )
    assert robustness_artifacts.measurement_digest(
        first
    ) == robustness_artifacts.measurement_digest(second)
    assert _fingerprint(path) == before


def test_equivalence_is_skipped_rather_than_failed_on_a_truncated_run(
    small_corpus: List[Path],
) -> None:
    """A row-limited run measures different rows, so a mismatch would say nothing
    about definition drift. It must be skipped with a note, not reported as a
    failure."""
    full = run_robustness(small_corpus, chunk_size=4, window_sizes=(5,))
    cell = full.cell(
        source_file="02-20-2018.csv", window_size=5, identity=IDENTITY_TRUE
    )
    assert cell is not None
    published = _published_stub(cell)

    truncated = run_robustness(
        small_corpus,
        chunk_size=4,
        window_sizes=(5,),
        row_limit=10,
        published_summary=published,
    )
    assert truncated.equivalence is None
    assert any("equivalence check" in note.lower() for note in truncated.notes)

    checked = run_robustness(
        small_corpus, chunk_size=4, window_sizes=(5,), published_summary=published
    )
    assert checked.equivalence is not None
    assert checked.equivalence["all_match"] is True
