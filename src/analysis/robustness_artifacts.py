"""JSON / CSV / Markdown views of a :mod:`src.analysis.continuity_robustness` run.

Writes under the configured results directory only. Nothing here reads or writes
``data/raw``, and no model input is produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.continuity_artifacts import strip_volatile
from src.analysis.continuity_robustness import (
    IDENTITY_PROXY,
    IDENTITY_TRUE,
    PROXY_IDENTITY_COLUMNS,
    TRUE_IDENTITY_COLUMNS,
    RobustnessResult,
)

SUMMARY_NAME = "robustness_summary.json"
TABLE_NAME = "window_size_comparison.csv"
REPORT_NAME = "T_SWEEP_ROBUSTNESS_REPORT.md"

ARTIFACT_NAMES: Sequence[str] = (SUMMARY_NAME, TABLE_NAME, REPORT_NAME)

#: Phrases that overstate what a descriptive analysis can establish. Checked by a
#: test as well as by the writer, so a future edit cannot quietly reintroduce one.
BANNED_PHRASES: Tuple[str, ...] = (
    "perfectly captures",
    "proves causality",
    "zero leakage",
    "fully validates",
)


def measurement_digest(result: RobustnessResult) -> str:
    """sha256 over every measurement, excluding timings."""
    payload = strip_volatile(result.to_dict())
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "---|" * len(headers))
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _arm(cell: Any) -> Dict[str, Any]:
    return cell.consecutive.to_dict(cell.window_size)


def _baseline(cell: Any) -> Optional[Dict[str, Any]]:
    if cell.random_baseline is None:
        return None
    return cell.random_baseline.to_dict(cell.window_size)


def _spread(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def representativeness(result: RobustnessResult, *, window_size: int = 5) -> Dict[str, Any]:
    """Where the reference file sits in the cross-file distribution.

    Uses the proxy identity, the only one measurable in every file. Reports rank
    and a z-score for each metric so "representative" is a measured position in a
    distribution rather than an impression.
    """
    cells = [
        cell
        for cell in result.cells
        if cell.window_size == window_size and cell.identity == IDENTITY_PROXY
    ]
    reference = next(
        (
            cell.source_file
            for cell in result.cells
            if cell.identity == IDENTITY_TRUE and cell.window_size == window_size
        ),
        None,
    )
    if reference is None or len(cells) < 2:
        return {
            "available": False,
            "reason": "needs the reference file plus at least one other file "
            "measured on the proxy identity",
        }

    metrics = (
        "A_pct_windows_all_same_key",
        "B_pct_windows_any_adjacent_same_key",
        "pct_adjacent_links_sharing_key",
        "C_mean_unique_keys_per_window",
        "pct_windows_fully_heterogeneous",
        "F_pct_windows_label_pure",
        "G_pct_windows_final_label_ne_majority",
    )
    per_file = {cell.source_file: _arm(cell) for cell in cells}
    out: Dict[str, Any] = {
        "available": True,
        "identity_used": IDENTITY_PROXY,
        "identity_columns": list(PROXY_IDENTITY_COLUMNS),
        "window_size": window_size,
        "reference_file": reference,
        "n_files_compared": len(cells),
        "caveat": "Measured on the (Dst Port, Protocol) proxy because the other "
        "files carry no endpoint identifiers. The proxy is a service-level key, so "
        "these values are not comparable to the true-identity numbers; only the "
        "*relative position* of the reference file within this distribution is.",
        "metrics": {},
    }
    for metric in metrics:
        values = {name: float(arm[metric]) for name, arm in per_file.items()}
        others = [v for name, v in values.items() if name != reference]
        stats = _spread(list(values.values()))
        own = values[reference]
        # Rank 1 = highest value among the compared files.
        ordered = sorted(values.items(), key=lambda kv: -kv[1])
        rank = next(i for i, (name, _) in enumerate(ordered, start=1) if name == reference)
        other_stats = _spread(others) if others else None
        std = other_stats["std"] if other_stats else 0.0
        z = (own - other_stats["mean"]) / std if other_stats and std > 0 else None
        out["metrics"][metric] = {
            "reference_value": own,
            "rank_of_reference_high_to_low": rank,
            "n_ranked": len(values),
            "all_files": values,
            "across_all_files": stats,
            "excluding_reference": other_stats,
            "z_score_vs_other_files": z,
            "position": (
                "unusually high"
                if z is not None and z >= 2
                else "unusually low"
                if z is not None and z <= -2
                else "within the spread of the other files"
                if z is not None
                else "not scoreable"
            ),
        }
    return out


def proxy_calibration(result: RobustnessResult, *, window_size: int = 5) -> Dict[str, Any]:
    """How far the proxy identity departs from the true identity, on the one file
    where both are measurable.

    Without this the other nine files' proxy numbers cannot be read at all.
    """
    true_cell = next(
        (
            cell
            for cell in result.cells
            if cell.identity == IDENTITY_TRUE and cell.window_size == window_size
        ),
        None,
    )
    if true_cell is None:
        return {"available": False, "reason": "no true-identity cell in this run"}
    proxy_cell = result.cell(
        source_file=true_cell.source_file,
        window_size=window_size,
        identity=IDENTITY_PROXY,
    )
    if proxy_cell is None:
        return {"available": False, "reason": "no proxy cell for the reference file"}

    true_arm, proxy_arm = _arm(true_cell), _arm(proxy_cell)
    metrics = (
        "A_pct_windows_all_same_key",
        "B_pct_windows_any_adjacent_same_key",
        "pct_adjacent_links_sharing_key",
        "C_mean_unique_keys_per_window",
        "pct_windows_fully_heterogeneous",
    )
    return {
        "available": True,
        "file": true_cell.source_file,
        "window_size": window_size,
        "note": "Both identities measured on the same rows with the same code. The "
        "proxy collapses distinct host pairs that share a destination service, so "
        "it reports more continuity than actually exists at the endpoint level.",
        "metrics": {
            metric: {
                "true_identity": float(true_arm[metric]),
                "proxy_identity": float(proxy_arm[metric]),
                "proxy_minus_true": float(proxy_arm[metric]) - float(true_arm[metric]),
            }
            for metric in metrics
        },
    }


def window_size_tradeoff(result: RobustnessResult) -> Dict[str, Any]:
    """Per-T continuity alongside the arithmetic cost of that choice.

    The cost figures are arithmetic on the measured window counts - not a
    measured model cost - and are labelled as such.
    """
    reference = next(
        (cell.source_file for cell in result.cells if cell.identity == IDENTITY_TRUE),
        None,
    )
    if reference is None:
        return {"available": False, "reason": "no true-identity cells in this run"}

    rows: Dict[str, Any] = {}
    for cell in sorted(
        (c for c in result.cells if c.identity == IDENTITY_TRUE and c.source_file == reference),
        key=lambda c: c.window_size,
    ):
        t = cell.window_size
        arm = _arm(cell)
        base = _baseline(cell)
        # If a window becomes a fully connected graph of T flow nodes, this is how
        # many intra-window edges that implies. Arithmetic, not a measurement.
        edges_per_window = t * (t - 1) // 2
        rows[str(t)] = {
            "n_windows": cell.n_windows,
            "trailing_rows_discarded": cell.n_trailing_rows_discarded,
            "A_pct_windows_all_same_key": arm["A_pct_windows_all_same_key"],
            "B_pct_windows_any_adjacent_same_key": arm[
                "B_pct_windows_any_adjacent_same_key"
            ],
            "B_random_baseline": (
                base["B_pct_windows_any_adjacent_same_key"] if base else None
            ),
            "pct_adjacent_links_sharing_key": arm["pct_adjacent_links_sharing_key"],
            "pct_adjacent_links_sharing_key_random": (
                base["pct_adjacent_links_sharing_key"] if base else None
            ),
            "C_mean_unique_keys_per_window": arm["C_mean_unique_keys_per_window"],
            "C_normalised_by_T": arm["C_mean_unique_keys_per_window"] / t,
            "D_median_unique_keys_per_window": arm["D_median_unique_keys_per_window"],
            "pct_windows_fully_heterogeneous": arm["pct_windows_fully_heterogeneous"],
            "F_pct_windows_label_pure": arm["F_pct_windows_label_pure"],
            "G_pct_windows_final_label_ne_majority": arm[
                "G_pct_windows_final_label_ne_majority"
            ],
            "median_span_seconds": arm["timestamp_diagnostics"]["median_span_seconds"],
            "p95_span_seconds": arm["timestamp_diagnostics"]["p95_span_seconds"],
            "pct_windows_entirely_within_one_second": arm["timestamp_diagnostics"][
                "pct_windows_entirely_within_one_second"
            ],
            "pct_windows_with_span_at_most_300s": arm["timestamp_diagnostics"][
                "pct_windows_with_span_at_most"
            ].get("300"),
            "cost_arithmetic": {
                "note": "Arithmetic on the measured window count, not a measured "
                "model cost. Assumes one graph per window with T flow nodes.",
                "nodes_per_window": t,
                "max_intra_window_edges_per_window": edges_per_window,
                "total_flow_nodes": cell.n_windows * t,
                "total_max_intra_window_edges": cell.n_windows * edges_per_window,
                "graphs_relative_to_T5": None,
            },
        }
    five = rows.get("5")
    if five:
        for t, payload in rows.items():
            payload["cost_arithmetic"]["graphs_relative_to_T5"] = (
                payload["n_windows"] / five["n_windows"] if five["n_windows"] else None
            )
            payload["cost_arithmetic"]["edges_relative_to_T5"] = (
                (
                    payload["cost_arithmetic"]["total_max_intra_window_edges"]
                    / five["cost_arithmetic"]["total_max_intra_window_edges"]
                )
                if five["cost_arithmetic"]["total_max_intra_window_edges"]
                else None
            )
    return {"available": True, "file": reference, "per_window_size": rows}


def build_report(result: RobustnessResult, *, digest: str) -> str:
    """The human-readable markdown report."""
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    payload = result.to_dict()
    inventory = result.inventory
    identifier_rich = [i for i in inventory if i["has_true_endpoint_identity"]]
    proxy_only = [i for i in inventory if not i["has_true_endpoint_identity"]]
    tradeoff = window_size_tradeoff(result)
    calibration = proxy_calibration(result)
    position = representativeness(result)

    add("# T=3 / T=5 / T=10 Window Continuity - Robustness Analysis")
    add("")
    add(
        f"Generated {result.generated_at_utc} | measurement digest "
        f"`{digest[:16]}` | elapsed {result.elapsed_seconds:.1f}s | seed "
        f"**{result.seed}**"
    )
    add("")
    add(
        "**Analysis only.** Every number was measured from the raw CSVs. No "
        "preprocessing, loader, windowing or architecture code was changed, no raw "
        "file was touched, and no model input was produced. This report presents "
        "evidence and does not select a representation."
    )
    add("")

    # -- 1. identifier inventory (Task 1) ----------------------------------- #
    add("## 1. Which files carry the identifier columns (Task 1)")
    add("")
    add(
        "Measured from each file's own header line, not inferred from the file name "
        "and not assumed from the 02-20 layout."
    )
    add("")
    add(
        _table(
            ["file", "columns", "variant", "Src IP", "Dst IP", "Src Port",
             "Dst Port", "Protocol", "full 5-tuple"],
            [
                [
                    item["source_file"],
                    item["n_columns"],
                    item["schema_variant"],
                    *[
                        "yes" if name in item["identifier_columns_present"] else "**no**"
                        for name in ("Src IP", "Dst IP", "Src Port", "Dst Port",
                                     "Protocol")
                    ],
                    "yes" if item["has_full_five_tuple"] else "**no**",
                ]
                for item in inventory
            ],
        )
    )
    add("")
    add(
        f"**{len(identifier_rich)} of {len(inventory)} files carry the full "
        f"identifier set**"
        + (
            f" ({', '.join(i['source_file'] for i in identifier_rich)}). "
            if identifier_rich
            else ". "
        )
        + f"The remaining {len(proxy_only)} file(s) have `Dst Port` and `Protocol` "
        "but no `Src IP`, `Dst IP` or `Src Port`, so **metrics A-E as defined in the "
        "first experiment cannot be computed on them at all.** This is a property of "
        "the corpus, not a limitation of the analysis: the CSE-CIC-IDS2018 release "
        "ships one 84-column file and nine 80-column files. The parquet copies carry "
        "78 columns with no identifiers and no `Timestamp`, so they are not usable "
        "here either."
    )
    add("")
    add(
        "Task 2 therefore has an honest scope limit. Rather than stopping there, "
        "every file is also measured on a deliberately weaker identity - "
        f"`{'`, `'.join(PROXY_IDENTITY_COLUMNS)}` - which exists everywhere. On "
        "02-20 both identities are measured on the same rows, which calibrates how "
        "much the proxy overstates continuity (section 5). Without that calibration "
        "the nine-file numbers would be uninterpretable."
    )
    add("")

    # -- 2. provenance ------------------------------------------------------ #
    add("## 2. What was measured")
    add("")
    add(
        _table(
            ["file", "raw rows", "embedded headers", "usable rows", "rows analyzed",
             "unparsed timestamps", "identity available"],
            [
                [
                    item["source_file"],
                    f"{item['n_raw_data_rows']:,}",
                    item["n_embedded_header_rows_excluded"],
                    f"{item['n_usable_rows']:,}",
                    f"{item['n_rows_analyzed']:,}",
                    f"{item['n_unparsed_timestamps']:,}",
                    "true + proxy" if item["has_true_endpoint_identity"] else "proxy only",
                ]
                for item in payload["files"]
            ],
        )
    )
    add("")
    add(f"**Ordering rule** (unchanged from the first experiment): {result.ordering_rule}")
    add("")
    add(
        f"**Seed**: {result.seed}, identical to the first experiment. Window sizes "
        f"swept: {', '.join(str(t) for t in result.window_sizes)}. Metric "
        "definitions generalise the T=5 set by substituting T for 5 and 'final "
        "flow' for 'fifth flow'; every other definition is character-for-character "
        "the same rule."
    )
    add("")

    # -- 3. equivalence ----------------------------------------------------- #
    if result.equivalence:
        add("## 3. The generalised code reproduces the published T=5 result")
        add("")
        add(
            "This sweep uses new code, so it is checked against the frozen "
            "`continuity_summary.json` rather than assumed to agree. A mismatch "
            "would mean a definition drifted, which would invalidate every "
            "cross-T comparison below."
        )
        add("")
        add(
            _table(
                ["metric", "this run", "published T=5", "abs difference", "match"],
                [
                    [
                        item["metric"],
                        f"{item['robustness_value']:,}"
                        if isinstance(item["robustness_value"], int)
                        else f"{item['robustness_value']:.6f}",
                        f"{item['published_value']:,}"
                        if isinstance(item["published_value"], int)
                        else f"{item['published_value']:.6f}",
                        f"{item['abs_difference']:.2e}"
                        if isinstance(item["abs_difference"], float)
                        else item["abs_difference"],
                        "yes" if item["match"] else "**NO**",
                    ]
                    for item in result.equivalence["comparisons"]
                ],
            )
        )
        add("")
        add(
            f"All metrics match: **{result.equivalence['all_match']}**."
            if result.equivalence["all_match"]
            else "**At least one metric does not match - the cross-T comparison "
            "below cannot be trusted until this is resolved.**"
        )
        add("")

    # -- 4. window size sweep (Task 3) -------------------------------------- #
    add("## 4. Window size sweep on the true endpoint identity (Task 3)")
    add("")
    if tradeoff["available"]:
        per_t = tradeoff["per_window_size"]
        add(
            f"`{tradeoff['file']}`, identity = "
            f"`{'`, `'.join(TRUE_IDENTITY_COLUMNS)}`. Same rows, same ordering, "
            "same seed at every T."
        )
        add("")
        add(
            _table(
                ["metric"] + [f"T={t}" for t in sorted(per_t, key=int)],
                [
                    [caption]
                    + [fmt.format(per_t[t][key]) for t in sorted(per_t, key=int)]
                    for key, caption, fmt in (
                        ("n_windows", "windows", "{:,}"),
                        ("trailing_rows_discarded", "trailing rows discarded", "{:,}"),
                        ("A_pct_windows_all_same_key",
                         "**A** all T share one pair (%)", "{:.4f}"),
                        ("B_pct_windows_any_adjacent_same_key",
                         "**B** >=1 adjacent link shares a pair (%)", "{:.4f}"),
                        ("B_random_baseline", "B random baseline (%)", "{:.4f}"),
                        ("pct_adjacent_links_sharing_key",
                         "adjacent *links* sharing a pair (%)", "{:.4f}"),
                        ("pct_adjacent_links_sharing_key_random",
                         "adjacent links, random baseline (%)", "{:.4f}"),
                        ("C_mean_unique_keys_per_window",
                         "**C** mean unique pairs (1..T)", "{:.4f}"),
                        ("C_normalised_by_T", "C / T (scale-free)", "{:.4f}"),
                        ("D_median_unique_keys_per_window",
                         "**D** median unique pairs", "{:.1f}"),
                        ("pct_windows_fully_heterogeneous",
                         "fully heterogeneous, T distinct pairs (%)", "{:.4f}"),
                        ("F_pct_windows_label_pure", "**F** label-pure (%)", "{:.4f}"),
                        ("G_pct_windows_final_label_ne_majority",
                         "**G** final label != majority (%)", "{:.4f}"),
                        ("median_span_seconds", "median span (s)", "{:,.1f}"),
                        ("p95_span_seconds", "p95 span (s)", "{:,.1f}"),
                        ("pct_windows_with_span_at_most_300s",
                         "span <= 300 s (%)", "{:.4f}"),
                        ("pct_windows_entirely_within_one_second",
                         "entirely within one clock second (%)", "{:.4f}"),
                    )
                ],
            )
        )
        add("")
        add(
            "**Metric B is not comparable across window sizes and the table shows "
            "why.** B asks whether *any* of T-1 adjacent links collides, so it rises "
            "with T mechanically, even when per-link continuity is unchanged. The "
            "adjacent-*links* row is the version that can be compared across T, "
            "because its denominator grows with T. Reading B alone across this table "
            "would manufacture a trend that the link-level measure does not show."
        )
        add("")
        add(
            "Likewise **A and C cannot be compared across T at face value**: A must "
            "fall as T grows (more rows must agree) and C must rise (more rows can "
            "differ). The `C / T` row normalises for that."
        )
        add("")

    # -- 5. proxy calibration ---------------------------------------------- #
    add("## 5. Calibrating the cross-file proxy")
    add("")
    if calibration["available"]:
        add(
            f"Both identities measured on the same {calibration['file']} rows with "
            "the same code, at T=5."
        )
        add("")
        add(
            _table(
                ["metric", "true (Src IP, Dst IP)", "proxy (Dst Port, Protocol)",
                 "proxy - true"],
                [
                    [
                        metric,
                        f"{values['true_identity']:.4f}",
                        f"{values['proxy_identity']:.4f}",
                        f"{values['proxy_minus_true']:+.4f}",
                    ]
                    for metric, values in calibration["metrics"].items()
                ],
            )
        )
        add("")
        add(
            "All five rows point the same way even though two of the deltas are "
            "negative: fewer unique keys and fewer fully-heterogeneous windows both "
            "mean *more* continuity, so a negative delta on those two rows is the "
            "proxy claiming more continuity, not less."
        )
        add("")
        add(
            "The proxy reports substantially more continuity than the endpoint "
            "identity does, which is expected: many distinct host pairs talk to the "
            "same destination port, so the proxy merges conversations that the true "
            "identity separates. **Proxy values are an upper bound on endpoint "
            "continuity and must never be quoted as endpoint continuity.** Their "
            "only legitimate use is comparing one file's position against another's, "
            "which is exactly what section 6 does."
        )
        add("")

    # -- 6. cross-file comparison (Tasks 2, 4, 5) -------------------------- #
    add("## 6. Cross-file comparison (Tasks 2, 4, 5)")
    add("")
    add(
        "Full comparison table across every (T, file, identity) combination is in "
        f"`{TABLE_NAME}`. The proxy-identity rows at T=5 are reproduced here."
    )
    add("")
    proxy_cells = sorted(
        (c for c in result.cells if c.identity == IDENTITY_PROXY and c.window_size == 5),
        key=lambda c: c.source_file,
    )
    if proxy_cells:
        add(
            _table(
                ["T", "file", "windows", "same-key %", "adjacent-key %",
                 "adj. links %", "mean unique keys", "label-pure %",
                 "final != majority %"],
                [
                    [
                        cell.window_size,
                        cell.source_file
                        + (" *(reference)*" if cell.source_file == position.get("reference_file") else ""),
                        f"{cell.n_windows:,}",
                        f"{_arm(cell)['A_pct_windows_all_same_key']:.4f}",
                        f"{_arm(cell)['B_pct_windows_any_adjacent_same_key']:.4f}",
                        f"{_arm(cell)['pct_adjacent_links_sharing_key']:.4f}",
                        f"{_arm(cell)['C_mean_unique_keys_per_window']:.4f}",
                        f"{_arm(cell)['F_pct_windows_label_pure']:.4f}",
                        f"{_arm(cell)['G_pct_windows_final_label_ne_majority']:.4f}",
                    ]
                    for cell in proxy_cells
                ],
            )
        )
        add("")
        # The single most informative cross-file result: label purity is a file
        # property, not a window property. Computed, not asserted.
        purity = {
            c.source_file: _arm(c)["F_pct_windows_label_pure"] for c in proxy_cells
        }
        lowest = min(purity.items(), key=lambda kv: kv[1])
        highest = max(purity.items(), key=lambda kv: kv[1])
        low_files = sorted(name for name, value in purity.items() if value < 90)
        if lowest[1] < 90:
            add(
                f"**Label purity is not a stable property of T=5 windowing - it "
                f"ranges from {lowest[1]:.4f}% (`{lowest[0]}`) to {highest[1]:.4f}% "
                f"(`{highest[0]}`) across the corpus**, with {len(low_files)} of "
                f"{len(purity)} files below 90%. The first experiment attributed "
                "02-20's 99.94% to long contiguous label blocks in capture order "
                "rather than to flow relatedness, and this spread is direct "
                "cross-file support for that reading: the same window construction "
                "on the same kind of data yields very different purity depending "
                "only on how each capture's labels happen to be laid out. A high F "
                "measures block layout. It is the metric most likely to be mistaken "
                "for a result, and 02-20 sits at the favourable end of it."
            )
            add("")

    if position.get("available"):
        add("### Is the 02-20 T=5 result representative? (Task 5)")
        add("")
        add(
            f"Position of `{position['reference_file']}` within the "
            f"{position['n_files_compared']}-file proxy-identity distribution at "
            "T=5. A z-score is computed against the *other* files only, so the "
            "reference file does not inflate its own baseline. 'Unusually high/low' "
            "means |z| >= 2."
        )
        add("")
        add(
            _table(
                ["metric", "02-20 value", "other files median", "other files range",
                 "z vs others", "rank", "verdict"],
                [
                    [
                        metric,
                        f"{item['reference_value']:.4f}",
                        f"{item['excluding_reference']['median']:.4f}"
                        if item["excluding_reference"]
                        else "n/a",
                        f"{item['excluding_reference']['min']:.4f} - "
                        f"{item['excluding_reference']['max']:.4f}"
                        if item["excluding_reference"]
                        else "n/a",
                        f"{item['z_score_vs_other_files']:+.2f}"
                        if item["z_score_vs_other_files"] is not None
                        else "n/a",
                        f"{item['rank_of_reference_high_to_low']} of {item['n_ranked']}",
                        item["position"],
                    ]
                    for metric, item in position["metrics"].items()
                ],
            )
        )
        add("")

    # -- 7. trade-off (Task 6) --------------------------------------------- #
    add("## 7. Is T=5 a reasonable trade-off? (Task 6)")
    add("")
    if tradeoff["available"]:
        per_t = tradeoff["per_window_size"]
        add(
            "The cost columns are **arithmetic on the measured window counts**, not "
            "measured model cost: they assume one graph per window with T flow nodes "
            "and count the maximum intra-window edges that implies. No model was "
            "built or timed."
        )
        add("")
        add(
            _table(
                ["T", "windows", "windows vs T=5", "nodes/window",
                 "max edges/window", "total max intra-window edges",
                 "edges vs T=5", "trailing rows lost"],
                [
                    [
                        t,
                        f"{per_t[t]['n_windows']:,}",
                        f"{per_t[t]['cost_arithmetic']['graphs_relative_to_T5']:.2f}x",
                        per_t[t]["cost_arithmetic"]["nodes_per_window"],
                        per_t[t]["cost_arithmetic"]["max_intra_window_edges_per_window"],
                        f"{per_t[t]['cost_arithmetic']['total_max_intra_window_edges']:,}",
                        f"{per_t[t]['cost_arithmetic']['edges_relative_to_T5']:.2f}x",
                        per_t[t]["trailing_rows_discarded"],
                    ]
                    for t in sorted(per_t, key=int)
                ],
            )
        )
        add("")
        add(build_tradeoff_prose(result))
        add("")

    # -- 8. challenging the assumption (Task 7) ---------------------------- #
    add(build_challenge_section(result))
    add("")

    # -- 9. conclusion (Task 8) -------------------------------------------- #
    add(build_conclusion_section(result))
    add("")

    # -- definitions -------------------------------------------------------- #
    add("## 10. Exact definitions")
    add("")
    for key in sorted(payload["definitions"]):
        add(f"- **`{key}`** - {payload['definitions'][key]}")
    add("")

    if result.notes:
        add("## 11. Notes")
        add("")
        for note in result.notes:
            add(f"- {note}")
        add("")

    return "\n".join(lines).rstrip() + "\n"


def build_tradeoff_prose(result: RobustnessResult) -> str:
    """Task 6 discussion, driven by the measured per-T numbers."""
    tradeoff = window_size_tradeoff(result)
    if not tradeoff["available"]:
        return ""
    per_t = tradeoff["per_window_size"]
    t3, t5, t10 = (per_t.get("3"), per_t.get("5"), per_t.get("10"))
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("Four things the sweep actually shows, taken one at a time.")
    add("")
    if t3 and t5 and t10:
        add(
            f"**Continuity per link is close to flat in T.** The comparable "
            f"adjacency measure - share of adjacent links whose two flows share a "
            f"pair - reads {t3['pct_adjacent_links_sharing_key']:.2f}% at T=3, "
            f"{t5['pct_adjacent_links_sharing_key']:.2f}% at T=5 and "
            f"{t10['pct_adjacent_links_sharing_key']:.2f}% at T=10. Growing the "
            "window does not buy denser local continuity; it mostly adds links of "
            "about the same quality. The apparent climb in metric B "
            f"({t3['B_pct_windows_any_adjacent_same_key']:.2f}% to "
            f"{t10['B_pct_windows_any_adjacent_same_key']:.2f}%) is the saturation "
            "artefact described in section 4, not an improvement."
        )
        add("")
        add(
            f"**Single-conversation windows get rarer as T grows.** Metric A falls "
            f"from {t3['A_pct_windows_all_same_key']:.4f}% at T=3 to "
            f"{t5['A_pct_windows_all_same_key']:.4f}% at T=5 and "
            f"{t10['A_pct_windows_all_same_key']:.4f}% at T=10. Even at T=3 the "
            "large majority of windows already hold more than one conversation, so "
            "no value of T in this range makes a global window a single "
            "conversation. Shrinking T buys a modest gain in purity, not a "
            "qualitative change of character."
        )
        add("")
        add(
            f"**Label purity is high at every T and stays uninformative.** "
            f"{t3['F_pct_windows_label_pure']:.4f}% at T=3, "
            f"{t5['F_pct_windows_label_pure']:.4f}% at T=5, "
            f"{t10['F_pct_windows_label_pure']:.4f}% at T=10. The first experiment "
            "traced this to long contiguous label blocks in capture order, and the "
            "sweep is consistent with that reading: purity degrades only slowly with "
            "T, which is what a blocky label column produces. A high F at any T is "
            "evidence about file layout, not about flow relatedness."
        )
        add("")
        add(
            f"**Cost is where T actually bites.** Going T=5 to T=10 halves the "
            f"window count ({t5['n_windows']:,} to {t10['n_windows']:,}) but raises "
            f"maximum intra-window edges by "
            f"{t10['cost_arithmetic']['edges_relative_to_T5']:.2f}x, since a fully "
            "connected window scales as T(T-1)/2. Going T=5 to T=3 cuts edge count "
            f"to {t3['cost_arithmetic']['edges_relative_to_T5']:.2f}x of T=5 while "
            f"raising window count to {t3['cost_arithmetic']['graphs_relative_to_T5']:.2f}x. "
            "Trailing-row loss is negligible at all three "
            f"({t3['trailing_rows_discarded']}, {t5['trailing_rows_discarded']} and "
            f"{t10['trailing_rows_discarded']} rows) and is not a reason to prefer "
            "any of them."
        )
        add("")
        add(
            "**Reading.** On these measurements T=5 is a defensible middle choice "
            "rather than an optimum: it sits between T=3 and T=10 on every "
            "continuity axis, and the axes themselves barely move, so no value in "
            "3-10 is strongly favoured by continuity. What separates them is cost "
            "and implementation complexity, where T=5 is unremarkable in both "
            "directions - materially cheaper than T=10 per window, without T=3's "
            "very short context. The measurements do not identify a continuity "
            "reason to change T, and they equally do not show that T=5 is optimal. "
            "Choosing the highest-continuity T would mean choosing T=3, and that "
            "would be optimising a metric that barely differs while giving up "
            "context length - which is why continuity score alone should not drive "
            "this."
        )
    return "\n".join(lines)


def build_challenge_section(result: RobustnessResult) -> str:
    """Task 7: challenge the global-window assumption without picking a winner."""
    tradeoff = window_size_tradeoff(result)
    per_t = tradeoff["per_window_size"] if tradeoff["available"] else {}
    t5 = per_t.get("5", {})
    calibration = proxy_calibration(result)
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("## 8. Challenging the global-window assumption (Task 7)")
    add("")
    add(
        "*If global windows are still semantically weak, what evidence would "
        "justify changing to another representation?*"
    )
    add("")
    add("State the weakness precisely first, because it is narrower than it sounds.")
    add("")
    if t5:
        add(
            f"At T=5 on the true identity, **{t5['A_pct_windows_all_same_key']:.4f}%** "
            f"of windows are one conversation and the median window holds "
            f"**{t5['D_median_unique_keys_per_window']:.1f}** distinct pairs. That is "
            "the weakness: a global window is a *slice of the capture stream*, not a "
            "*conversation*. It is not the stronger claim that the slice is "
            "meaningless - the first experiment measured every metric moving far "
            "from its random baseline, and this sweep shows that holds at T=3 and "
            "T=10 too."
        )
        add("")
    add(
        "So the assumption under challenge is specifically: *that a T-row slice of "
        "capture order is a useful unit of temporal context for intrusion "
        "detection.* Four kinds of evidence would justify abandoning it, and none of "
        "them is in hand yet:"
    )
    add("")
    add(
        "1. **A controlled grouping comparison that global windows lose.** Same "
        "model, same rows, same features, same split, same seed; only the grouping "
        "rule varies across global-consecutive, within-conversation, and seeded "
        "random. If global-consecutive lands near the random control on held-out "
        "performance, the positional framing is not earning its cost and should go. "
        "If it lands near within-conversation grouping, it is an adequate cheap "
        "proxy. This is the decisive test and it requires training, which is out of "
        "scope here."
    )
    add(
        "2. **Demonstrated dependence on collector write order.** The ordering is an "
        "artefact of how CICFlowMeter emitted rows. If a model trained on one day's "
        "global windows degrades sharply on another day while a "
        "conversation-grouped model does not, the global window was encoding "
        "emission order rather than network behaviour. Note this project can only "
        "run that comparison on one day at the endpoint level, which is itself a "
        "finding - see the uncertainty list below."
    )
    add(
        "3. **Continuity that collapses under the duplicate/conflict policy.** The "
        "measured continuity has not yet been re-measured after the project's "
        "de-duplication rules are applied. If a large share of the adjacent "
        "same-pair links turn out to be duplicate or near-duplicate rows, then part "
        "of the measured locality is a data-quality artefact rather than temporal "
        "structure. This is cheap to test and requires no model."
    )
    add(
        "4. **A representation whose cost is justified by the gain.** "
        "Conversation-based grouping is not free: it needs the endpoint identifiers, "
        + (
            "and **only 1 of the 10 files in this corpus has them**. "
            if calibration.get("available")
            else ""
        )
        + "A representation that cannot be built on nine tenths of the available "
        "data has an evidential burden of its own, and 'higher continuity by "
        "construction' does not discharge it - a conversation-grouped window has "
        "A=100% by definition, which makes that metric uninformative about whether "
        "it detects intrusions better."
    )
    add("")
    add(
        "**No representation is recommended here.** In particular these "
        "measurements do not make the case for IP-pair or 5-tuple grouping. They "
        "establish that global T-windows are not conversation windows at any T in "
        "3-10, which is a statement about what the windows *are*, not about which "
        "representation detects attacks better. The identifier inventory in section "
        "1 is a real obstacle to conversation-based grouping on this corpus and "
        "counts as evidence against a wholesale switch, independently of continuity."
    )
    return "\n".join(lines)


def build_conclusion_section(result: RobustnessResult) -> str:
    """Task 8: separate measurement, interpretation and uncertainty."""
    tradeoff = window_size_tradeoff(result)
    per_t = tradeoff["per_window_size"] if tradeoff["available"] else {}
    t3, t5, t10 = per_t.get("3", {}), per_t.get("5", {}), per_t.get("10", {})
    position = representativeness(result)
    inventory = result.inventory
    n_true = sum(1 for i in inventory if i["has_true_endpoint_identity"])
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("## 9. Conclusion (Task 8)")
    add("")
    add("### 9.1 What was measured")
    add("")
    if t5:
        add(
            f"- **{n_true} of {len(inventory)}** corpus files carry the endpoint "
            "identifiers. Metrics A-E are computable on that one file only."
        )
        add(
            f"- On the true identity, single-conversation windows fall with T: "
            f"**{t3.get('A_pct_windows_all_same_key', 0):.4f}%** (T=3), "
            f"**{t5.get('A_pct_windows_all_same_key', 0):.4f}%** (T=5), "
            f"**{t10.get('A_pct_windows_all_same_key', 0):.4f}%** (T=10)."
        )
        add(
            f"- Per-link adjacency is near-flat in T: "
            f"**{t3.get('pct_adjacent_links_sharing_key', 0):.4f}%**, "
            f"**{t5.get('pct_adjacent_links_sharing_key', 0):.4f}%**, "
            f"**{t10.get('pct_adjacent_links_sharing_key', 0):.4f}%** - against "
            f"random baselines of "
            f"**{t3.get('pct_adjacent_links_sharing_key_random', 0):.4f}%**, "
            f"**{t5.get('pct_adjacent_links_sharing_key_random', 0):.4f}%**, "
            f"**{t10.get('pct_adjacent_links_sharing_key_random', 0):.4f}%**."
        )
        add(
            f"- Label purity stays high at every T: "
            f"**{t3.get('F_pct_windows_label_pure', 0):.4f}%**, "
            f"**{t5.get('F_pct_windows_label_pure', 0):.4f}%**, "
            f"**{t10.get('F_pct_windows_label_pure', 0):.4f}%**."
        )
    if position.get("available"):
        verdicts = [
            item["position"] for item in position["metrics"].values()
        ]
        n_within = sum(1 for v in verdicts if v.startswith("within"))
        add(
            f"- On the cross-file proxy identity at T=5, 02-20 sits **within the "
            f"spread of the other {position['n_files_compared'] - 1} files on "
            f"{n_within} of {len(verdicts)} metrics**."
        )
        purity = position["metrics"]["F_pct_windows_label_pure"]["all_files"]
        add(
            f"- Across the corpus, T=5 label purity ranges from "
            f"**{min(purity.values()):.4f}%** to **{max(purity.values()):.4f}%** - "
            "it is a property of each file's label layout, not of the windowing."
        )
    add("")
    add("### 9.2 Semantic interpretation")
    add("")
    add(
        "These are readings of the numbers above, not further measurements. A global "
        "T-window in this corpus is a *contiguous slice of the capture stream* whose "
        "flows overlap partially in endpoints. It is measurably more coherent than a "
        "random regrouping of the same rows, at every T tested. It is not a single "
        "conversation at any T tested, and the gap is large rather than marginal. "
        "The near-flat per-link adjacency across T=3 to T=10 reads as local "
        "continuity being a property of the capture ordering itself rather than "
        "something a longer window accumulates - which in turn means the choice of T "
        "in this range is not a continuity decision. High label purity is best read "
        "as a description of how the label column is laid out in file order."
    )
    add("")
    add("### 9.3 Remaining uncertainty")
    add("")
    add(
        "- **Cross-file generalisation of the endpoint metrics is untested and, on "
        "this corpus, untestable.** Nine files lack the identifiers. The proxy "
        "comparison places 02-20 inside the others' spread on a service-level key, "
        "which is weaker evidence than a direct replication and is not a substitute "
        "for it."
    )
    add(
        "- **Predictive value is entirely unmeasured.** Nothing here says whether "
        "T=5 global windows help a detector. Every metric in this report is "
        "descriptive; the grouping comparison that would settle it needs training."
    )
    add(
        "- **The duplicate/conflict policy has not been applied.** Some measured "
        "adjacency may be duplicate rows rather than temporal structure. Unknown in "
        "size until measured."
    )
    add(
        "- **Timestamp remains diagnostic only.** The 12-hour clock has no AM/PM "
        "marker and was not reconstructed, so span figures indicate locality "
        "roughly and cannot be read as elapsed time. The bimodal span distribution "
        "means row adjacency and time adjacency coincide for some windows and not "
        "others."
    )
    add(
        "- **Only T in {3, 5, 10} was tested,** on one file, at one seed for the "
        "baseline. The near-flatness observed within that range is not evidence "
        "about T outside it."
    )
    add("")
    add(
        "**Net.** The measurements neither justify changing T away from 5 nor "
        "establish that 5 is optimal; within 3-10 the continuity axes barely move, "
        "so cost and complexity are the discriminating factors and T=5 is a "
        "reasonable middle. The measurements do weaken the semantic claim that a "
        "global T-window captures one communication context - at every T tested it "
        "does not. Whether that weakness matters for detection performance is not "
        "answered by any number in this report."
    )
    return "\n".join(lines)


def write_artifacts(result: RobustnessResult, output_dir: str | Path) -> Dict[str, Path]:
    """Write the JSON summary, comparison CSV and markdown report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = measurement_digest(result)
    written: Dict[str, Path] = {}

    payload = result.to_dict()
    payload["measurement_digest_sha256"] = digest
    payload["representativeness"] = representativeness(result)
    payload["proxy_calibration"] = proxy_calibration(result)
    payload["window_size_tradeoff"] = window_size_tradeoff(result)
    summary_path = out / SUMMARY_NAME
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    written[SUMMARY_NAME] = summary_path

    table_path = out / TABLE_NAME
    result.comparison_frame().to_csv(table_path, index=False)
    written[TABLE_NAME] = table_path

    report = build_report(result, digest=digest)
    lowered = report.lower()
    found = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
    if found:  # pragma: no cover - guard against a future edit
        raise ValueError(f"report contains overstated phrasing: {found}")
    report_path = out / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    written[REPORT_NAME] = report_path

    return written
