"""JSON / CSV / Markdown views of a :mod:`src.analysis.window_continuity` run.

Writes under the configured results directory only. Nothing here reads or writes
``data/raw``, and no model input is produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.analysis.window_continuity import (
    DEFINITIONS,
    READ_COLUMNS,
    WINDOW_SIZE,
    ContinuityResult,
    WindowStats,
)

#: Keys that legitimately differ between two identical-input runs.
VOLATILE_KEYS = frozenset({"generated_at_utc", "elapsed_seconds"})

SUMMARY_NAME = "continuity_summary.json"
WINDOW_CSV_NAME = "per_window_continuity.csv"
REPORT_NAME = "T5_CONTINUITY_REPORT.md"

ARTIFACT_NAMES: Sequence[str] = (SUMMARY_NAME, WINDOW_CSV_NAME, REPORT_NAME)


def strip_volatile(payload: Any) -> Any:
    """Recursively drop wall-clock keys so two runs can be compared."""
    if isinstance(payload, dict):
        return {
            key: strip_volatile(value)
            for key, value in sorted(payload.items())
            if key not in VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [strip_volatile(item) for item in payload]
    return payload


def measurement_digest(result: ContinuityResult) -> str:
    """sha256 over every measurement, excluding timings.

    Two runs over an unchanged file must produce the same digest.
    """
    payload = strip_volatile(result.to_dict())
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "---|" * len(headers))
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def _arm_rows(stats: WindowStats) -> Dict[str, Any]:
    payload = stats.to_dict()
    return {
        "endpoint": payload["endpoint_continuity"],
        "label": payload["label_continuity"],
        "time": payload["timestamp_diagnostics"],
    }


def build_evidence_section(result: ContinuityResult, *, section: int = 7) -> str:
    """Answer the eight interpretation questions from measured values only.

    Every number interpolated here comes from ``result``. The section states what
    the measurements support and what they do not; it deliberately does not
    recommend an architecture.
    """
    consecutive = _arm_rows(result.consecutive)
    endpoint = consecutive["endpoint"]
    label = consecutive["label"]
    time_stats = consecutive["time"]
    random_arm = _arm_rows(result.random_baseline) if result.random_baseline else None

    def rnd(group: str, key: str) -> Optional[float]:
        if random_arm is None:
            return None
        return random_arm[group][key]

    def vs(group: str, key: str, fmt: str = "{:.4f}") -> str:
        """Render 'x (random baseline y)' when a baseline exists."""
        value = fmt.format(consecutive[group][key])
        other = rnd(group, key)
        return value if other is None else f"{value} (random baseline {fmt.format(other)})"

    a = endpoint["A_pct_windows_all_five_same_src_dst_pair"]
    b = endpoint["B_pct_windows_any_adjacent_same_src_dst_pair"]
    c = endpoint["C_mean_unique_pairs_per_window"]
    d = endpoint["D_median_unique_pairs_per_window"]
    het = endpoint["pct_windows_fully_heterogeneous_five_distinct_pairs"]
    adj_pairs = endpoint["pct_adjacent_flow_pairs_sharing_src_dst"]
    five_tuple = endpoint["pct_windows_all_five_same_five_tuple"]
    mean_tuples = endpoint["mean_unique_five_tuples_per_window"]
    f = label["F_pct_windows_label_pure"]
    g = label["G_pct_windows_fifth_label_ne_majority"]
    same_second = time_stats["pct_windows_entirely_within_one_second"]
    buckets = time_stats.get("pct_windows_with_span_at_most") or {}
    tight = buckets.get("300")

    # Windows holding 2-4 distinct pairs: partial overlap, neither one
    # conversation nor five unrelated ones.
    dist = endpoint["E_pct_windows_by_unique_pair_count"]
    partial = sum(dist[str(k)] for k in range(2, WINDOW_SIZE))

    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add(f"## {section}. Evidence for the interpretation questions")
    add("")
    add(
        "Measured values only. This section reports what the numbers show and "
        "where they are ambiguous; **it does not select an architecture.**"
    )
    add("")

    add(
        "**1. Do five globally consecutive capture-order flows generally belong "
        "to the same communication context?**"
    )
    add("")
    add(
        f"No, not as a rule - but they are far from unrelated. Only **{a:.4f}%** of "
        f"windows are a single (Src IP, Dst IP) conversation, and the median window "
        f"holds **{d:.1f}** distinct pairs (mean **{c:.4f}**). At the same time only "
        f"**{het:.4f}%** are fully heterogeneous, so **{partial:.4f}%** sit in "
        "between: some endpoint overlap, but more than one conversation. Against the "
        + (
            f"random baseline (**{rnd('endpoint', 'C_mean_unique_pairs_per_window'):.4f}** "
            f"pairs per window, **{rnd('endpoint', 'pct_windows_fully_heterogeneous_five_distinct_pairs'):.4f}%** "
            "fully heterogeneous) the consecutive grouping is clearly more coherent "
            "than chance. So the honest reading is *partial* shared context, not "
            "*one* shared context."
            if random_arm
            else "file's own marginal distributions this is not chance, but it is "
            "still partial shared context, not one shared context."
        )
    )
    add("")

    add("**2. How often do all five flows share the same Src/Dst pair?**")
    add("")
    add(
        f"**{a:.4f}%** of windows "
        f"({endpoint['counts']['all_same_pair']:,} of {result.n_windows:,}). "
        f"Tightening the identity to the full 5-tuple drops this to "
        f"**{five_tuple:.4f}%**, and the mean number of distinct 5-tuples per window "
        f"is **{mean_tuples:.4f}** out of 5 - so even inside an all-same-pair window "
        "the five flows are almost always different connections between the same two "
        "hosts, not one connection observed five times."
    )
    add("")

    add("**3. How often do adjacent flows share a Src/Dst pair?**")
    add("")
    add(
        f"At the window level **{b:.4f}%** of windows contain at least one adjacent "
        f"same-pair link. At the link level **{adj_pairs:.4f}%** of all "
        f"{result.n_windows * (WINDOW_SIZE - 1):,} adjacent flow pairs share a pair, "
        f"and the mean window has "
        f"**{endpoint['mean_adjacent_same_pair_per_window_max_4']:.4f}** such links "
        "out of 4."
    )
    if random_arm:
        add("")
        add(
            f"Metric B must be read against the baseline rather than on its own. "
            f"Random regrouping of the same rows yields "
            f"**{rnd('endpoint', 'B_pct_windows_any_adjacent_same_src_dst_pair'):.4f}%**, "
            f"so B is not chance-driven *in this file*. But B is structurally "
            "chance-inflated whenever pair cardinality is low: it asks only whether "
            "1 of 4 slots collides, so a file with a handful of heavy-hitter pairs "
            "can score high on B with no real locality at all. B is the weakest of "
            "the seven metrics and should not carry an argument by itself."
        )
    add("")

    add("**4. How often are the windows completely heterogeneous?**")
    add("")
    add(
        f"**{het:.4f}%** of windows have five distinct Src/Dst pairs "
        f"({endpoint['n_windows_by_unique_pair_count'][str(WINDOW_SIZE)]:,} windows)"
        + (
            f", against **{rnd('endpoint', 'pct_windows_fully_heterogeneous_five_distinct_pairs'):.4f}%** "
            "under random regrouping."
            if random_arm
            else "."
        )
    )
    add("")

    add("**5. How often does the fifth-flow label disagree with the majority?**")
    add("")
    add(
        f"**{g:.4f}%** of windows ({label['counts']['fifth_ne_majority']:,} of "
        f"{result.n_windows:,})"
        + (
            f", against **{rnd('label', 'G_pct_windows_fifth_label_ne_majority'):.4f}%** "
            "under random regrouping"
            if random_arm
            else ""
        )
        + f". Label purity is **{f:.4f}%**, so mixed-label windows are "
        f"**{label['pct_windows_mixed_label']:.4f}%** "
        f"({label['counts']['mixed_label']:,} windows) in total."
    )
    add("")
    add(
        "**This near-perfect label purity is weak evidence for temporal context and "
        "must not be read as a model result.** It is a property of how the capture is "
        "laid out: the labels sit in long contiguous blocks in file order, so almost "
        "every window falls strictly inside one block and inherits one label for free. "
        "A high F says the label column is blocky; it does not say the five flows are "
        "one conversation. It also means a T=5 window carries almost no label "
        "transition to learn from - the boundary cases are the "
        f"{label['counts']['mixed_label']:,} mixed windows."
    )
    add("")

    add(
        "**6. Does the evidence support or weaken the idea that T=5 global windows "
        "provide meaningful temporal context?**"
    )
    add("")
    add("Mixed, and the two sides are worth separating.")
    add("")
    add(
        "*Supports:* every metric moves substantially away from chance in the "
        "direction of coherence"
        + (
            f" - fully heterogeneous {het:.2f}% vs "
            f"{rnd('endpoint', 'pct_windows_fully_heterogeneous_five_distinct_pairs'):.2f}%, "
            f"mean pairs {c:.2f} vs "
            f"{rnd('endpoint', 'C_mean_unique_pairs_per_window'):.2f}, "
            f"label purity {f:.2f}% vs {rnd('label', 'F_pct_windows_label_pure'):.2f}%"
            if random_arm
            else ""
        )
        + ". Consecutive rows are demonstrably not an arbitrary sample: the ordering "
        "carries real structure, and roughly "
        + (f"{tight:.1f}% " if tight is not None else "")
        + "of windows are locally tight in time"
        + (" (span <= 300 s)" if tight is not None else "")
        + f", with **{same_second:.4f}%** landing entirely inside one clock second."
    )
    add("")
    add(
        "*Weakens:* the shared context is partial rather than actual. A "
        f"median of {d:.1f} distinct conversations per window means the majority of "
        "windows mix traffic from different host pairs, so a sequence model reading "
        "five consecutive rows is mostly reading interleaved conversations, not one "
        "conversation's evolution. The strongest-looking number (label purity "
        f"{f:.4f}%) is explained by block layout, not by flow relatedness. And the "
        "span distribution is bimodal: many windows are tight bursts while the rest "
        "straddle most of the capture day, so 'temporal locality' is not a uniform "
        "property of the windowing - it holds for some windows and not others."
    )
    add("")
    add(
        "Net: the evidence supports that consecutive ordering carries *some* real "
        "signal, and does not support the stronger claim that a T=5 global window is "
        "a coherent communication episode."
    )
    add("")

    add("**7. What is the strongest argument AGAINST using global T=5 windows?**")
    add("")
    add(
        f"Most windows are not one conversation. With only **{a:.4f}%** of windows "
        f"single-pair and a median of **{d:.1f}** distinct pairs, the sequence a model "
        "sees at T=5 is predominantly an interleaving of unrelated flows that happen "
        "to be adjacent in the capture file. Any 'temporal' relation it learns is "
        "then partly a property of collector write order rather than of network "
        "behaviour - and that ordering is an artefact of how CICFlowMeter emitted "
        "rows, which does not have to reproduce on another capture or another day. "
        "The bimodal span distribution sharpens this: adjacency in row position does "
        "not reliably mean adjacency in time."
    )
    add("")
    add(
        "The secondary argument is that label purity, the metric that looks most "
        "favourable, is the one most easily explained without any temporal mechanism."
    )
    add("")

    add(
        "**8. What additional experiment would most decisively resolve the "
        "uncertainty?**"
    )
    add("")
    add(
        "Train the *same* model on the *same* rows under three grouping rules and "
        "compare held-out performance:"
    )
    add("")
    add(
        "1. T=5 global consecutive windows (the current proposal).\n"
        "2. T=5 windows grouped within a Src/Dst pair (or 5-tuple), so every window "
        "is one conversation by construction.\n"
        "3. T=5 windows built by seeded random regrouping - the baseline measured "
        "here, used as a control."
    )
    add("")
    add(
        "Holding architecture, features, split and seed fixed makes grouping the only "
        "variable. If (1) beats (3) but not (2), the context that matters is "
        "conversational rather than positional; if (1) matches (2), global adjacency "
        "is a cheap adequate proxy; if (1) matches (3), T=5 global windows add "
        "nothing over shuffling and the temporal framing is not earning its cost. "
        "This experiment is decisive in a way no descriptive statistic can be, "
        "because it measures the quantity actually at stake - predictive value - "
        "rather than endpoint overlap."
    )
    add("")
    add(
        "Two cheaper checks are worth running first, and neither requires a model: "
        "repeat this analysis on the other capture days to see whether the continuity "
        "profile is stable or specific to this file (only this file carries endpoint "
        "identifiers, so a same-file proxy such as a Flow-ID-free variant would be "
        "needed), and measure how much of the measured continuity survives after the "
        "project's duplicate/conflict policy is applied."
    )
    add("")
    add(
        "**No architecture conclusion is drawn here.** In particular the measurements "
        "above do not establish that IP-pair windows are the right choice - they "
        "establish that global T=5 windows are not single-conversation windows, which "
        "is a different and weaker statement."
    )

    return "\n".join(lines)


def build_report(result: ContinuityResult, *, digest: str) -> str:
    """The human-readable markdown report."""
    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    consecutive = _arm_rows(result.consecutive)
    random_arm = _arm_rows(result.random_baseline) if result.random_baseline else None

    add("# T=5 Window Continuity - Empirical Analysis")
    add("")
    add(
        f"Generated {result.generated_at_utc} | measurement digest "
        f"`{digest[:16]}` | elapsed {result.elapsed_seconds:.1f}s"
    )
    add("")
    add(
        "**Analysis only.** Every number was measured from the raw CSV. No "
        "preprocessing, loader or windowing code was changed, no raw file was "
        "touched, and no model input was produced. This report presents evidence; "
        "it does not choose an architecture."
    )
    add("")

    # -- provenance --------------------------------------------------------- #
    add("## 1. What was measured")
    add("")
    add("```")
    add(f"dataset                  : {result.source_file}")
    add(f"path                     : {result.source_path}")
    add(f"schema variant           : {result.schema_variant}")
    add(f"raw data rows            : {result.n_raw_rows:,}")
    add(f"embedded header rows     : {result.n_embedded_header_rows} (excluded)")
    add(f"usable rows              : {result.n_usable_rows:,}")
    add(f"rows analyzed            : {result.n_rows_analyzed:,}")
    add(f"T = {WINDOW_SIZE} windows            : {result.n_windows:,} (non-overlapping)")
    add(f"trailing rows discarded  : {result.n_trailing_rows_discarded}")
    add(f"seed                     : {result.seed}")
    add(f"unparsed timestamps      : {result.n_unparsed_timestamps:,}")
    add("```")
    add("")
    add(f"**Ordering rule**: {result.ordering_rule}")
    add("")
    add(
        f"Only {len(READ_COLUMNS)} of the file's {result.n_columns_in_file} columns "
        "were read (" + ", ".join(f"`{name}`" for name in READ_COLUMNS) + "), plus "
        "each row's original position. The 84-column variant is the only file "
        "carrying endpoint identifiers; that layout is not assumed for any other "
        "file."
    )
    add("")

    # -- headline ----------------------------------------------------------- #
    add("## 2. Headline continuity metrics")
    add("")
    endpoint = consecutive["endpoint"]
    label = consecutive["label"]
    headers = ["metric", "definition", "consecutive T=5"]
    if random_arm:
        headers += ["random baseline", "difference"]

    def row(key: str, definition: str, section: str, fmt: str = "{:.4f}%") -> List[Any]:
        left = consecutive[section][key]
        cells: List[Any] = [definition, fmt.format(left)]
        if random_arm:
            right = random_arm[section][key]
            cells += [fmt.format(right), f"{left - right:+.4f}"]
        return cells

    rows = [
        ["**A**"] + row(
            "A_pct_windows_all_five_same_src_dst_pair",
            "all 5 flows share one (Src IP, Dst IP)",
            "endpoint",
        ),
        ["**B**"] + row(
            "B_pct_windows_any_adjacent_same_src_dst_pair",
            ">=1 adjacent flow pair shares a Src/Dst pair",
            "endpoint",
        ),
        ["**C**"] + row(
            "C_mean_unique_pairs_per_window",
            "mean unique Src/Dst pairs per window (1-5)",
            "endpoint",
            "{:.4f}",
        ),
        ["**D**"] + row(
            "D_median_unique_pairs_per_window",
            "median unique Src/Dst pairs per window",
            "endpoint",
            "{:.1f}",
        ),
        ["**F**"] + row(
            "F_pct_windows_label_pure", "all 5 labels identical", "label"
        ),
        ["**G**"] + row(
            "G_pct_windows_fifth_label_ne_majority",
            "5th label != window majority label",
            "label",
        ),
    ]
    add("\n".join(_table(headers, rows)))
    add("")

    # -- E: distribution ---------------------------------------------------- #
    add("### E. Distribution of unique Src/Dst pairs per window")
    add("")
    dist_headers = ["unique pairs", "windows", "% of windows"]
    if random_arm:
        dist_headers += ["% random baseline"]
    dist_rows = []
    for k in range(1, WINDOW_SIZE + 1):
        key = str(k)
        cells: List[Any] = [
            k,
            f"{endpoint['n_windows_by_unique_pair_count'][key]:,}",
            f"{endpoint['E_pct_windows_by_unique_pair_count'][key]:.4f}%",
        ]
        if random_arm:
            cells.append(
                f"{random_arm['endpoint']['E_pct_windows_by_unique_pair_count'][key]:.4f}%"
            )
        dist_rows.append(cells)
    add("\n".join(_table(dist_headers, dist_rows)))
    add("")

    # -- endpoint detail ---------------------------------------------------- #
    add("## 3. Endpoint continuity in detail")
    add("")
    detail_headers = ["measurement", "consecutive T=5"]
    if random_arm:
        detail_headers.append("random baseline")

    def detail(key: str, caption: str, fmt: str = "{:.4f}") -> List[Any]:
        cells: List[Any] = [caption, fmt.format(endpoint[key])]
        if random_arm:
            cells.append(fmt.format(random_arm["endpoint"][key]))
        return cells

    add(
        "\n".join(
            _table(
                detail_headers,
                [
                    detail("mean_unique_src_ip_per_window", "mean unique Src IPs (1-5)"),
                    detail("mean_unique_dst_ip_per_window", "mean unique Dst IPs (1-5)"),
                    detail(
                        "mean_unique_five_tuples_per_window",
                        "mean unique 5-tuples (1-5)",
                    ),
                    detail(
                        "pct_windows_all_five_same_five_tuple",
                        "% windows where all 5 share one 5-tuple",
                        "{:.4f}",
                    ),
                    detail(
                        "mean_adjacent_same_pair_per_window_max_4",
                        "mean adjacent same-pair count (0-4)",
                    ),
                    detail(
                        "pct_adjacent_flow_pairs_sharing_src_dst",
                        "% of all adjacent flow pairs sharing a Src/Dst pair",
                        "{:.4f}",
                    ),
                    detail(
                        "pct_windows_fully_heterogeneous_five_distinct_pairs",
                        "% windows fully heterogeneous (5 distinct pairs)",
                        "{:.4f}",
                    ),
                ],
            )
        )
    )
    add("")

    # -- labels ------------------------------------------------------------- #
    add("## 4. Label continuity")
    add("")
    label_headers = ["unique labels in window", "windows", "% of windows"]
    if random_arm:
        label_headers.append("% random baseline")
    label_rows = []
    for key in sorted(label["n_windows_by_unique_label_count"], key=int):
        cells: List[Any] = [
            key,
            f"{label['n_windows_by_unique_label_count'][key]:,}",
            f"{label['pct_windows_by_unique_label_count'][key]:.4f}%",
        ]
        if random_arm:
            cells.append(
                f"{random_arm['label']['pct_windows_by_unique_label_count'].get(key, 0.0):.4f}%"
            )
        label_rows.append(cells)
    add("\n".join(_table(label_headers, label_rows)))
    add("")
    add(
        f"Label-pure: **{label['F_pct_windows_label_pure']:.4f}%** "
        f"({label['counts']['label_pure']:,} windows). Mixed-label: "
        f"**{label['pct_windows_mixed_label']:.4f}%** "
        f"({label['counts']['mixed_label']:,}). Fifth label equals the majority in "
        f"**{label['pct_windows_fifth_label_eq_majority']:.4f}%**, disagrees in "
        f"**{label['G_pct_windows_fifth_label_ne_majority']:.4f}%** "
        f"({label['counts']['fifth_ne_majority']:,} windows)."
    )
    add("")
    add(
        "Majority ties are broken by ascending label string, so "
        "`fifth_is_majority` is conservative: under a tie the fifth label counts "
        "as the majority only if it also wins that tiebreak."
    )
    add("")

    # -- time --------------------------------------------------------------- #
    add("## 5. Temporal diagnostics")
    add("")
    time_arm = consecutive["time"]
    time_headers = ["measurement", "consecutive T=5"]
    if random_arm:
        time_headers.append("random baseline")
    time_rows = []
    for key, caption, fmt in (
        ("median_span_seconds", "median window span (s)", "{:,.1f}"),
        ("p95_span_seconds", "95th percentile span (s)", "{:,.1f}"),
        ("max_span_seconds", "maximum span (s)", "{:,.1f}"),
        ("mean_span_seconds", "mean span (s)", "{:,.1f}"),
        (
            "pct_windows_entirely_within_one_second",
            "% windows entirely within one clock second",
            "{:.4f}",
        ),
    ):
        cells: List[Any] = [caption, fmt.format(time_arm[key])]
        if random_arm:
            cells.append(fmt.format(random_arm["time"][key]))
        time_rows.append(cells)
    add("\n".join(_table(time_headers, time_rows)))
    add("")
    buckets = time_arm.get("pct_windows_with_span_at_most") or {}
    if buckets:
        add("Cumulative span distribution (consecutive arm):")
        add("")
        add(
            "\n".join(
                _table(
                    ["window span", "% of windows at or below"],
                    [
                        [
                            "same second (0 s)" if key == "0" else f"<= {int(key):,} s",
                            f"{value:.4f}%",
                        ]
                        for key, value in sorted(buckets.items(), key=lambda kv: int(kv[0]))
                    ],
                )
            )
        )
        add("")
        tight = buckets.get("300", 0.0)
        add(
            f"The distribution is **bimodal**, so the median alone is misleading: "
            f"about {tight:.1f}% of windows span 5 minutes or less while the rest "
            "stretch across most of the capture day. A window is either a tight "
            "burst of near-simultaneous flows or an arbitrary jump between distant "
            "parts of the file, with little in between. The median sits in the empty "
            "middle and describes neither mode."
        )
        add("")
    add(
        "`span_seconds` = max(parsed Timestamp) - min(parsed Timestamp) within the "
        "window. **Timestamp is diagnostic only and never orders rows** - the "
        "ordering key is the capture-order proxy above. The column is non-monotonic "
        "in file order and its clock is 12-hour with no AM/PM marker, so a span is "
        "a rough locality indicator, not a measured elapsed time. In particular a "
        "large span does not prove the five flows are far apart in capture time; it "
        "proves the timestamp column does not increase monotonically with row "
        "position."
    )
    add("")

    # -- baseline ----------------------------------------------------------- #
    if random_arm:
        add("## 6. Random-grouping baseline")
        add("")
        add(
            f"The identical {result.n_rows_analyzed:,} usable rows were regrouped "
            f"into {result.n_windows:,} windows of {WINDOW_SIZE} by a seeded "
            f"permutation (seed **{result.seed}**) and measured with the same code. "
            "Row content and marginal label/endpoint distributions are held fixed, "
            "so any difference isolates the effect of *consecutive adjacency*."
        )
        add("")
        add(
            "**This baseline is a measuring stick only.** It is not used for "
            "training, and it does not alter the project's split methodology."
        )
        add("")
        comparison = result.comparison()
        add(
            "\n".join(
                _table(
                    ["metric", "consecutive", "random", "difference"],
                    [
                        [
                            key,
                            f"{values['consecutive']:.4f}",
                            f"{values['random']:.4f}",
                            f"{values['difference']:+.4f}",
                        ]
                        for key, values in comparison.items()
                    ],
                )
            )
        )
        add("")

    # -- evidence ----------------------------------------------------------- #
    section = 7 if random_arm else 6
    add(build_evidence_section(result, section=section))
    add("")

    # -- definitions -------------------------------------------------------- #
    add(f"## {section + 1}. Exact definitions")
    add("")
    for key in sorted(DEFINITIONS):
        add(f"- **`{key}`** - {DEFINITIONS[key]}")
    add("")

    if result.notes:
        add(f"## {section + 2}. Notes")
        add("")
        for note in result.notes:
            add(f"- {note}")
        add("")

    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(result: ContinuityResult, output_dir: str | Path) -> Dict[str, Path]:
    """Write the JSON summary, per-window CSV and markdown report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = measurement_digest(result)

    written: Dict[str, Path] = {}

    payload = result.to_dict()
    payload["measurement_digest_sha256"] = digest
    summary_path = out / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    written[SUMMARY_NAME] = summary_path

    csv_path = out / WINDOW_CSV_NAME
    if result.per_window is not None:
        frame = result.per_window.copy()
        # 1.6M rows x 5 boolean columns written as "True"/"False" is ~40 MB of
        # pure formatting. 0/1 says the same thing and reads back as bool.
        for column in frame.columns:
            if frame[column].dtype == bool:
                frame[column] = frame[column].astype("int8")
        # Spans are whole seconds; a nullable integer keeps the NaN for windows
        # with an unparseable timestamp without printing ".0" 1.6M times.
        frame["span_seconds"] = frame["span_seconds"].astype("Int64")
        frame.to_csv(csv_path, index=False)
    else:  # pragma: no cover - per_window is always populated by analyze_file
        csv_path.write_text("", encoding="utf-8")
    written[WINDOW_CSV_NAME] = csv_path

    report_path = out / REPORT_NAME
    report_path.write_text(build_report(result, digest=digest), encoding="utf-8")
    written[REPORT_NAME] = report_path

    return written
