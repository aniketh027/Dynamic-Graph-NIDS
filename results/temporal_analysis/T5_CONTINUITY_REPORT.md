# T=5 Window Continuity - Empirical Analysis

Generated 2026-09-06T12:18:13.332161+00:00 | measurement digest `b831c47c5e1ecccf` | elapsed 114.4s

**Analysis only.** Every number was measured from the raw CSV. No preprocessing, loader or windowing code was changed, no raw file was touched, and no model input was produced. This report presents evidence; it does not choose an architecture.

## 1. What was measured

```
dataset                  : 02-20-2018.csv
path                     : C:\Project\ComputerNetworks\data\raw\csv\02-20-2018.csv
schema variant           : extended_84
raw data rows            : 7,948,748
embedded header rows     : 0 (excluded)
usable rows              : 7,948,748
rows analyzed            : 7,948,748
T = 5 windows            : 1,589,749 (non-overlapping)
trailing rows discarded  : 3
seed                     : 42
unparsed timestamps      : 0
```

**Ordering rule**: deterministic capture-order proxy: (source file capture date, Original_Index). Within one file this is the file's own data-row order; no sort is applied and Timestamp is never an ordering key.

Only 7 of the file's 84 columns were read (`Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`, `Timestamp`, `Label`), plus each row's original position. The 84-column variant is the only file carrying endpoint identifiers; that layout is not assumed for any other file.

## 2. Headline continuity metrics

| metric | definition | consecutive T=5 | random baseline | difference |
|---|---|---|---|---|
| **A** | all 5 flows share one (Src IP, Dst IP) | 7.7894% | 0.0000% | +7.7894 |
| **B** | >=1 adjacent flow pair shares a Src/Dst pair | 66.4380% | 0.3813% | +66.0567 |
| **C** | mean unique Src/Dst pairs per window (1-5) | 3.2933 | 4.9905 | -1.6971 |
| **D** | median unique Src/Dst pairs per window | 3.0 | 5.0 | -2.0000 |
| **F** | all 5 labels identical | 99.9367% | 68.6379% | +31.2988 |
| **G** | 5th label != window majority label | 0.0146% | 7.1930% | -7.1784 |

### E. Distribution of unique Src/Dst pairs per window

| unique pairs | windows | % of windows | % random baseline |
|---|---|---|---|
| 1 | 123,832 | 7.7894% | 0.0000% |
| 2 | 287,265 | 18.0698% | 0.0000% |
| 3 | 456,069 | 28.6881% | 0.0077% |
| 4 | 443,897 | 27.9225% | 0.9394% |
| 5 | 278,686 | 17.5302% | 99.0529% |

## 3. Endpoint continuity in detail

| measurement | consecutive T=5 | random baseline |
|---|---|---|
| mean unique Src IPs (1-5) | 2.5656 | 4.9803 |
| mean unique Dst IPs (1-5) | 2.4623 | 4.1982 |
| mean unique 5-tuples (1-5) | 4.7405 | 4.9986 |
| % windows where all 5 share one 5-tuple | 0.0026 | 0.0000 |
| mean adjacent same-pair count (0-4) | 1.2850 | 0.0038 |
| % of all adjacent flow pairs sharing a Src/Dst pair | 32.1261 | 0.0958 |
| % windows fully heterogeneous (5 distinct pairs) | 17.5302 | 99.0529 |

## 4. Label continuity

| unique labels in window | windows | % of windows | % random baseline |
|---|---|---|---|
| 1 | 1,588,743 | 99.9367% | 68.6379% |
| 2 | 1,006 | 0.0633% | 31.3621% |

Label-pure: **99.9367%** (1,588,743 windows). Mixed-label: **0.0633%** (1,006). Fifth label equals the majority in **99.9854%**, disagrees in **0.0146%** (232 windows).

Majority ties are broken by ascending label string, so `fifth_is_majority` is conservative: under a tie the fifth label counts as the majority only if it also wins that tiebreak.

## 5. Temporal diagnostics

| measurement | consecutive T=5 | random baseline |
|---|---|---|
| median window span (s) | 24,022.0 | 32,421.0 |
| 95th percentile span (s) | 39,321.0 | 40,060.0 |
| maximum span (s) | 43,199.0 | 43,197.0 |
| mean span (s) | 18,271.4 | 31,030.1 |
| % windows entirely within one clock second | 4.5756 | 0.0000 |

Cumulative span distribution (consecutive arm):

| window span | % of windows at or below |
|---|---|
| same second (0 s) | 4.5756% |
| <= 1 s | 5.8360% |
| <= 10 s | 11.0985% |
| <= 60 s | 15.1743% |
| <= 300 s | 30.0356% |
| <= 3,600 s | 38.5347% |
| <= 21,600 s | 47.8589% |

The distribution is **bimodal**, so the median alone is misleading: about 30.0% of windows span 5 minutes or less while the rest stretch across most of the capture day. A window is either a tight burst of near-simultaneous flows or an arbitrary jump between distant parts of the file, with little in between. The median sits in the empty middle and describes neither mode.

`span_seconds` = max(parsed Timestamp) - min(parsed Timestamp) within the window. **Timestamp is diagnostic only and never orders rows** - the ordering key is the capture-order proxy above. The column is non-monotonic in file order and its clock is 12-hour with no AM/PM marker, so a span is a rough locality indicator, not a measured elapsed time. In particular a large span does not prove the five flows are far apart in capture time; it proves the timestamp column does not increase monotonically with row position.

## 6. Random-grouping baseline

The identical 7,948,748 usable rows were regrouped into 1,589,749 windows of 5 by a seeded permutation (seed **42**) and measured with the same code. Row content and marginal label/endpoint distributions are held fixed, so any difference isolates the effect of *consecutive adjacency*.

**This baseline is a measuring stick only.** It is not used for training, and it does not alter the project's split methodology.

| metric | consecutive | random | difference |
|---|---|---|---|
| A_pct_windows_all_five_same_src_dst_pair | 7.7894 | 0.0000 | +7.7894 |
| B_pct_windows_any_adjacent_same_src_dst_pair | 66.4380 | 0.3813 | +66.0567 |
| C_mean_unique_pairs_per_window | 3.2933 | 4.9905 | -1.6971 |
| pct_windows_fully_heterogeneous_five_distinct_pairs | 17.5302 | 99.0529 | -81.5227 |
| F_pct_windows_label_pure | 99.9367 | 68.6379 | +31.2988 |
| G_pct_windows_fifth_label_ne_majority | 0.0146 | 7.1930 | -7.1784 |

## 7. Evidence for the interpretation questions

Measured values only. This section reports what the numbers show and where they are ambiguous; **it does not select an architecture.**

**1. Do five globally consecutive capture-order flows generally belong to the same communication context?**

No, not as a rule - but they are far from unrelated. Only **7.7894%** of windows are a single (Src IP, Dst IP) conversation, and the median window holds **3.0** distinct pairs (mean **3.2933**). At the same time only **17.5302%** are fully heterogeneous, so **74.6804%** sit in between: some endpoint overlap, but more than one conversation. Against the random baseline (**4.9905** pairs per window, **99.0529%** fully heterogeneous) the consecutive grouping is clearly more coherent than chance. So the honest reading is *partial* shared context, not *one* shared context.

**2. How often do all five flows share the same Src/Dst pair?**

**7.7894%** of windows (123,832 of 1,589,749). Tightening the identity to the full 5-tuple drops this to **0.0026%**, and the mean number of distinct 5-tuples per window is **4.7405** out of 5 - so even inside an all-same-pair window the five flows are almost always different connections between the same two hosts, not one connection observed five times.

**3. How often do adjacent flows share a Src/Dst pair?**

At the window level **66.4380%** of windows contain at least one adjacent same-pair link. At the link level **32.1261%** of all 6,358,996 adjacent flow pairs share a pair, and the mean window has **1.2850** such links out of 4.

Metric B must be read against the baseline rather than on its own. Random regrouping of the same rows yields **0.3813%**, so B is not chance-driven *in this file*. But B is structurally chance-inflated whenever pair cardinality is low: it asks only whether 1 of 4 slots collides, so a file with a handful of heavy-hitter pairs can score high on B with no real locality at all. B is the weakest of the seven metrics and should not carry an argument by itself.

**4. How often are the windows completely heterogeneous?**

**17.5302%** of windows have five distinct Src/Dst pairs (278,686 windows), against **99.0529%** under random regrouping.

**5. How often does the fifth-flow label disagree with the majority?**

**0.0146%** of windows (232 of 1,589,749), against **7.1930%** under random regrouping. Label purity is **99.9367%**, so mixed-label windows are **0.0633%** (1,006 windows) in total.

**This near-perfect label purity is weak evidence for temporal context and must not be read as a model result.** It is a property of how the capture is laid out: the labels sit in long contiguous blocks in file order, so almost every window falls strictly inside one block and inherits one label for free. A high F says the label column is blocky; it does not say the five flows are one conversation. It also means a T=5 window carries almost no label transition to learn from - the boundary cases are the 1,006 mixed windows.

**6. Does the evidence support or weaken the idea that T=5 global windows provide meaningful temporal context?**

Mixed, and the two sides are worth separating.

*Supports:* every metric moves substantially away from chance in the direction of coherence - fully heterogeneous 17.53% vs 99.05%, mean pairs 3.29 vs 4.99, label purity 99.94% vs 68.64%. Consecutive rows are demonstrably not an arbitrary sample: the ordering carries real structure, and roughly 30.0% of windows are locally tight in time (span <= 300 s), with **4.5756%** landing entirely inside one clock second.

*Weakens:* the shared context is partial rather than actual. A median of 3.0 distinct conversations per window means the majority of windows mix traffic from different host pairs, so a sequence model reading five consecutive rows is mostly reading interleaved conversations, not one conversation's evolution. The strongest-looking number (label purity 99.9367%) is explained by block layout, not by flow relatedness. And the span distribution is bimodal: many windows are tight bursts while the rest straddle most of the capture day, so 'temporal locality' is not a uniform property of the windowing - it holds for some windows and not others.

Net: the evidence supports that consecutive ordering carries *some* real signal, and does not support the stronger claim that a T=5 global window is a coherent communication episode.

**7. What is the strongest argument AGAINST using global T=5 windows?**

Most windows are not one conversation. With only **7.7894%** of windows single-pair and a median of **3.0** distinct pairs, the sequence a model sees at T=5 is predominantly an interleaving of unrelated flows that happen to be adjacent in the capture file. Any 'temporal' relation it learns is then partly a property of collector write order rather than of network behaviour - and that ordering is an artefact of how CICFlowMeter emitted rows, which does not have to reproduce on another capture or another day. The bimodal span distribution sharpens this: adjacency in row position does not reliably mean adjacency in time.

The secondary argument is that label purity, the metric that looks most favourable, is the one most easily explained without any temporal mechanism.

**8. What additional experiment would most decisively resolve the uncertainty?**

Train the *same* model on the *same* rows under three grouping rules and compare held-out performance:

1. T=5 global consecutive windows (the current proposal).
2. T=5 windows grouped within a Src/Dst pair (or 5-tuple), so every window is one conversation by construction.
3. T=5 windows built by seeded random regrouping - the baseline measured here, used as a control.

Holding architecture, features, split and seed fixed makes grouping the only variable. If (1) beats (3) but not (2), the context that matters is conversational rather than positional; if (1) matches (2), global adjacency is a cheap adequate proxy; if (1) matches (3), T=5 global windows add nothing over shuffling and the temporal framing is not earning its cost. This experiment is decisive in a way no descriptive statistic can be, because it measures the quantity actually at stake - predictive value - rather than endpoint overlap.

Two cheaper checks are worth running first, and neither requires a model: repeat this analysis on the other capture days to see whether the continuity profile is stable or specific to this file (only this file carries endpoint identifiers, so a same-file proxy such as a Flow-ID-free variant would be needed), and measure how much of the measured continuity survives after the project's duplicate/conflict policy is applied.

**No architecture conclusion is drawn here.** In particular the measurements above do not establish that IP-pair windows are the right choice - they establish that global T=5 windows are not single-conversation windows, which is a different and weaker statement.

## 8. Exact definitions

- **`A`** - Percentage of windows with all_same_pair.
- **`B`** - Percentage of windows with any_adjacent_same_pair.
- **`C`** - Mean of n_unique_pairs over all windows.
- **`D`** - Median of n_unique_pairs over all windows.
- **`E`** - Percentage of windows at each n_unique_pairs value 1..5.
- **`F`** - Percentage of windows with label_pure.
- **`G`** - Percentage of windows where the 5th row's label != the majority label.
- **`all_same_pair`** - n_unique_pairs == 1.
- **`any_adjacent_same_pair`** - n_adjacent_same_pair >= 1.
- **`fifth_is_majority`** - The 5th row's label equals the window's majority label.
- **`five_tuple`** - The tuple (Src IP, Src Port, Dst IP, Dst Port, Protocol).
- **`fully_heterogeneous`** - n_unique_pairs == 5 (no two flows share a Src/Dst pair).
- **`label_pure`** - n_unique_labels == 1.
- **`majority_label`** - The label with the highest count in the window. Ties are broken by ascending label string, so the rule is deterministic; 'fifth_is_majority' is therefore a conservative measure - under a tie the fifth label counts as the majority only if it wins that alphabetical tiebreak.
- **`n_adjacent_same_pair`** - Number of adjacent positions j in {0,1,2,3} where row j and row j+1 have the same (Src IP, Dst IP) pair. Range 0..4.
- **`n_unique_dst_ip`** - Count of distinct Dst IP values among the window's 5 rows.
- **`n_unique_five_tuples`** - Count of distinct 5-tuples among the 5 rows; 1..5.
- **`n_unique_pairs`** - Count of distinct (Src IP, Dst IP) pairs among the 5 rows; 1..5.
- **`n_unique_src_ip`** - Count of distinct Src IP values among the window's 5 rows.
- **`pct_windows_with_span_at_most`** - Cumulative share of span-valid windows whose span is at or below each threshold in seconds. Reported because the span distribution is bimodal - tight bursts versus whole-day spans - so the median describes neither mode.
- **`random_baseline`** - The identical usable rows regrouped into windows of 5 by a seeded permutation, measured with the same code. Isolates the effect of consecutive adjacency: row content and marginal distributions are held fixed. Never used for training or splitting.
- **`same_second`** - span_seconds == 0 (all 5 rows fall in one clock second).
- **`span_seconds`** - max(parsed Timestamp) - min(parsed Timestamp) over the window's 5 rows, in seconds. Diagnostic only; Timestamp never orders rows. A window with any unparseable timestamp is excluded from the span statistics only.
- **`src_dst_pair`** - The tuple (Src IP, Dst IP), compared as an exact string pair.
- **`window`** - Non-overlapping group of 5 consecutive rows in the ordering sequence: window i holds sequence positions [5i, 5i+5). Trailing rows that cannot fill a window are discarded.
