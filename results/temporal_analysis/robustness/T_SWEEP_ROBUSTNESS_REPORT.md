# T=3 / T=5 / T=10 Window Continuity - Robustness Analysis

Generated 2026-09-09T11:34:36.026054+00:00 | measurement digest `b733973a64bcc842` | elapsed 373.1s | seed **42**

**Analysis only.** Every number was measured from the raw CSVs. No preprocessing, loader, windowing or architecture code was changed, no raw file was touched, and no model input was produced. This report presents evidence and does not select a representation.

## 1. Which files carry the identifier columns (Task 1)

Measured from each file's own header line, not inferred from the file name and not assumed from the 02-20 layout.

| file | columns | variant | Src IP | Dst IP | Src Port | Dst Port | Protocol | full 5-tuple |
|---|---|---|---|---|---|---|---|---|
| 02-14-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 02-15-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 02-16-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 02-20-2018.csv | 84 | extended_84 | yes | yes | yes | yes | yes | yes |
| 02-21-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 02-22-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 02-23-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 02-28-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 03-01-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |
| 03-02-2018.csv | 80 | standard_80 | **no** | **no** | **no** | yes | yes | **no** |

**1 of 10 files carry the full identifier set** (02-20-2018.csv). The remaining 9 file(s) have `Dst Port` and `Protocol` but no `Src IP`, `Dst IP` or `Src Port`, so **metrics A-E as defined in the first experiment cannot be computed on them at all.** This is a property of the corpus, not a limitation of the analysis: the CSE-CIC-IDS2018 release ships one 84-column file and nine 80-column files. The parquet copies carry 78 columns with no identifiers and no `Timestamp`, so they are not usable here either.

Task 2 therefore has an honest scope limit. Rather than stopping there, every file is also measured on a deliberately weaker identity - `Dst Port`, `Protocol` - which exists everywhere. On 02-20 both identities are measured on the same rows, which calibrates how much the proxy overstates continuity (section 5). Without that calibration the nine-file numbers would be uninterpretable.

## 2. What was measured

| file | raw rows | embedded headers | usable rows | rows analyzed | unparsed timestamps | identity available |
|---|---|---|---|---|---|---|
| 02-14-2018.csv | 1,048,575 | 0 | 1,048,575 | 1,048,575 | 0 | proxy only |
| 02-15-2018.csv | 1,048,575 | 0 | 1,048,575 | 1,048,575 | 0 | proxy only |
| 02-16-2018.csv | 1,048,575 | 1 | 1,048,574 | 1,048,574 | 0 | proxy only |
| 02-20-2018.csv | 7,948,748 | 0 | 7,948,748 | 7,948,748 | 0 | true + proxy |
| 02-21-2018.csv | 1,048,575 | 0 | 1,048,575 | 1,048,575 | 0 | proxy only |
| 02-22-2018.csv | 1,048,575 | 0 | 1,048,575 | 1,048,575 | 0 | proxy only |
| 02-23-2018.csv | 1,048,575 | 0 | 1,048,575 | 1,048,575 | 0 | proxy only |
| 02-28-2018.csv | 613,104 | 33 | 613,071 | 613,071 | 0 | proxy only |
| 03-01-2018.csv | 331,125 | 25 | 331,100 | 331,100 | 0 | proxy only |
| 03-02-2018.csv | 1,048,575 | 0 | 1,048,575 | 1,048,575 | 0 | proxy only |

**Ordering rule** (unchanged from the first experiment): deterministic capture-order proxy: (source file capture date, Original_Index). Within one file this is the file's own data-row order; no sort is applied and Timestamp is never an ordering key.

**Seed**: 42, identical to the first experiment. Window sizes swept: 3, 5, 10. Metric definitions generalise the T=5 set by substituting T for 5 and 'final flow' for 'fifth flow'; every other definition is character-for-character the same rule.

## 3. The generalised code reproduces the published T=5 result

This sweep uses new code, so it is checked against the frozen `continuity_summary.json` rather than assumed to agree. A mismatch would mean a definition drifted, which would invalidate every cross-T comparison below.

| metric | this run | published T=5 | abs difference | match |
|---|---|---|---|---|
| n_windows | 1,589,749 | 1,589,749 | 0 | yes |
| n_rows_analyzed | 7,948,748 | 7,948,748 | 0 | yes |
| A_pct_windows_all_five_same_src_dst_pair | 7.789406 | 7.789406 | 0.00e+00 | yes |
| B_pct_windows_any_adjacent_same_src_dst_pair | 66.437972 | 66.437972 | 0.00e+00 | yes |
| C_mean_unique_pairs_per_window | 3.293342 | 3.293342 | 0.00e+00 | yes |
| D_median_unique_pairs_per_window | 3.000000 | 3.000000 | 0.00e+00 | yes |
| pct_windows_fully_heterogeneous_five_distinct_pairs | 17.530189 | 17.530189 | 0.00e+00 | yes |
| pct_adjacent_flow_pairs_sharing_src_dst | 32.126078 | 32.126078 | 0.00e+00 | yes |
| F_pct_windows_label_pure | 99.936720 | 99.936720 | 0.00e+00 | yes |
| G_pct_windows_final_label_ne_majority | 0.014593 | 0.014593 | 0.00e+00 | yes |

All metrics match: **True**.

## 4. Window size sweep on the true endpoint identity (Task 3)

`02-20-2018.csv`, identity = `Src IP`, `Dst IP`. Same rows, same ordering, same seed at every T.

| metric | T=3 | T=5 | T=10 |
|---|---|---|---|
| windows | 2,649,582 | 1,589,749 | 794,874 |
| trailing rows discarded | 2 | 3 | 8 |
| **A** all T share one pair (%) | 17.5613 | 7.7894 | 2.2981 |
| **B** >=1 adjacent link shares a pair (%) | 46.6931 | 66.4380 | 86.5995 |
| B random baseline (%) | 0.1959 | 0.3813 | 0.8593 |
| adjacent *links* sharing a pair (%) | 32.1272 | 32.1261 | 32.1316 |
| adjacent links, random baseline (%) | 0.0983 | 0.0958 | 0.0962 |
| **C** mean unique pairs (1..T) | 2.2626 | 3.2933 | 5.5963 |
| C / T (scale-free) | 0.7542 | 0.6587 | 0.5596 |
| **D** median unique pairs | 2.0 | 3.0 | 6.0 |
| fully heterogeneous, T distinct pairs (%) | 43.8231 | 17.5302 | 2.4267 |
| **F** label-pure (%) | 99.9598 | 99.9367 | 99.8834 |
| **G** final label != majority (%) | 0.0124 | 0.0146 | 0.0152 |
| median span (s) | 7,691.0 | 24,022.0 | 31,928.0 |
| p95 span (s) | 36,845.0 | 39,321.0 | 41,252.0 |
| span <= 300 s (%) | 33.6853 | 30.0356 | 24.5513 |
| entirely within one clock second (%) | 9.0353 | 4.5756 | 2.1140 |

**Metric B is not comparable across window sizes and the table shows why.** B asks whether *any* of T-1 adjacent links collides, so it rises with T mechanically, even when per-link continuity is unchanged. The adjacent-*links* row is the version that can be compared across T, because its denominator grows with T. Reading B alone across this table would manufacture a trend that the link-level measure does not show.

Likewise **A and C cannot be compared across T at face value**: A must fall as T grows (more rows must agree) and C must rise (more rows can differ). The `C / T` row normalises for that.

## 5. Calibrating the cross-file proxy

Both identities measured on the same 02-20-2018.csv rows with the same code, at T=5.

| metric | true (Src IP, Dst IP) | proxy (Dst Port, Protocol) | proxy - true |
|---|---|---|---|
| A_pct_windows_all_same_key | 7.7894 | 18.7270 | +10.9376 |
| B_pct_windows_any_adjacent_same_key | 66.4380 | 82.3466 | +15.9086 |
| pct_adjacent_links_sharing_key | 32.1261 | 47.1047 | +14.9787 |
| C_mean_unique_keys_per_window | 3.2933 | 2.5746 | -0.7187 |
| pct_windows_fully_heterogeneous | 17.5302 | 4.1282 | -13.4020 |

All five rows point the same way even though two of the deltas are negative: fewer unique keys and fewer fully-heterogeneous windows both mean *more* continuity, so a negative delta on those two rows is the proxy claiming more continuity, not less.

The proxy reports substantially more continuity than the endpoint identity does, which is expected: many distinct host pairs talk to the same destination port, so the proxy merges conversations that the true identity separates. **Proxy values are an upper bound on endpoint continuity and must never be quoted as endpoint continuity.** Their only legitimate use is comparing one file's position against another's, which is exactly what section 6 does.

## 6. Cross-file comparison (Tasks 2, 4, 5)

Full comparison table across every (T, file, identity) combination is in `window_size_comparison.csv`. The proxy-identity rows at T=5 are reproduced here.

| T | file | windows | same-key % | adjacent-key % | adj. links % | mean unique keys | label-pure % | final != majority % |
|---|---|---|---|---|---|---|---|---|
| 5 | 02-14-2018.csv | 209,715 | 42.5544 | 88.7657 | 63.5685 | 2.0768 | 96.8920 | 1.0047 |
| 5 | 02-15-2018.csv | 209,715 | 17.4108 | 83.9849 | 47.6799 | 2.5760 | 98.1122 | 0.6757 |
| 5 | 02-16-2018.csv | 209,714 | 24.2220 | 49.8641 | 33.8813 | 3.0339 | 33.8013 | 17.1305 |
| 5 | 02-20-2018.csv *(reference)* | 1,589,749 | 18.7270 | 82.3466 | 47.1047 | 2.5746 | 99.9367 | 0.0146 |
| 5 | 02-21-2018.csv | 209,715 | 32.9280 | 76.0370 | 49.3554 | 2.6954 | 34.3829 | 23.5815 |
| 5 | 02-22-2018.csv | 209,715 | 13.2342 | 76.2587 | 41.4557 | 2.8317 | 99.9375 | 0.0272 |
| 5 | 02-23-2018.csv | 209,715 | 13.0472 | 74.7343 | 40.5308 | 2.8734 | 99.9585 | 0.0110 |
| 5 | 02-28-2018.csv | 122,614 | 14.1411 | 75.7124 | 42.3630 | 2.8186 | 80.3685 | 5.7897 |
| 5 | 03-01-2018.csv | 66,220 | 17.1368 | 70.5950 | 41.1666 | 3.0008 | 99.9955 | 0.0030 |
| 5 | 03-02-2018.csv | 209,715 | 32.3382 | 83.3598 | 55.3567 | 2.3292 | 96.8228 | 0.6471 |

**Label purity is not a stable property of T=5 windowing - it ranges from 33.8013% (`02-16-2018.csv`) to 99.9955% (`03-01-2018.csv`) across the corpus**, with 3 of 10 files below 90%. The first experiment attributed 02-20's 99.94% to long contiguous label blocks in capture order rather than to flow relatedness, and this spread is direct cross-file support for that reading: the same window construction on the same kind of data yields very different purity depending only on how each capture's labels happen to be laid out. A high F measures block layout. It is the metric most likely to be mistaken for a result, and 02-20 sits at the favourable end of it.

### Is the 02-20 T=5 result representative? (Task 5)

Position of `02-20-2018.csv` within the 10-file proxy-identity distribution at T=5. A z-score is computed against the *other* files only, so the reference file does not inflate its own baseline. 'Unusually high/low' means |z| >= 2.

| metric | 02-20 value | other files median | other files range | z vs others | rank | verdict |
|---|---|---|---|---|---|---|
| A_pct_windows_all_same_key | 18.7270 | 17.4108 | 13.0472 - 42.5544 | -0.40 | 5 of 10 | within the spread of the other files |
| B_pct_windows_any_adjacent_same_key | 82.3466 | 76.0370 | 49.8641 - 88.7657 | +0.62 | 4 of 10 | within the spread of the other files |
| pct_adjacent_links_sharing_key | 47.1047 | 42.3630 | 33.8813 - 63.5685 | +0.11 | 5 of 10 | within the spread of the other files |
| C_mean_unique_keys_per_window | 2.5746 | 2.8186 | 2.0768 - 3.0339 | -0.37 | 8 of 10 | within the spread of the other files |
| pct_windows_fully_heterogeneous | 4.1282 | 10.2239 | 2.5902 - 18.8644 | -1.02 | 8 of 10 | within the spread of the other files |
| F_pct_windows_label_pure | 99.9367 | 96.8920 | 33.8013 - 99.9955 | +0.63 | 4 of 10 | within the spread of the other files |
| G_pct_windows_final_label_ne_majority | 0.0146 | 0.6757 | 0.0030 - 23.5815 | -0.62 | 8 of 10 | within the spread of the other files |

## 7. Is T=5 a reasonable trade-off? (Task 6)

The cost columns are **arithmetic on the measured window counts**, not measured model cost: they assume one graph per window with T flow nodes and count the maximum intra-window edges that implies. No model was built or timed.

| T | windows | windows vs T=5 | nodes/window | max edges/window | total max intra-window edges | edges vs T=5 | trailing rows lost |
|---|---|---|---|---|---|---|---|
| 3 | 2,649,582 | 1.67x | 3 | 3 | 7,948,746 | 0.50x | 2 |
| 5 | 1,589,749 | 1.00x | 5 | 10 | 15,897,490 | 1.00x | 3 |
| 10 | 794,874 | 0.50x | 10 | 45 | 35,769,330 | 2.25x | 8 |

Four things the sweep actually shows, taken one at a time.

**Continuity per link is close to flat in T.** The comparable adjacency measure - share of adjacent links whose two flows share a pair - reads 32.13% at T=3, 32.13% at T=5 and 32.13% at T=10. Growing the window does not buy denser local continuity; it mostly adds links of about the same quality. The apparent climb in metric B (46.69% to 86.60%) is the saturation artefact described in section 4, not an improvement.

**Single-conversation windows get rarer as T grows.** Metric A falls from 17.5613% at T=3 to 7.7894% at T=5 and 2.2981% at T=10. Even at T=3 the large majority of windows already hold more than one conversation, so no value of T in this range makes a global window a single conversation. Shrinking T buys a modest gain in purity, not a qualitative change of character.

**Label purity is high at every T and stays uninformative.** 99.9598% at T=3, 99.9367% at T=5, 99.8834% at T=10. The first experiment traced this to long contiguous label blocks in capture order, and the sweep is consistent with that reading: purity degrades only slowly with T, which is what a blocky label column produces. A high F at any T is evidence about file layout, not about flow relatedness.

**Cost is where T actually bites.** Going T=5 to T=10 halves the window count (1,589,749 to 794,874) but raises maximum intra-window edges by 2.25x, since a fully connected window scales as T(T-1)/2. Going T=5 to T=3 cuts edge count to 0.50x of T=5 while raising window count to 1.67x. Trailing-row loss is negligible at all three (2, 3 and 8 rows) and is not a reason to prefer any of them.

**Reading.** On these measurements T=5 is a defensible middle choice rather than an optimum: it sits between T=3 and T=10 on every continuity axis, and the axes themselves barely move, so no value in 3-10 is strongly favoured by continuity. What separates them is cost and implementation complexity, where T=5 is unremarkable in both directions - materially cheaper than T=10 per window, without T=3's very short context. The measurements do not identify a continuity reason to change T, and they equally do not show that T=5 is optimal. Choosing the highest-continuity T would mean choosing T=3, and that would be optimising a metric that barely differs while giving up context length - which is why continuity score alone should not drive this.

## 8. Challenging the global-window assumption (Task 7)

*If global windows are still semantically weak, what evidence would justify changing to another representation?*

State the weakness precisely first, because it is narrower than it sounds.

At T=5 on the true identity, **7.7894%** of windows are one conversation and the median window holds **3.0** distinct pairs. That is the weakness: a global window is a *slice of the capture stream*, not a *conversation*. It is not the stronger claim that the slice is meaningless - the first experiment measured every metric moving far from its random baseline, and this sweep shows that holds at T=3 and T=10 too.

So the assumption under challenge is specifically: *that a T-row slice of capture order is a useful unit of temporal context for intrusion detection.* Four kinds of evidence would justify abandoning it, and none of them is in hand yet:

1. **A controlled grouping comparison that global windows lose.** Same model, same rows, same features, same split, same seed; only the grouping rule varies across global-consecutive, within-conversation, and seeded random. If global-consecutive lands near the random control on held-out performance, the positional framing is not earning its cost and should go. If it lands near within-conversation grouping, it is an adequate cheap proxy. This is the decisive test and it requires training, which is out of scope here.
2. **Demonstrated dependence on collector write order.** The ordering is an artefact of how CICFlowMeter emitted rows. If a model trained on one day's global windows degrades sharply on another day while a conversation-grouped model does not, the global window was encoding emission order rather than network behaviour. Note this project can only run that comparison on one day at the endpoint level, which is itself a finding - see the uncertainty list below.
3. **Continuity that collapses under the duplicate/conflict policy.** The measured continuity has not yet been re-measured after the project's de-duplication rules are applied. If a large share of the adjacent same-pair links turn out to be duplicate or near-duplicate rows, then part of the measured locality is a data-quality artefact rather than temporal structure. This is cheap to test and requires no model.
4. **A representation whose cost is justified by the gain.** Conversation-based grouping is not free: it needs the endpoint identifiers, and **only 1 of the 10 files in this corpus has them**. A representation that cannot be built on nine tenths of the available data has an evidential burden of its own, and 'higher continuity by construction' does not discharge it - a conversation-grouped window has A=100% by definition, which makes that metric uninformative about whether it detects intrusions better.

**No representation is recommended here.** In particular these measurements do not make the case for IP-pair or 5-tuple grouping. They establish that global T-windows are not conversation windows at any T in 3-10, which is a statement about what the windows *are*, not about which representation detects attacks better. The identifier inventory in section 1 is a real obstacle to conversation-based grouping on this corpus and counts as evidence against a wholesale switch, independently of continuity.

## 9. Conclusion (Task 8)

### 9.1 What was measured

- **1 of 10** corpus files carry the endpoint identifiers. Metrics A-E are computable on that one file only.
- On the true identity, single-conversation windows fall with T: **17.5613%** (T=3), **7.7894%** (T=5), **2.2981%** (T=10).
- Per-link adjacency is near-flat in T: **32.1272%**, **32.1261%**, **32.1316%** - against random baselines of **0.0983%**, **0.0958%**, **0.0962%**.
- Label purity stays high at every T: **99.9598%**, **99.9367%**, **99.8834%**.
- On the cross-file proxy identity at T=5, 02-20 sits **within the spread of the other 9 files on 7 of 7 metrics**.
- Across the corpus, T=5 label purity ranges from **33.8013%** to **99.9955%** - it is a property of each file's label layout, not of the windowing.

### 9.2 Semantic interpretation

These are readings of the numbers above, not further measurements. A global T-window in this corpus is a *contiguous slice of the capture stream* whose flows overlap partially in endpoints. It is measurably more coherent than a random regrouping of the same rows, at every T tested. It is not a single conversation at any T tested, and the gap is large rather than marginal. The near-flat per-link adjacency across T=3 to T=10 reads as local continuity being a property of the capture ordering itself rather than something a longer window accumulates - which in turn means the choice of T in this range is not a continuity decision. High label purity is best read as a description of how the label column is laid out in file order.

### 9.3 Remaining uncertainty

- **Cross-file generalisation of the endpoint metrics is untested and, on this corpus, untestable.** Nine files lack the identifiers. The proxy comparison places 02-20 inside the others' spread on a service-level key, which is weaker evidence than a direct replication and is not a substitute for it.
- **Predictive value is entirely unmeasured.** Nothing here says whether T=5 global windows help a detector. Every metric in this report is descriptive; the grouping comparison that would settle it needs training.
- **The duplicate/conflict policy has not been applied.** Some measured adjacency may be duplicate rows rather than temporal structure. Unknown in size until measured.
- **Timestamp remains diagnostic only.** The 12-hour clock has no AM/PM marker and was not reconstructed, so span figures indicate locality roughly and cannot be read as elapsed time. The bimodal span distribution means row adjacency and time adjacency coincide for some windows and not others.
- **Only T in {3, 5, 10} was tested,** on one file, at one seed for the baseline. The near-flatness observed within that range is not evidence about T outside it.

**Net.** The measurements neither justify changing T away from 5 nor establish that 5 is optimal; within 3-10 the continuity axes barely move, so cost and complexity are the discriminating factors and T=5 is a reasonable middle. The measurements do weaken the semantic claim that a global T-window captures one communication context - at every T tested it does not. Whether that weakness matters for detection performance is not answered by any number in this report.

## 10. Exact definitions

- **`A_pct_windows_all_same_key`** - Percentage of windows whose T rows all share one identity value (n_unique_keys == 1). Generalises metric A from the T=5 experiment.
- **`B_pct_windows_any_adjacent_same_key`** - Percentage of windows with at least one adjacent position j in {0..T-2} where row j and row j+1 share the identity value. Generalises metric B. Note B is structurally easier to satisfy as T grows, because it asks whether any 1 of T-1 links collides.
- **`C_mean_unique_keys_per_window`** - Mean of n_unique_keys over all windows; range 1..T.
- **`D_median_unique_keys_per_window`** - Median of n_unique_keys over all windows.
- **`F_pct_windows_label_pure`** - Percentage of windows whose T labels are identical.
- **`G_pct_windows_final_label_ne_majority`** - Percentage of windows where the final (T-th) row's label differs from the window's majority label. Ties are broken by ascending label string, so the measure is deterministic and conservative.
- **`T`** - Window size: the number of consecutive rows per non-overlapping window.
- **`identity`** - 'src_dst_ip_pair' = the exact (Src IP, Dst IP) pair, available only in the 84-column variant. 'dst_port_protocol' = the (Dst Port, Protocol) pair, available in every file. The proxy is a service-level key, not a conversation key: distinct host pairs talking to the same port collapse onto one proxy value, so proxy continuity is an upper bound on real endpoint continuity, never a substitute for it.
- **`pct_adjacent_links_sharing_key`** - Share of all n_windows*(T-1) adjacent links whose two rows share the identity value. Unlike B this does not saturate as T grows, so it is the adjacency measure that can be compared across window sizes.
- **`pct_windows_fully_heterogeneous`** - Percentage of windows with n_unique_keys == T, i.e. no two rows in the window share the identity value. This is the T-generalisation of the '5 distinct pairs' figure.
- **`pct_windows_with_span_at_most`** - Cumulative share of span-valid windows at or below each threshold in seconds. Reported because the span distribution is bimodal, so a median describes neither mode.
- **`random_baseline`** - The identical rows regrouped into windows of T by a seeded permutation and measured with the same code. Row content and marginal distributions are held fixed, so a difference isolates the effect of consecutive adjacency. Never used for training or splitting.
- **`span_seconds`** - max(parsed Timestamp) - min(parsed Timestamp) over the window's T rows, in seconds. Diagnostic only: Timestamp never orders rows. A window containing an unparseable timestamp is excluded from span statistics only.
- **`window`** - Non-overlapping group of T consecutive rows in the deterministic capture-order proxy: window i holds sequence positions [T*i, T*i+T). Trailing rows that cannot fill a window are discarded.
