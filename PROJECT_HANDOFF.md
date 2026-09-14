# Project Handoff — Frozen Through Stage C

## Project Goal

The project studies automated feature extraction and dynamic heterogeneous temporal graph optimization for efficient network intrusion detection using CSE-CIC-IDS2018. The verified implementation currently stops after dataset profiling and Stage C preprocessing. No graph or model result exists yet.

## Authority and Source Hierarchy

Use sources in this order when continuing the work:

1. The current task's explicit requirements and stage boundary.
2. `docs/STAGE_C_VERIFICATION.md` and `results/stage_c_full_validation.json` for the independently verified Stage C milestone.
3. Frozen implementation and tests in `src/preprocessing/`, `scripts/`, and `tests/`.
4. Compact Stage B and temporal-analysis evidence in `results/data_profile/` and `results/temporal_analysis/`.
5. The original report and extracted text as background only; their experimental performance claims are not verified authority.

If a later document conflicts with verified Stage C facts, stop and resolve the conflict before changing code or claims.

## Frozen Status

Stage C passed an independent adversarial re-audit and is frozen. Do not change its cleaning cascade, exact-comparison logic, transforms, window construction, split behavior, verified counts, or tests without a separately approved methodology change and a new audit.

## Verified Dataset Facts

| Measurement | Value |
| --- | ---: |
| Raw rows | 16,233,002 |
| Embedded headers removed | 59 |
| Usable rows | 16,232,943 |
| Conflicting rows | 423,390 |
| Conflicting groups | 22,373 |
| Canonical exact duplicates excluded | 213,316 |
| Temporal duplicates retained | 3,651,448 |
| Clean rows | 15,596,237 |
| T=5 windows | 3,119,247 |
| Trailing rows | 2 |

The raw dataset consists of 10 CSV files. All 10 remained byte-identical across verified processing and validation.

## Feature and Capture-Order Conventions

- Exactly 78 columns are model-input features.
- `Timestamp` and `Label` are not model-input features.
- Files are ordered by source-file capture date.
- Within a file, order is the synthetic 0-based `Original_Index`.
- `Original_Index` is assigned before embedded-header filtering.
- This ordering is a deterministic capture-order proxy, not guaranteed physical chronology, because the source timestamps lack AM/PM.

## T=5 Temporal Representation

A sample is a **contiguous capture-order temporal window** of five consecutive clean flow records. The target is the label of the fifth/final flow. These windows are not guaranteed to be individual network conversations. `T=5` is a reasonable engineering choice; it is not proven mathematically optimal, and its predictive value remains unmeasured.

## Canonical Duplicate Cascade

The mutually exclusive cascade runs before windowing:

1. **Conflicting:** group by the 78-feature vector. If a group has more than one label, exclude every row in that group.
2. **Exact duplicate:** among nonconflicting rows, group by `(78 features, Timestamp, Label)`. Retain the first capture-order occurrence and exclude later occurrences.
3. **Temporal duplicate:** among retained rows, flag repeated `(78 features, Label)` states at later positions but retain them.
4. **Unique:** retained rows not classified as temporal duplicates.

Fingerprints select candidate buckets only. Equality is confirmed using exact payload/canonical-key comparisons, including under forced fingerprint collisions.

## Evaluation Configurations

### E1 — Strict Global Capture-Order Multiclass Diagnostic

Global 70/10/20 split, no shuffle, no stratification. E1 is diagnostic only and is not the primary multiclass performance evaluation.

### E2 — Primary Multiclass Supervised Experiment

Group windows by final-flow class, preserve chronology within each class, split each class 70/10/20, and recombine partitions. E2 is not globally chronological. Reporting eligibility is train ≥ 100, validation ≥ 30, and test ≥ 30. E2 preprocessing is fitted on E2-train only.

### E3 — Primary Strict Global Capture-Order Binary Experiment

Use the global E1 boundaries with `Benign → 0` and every attack label `→ 1`. E1 and E3 may share train-fitted feature state because their feature windows and boundaries are identical.

## Verification Evidence

- Corrected full suite: **460 passed, 0 failed, 0 skipped**.
- Two complete corrected full-corpus runs: 33 canonical artifacts each.
- Relative file sets identical; all 33 corresponding SHA-256 values identical.
- Comparison result: missing = 0, extra = 0, differing = 0.
- All 10 raw CSV files remained byte-identical.
- Independent re-audit verdict: **PASS — Stage C can be frozen**.

Large arrays and local recovery copies are deliberately excluded from Git. Compact validation evidence is in `results/stage_c_full_validation.json`.

## Known Methodological Limitations

1. Capture order is deterministic but not guaranteed physical chronology because timestamps lack AM/PM.
2. T=5 windows are not guaranteed individual conversations.
3. Repeated temporal/network states occur across partitions. Indices do not overlap, but evaluation is not exclusively on previously unseen states.
4. E2 preserves within-class chronology rather than one global timeline.
5. The predictive value of T=5 has not been measured.
6. Original report performance claims are not verified results.
7. Graph construction, HTGraph, GATv2, HTGNN, and pruning are not implemented.

## Current Boundary and Exact Next Step

Stop at the frozen Stage C checkpoint. Do not train a model or construct a graph.

The exact next step is a **feature engineering / HTGraph construction design review**. The review must define the proposed feature and graph schema, temporal/node/edge semantics, data boundaries, evaluation interaction, memory strategy, and acceptance criteria before implementation begins.
