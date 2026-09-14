# Stage C Verification and Freeze Record

## Scope

This record summarizes the independent adversarial audit, collision-safe correction, corrected validation, and final freeze decision for CSE-CIC-IDS2018 Stage C preprocessing. It does not claim any feature-engineering, graph, model, pruning, or predictive result.

## Original Fingerprint Defect and Initial Audit

The original duplicate classifier used deterministic 64-bit fingerprints as if matching fingerprints proved equality. A hash is suitable for locating a candidate bucket, but different feature vectors or canonical row/state keys can collide. Forced-collision probes demonstrated that distinct payloads could therefore be merged or misclassified. The independent audit verdict was **FAIL** because duplicate-group membership and the cleaning cascade were not collision-safe by construction.

## Collision-Safe Correction

The corrected implementation keeps hashing as a bounded-memory bucket-selection mechanism but confirms equality using exact data:

- Level-1 feature payloads are stored in a temporary disk-backed fixed-record store and compared exactly within matching fingerprint buckets.
- Level-2 exact-row and Level-3 temporal-state decisions use exact canonical keys associated with verified feature groups.
- A fingerprint alone is never accepted as equality proof.
- The canonical precedence remains conflicting → exact duplicate → retained temporal duplicate/unique.
- Disk-backed and chunked full-corpus processing remains intact.

Dedicated forced-collision tests cover distinct and identical feature vectors, exact rows, and temporal states, including the corpus `build_keep_index` path. They prove that a collision does not merge distinct payloads and that genuinely identical payloads still group correctly.

## Corrected Validation

The corrected suite completed with **460 passed, 0 failed, 0 skipped**.

Two complete corrected full-corpus executions independently produced 33 canonical artifacts each. Their relative file sets and every corresponding SHA-256 digest were identical:

| Comparison | Result |
| --- | ---: |
| Files in each run | 33 |
| Missing | 0 |
| Extra | 0 |
| Differing | 0 |

The collision-safe correction did not change the verified Stage C numerical results:

| Measurement | Verified value |
| --- | ---: |
| Raw rows | 16,233,002 |
| Embedded headers removed | 59 |
| Usable rows | 16,232,943 |
| Conflicting rows / groups | 423,390 / 22,373 |
| Canonical exact duplicates excluded | 213,316 |
| Temporal duplicates retained | 3,651,448 |
| Clean rows | 15,596,237 |
| T=5 windows / trailing rows | 3,119,247 / 2 |

All 10 raw CSV files remained byte-identical. Compact machine-readable evidence is stored in `results/stage_c_full_validation.json`; the multi-gigabyte generated arrays and verbose logs remain local and Git-ignored.

## Independent Re-audit and Final Status

The independent re-audit verdict was **PASS — Stage C can be frozen**.

Stage C is therefore frozen at this checkpoint. Its algorithms, cleaning and duplicate semantics, transforms, T=5 window behavior, E1/E2/E3 split behavior, tests, verified counts, and raw inputs must not be modified as part of downstream repository preparation.

The next authorized activity is only a **feature engineering / HTGraph construction design review**. Graph construction and model training have not started.
