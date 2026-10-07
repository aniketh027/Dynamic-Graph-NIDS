# Stage E.11 first official test evaluation: preparation only

The authoritative checkout is `C:\Project\ComputerNetworks`. The owner explicitly
authorized preparation despite the stale E.9 handoff. This document does not
declare E.11 executed, passed, independently audited, or frozen. No real test
predictive arrays were opened during preparation. Git and E.10 remain unchanged.

## Reconciled E.10 inputs

Study `e10-official-20261001-v1` completed 45 unique official training runs:
27 LR-selection runs at seed 2025 and 18 selected-LR runs at seeds 2026/2027.
The selected seed-2025 run is reused, producing 27 evaluation checkpoints,
three seeds for each of nine task/family configurations. All 45 results are
PASS with COMPLETE lifecycle records. Physical artifact hashes and loaded
checkpoint provenance were checked, and the final report was independently
rebuilt in memory exactly. No aggregation command or training was executed.

The reviewed correction is pinned by its policy-file SHA256 and the reviewed
runtime/runner digests. E.11 also pins the completed E.10 report and the frozen
Stage C integrity report. It verifies the existing LR freeze without selecting
a new LR, changing any original record, or trusting incidental Git status.

| Family | E1 LR | E2 LR | E3 LR |
| --- | --- | --- | --- |
| HTGNN | 0.001 | 0.001 | 0.001 |
| Final-flow MLP | 0.001 | 0.001 | 0.001 |
| Flattened-five-flow MLP | 0.0001 | 0.0003 | 0.0001 |

E.10 records an empty test-open audit and no predictive test access. That claim
is supported by the frozen runner's lockout and validated records; it is not
independent historical operating-system telemetry. Original preprocessing and
earlier authorized integrity stages necessarily inspected test artifacts.

## Prespecified evaluation methodology

Evaluate the frozen best checkpoint for every task/family/seed exactly once,
in deterministic family, task, seed order. Each checkpoint traverses its full
test split sequentially with batch size 256, using D4 read-only memory maps,
D6 bounded logical batches, and E.2 conversion. No new preprocessing, training,
LR selection, calibration, class weighting, threshold tuning, or pruning is
performed. `T=5` remains five flow records; the fifth record supplies the label.

Models run in `eval()` and `torch.inference_mode()` with gradients disabled.
Model state is hashed before and after inference. The E.7 checkpoint loader
validates model/config/state and optimizer binding; it creates a temporary AdamW
for validation, without restoring optimizer state or executing optimizer steps.
E1/E2 retain all 15 logits. Argmax uses the frozen full vocabulary, including
ineligible classes. E3 retains one raw logit per example.

The frozen E.7 batch helper and E.9 training adapter reject test splits. E.11
therefore adds a separate test-only traversal and dense inference accessor.
It uses the same E.2 graph-major feature order: final-flow MLP reads time position
five; flattened MLP reconstructs temporal-major `[B,5,78]`. It never relabels test
data as validation or changes frozen source. No attention or pruned topology is
requested.

### E1: strict global capture-order multiclass diagnostic

The primary diagnostic is support-restricted macro F1 over classes having
positive train and test support, from the frozen availability metadata. As in
the E.7 support-restricted objective, select averaging labels but retain the
full test confusion matrix when counting false positives/negatives. Unseen
true-label rows can therefore contribute false positives to a supported class.
Do not filter predictions or recalculate argmax over supported classes.

Also report restricted macro precision/recall and weighted F1; full-test
descriptive accuracy, supported-test-class macro/weighted F1; all 15 per-class
precision/recall/F1/support; and the 15x15 confusion matrix. Each class records
train/validation/test support and its reporting status. Absent or unseen
classes are explicitly labelled. Heartbleed is absent from the corpus.

In this frozen split, **Benign alone** has both train and test support. The
restricted macro F1 is consequently Benign F1. Five attack classes are unseen
in training. These results are diagnostics of the capture-order split and
must not be presented as the primary multiclass benchmark.

### E2: primary supervised multiclass benchmark

The primary metric is eligible-class macro F1 under the unchanged Stage C rule:
train >=100, validation >=30, test >=30. There are 12 eligible classes in the
frozen metadata. The three rare corpus classes remain in the dataset and the
all-test report; Heartbleed remains explicitly absent.

For primary reporting, include test examples whose **true class** is eligible,
while preserving the original 15-way argmax. A prediction of an ineligible
class on an eligible example is still an error. Average precision/recall/F1
over the 12 eligible labels. This implements Stage C's eligible-class evaluation
set, rather than suppressing logits or discarding incorrect predictions.

Report primary macro precision/recall/F1, accuracy, and weighted F1, plus full-test
descriptive metrics, all 15 per-class supports/scores, the eligibility policy,
and the full 15x15 confusion matrix. Rare-class scores are descriptive and do
not enter the primary macro average. E2 preserves chronology within each class
and is not one globally chronological split.

### E3: primary binary benchmark

The primary metric is PR-AUC defined as **non-interpolated average precision**
with Attack=1, calculated from raw logits. This matches the E.7/E.10 selection
definition. It is not trapezoidal PR-curve integration; see the official
[scikit-learn average-precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Also report ROC-AUC, Attack prevalence, accuracy, Attack precision/recall/F1,
specificity, false-positive rate, balanced accuracy, MCC, both class supports,
and the 2x2 confusion matrix. Thresholded metrics use the prespecified natural
decision threshold **logit >=0 (sigmoid probability >=0.5)**. This is an E.11
reporting convention chosen before evaluation, not an E.10-selected threshold
or a test-tuned operating point. No threshold option exists in the CLI.

Undefined ranking metrics on a single-class split are JSON null with a reason,
not fabricated zero scores. Zero predicted-positive precision and zero-denominator
F1/MCC use an explicit zero convention. Missing-class recall/specificity are null.

### Seeds and memory

Publish every seed's complete metrics. For each of the nine configurations,
report scalar-metric means and sample standard deviations (`ddof=1`) across exactly
2025, 2026, 2027. This is descriptive seed variation, not an ensemble, pooled
prediction result, significance test, or confidence interval.

Multiclass confusion accumulation is constant memory. Exact binary AP/ROC-AUC
uses temporary disk-backed scalar scores and targets, never full feature or
graph collections. Ranking allocates O(N) scalar working memory in scikit-learn;
the fixed frozen test size bounds it. Score maps are removed after each checkpoint.

## Provenance and one-time safeguards

`audit_e10` reads results, checkpoints, compact split/scaler metadata and source
only. A process-wide audit hook rejects predictive `.npy` access during preparation.
It validates canonical metadata, all bound file hashes, all 45 checkpoints, the
27 selected identities, cross-phase reviewed provenance, frozen upstream source,
and split/scaler evidence. Stored predictive-array hashes are copied from frozen
Stage C evidence during preparation, **not claimed to have been rehashed**.

The owner-only path requires an explicit acknowledgement and creates an immutable
`logs/stage_e11/<E10-study>/test_consumption.json` using exclusive creation before
any real test file is read, even for hashing. This marker covers all E.11 run IDs
for the E.10 study and serializes concurrent invocation. An existing marker blocks
another evaluation. Direct traversal requires the active owner reservation context.
The guard rejects overwriting, deleting or renaming the marker. No resume, retry,
overwrite, threshold, LR, weighting, pruning,
or subset-evaluation option is exposed. An interrupted/failed run retains the
marker and any completed results; re-execution requires a separately reviewed
owner recovery decision. Do not delete the marker to retry.

The owner path hashes real test files before and after inference, compares supports
with frozen metadata, checks every sequential source index and complete coverage,
rehashes E.10 artifacts, metadata, and execution source, and publishes PASS only
after postflight succeeds. Per-checkpoint results bind checkpoint SHA256, exact
checkpoint metadata, E.10 result hash, configuration/seed/LR and traversal hash.
The manifest binds the complete E.10 audit, source digests, environment and fixed
methodology. Canonical JSON hashes are distinct from physical report-file hashes.

Writes are restricted to new E.11 logs/results; source, datasets, handoff, and
E.10 are protected by the process audit guard. The guard is an accidental-use
boundary for this process, not a security sandbox against arbitrary Python code
execution, malicious native extensions or external filesystem mutation. The
study marker enforces one-time use within this authoritative checkout; copies
of the repository or deliberate marker removal are outside that boundary.

## Commands: preparation and owner execution

Use the existing isolated `.venv`; install nothing. From the authoritative root,
the safe audit command reads no predictive arrays and writes no artifacts:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B scripts\run_stage_e11_test_evaluation.py audit --official-study-id e10-official-20261001-v1
```

**OWNER ONLY, later. This command has not been executed during preparation:**

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B scripts\run_stage_e11_test_evaluation.py evaluate --official-study-id e10-official-20261001-v1 --run-id e11-official-20261006-v1 --device cuda --owner-confirmed-manual-test-evaluation
```

Expected outputs, relative to the authoritative root:

```text
logs/stage_e11/e10-official-20261001-v1/test_consumption.json
logs/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/manifest.json
logs/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/status.json
results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/<E10-candidate-id>.json  (27)
results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/stage_e11_report.json
results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/stage_e11_report.sha256
```

The run ID is an artifact identity, not a scheduler or authorization to run later.
The owner may choose a fresh date-labelled ID; it does not bypass the study marker.

## Handoff and next boundary

Leave `PROJECT_HANDOFF.md` unchanged during preparation. Update it separately
after owner review of the E.10 reconciliation, preserving the distinction between
completed owner-executed E.10 and an independently audited/frozen checkpoint.
The update should identify the authoritative checkout, reviewed dirty-source
provenance, 45 completed runs, frozen LR plan, 27 final checkpoints, test lockout,
and the E.11 preparation boundary. Its bytes are not checkpoint input, but a
separate scoped update keeps this task's E.10-preservation evidence exact.

The next safe boundary is owner review and manual execution of this prepared
first E.11 evaluation. No downstream pruning, retraining, tuning, or release work
is authorized. A complete-suite run is unsuitable for this preparation task:
existing tests can train synthetic models or open real test arrays. Run only
the reviewed unit/synthetic E.11 and result/provenance regression tests.

## Preparation validation, 2026-10-06

- 69 E.11 unit/synthetic tests passed, including all nine model/task checkpoint
  inference pairs, actual D4/D6/E.2 synthetic traversal with a final partial batch,
  independently checked metric definitions, corrupted provenance, ordered coverage,
  immutable test reservation, owner acknowledgement, and failure/retry lockout.
- 55 existing E.10 result/provenance tests and 3 E.9 source/artifact-provenance
  tests passed. The combined 127-test invocation ran inside the authoritative
  checkout's predictive-array lockout and write guard.
- Synthetic checkpoint fixtures construct AdamW state directly without training,
  backward passes, or optimizer steps. The actual checkpoint loader is exercised.
- Existing PyG/Python annotation deprecation warnings remain; validation had no
  test failures or skips. The full suite was intentionally not executed under
  this task's unit/synthetic-only restriction.
- A read-only E.11 audit verified all 45 actual E.10 checkpoints and the compact
  frozen metadata. Real predictive test-array hashes remain unverified during
  preparation and will be checked only by the owner execution path.
