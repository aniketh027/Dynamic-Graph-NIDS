# E.11 dtype failure audit and reviewed owner continuation

This records a harness defect after the owner's first official attempt. It
does not declare the test set unused, clear its consumption state, or authorize
Codex to evaluate test data. The original E.10 and E.11 artifacts remain exact.

## Audited failure

Original run: `e11-official-20261006-v1`, E.10 study
`e10-official-20261001-v1`. Status is FAILED with six completed checkpoints and
the error `invalid frozen output shape or nonfinite logits`.

The next candidate is checkpoint 7:

```text
e10-htgnn-e3-lr1e-3-s2025-b256-b0ed07a45fa6
checkpoints/stage_e10/e10-official-20261001-v1/lr_selection/e3/htgnn/e10-htgnn-e3-lr1e-3-s2025-b256-b0ed07a45fa6/best.pt
SHA256: a26bfded6f661e5d73b673ddff7fb32e23d6e58b82a8ce071fe56a00e1f62e39
```

E.11 required `targets.dtype == torch.long` for every task. The frozen E.2
adapter intentionally emits E3 labels as `float32[B]`, as required by the E.7
binary loss contract. Thus a legitimate first binary batch is always rejected.
The broad exception text obscured which clause failed.

The actual frozen HTGNN E3 checkpoint was loaded with its full hash/config/state
binding and forwarded on synthetic batches of sizes 1, 2 and 256 on CPU and CUDA.
It returns finite `float32[B]` logits. Shape and checkpoint binding are correct;
the target-dtype check is the deterministic defect. No original failing test
logits were recorded, so their finiteness cannot independently be reconstructed
without accessing the test partition. The audit does not make that claim.

The evaluator correction changes only target validation: float32 for E3, long
for E1/E2, and gives separate shape/finiteness/target diagnostics. The frozen model
and adapters are unchanged. Neither logits nor labels are reshaped or cast. `[B]`
is the canonical frozen binary output; `[B,1]` is the internal linear-head output
that frozen forward already squeezes. Scalar, extra-axis, multiclass, wrong-batch,
wrong-dtype and NaN/Inf outputs remain rejected. Binary labels must still be 0/1.

The original synthetic inference fixture incorrectly assigned long labels to E3,
masking the defect. It now uses the task-specific dtype, and an additional end-to-end
regression runs the actual E.2 adapter on synthetic data into all three E3 families.

## Original evidence preservation

Six persisted results cover HTGNN E1 and E2, seeds 2025/2026/2027. Their canonical
JSON hashes, physical file hashes, checkpoint hashes/metadata, configurations,
full sample counts, sequential traversal digests and confusion-derived metrics
were independently verified without predictive arrays. E1 has 623,851 examples
per result; E2 has 623,866. Their model-state-unchanged evidence is true.

Checkpoint 7 has no persisted result or metric. No scratch score directory remains.
The dtype check occurs before any score is assigned or a batch is counted, so the
failed checkpoint needs to restart from its first sample. It may already have
performed a first-batch forward; this is an interrupted attempt, not an untouched
test set. There is no final E.11 PASS report.

The original study marker remains `CONSUMED_OR_RESERVED`, binds the original run,
E.10 report and original execution-source hashes, and has physical SHA256:

```text
f1b528868e61c9650acb75692ae933bab1552aedb8b7fbffb6c53a25b58b0a0a
```

It is not a completion flag and must never be deleted, edited, replaced, or
recreated. The original `evaluate` command continues to reject this study.

## Narrow recovery procedure

A separately acknowledged owner continuation is implemented only for this exact
audited failure. It pins the original marker, original manifest/status, and all
six result-file hashes. Unknown failure states, changed files, extra original
outputs/scratch state, changed frozen source/E.10/split evidence, or changed
original environment/library versions fail before any test file is read.

The corrected evaluator digest is pinned in the separate recovery module. Original
model, adapter, metric and CLI sources remain required to match the failed run;
only the dtype validator and its regression tests may differ from original E.11
source. The recovery manifest records both source identities and the immutable
original artifacts. No old manifest is normalized to the corrected code.

Recovery reuses results 1–6 in place and evaluates only checkpoints 7–27 in their
original order. Checkpoint 7 restarts from its first sample, with no fake partial
resume. The owner path hashes test files before/after the continuation, checks
supports and sequential coverage, and revalidates original evidence and source.
It does not re-evaluate checkpoints 1–6, train, tune, prune, rank new LRs, select a
seed, or change metrics. Re-reading test files for owner-only integrity checks
does not produce another inference result for the completed checkpoints.

An atomically created new recovery directory reserves this single continuation;
its `recovery_authorization.json` binds the original consumed marker instead of
granting a fresh unused-test identity. Existing recovery paths block another
automatic attempt. This directory and any partial continuation outputs remain if
the continuation fails; another recovery then requires separate review. Original
FAILED status and results stay unchanged even if recovery later succeeds.

Read-only audit command, safe for preparation:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B scripts\recover_stage_e11_test_evaluation.py audit
```

**OWNER ONLY, later, after reviewing this correction and continuation. This command
was not executed against the authoritative checkout during diagnosis:**

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B scripts\recover_stage_e11_test_evaluation.py recover --owner-confirmed-reviewed-failure-recovery
```

It is fixed to the reviewed study/run, CUDA device, six-result prefix, and recovery
ID `e11-recovery-20261006-v1`. There is no user-selected start index, replacement
marker, force switch, overwrite, or generic resume option.

New owner-created paths:

```text
logs/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/recovery_authorization.json
logs/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/status.json
results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/<candidate-id>.json  (21)
results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/stage_e11_report.json
results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/stage_e11_report.sha256
```

The recovered report combines six immutable earlier results with 21 continued
results and explicitly records the failure and source correction. It is an
interrupted-and-recovered first official evaluation, not an uninterrupted run.
The change is a contract repair, with no test-informed methodological selection.
No real-test finiteness claim is made before owner continuation. Failure on
nonfinite logits under the corrected validator remains a separate hard failure.

No existing E.11 artifact, E.10 artifact, checkpoint, dataset, handoff or Git state
was changed during this failure audit/correction. A predictive-array audit guard
covered all Python diagnosis/tests in the authoritative checkout; synthetic arrays
lived in temporary fixture directories only.

## Correction validation

The combined guarded run passed 190 tests: 107 E.11 evaluator tests, 25 recovery
tests, 55 frozen E.10 result/provenance regressions and three E.9 provenance
regressions. It recorded zero real predictive-array open attempts and zero writes
to the original E.10/E.11 artifacts. No optimizer step or training was executed.
PyG/Python annotation deprecation warnings remain. The full project suite was not
run because this task permits unit/synthetic validation only.

Tests cover canonical E3 outputs at batch sizes 1, 2, 243 and 256; invalid scalar,
extra-axis/broadcast and multiclass outputs; NaN/Inf rejection; invalid/nonbinary
labels; unchanged E1/E2 contracts; actual synthetic E.2 float32 targets entering
all three E3 families; complete-prefix validation; changed marker/results/source;
and suffix-only continuation/failure/retry behavior using temporary fixtures.

Reversing the evaluator correction in memory reproduces the original evaluator
SHA256 `79da1d5a87f50d21752bed951988042ffdde8ee3cfc1f62a6d39c8038d348e84`
exactly. Its corrected SHA256 is
`ce41f85f285a0557ed9898c5c45deb6ba534e42804dab39e3cea521cf4c61da3`.
The old six results, original marker/manifest/status and all E.10 evidence remain
byte-identical. No recovery namespace was created in the authoritative checkout.
