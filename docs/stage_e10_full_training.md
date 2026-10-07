# Stage E.10 Owner-Executed Full Train and Validation

Stage E.10 is a train/validation-only model-selection stage. Codex may maintain
and unit-test this machinery, but only the owner may execute an ML job. The CLI
therefore has no bulk-training command: it exposes non-training dry runs,
exactly one owner-confirmed job at a time, and result-only aggregation.

## Frozen protocol and exact matrix

- Experiments: separate E1, E2, and E3.
- Families: htgnn, final_flow_mlp, and flattened_five_flow_mlp.
- LR candidates: 1e-4, 3e-4, and 1e-3.
- LR-selection seed: 2025.
- Final seeds: 2025, 2026, and 2027.
- Frozen batch size: 256.
- Training schedule: maximum 30 epochs, minimum 3 epochs, patience 5, and
  min_delta=1e-4.
- Selection metrics: E1 support-restricted validation macro F1, E2
  eligible-class validation macro F1, and E3 validation PR-AUC.
- LR ties use the approved order 1e-4, 3e-4, then 1e-3.
- Losses remain unweighted. E.10 does not select class weights or an E3
  threshold.

Phase A is the Cartesian product 3 experiments x 3 families x 3 LRs x
seed 2025, or 27 jobs. Phase B freezes one LR for each of the nine
family/experiment pairs without training. Phase C runs those nine selected
configurations for seeds 2026 and 2027, or 18 jobs. The selected seed-2025
Phase A job is reused as the first final seed, so the owner executes 45 unique
ML jobs rather than 54.

## Official identity and accidental artifact policy

An official study ID must match e10-official-YYYYMMDD-vN, for example
e10-official-20261001-v1. Every manifest, result, and lifecycle status binds
that identity, run_class=official_owner_execution, and
execution_authority=owner_manual_only. Aggregation enumerates exact expected
candidate paths under that identity; it never recursively discovers runs.

The existing e10-20261001-fulltrain tree and all of its children are:

> pre-authorization diagnostic runs — excluded from official E.10 model selection

They must be preserved, not resumed, and not copied into an official tree.
That ID fails official-ID validation and its older artifacts lack the required
official metadata and COMPLETE lifecycle record.

## Reviewed provenance compatibility (2026-10-03)

The owner authorized this correction for `e10-official-20261001-v1` after a
read-only audit of all 27 Phase-A candidates. The historical handoff still
describes the frozen E.9 checkpoint; this correction neither updates that
checkpoint nor authorizes further training. The next owner action, after
validation, is Phase B result-only aggregation. Phase C needs its own execution
authorization.

Original manifests, statuses, results, JSONL and derived histories, checkpoints,
and the interrupted-attempt archive remain immutable. The compatibility
projection is computed in memory and never written back into an old artifact.
Canonical metadata hashes and live artifact hashes are still verified first.
Official results are loaded from exact expected paths, so the interrupted E3
attempt outside the official tree remains excluded.

Previously, compatibility compared all of `source_git_identity` and
`environment_identity`, plus stage/version, Stage C provenance, and upstream
evidence. Source hashes were recorded but were not checked against approved
execution-source revisions. Six identities arose from incidental Git-status
entries and a Windows build change.

The corrected comparison:

- Retains `status_short` verbatim in manifests but excludes it from compatibility:
  wrappers, verification ZIPs, and recovery directories do not change training.
- Requires the frozen HEAD/origin identity and tag. Tags, dirty state, and every
  other Git field except `status_short` remain part of strict equality.
- Requires complete environment metadata. Only for `e10-official-20261001-v1`,
  the explicitly approved `Windows-11-10.0.26200-SP0` and
  `Windows-11-10.0.26300-SP0` map to one compatibility class. For other studies,
  these strings remain distinct. Unknown platforms/builds fail closed.
- Retains strict equality for Python/implementation, Torch, PyG, CUDA,
  GPU/name/memory, availability, and any additional environment fields.
- Retains stage/version/study validation, Stage C provenance, all three split
  hashes, and the complete upstream evidence mapping.
- Requires an approved full `src/training/full_training.py` digest and the
  unchanged CLI runner digest. An unknown runtime or runner is rejected even
  when its version label matches. Test/document hashes remain recorded and
  bound by the manifest; they are not execution-source compatibility keys.

This is an explicitly authorized compatibility decision for the two observed
OS builds, not evidence of bitwise GPU equivalence or unchanged drivers.

### Reviewed source revisions and future Phase C

`src/training/e10_provenance_policy.py` holds the explicit runtime allowlist:

| Runtime | Exact SHA-256 |
| --- | --- |
| Before live progress | `b1aac183565c08e0d8d5307d466f5e3a60762f6604e9390bfddd3a54120f4023` |
| After live progress | `52222fb503466aee88e79f88c7defd7e12d90891d75ec507317a1bc8f25bb7db` |
| This provenance correction | `5bac1f7b37b718230604aebfdf035a0f42c25e99017d3dbd0e737eaa0d89fd77` |

The first two E1 HTGNN candidates used the first revision; the other 25 used
the second. Reversing the recovered historical patch in memory reproduced the
first digest exactly. It changed flushed output, complete-line observation,
timing observation, and aggregation compatibility, without changing model,
optimizer, RNG, loss, early stopping, checkpoint selection, or training calls.
Timing measurements can differ. The present correction changes only the
provenance comparison and imports its data-only review policy.

Future Phase-C manifests will record the third runtime digest, the unchanged
runner digest `3313c700240f88358d5a0de742ac3d51ef53bc9e043eecb91d003aacdbfa02c5`,
and the then-current physical hashes of the E.10 tests and this document.
No historical source hash is copied into a new manifest. `_e10_source_hashes`
and manifest construction remain unchanged.

All three reviewed runtime digests map to
`e10-v2-reviewed-training-semantics-20261003` only after allowlist validation.
`aggregate_final_report` uses this same policy for all 27 Phase-A records,
all 18 Phase-C records, and the cross-phase identity comparison. Thus seeds
2026/2027 can join the reused seed-2025 candidates without another false
source mismatch. Unknown runtime/runner revisions still fail.

The separate policy is part of the trusted aggregation validator, not a
training source or model input. It avoids embedding a runtime file's own hash
inside that file. No live-file hash is automatically trusted and no arbitrary
code block is exempted from the runtime digest. Any later runtime or runner
edit requires another explicit review, a pinned digest, and regression tests;
do not update the allowlist just to make a failed run pass.

## Owner workflow (PowerShell)

Run from C:\Project\ComputerNetworks. Choose a fresh ID after confirming its
directories do not already exist:

    $Study = 'e10-official-20261001-v1'
    Test-Path "logs\stage_e10\$Study"
    Test-Path "checkpoints\stage_e10\$Study"
    Test-Path "results\stage_e10\$Study"

All three checks must print False.

### Phase A: LR-selection jobs

This safe command prints the exact 27 PowerShell commands and expected paths.
It does not run preflight, load arrays, create a model/optimizer, or train:

    .\.venv\Scripts\python.exe scripts\run_stage_e10_full_training.py dry-run --official-study-id $Study --phase lr-selection --format powershell

The owner must inspect the output and execute each printed run-job command
individually. A concrete first job is:

    .\.venv\Scripts\python.exe scripts\run_stage_e10_full_training.py run-job --official-study-id $Study --phase lr-selection --experiment E1 --family htgnn --learning-rate 0.0001 --seed 2025 --device cuda --owner-confirmed-manual-execution

Each invocation can execute only that exact experiment/family/LR/seed.

### Phase B: result-only LR aggregation and freeze

After all 27 Phase A statuses are COMPLETE, run:

    .\.venv\Scripts\python.exe scripts\run_stage_e10_full_training.py aggregate-lr --official-study-id $Study

This reads only the exact 27 official results and their bound artifacts. It
rejects missing, failed, interrupted, diagnostic, cross-study, changed, or
misconfigured results. It writes:

- results/stage_e10/<study>/lr_selection_report.json
- results/stage_e10/<study>/selected_learning_rates.json

It does not call preflight, open predictive arrays, create a model/optimizer,
or train.

### Phase C: final seeds 2026 and 2027

Print the exact 18 selected-LR commands:

    .\.venv\Scripts\python.exe scripts\run_stage_e10_full_training.py dry-run --official-study-id $Study --phase final-seed --selection-plan "results\stage_e10\$Study\selected_learning_rates.json" --format powershell

Inspect and execute each printed command individually. The CLI rejects seed
2025 in this phase and rejects an LR that differs from the frozen selection
plan. After all 18 statuses are COMPLETE, aggregate the final validation-only
report:

    .\.venv\Scripts\python.exe scripts\run_stage_e10_full_training.py aggregate-final --official-study-id $Study --selection-plan "results\stage_e10\$Study\selected_learning_rates.json"

## Test lockout and selection boundaries

The frozen training harness constructs only train and validation datasets and
iterates only those two split names. E.10 additionally installs a process audit
hook that raises immediately if any E1/E2/E3 X_test.npy or y_test.npy is
opened while a job is active. Training loss, validation metrics, early
stopping, checkpoint choice, and LR ranking therefore remain validation-only.
Class weighting is disabled, and E.10 performs no threshold selection.
Result-only aggregation never calls the data layer.

## Interruption and no-overwrite behavior

Before a job enters the frozen harness it exclusively creates its manifest and
a RUNNING status. Ctrl+C atomically changes the status to INTERRUPTED and
records the number of complete JSONL epochs observed. Other exceptions produce
FAILED. A successful immutable result.json is followed by COMPLETE. Abrupt
process termination can leave RUNNING; that is also ineligible.

Checkpoints, epoch logs, and any partial files remain in place. A rerun with
the same candidate identity fails because every artifact path, including the
status path, must be fresh. Aggregation accepts only COMPLETE plus a valid
result, manifest, canonical hashes, configuration identity, and live artifact
hashes.

## Artifact layout

    logs/stage_e10/<study>/<phase>/<experiment>/<family>/<candidate>/
      manifest.json
      status.json
      epochs.jsonl
      training_history.json
      validation_history.json
      result.json
    checkpoints/stage_e10/<study>/<phase>/<experiment>/<family>/<candidate>/
      best.pt
    results/stage_e10/<study>/
      lr_selection_report.json
      selected_learning_rates.json
      stage_e10_report.json
      stage_e10_report.sha256

Detailed logs and checkpoints remain Git-ignored. All predictive values in the
reports are validation-only. E.10 produces no test metrics, final scientific
claim, calibration result, threshold, pruning result, E.11 work, or release
checkpoint.
