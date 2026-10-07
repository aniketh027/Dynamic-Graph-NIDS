# Stage E.12B — fixed-mask comparison, implementation only

The owner authorized a separate E.12A freeze attestation and E.12B implementation
and synthetic/allowed-validation testing. Real E.12B execution remains unauthorized.
The authoritative checkout is `C:\Project\ComputerNetworks`. The separately
reconciled live `PROJECT_HANDOFF.md` stays byte-identical during this correction.
E.12B binds its exact current bytes and the original E.9 historical pin through
the separate authority attestation described below.
No Git release, checkpoint tag, history rewrite or upstream evidence mutation is
part of this implementation. No real test arrays may be opened during development.

## Frozen policy and immutable evidence

Run `e12a-20261006-v1` selected requested per-category pruning ratio 0.25. The
canonical policy digest is
`f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa`;
the physical proposed-policy digest is
`28727d3d7bfe94f0d624639859d5a3c965dbda1ccbbab760a2ec7bc091f91317`.

The separate immutable attestation is
`results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1-freeze/freeze_attestation.json`.
Its physical digest is
`7f92187689993ea049cf4a364746301f1175d18ae25402e5a0518f2094032e51`.
It binds the original policy, original seven artifacts, source hashes, three E2
checkpoint identities, exact retained/removed IDs, review PASS, owner freeze and
owner approval of the 5% relative macro-F1 tolerance. Original E.12A files remain
unchanged, including their proposed/pending statuses. Freeze is represented by
the separate attestation, not by rewriting historical evidence.

The frozen selection rule was the greatest declared candidate for which EVERY
E2 seed's pruned eligible-class validation macro-F1 is at least 0.95 times that
seed's unpruned value. Declared candidates were 0%, 10%, 25%, 50%, 75%; admissible
candidates were 0%, 10%, 25%. E.12B performs no selection or margin check that can
change the mask. The tolerance is a validation acceptance rule, not a statistical
noninferiority confidence bound or a guarantee about test results.

Removed original feature IDs:
`6,10,29,32,34,35,40,41,42,50,52,57,64,70,71,72,73`.
Retained original IDs:
`0,1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,33,36,37,38,39,43,44,45,46,47,48,49,51,53,54,55,56,58,59,60,61,62,63,65,66,67,68,69,74,75,76,77`.
Category retention is `[2,18,18,4,9,5,3,2]`. Frozen taxonomy and edges remain the
authority; compaction reuses E.12A's mapping primitive and original feature IDs.

| Graph quantity | Original | Fixed mask |
|---|---:|---:|
| Feature nodes | 78 | 61 |
| Category nodes | 8 | 8 |
| Sample nodes | 1 | 1 |
| Total nodes | 87 | 70 |
| Feature-category edges | 78 | 61 |
| Category-sample edges | 8 | 8 |
| Total edges | 86 | 69 |

This is a single offline training-attention-derived mask applied at inference.
It is not an adaptive per-example mask, causal feature explanation, weight
pruning method, retraining method or parameter reduction. T=5 remains five flow
records and the target remains the final flow's label.

## Nine comparisons and existing E.11 baselines

Evaluate only HTGNN seeds 2025, 2026, 2027 for each task, in E2/E1/E3 order.
E2 is the PRIMARY experiment because its training attention and validation
selected this mask. E1 is a diagnostic transfer experiment; E3 is a binary
transfer experiment. Their inclusion is fixed by scientific roles before test
access, without performance-based inclusion, retuning, calibration or thresholds.
MLPs have no graph to prune and are excluded.

Reuse nine existing E.11 unpruned predictive records from the completed recovered
27-record report. Its physical digest is
`ace0d585fbb89b0f3df4376b2f69e35bd3172ab663cc66b369b769e84a964aac`.
E1/E2 records are the preserved original six-record prefix; E3 records are in
`recovery/e11-recovery-20261006-v1`. Verify physical bytes, canonical result
digests, exact report membership, E.10 result/checkpoint bindings and full
sequential coverage. All nine physical result digests are also explicitly pinned
in the reviewed E.12B source, including the E3 recovery results. Do not call the
E.11 executor, original reservation reader,
or its checkpoint loader: the latter constructs AdamW for historical validation.
The new inference loader constructs no optimizer and restores only model tensors.
The serialized historical optimizer payload is never instantiated or restored.

E1 uses frozen support-restricted macro-F1 and all original secondary metrics.
E2 uses E.11 eligible-class macro-F1 with the frozen eligible TRUE-ROW filter;
all 15 logits remain available for prediction. This differs from E.12A's E7
validation F1, which included every validation row for false-positive accounting.
E3 uses average-precision PR-AUC, ROC-AUC and the frozen binary secondary metrics,
with raw-logit decision threshold 0 (probability 0.5), without tuning.
Reuse E.11 metric functions; include full confusion matrices/per-class reporting.
Report each seed's baseline, pruned result, absolute change and relative drop,
and task-wise means/sample standard deviations (ddof=1) for all scalar metrics
and paired changes. Undefined values remain null with their frozen explanations.
The manifest retains complete baseline metric records and both hash forms.

Test traversal uses authoritative D4 -> D6 -> E2 APIs with sequential bounded
256-example batches, including the final partial batch. No full graph collection
is constructed. E3 exact ranking metrics use temporary disk-backed score/target
maps, closed before cleanup. No raw features or large predictive arrays are saved.
Outputs stay `[B,15]` for E1/E2 and `[B]` for E3. E3 targets stay float32 `[B]`.
The explicit E.12B adapter validates true test provenance; it never relabels test
data to bypass E.12A's train/validation restriction. Zero-mask diagnostics are
permitted only on non-test inputs. Every test forward uses the fixed 61-feature mask.

## Validation-only efficiency protocol

Predeclare the first four sequential validation batches per task, starts
`0,256,512,768`, each with 256 examples. The same memberships apply to every
seed and both paths. Use float32 and the same device, checkpoint and deterministic
arithmetic context for both paths. Keep cuDNN benchmark disabled and synchronize
the exact CUDA device before starting and before stopping each timed scope.

For each batch/path, perform five warmups. Then collect 20 paired repetitions,
alternating unpruned/pruned and pruned/unpruned order. This gives 80 observations
per path/checkpoint. Report median, p95, mean, sample standard deviation, coefficient
of variation, range and examples/second derived from median batch latency.
Retain paired timings, paired speed-ratio median/std and input traversal digest.

Forward timing includes contract checks, feature compaction, encoders and head;
the original input is already resident. Separately time combined lazy batch
loading/D6/E2 conversion and input cloning/device transfer. Dataset initialization,
checkpoint loading, ranking and integrity hashing are excluded from these scopes.
Do not combine the scopes into an unqualified end-to-end deployment claim.
Never benchmark by repeating official unpruned test predictions.

GPU allocated/reserved before/after and peak values are reported only if all
post-warmup observations of that path reproduce exactly; otherwise memory bytes
are null with a reason. CPU allocator memory is unavailable. These measurements
include model and full adapter inputs, describe the PyTorch caching allocator,
and do not represent total device/driver memory. Parameter count remains unchanged;
the adapter initially builds the complete graph. No input-memory saving is assumed.

Historical E.11 CUDA arithmetic was not certified bitwise equivalent to the
strict deterministic E.12 context. Zero-mask CPU/CUDA equivalence with original
models on allowed inputs is required; runtime/settings identities and this
remaining historical limitation are explicitly recorded. E.11 test prediction
must not be rerun to resolve it. Small-GPU timing and allocator variability may
prevent an efficiency improvement claim even when node/edge reduction is exact.

## Safeguards, reservation and failure

`prepare(root)` is read-only and test-locked. It authenticates the pinned freeze,
all original E.12A files/source, E.10/E.11 report bindings, frozen model sources,
split/preprocessing metadata, metric libraries and hardware/runtime identity.
It checks validation input bytes and all nine checkpoint tensors without training.
Test digests are obtained from frozen Stage C metadata, not by opening test files.

The owner execution command must supply the exact policy digest, freeze physical
digest, reviewed four-file implementation-bundle digest and explicit one-time
pruned-test acknowledgement. Wrong/missing values fail before reservation.
The implementation digest is canonical SHA256 of the four source-path/physical
SHA256 pairs returned by preparation; any later edit requires a new review.

Execution first records an exclusive manifest and performs only validation
benchmarks while test access remains locked. It rechecks provenance, then creates
`logs/stage_e12b/e10-official-20261001-v1/test_consumption.json` exclusively.
This independent study-wide immutable consumption ledger records owner authority,
policy/freeze/source/manifest/checkpoint bindings. Only its active, verified
execution context unlocks test hashing/loading/inference. Existing ledger or run
namespace rejects subsequent execution; there is no resume/retry switch.

After reservation, verify all six test-array physical digests before predictive
traversal and again afterward. Verify full counts/order, unchanged tensor states,
sources, validation inputs and upstream evidence before writing completion.
No optimizer, backward, training, fine-tuning, mask selection or threshold tuning
exists in the execution path. A process-wide Python audit hook blocks writes
outside the new run namespace, mutations of the ledger, and unapproved arrays,
including worker-thread access. It is an accidental-access safeguard rather than
a security sandbox against arbitrary native code/external mutation or hard links.

Any failure stops immediately and records exclusive failure evidence when the
run directory exists. A reserved ledger stays reserved. No automatic retry,
overwrite, cleanup of historical evidence or second test consumption occurs.
Failures before ledger creation still leave a run namespace that rejects replay.
Preparation failures before any run namespace opens no test arrays and create no
evaluation results. No writes to E.10/E.11/E.12A are allowed.

## Prepared owner interface and outputs

Metadata preflight (no test arrays, no output writes):

```powershell
.\.venv\Scripts\python.exe -B scripts\run_stage_e12b.py prepare
```

The owner-only `execute` interface requires `--run-id`, `--device`,
`--policy-sha256`, `--freeze-sha256`, `--implementation-sha256` and
`--owner-confirmed-one-time-pruned-test-execution`. The implementation report
provides an exact prepared command with the reviewed bundle hash. This document
does not authorize running that command.

Expected new paths for run `e12b-20261006-v1`:

- `logs/stage_e12b/e10-official-20261001-v1/test_consumption.json`
- `logs/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/manifest.json`
- Terminal `status.json` or `failure.json` in that log directory.
- `results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/validation_benchmarks.json`
- Nine `<candidate_id>.json` pruned results in that result directory.
- `stage_e12b_report.json` and its physical `.sha256` sidecar there.

No real predictive or benchmark results exist from implementation testing.
Completion remains pending independent review. The next safe boundary is an
independent E.12B implementation/protocol pre-execution audit, followed only
after approval by separately explicit owner authorization of one-time execution.
The reconciled handoff records completed E.10/recovered E.11, artifact-frozen
E.12A, and implemented/tested but unexecuted E.12B. Its recorded four-file digest
and inherited-blocker description predate this separately authorized correction;
the correction's separate evidence records the replacement digest without
rewriting that pinned handoff.

## Separate historical-to-live authority reconciliation

The original E.12B preparation called frozen E.12A `prepare`, whose E.9
source-hash loop compared `e9_source_sha256["PROJECT_HANDOFF.md"]` with the live
handoff. That conflated a historical documentation binding with current dynamic
project authority. The historical pin remains
`0eb4f09d989a8c641736be9637f7677cdccf3e675d73425bbd0e7cf22a5df9f6`.
Git object `2fa823943ba318fe32b2867bc8e2b1f799d78ac1:PROJECT_HANDOFF.md`
reproduces those exact bytes. The separately authorized reconciled live pin is
`58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5`.

The separate namespace is
`results/provenance/authority/handoff-reconciliation-20261006-v1/`.
`attestation.json` has physical SHA-256
`af4b21ea289932dfad648c6affd13a4cc9f91944bcc8d4eca7a94397d3422895`
and canonical content SHA-256
`4e338a1582834aea9a66a77a6ba2de4517ee28c6c61654701f1405ae01549d4b`.
The physical digest is independently pinned in E.12B source and matched by its
sidecar. The attestation binds both handoff hashes, the historical E.9 commit,
original E.9 contract physical/canonical digests, reconciliation reason, UTC
timestamp/run ID, exact owner request bytes, applicable authority-instruction
digests, and compact E.9/E.10/E.11/E.12A review evidence. The current handoff
supersedes the historical handoff only as the live authority/status record.
It grants no experimental/source exception, execution approval or Git release.

`verify_authority_reconciliation(root)` verifies the independently pinned
attestation bytes, canonical content and serialization, sidecar, explicit
old/new/scope/commit bindings, owner evidence and authority/review hashes.
It verifies the original E.9 contract and unchanged historical handoff pin,
reads the original handoff with read-only `git cat-file blob`, verifies the
current live handoff, and still compares every other frozen E.9/upstream source
against its original live pin. There is no fallback accepting an old live
handoff, missing attestation, newer file, or arbitrary later edit.

Because E.12A source is frozen, E.12B's `_prepare_upstream` retains the compact
E.12A preflight contract locally, using the existing E.9/E.10 APIs. It substitutes
only the authenticated historical/live handoff comparison, then retains E.10
record membership, execution-source pins, semantic E.9 observation verification,
upstream evidence, environment, Stage C, split metadata and train/validation
integrity checks. It adds the authority attestation and both hashes to the
returned preparation record, which the existing manifest/revalidation lifecycle
binds. No loader, graph topology, split, policy or selection implementation is
duplicated. E.10/E.11/E.12A preparation code remains unchanged and can still
reject the reconciled live handoff in their historical execution paths.

The four-file E.12B bundle has changed; the historical digest
`0fd1177ef4128a7384f5ef4c5b8cecdf0e1469e0448499d6f5df9d279e930002`
must not be reused in an owner command. The new digest is recorded in the
separate correction report after final source/test/document edits. The existing
owner interface checks that digest before reserving any execution namespace.

Restart independent review in the authoritative checkout by reading the
unchanged handoff, this correction, the authority attestation and owner request,
and the frozen E.12A attestation. Reproduce the authority validator inside
`file_guard(root)`; inspect the exact scoped diff and the old/new four-file
digests. Run `python -B -m pytest -q tests/test_stage_e12b.py -p no:cacheprovider`
in the approved `.venv`, followed by the allowed synthetic upstream regressions
listed in the correction report. Do not execute E.12B, read real test arrays,
rerun E.11, train, select another pruning mask, or grant owner execution approval
as part of this implementation correction. The full project suite includes
real-data/training tests and is outside this request's allowed verification scope.
