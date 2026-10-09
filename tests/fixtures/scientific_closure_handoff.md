# Project Handoff — E.9 Git Checkpoint; Experimental Scope through E.12B Complete

## Project Goal

The project studies automated feature extraction and dynamic heterogeneous temporal graph optimization for efficient network intrusion detection using CSE-CIC-IDS2018. The frozen checkpoint includes the Stage E.6 bounded forward/resource pilot, the independently audited E.2 provenance-cache correction, the corrected Stage E.7 training harness, the independently audited Stage E.8 bounded optimization pilot, and the independently audited Stage E.9 model-binding and bounded resource-pilot checkpoint. Stage E.9 freezes HTGNN training batch size 256 under its hardware-specific resource rule; that checkpoint alone did not establish full-training throughput, predictive performance, a final learning rate, locked-test results, or pruning. The authoritative working tree subsequently completed owner-authorized E.10 training/validation, E.11 interrupted-and-recovered first-test evaluation, and E.12A validation-only pruning selection and policy freeze. E.12B is COMPLETE and independently audited PASS, as confirmed by the owner on 2026-10-06. The approved experimental scope is complete; final report/release/documentation is now the active boundary. No further ML experiment is currently authorized or required. Final measured results and exact frozen-evidence identifiers are in `docs/final_experimental_results.md`.

## Authority and Evidence by Purpose

Use each source for its purpose, consistently with `AGENTS.md` and the
`nids-release-audit` skill, rather than treating Git history or stage documents
as one linear authority chain:

- The current user's explicit scope and approvals define authorized work.
- This handoff records verified dynamic state and the next safe boundary.
- Applicable stage documentation defines its approved/intended contract;
  historical preparation statements are scoped to the stage/time they describe.
- Actual source and tests are implementation truth. Compact evidence supplies
  reproducible measurements and provenance; stored PASS fields alone are not
  an independent release audit.
- The original report is background only; its performance claims are not
  verified authority.

Current working-tree evidence and its identifiers are reconciled in
**Authoritative Working-Tree Experimental State** below. Relevant contracts are
`docs/stage_e10_full_training.md`, `docs/stage_e11_test_evaluation.md`,
`docs/stage_e11_failure_recovery.md`, `docs/stage_e12_attention_pruning.md`,
and `docs/stage_e12b_final_comparison.md`. Their earlier statements that runs,
recovery, policy freeze, or E.12B implementation had not yet occurred remain
historical preparation records; the current state below supersedes those
statements without rewriting provenance-bound files.

Historical frozen evidence remains authoritative for its own contract:

- E.9: `results/stage_e9/stage_e9_contract.json`,
  `results/stage_e9/htgnn_batch_size_pilot_observations.json`, and
  `docs/stage_e9_baselines_batch_size_pilot.md`.
- E.8: `results/stage_e8/htgnn_optimization_pilot_contract.json`,
  `results/stage_e8/htgnn_optimization_pilot_observations.json`, and
  `docs/stage_e8_optimization_pilot.md`.
- E.7: `results/stage_e7/htgnn_training_harness_validation.json` and
  `docs/stage_e7_training_harness.md`.
- E.3-E.5: their feature-encoder, hierarchical-encoder, and classifier reports
  under `results/stage_e3/` through `results/stage_e5/`, and corresponding
  stage documents.
- E.1-E.2: `results/stage_e1/`, `results/stage_e2/`,
  `docs/stage_e2_heterodata_adapter.md`, and
  `docs/stage_e2_provenance_cache.md`.
- D1-D8: `docs/STAGE_D_VERIFICATION.md` and `results/stage_d1/` through
  `results/stage_d8/`; C: `docs/STAGE_C_VERIFICATION.md` and
  `results/stage_c_full_validation.json`.
- Stage B/temporal background: `results/data_profile/` and
  `results/temporal_analysis/`; implementation/tests: `src/`, `scripts/`,
  and `tests/`.

Stop on a new material inconsistency and resolve it before execution or changing
claims. This owner-authorized reconciliation resolves the known stale-handoff
exception only; it does not authorize execution, Git release, or a source repair.

## Frozen Status

Stage C passed an independent adversarial re-audit and is frozen. Stage D D1-D8 also passed an independent adversarial release audit and is frozen. Stage E.1-E.2 passed independent environment, conversion, CUDA-transfer, storage-ownership, provenance, and integrity checks. Stage E.3 passed an independent causal-contract, architecture, integration, device, evidence, and regression audit and is frozen. Stage E.4 passed an independent exact-topology, attention-semantics, architecture, isolation, leakage, device, integration, evidence, and regression audit and is frozen. Stage E.5 passed an independent task/head-contract, readout, topology, leakage, isolation, attention, trainability, state-dictionary, CPU/CUDA, nine-pair integration, deterministic-evidence, and regression audit and is frozen. Do not change Stage C cleaning, transforms, windows, splits, the Stage D graph-input layer, or the Stage E.1-E.5 environment, predictive adapter contract, encoders, classifier contract, verified counts, hashes, or tests without a separately approved change and a new audit.

Stage E.6 passed an independent release audit and is frozen. Its verified
implementation commit is `a959ee93751c585aeaf50341b8739ac0034ffbeb`. The
bounded real-data pilot uses train data only, batch size 32, five CUDA warmups,
and twenty measured repetitions for each of E1, E2, and E3. It separates E.2
conversion, host-to-device transfer, normal forward, and attention-return
forward. Bounded batch-32 CUDA forward was operational on the verified NVIDIA
GeForce RTX 4050 Laptop GPU. No architecture modification occurred. This
result does not prove training feasibility and does not establish an optimal
batch size.

At the frozen E.6 checkpoint, the independent audit classified repeated E.2
provenance/report validation as
**clearly material overhead** in this environment and measurement procedure.
Paired component/complete-conversion medians were E1 13.03/16.24 ms
(approximately 80.2%), E2 12.17/14.21 ms (approximately 85.7%), and E3
12.61/14.91 ms (approximately 84.6%). These percentages are observational, not
universal constants. E.2 was not modified, and no caching or optimization was
implemented. Any E.2 optimization requires a separately authorized boundary.

Canonical contract evidence is in
`results/stage_e6/htgnn_resource_pilot_contract.json`; noisy hardware-specific
observations are deliberately separate in
`results/stage_e6/htgnn_resource_pilot_observations.json`.

## Frozen E.2 Provenance-Cache Correction

A separately authorized correction, implemented in commit
`062fb227edf02f2be65c2d45fafa2481f2be138b`, passed an independent audit and
is frozen at checkpoint tag `v0.9.1-e2-provenance-cache`. Its engineering
purpose is to eliminate repeated D8 and E.1 provenance/report validation
overhead while preserving the frozen E.2 predictive, topology, and ownership
contract. It is not a new research experiment.

The correction uses a lazy, process-local, thread-safe, one-entry cache for
successful validation of only the compact frozen D8 and E.1 provenance
identity. Memory remains bounded, and no logical batch, graph, tensor, label,
model output, or model state is cached. The first call still performs full
validation. The cache key binds the result to the resolved root, approved
constants and hashes, resolved paths, file identity, mode/link count, size,
and nanosecond timestamps. Ordinary filesystem-detectable changes miss the
cache and return to the original full validation path; a change during
validation fails closed. The immutable cache value contains only identity
strings, and each caller receives a fresh dictionary.

Every conversion still validates the D6 batch provenance, experiment, split,
read-only arrays, ownership, target domain, and exact topology and still
creates new CPU-owned tensors. Frozen pre-correction output fingerprints match
corrected cold and warm output exactly for bounded real batches from all nine
E1/E2/E3 train, validation, and test pairs. No training, pruning, E.7 behavior,
predictive input, topology, label, model architecture, or frozen predictive
semantic change was introduced.

The documented coherence boundary is filesystem-detectable identity. A
privileged same-process or filesystem adversary that deliberately rewrites an
artifact and restores every cache identity field remains outside that
boundary; detecting that case would require re-reading and rehashing the
artifacts on every conversion and defeat the optimization. The cache is not
claimed to be cryptographically tamper-proof.

Canonical correction evidence is
`results/stage_e2/provenance_cache_validation.json`, with canonical SHA-256
`cd2b8cd961c9ebfe6bb6e21c69b600db799897b8d7df3a3bcd0e4e2f234539bc`
and physical SHA-256
`ecba8747cb54a138f502297feb189bb23f2ddbe454473b8cc822668dac91982d`.
Hardware-specific timing observations remain separate in
`results/stage_e2/provenance_cache_observations.json`. In the independent
paired audit, batch-32 cold/warm conversion medians were 10.8813/3.0034 ms for
E1, 12.0495/2.9959 ms for E2, and 11.8971/3.0711 ms for E3. Warm repeated
conversion was approximately 25–28% of cold conversion in that environment
and measurement procedure. These observations are not universal constants or
performance guarantees. The project-level conclusion is that the Stage E.6
provenance bottleneck was materially reduced before the training-harness
stage.

The independent audit reconfirmed 20 focused cache tests, 183 frozen E.2-E.6
regression tests, and the complete suite at 894 passed, 0 failed, and 0
skipped. Stage C integrity remained 33/33 and raw CSV integrity remained
10/10. The real batch-32 CUDA path passed with E1/E2 logits `[32,15]`, E3
logits `[32]`, and unchanged parameter counts of 7,183 for E1/E2 and 6,721 for
E3.

## Frozen Stage E.7 Training Harness

Stage E.7 passed an independent adversarial release re-audit and is frozen at
checkpoint tag `v0.10-stage-e7-training-harness`. Its original implementation
commit is `f3961cde141ac2b452fd21c62982b35afca6dcac`; the required log-ownership
correction is commit `4e0a6bd522dffa6c59474ded7caed8d3acd535b4`.

The production harness preserves the frozen D4-to-D6-to-E.2-to-E.5 path. E1
and E2 use unweighted cross-entropy on raw `[B,15]` logits; E3 uses unweighted
binary cross-entropy with logits on raw `[B]` logits. AdamW uses an explicit
learning rate and weight decay `1e-4`, with no baseline scheduler. Training is
seeded and shuffled without split-sized materialization, validation remains
sequential, and test data is excluded from training and checkpoint selection.
The best validation checkpoint is state-dict based and is selected with the
task-authorized validation objective and the frozen early-stopping contract.

One JSONL log path belongs to exactly one training run. Before model, data, or
optimizer setup, E.7 resolves checkpoint and log paths, rejects equivalent
checkpoint/log targets, and reserves a fresh log atomically with exclusive
creation. Every existing path, including an empty file, is rejected; logs are
never silently appended, truncated, substituted, or reused across runs.

The final independent audit reproduced 42 focused E.7 tests, 203 frozen
E.2-E.6 regression tests, and 936 complete-suite tests, with zero failures or
skips. Stage C integrity remained 33/33 and raw CSV integrity remained 10/10.
The canonical E.7 evidence SHA-256 is
`7a23c435850f577a9450b8edef56c03c89b7e567ccac9307c6b493c20c3e81a7`;
the physical file SHA-256 is
`fa82febc3b8ef14079c55f933f1e4109f0be9154ba38d57d5fab906eaf83809d`.
Only small bounded CPU/CUDA smoke runs were performed. No full training,
three-seed study, hyperparameter sweep, test evaluation, pruning, or
predictive-performance study was run. Later expensive work remains subject to
the hybrid execution rule and may be run manually by the user only after its
separate stage is authorized.

That release statement records the historical E.7 checkpoint. A later
independent E.8 audit found an indirect methodology defect: E.7 loaded its E2
validation class set from D5 reporting eligibility, which had been derived
from train, validation, **and test** support. The real corpus happened to yield
the same class codes under a train+validation-only rule, but the dependency was
still test leakage. The explicitly authorized correction now derives
`validation_selection_class_codes` only from the authoritative E2 train and
validation target memory maps using train support ≥100 and validation support
≥30. It does not consult test labels, support, membership, metadata, or
final-reporting eligibility. This corrected E.7 source passed the independent
E.7/E.8 correction re-audit and is frozen as part of the Stage E.8 checkpoint.
The historical E.7 tag is not rewritten.

## Frozen Stage E.8 Corrected Optimization Pilot

Stage E.8 has a regenerated implementation-level bounded optimization pilot
built around the corrected E.7 harness. It compares only the approved learning
rates `1e-4`,
`3e-4`, and `1e-3` for E1, E2, and E3 with the frozen unweighted task losses,
single seed `2025`, batch size 32, three epochs, 16 train batches, and 128
sequential validation batches per epoch. It does not implement weighted-loss
ablations, full training, a three-seed study, batch-size optimization,
baselines, test evaluation, or pruning.

The first independent E.8 audit was **BLOCKED** by the indirect E2
test-informed selection dependency described above, and E.8 was never frozen.
The corrected bounded CUDA pilot reran all nine candidates under one consistent
corrected harness identity without E2 selection access to test information. Its
compact deterministic contract and separate observational output are under
`results/stage_e8/`, and the exact bounds and limitations are documented in
`docs/stage_e8_optimization_pilot.md`. The E2 validation prefix does not cover
every validation-selection class, and all rankings remain bounded observations
rather than a finalized production configuration or project-performance result.

The corrected implementation and regenerated pilot are recorded by correction
commit `517d37783d518e04200f400a0f5a238255a887fa`. They passed an independent
adversarial re-audit and are frozen at checkpoint tag
`v0.11-stage-e8-optimization-pilot`. The audit independently reproduced the
test-blind E2 selection boundary, live candidate/checkpoint/log reconciliation,
bounded checkpoint metrics, deterministic evidence hashes, frozen-data
integrity, focused tests, frozen-stage regressions, and the complete suite.
The corrected E2 bounded observations remain a tie between `1e-4` and `3e-4`
at `0.02813852813852814`; this numerical equality with the rejected package is
expected because the real test-ineligible classes also fail the train or
validation thresholds. It does not weaken the methodological correction. E2
validation-prefix support remains incomplete. The E3 validation prefix remains
4,095 attacks and one benign example, so its near-one PR-AUC is highly
composition-dependent. No final learning rate is selected.

Correction validation passed 44 focused E.7 tests, 20 focused E.8 tests, 203
frozen E.2-E.6 regressions, 247 E.2-E.7 regressions, and 962 complete-suite
tests, with zero failures or skips. Stage C integrity remained 33/33 and raw
CSV integrity remained 10/10. The corrected E.7 canonical/physical evidence
SHA-256 values are
`9b6669e35344757dde130b7ffc6f089192aae56494a009ba3e76994e7920062a` /
`3ea99b2bb39e9f8a5e6e1b04ef46ba837037d0797cde38dadfd9bb9a1b00e67b`.
The corrected E.8 canonical/physical contract SHA-256 values are
`2c34c71c99c2aee4dba69a909e7094eadd4aa12b1cd7627e318b43c4813746ee` /
`2015a4b71c7e70f6a9600f76b34388a953b27b33bca356e539e1c1570da1ee62`.
The independent correction/E.8 release-audit verdict is **PASS**. These hashes
identify the frozen deterministic evidence; the candidate metrics remain
bounded observations with the limitations above.

## Frozen Stage E.9 Engineering Contract

This section defines the complete authoritative Stage E.9 engineering
contract. It does not implement or execute E.9, create baseline source files,
run the batch-size pilot, authorize E.10, or modify any frozen model, training,
or data source.

### Final-Flow MLP Baseline

The final-flow baseline consumes only the fifth/final flow from the frozen
preprocessed `T=5` window. Its exact input shape is `[B,78]`. Production code
must derive this tensor as the final temporal position from the same
authoritative preprocessed window and must not access temporal positions 1-4.

Its architecture is exactly:

```text
Linear(78,64, bias=True)
LayerNorm(64)
GELU
Dropout(0.10)
Linear(64,32, bias=True)
LayerNorm(32)
GELU
Dropout(0.10)
```

The task head is `Linear(32,15, bias=True)` for E1 and E2 and
`Linear(32,1, bias=True)` for E3. Forward returns raw logits. The canonical
public E3 output shape follows the frozen E.5 convention and is `[B]`. No
sigmoid or softmax is applied. This baseline contains no BatchNorm, residual
connection, TCN, graph operation, attention, or pooling.

### Flattened Five-Flow MLP Baseline

The flattened baseline consumes the full frozen ordered `T=5` window with
shape `[B,5,78]`. It flattens deterministically in temporal-major order: flow
1 features 0-77, then flow 2 features 0-77, then flows 3, 4, and 5 in the same
feature order, producing `[B,390]`.

Its architecture is exactly:

```text
Linear(390,64, bias=True)
LayerNorm(64)
GELU
Dropout(0.10)
Linear(64,32, bias=True)
LayerNorm(32)
GELU
Dropout(0.10)
```

It uses the same E1/E2/E3 task-head and raw-logit output contract as the
final-flow baseline, including canonical public E3 output shape `[B]`. No
sigmoid or softmax is applied. This baseline contains no BatchNorm, residual
connection, TCN, graph operation, attention, or temporal pooling before
flattening.

### Baseline Fairness Contract

Both baselines must reuse the authoritative preprocessing, `T=5` windows,
split membership, labels, task mapping, training losses, AdamW configuration,
validation objectives, early stopping, and checkpoint/log contracts from the
frozen upstream stages. E.9 must not add new split logic, preprocessing, label
mapping, class-eligibility authority, or training implementation. It must
measure the actual parameter count of each baseline and must not alter either
architecture merely to force exact parameter matching with HTGNN.

### HTGNN Batch-Size Pilot

The resource pilot task is E2 because E2 is the primary supervised multiclass
project experiment. The pilot seed is `2025`. Its learning rate is `3e-4`, a
predeclared engineering value for this resource pilot only and the middle
value of the already approved `1e-4`, `3e-4`, `1e-3` grid. Using it in E.9
does not finalize or select a learning rate; Stage E.8 did not finalize any
task learning rate.

The pilot must reuse the frozen E.7 E2 baseline: unweighted
`CrossEntropyLoss`, AdamW, `weight_decay=1e-4`, and no scheduler. The exact
batch-size candidates are `32`, `64`, `128`, and `256`; E.9 must not probe
beyond `256`.

For each batch-size candidate, the pilot must:

- use only the authoritative E2 train stream;
- seed traversal with `2025` and restart from the same deterministic seeded
  training traversal for every candidate;
- use no validation data for resource measurement and no test data;
- perform 5 warmup optimizer steps followed by 20 measured optimizer steps;
- include host-to-device transfer when applicable, forward,
  `CrossEntropyLoss`, backward, and optimizer step in every step; and
- discard the pilot-trained weights rather than preserve them as project
  results.

This is a bounded resource pilot, not model-quality training.

### Timing and GPU-Memory Measurement

Measured CUDA training steps must use correct CUDA synchronization, exclude
warmups, retain the individual measured samples, and report the median, p95
or an equivalent documented spread, and examples/windows per second.
Asynchronous CUDA launch time must not be treated as completed execution time.

The pilot must verify the live GPU and total VRAM. For every candidate it must
record baseline allocated memory, baseline reserved memory, peak allocated
memory, and peak reserved memory. Peak statistics must be reset appropriately
after warmup and before measurement. Zero additional reserved-memory growth
must not be reported or interpreted as zero reserved-memory usage.

### Resource Eligibility and Batch-Size Selection

A candidate is `RESOURCE-ELIGIBLE` only when all bounded warmup and measured
training steps complete, no CUDA out-of-memory condition occurs, losses remain
finite, gradients remain finite, optimizer steps succeed, and peak absolute
CUDA reserved memory remains strictly below 80% of live total GPU VRAM. The
80% rule is the project headroom threshold. E.9 must not intentionally
approach out-of-memory conditions beyond the approved candidate set.

Stage E.9 is authorized to finalize the HTGNN training batch size using this
deterministic rule:

1. Compute the median examples/second for every resource-eligible candidate.
2. Identify the maximum median throughput.
3. Define the throughput-tied set as candidates whose median throughput is at
   least 95% of that maximum.
4. Select the smallest batch size in the throughput-tied set.

This rule preserves memory headroom when a larger batch provides less than a
5% throughput advantage. If no candidate is resource-eligible, E.9 must block
rather than invent another batch size. If exactly one candidate is eligible,
that candidate is selected.

### E.9 Authority and Execution Limits

Stage E.9 may finalize only the HTGNN training batch size. It may not finalize
a learning rate, weighted-loss configuration, task performance, final model
checkpoint, or test performance. Stage E.8 remains a bounded learning-rate
observation only.

Codex may execute the E.9 pilot directly only while it remains short and
bounded. If execution becomes clearly expensive, Codex must provide exact
commands for the user to run manually; methodological correctness must not be
reduced merely to avoid manual execution.

The E.9 checkpoint itself did not authorize E.10; later owner-authorized completion is recorded below.

## Frozen Stage E.9 Implementation, Four Blocked Audits, and Corrections

The frozen E.9 contract was recorded in commit
`76d1bf81f2e18e3d0b99755a2d87bff9d96f3f0f`. The first E.9 implementation
was recorded in commit `a717a2af00d1f333f462b27ad8dad7f9e05ac31b`.
Its independent audit verdict was **BLOCKED**. The bounded CUDA measurements
themselves reproduced, but three production defects were confirmed:

1. the resource rule selected batch 256 while production configuration and
   iteration still rejected values above 128;
2. baseline checkpoints could not reliably distinguish task, model family,
   architecture, or input contract and the claimed checkpoint/log smoke wrote
   neither artifact; and
3. the observation validator accepted corrupted source, device, throughput,
   and related critical evidence.

The explicitly authorized correction preserves the frozen candidate set and
selection method. The production maximum is now 256, while the default remains
32. Boundary checks accept 128 and 256 and reject 257 and 512. The actual E2
production iterator emitted 256 examples, and one bounded HTGNN production
train step produced raw `[256,15]` logits, finite cross-entropy and gradients,
and a successful optimizer update.

The E.7 checkpoint/training machinery now supports first-class `htgnn`,
`final_flow_mlp`, and `flattened_five_flow_mlp` identities. Checkpoint schema
v2 binds task, family, architecture version, input contract, parameter count,
training configuration, validation-selection state, model state, and optimizer
state before weight loading. Existing valid schema-v1 HTGNN checkpoints remain
loadable. Both baseline families use the shared harness for all E1/E2/E3 task
contracts, including real temporary JSONL and best-checkpoint creation,
matching-model reload, and output reproduction. The E.7 exclusive one-log-per-
run and checkpoint/log alias rules remain intact.

Observation validation now checks exact candidate coverage/order, the frozen
configuration identity, current source bindings, audited CUDA device identity,
raw timing/loss counts, timing-derived throughput and summaries, memory
relationships, eligibility, selection, and final production compatibility.
The deterministic evidence binds the exact observation-file bytes by SHA-256.
Adversarial mutations of every audit-critical field are rejected.

Production-integration correction commit
`5f945149859b9b7d2002e712525b7b34ed82e1be` was then independently audited.
That second audit also returned **BLOCKED** for two remaining defects. First,
the loader applied checkpoint model weights before discovering a missing or
malformed optimizer payload, and strict model loading could copy multiple
tensors before reporting a missing model key. Second, internally coordinated
edits could label a complete finite trace as nonfinite or CUDA OOM, recompute
the related summaries, and pass the forged record to deterministic evidence
generation for new hash binding.

The authorized fail-closed correction keeps the one production checkpoint
loader and adds a two-phase transaction. Preflight now validates the complete
model key set, tensor shapes/dtypes/layouts, model identity and parameter
contract, optimizer fields, parameter-group structure, AdamW state tensors,
and a disposable optimizer load before target mutation. Exact model and
optimizer snapshots are then captured; any exception after mutation starts
restores both targets, including optimizer state and parameter-group
configuration, and re-raises the original failure. Adversarial tests prove
model and nested optimizer state equality after every required rejection and
also force a post-preflight optimizer failure to exercise rollback.

Fail-closed correction commit
`43f4582063e32e68bde411dc1c46c19c52afe7a8` was then independently audited.
That third audit verified checkpoint transactionality, exact rollback,
schema-v1 HTGNN safety, baseline integration, and the raw-observation state
machine, but remained **BLOCKED** because the loader accepted five semantically
invalid AdamW payloads: optimizer LR `9.0` contradicting metadata LR `3e-4`,
weight decay `0.5` contradicting frozen `1e-4`, one missing parameter-state
entry, `NaN` in `exp_avg`, and `NaN` in a step counter.

The authorized optimizer-state correction keeps schema v2 and the single
production loader. Every load, including model-only evaluation, now binds the
serialized LR to checkpoint `training_config`, binds weight decay and all
present semantic AdamW group fields to a freshly constructed frozen optimizer,
and requires the exact serialized parameter mapping and complete state coverage
for every grouped trainable parameter. Under frozen `amsgrad=False`, each state
contains exactly scalar float32 `step`, `exp_avg`, and `exp_avg_sq`; step must
be finite, nonnegative, and integral, both moments must be finite and match the
parameter shape/dtype/layout, and the second moment must be nonnegative.
Unknown, missing, extra, nonfinite, negative-second-moment, or malformed state
is rejected before mutation. The production save lifecycle reaches checkpoints
only after optimizer updates, and bounded real HTGNN plus all baseline task
paths confirm complete AdamW state. Existing schema-v1 compatibility remains
HTGNN-only and uses the same training-configuration and optimizer-state checks.

Optimizer-state correction commit
`dc85cbd39346207c572a7e3e7434dc51d65ebca4` was then independently audited.
That fourth audit confirmed the metadata/group/state semantic checks, exact
rollback, valid schema-v2 and schema-v1 round trips, observation fail-closed
behavior, and batch-256 production compatibility, but remained **BLOCKED**
because the loader accepted an optimizer belonging to a different identical
model and a same-model optimizer with same-shaped parameters reordered. In the
latter case, saved AdamW moments were installed on the wrong named parameters.

The authorized model-binding correction derives the authoritative ordered
trainable sequence from the supplied model's `named_parameters()` registration
order. Before either save or load can trust optimizer ordinals, the frozen
single AdamW group must contain those exact Python `Parameter` objects in that
exact order, without omissions, foreign objects, duplicates, extras, reversal,
or unsupported groups. Save validates before creating its output directory.
Schema v2 remains version 2 but now requires the model-derived
`optimizer_parameter_names_by_group` metadata; pre-correction v2 checkpoints
without this proof fail closed. The narrow schema-v1 HTGNN path remains
supported without that newer field, while any supplied live optimizer is still
identity/order checked against the target HTGNN model before legacy validation.

Model-binding correction commit
`eb7d075943cbd3f209934d1c962eb60c99f3f64e` then passed an independent
read-only audit. Foreign, reordered, omitted, substituted, duplicated, extra,
reversed, and regrouped optimizer parameters failed both save and load before
valid state installation; rejected saves created no output artifact, and
rejected loads preserved model parameters and buffers plus optimizer state,
groups, settings, and live bindings exactly. A forced post-preflight failure
also restored both targets exactly. Current schema-v2 checkpoints require the
model-derived optimizer-name mapping; older schema-v2 checkpoints without it
fail closed. Schema-v1 compatibility remains HTGNN-only with live binding and
AdamW validation. All nine HTGNN/final-flow/flattened-family and E1/E2/E3
combinations round-tripped exact model and optimizer state.

Candidate observations now contain a compact ordered raw trace for each
attempted warmup and measured step. Completion, finite loss, finite gradients,
optimizer success, CUDA OOM, failure stage, measured latency, candidate status,
counts, summaries, and eligibility must form one legal state. The validator
derives the stored summaries and status from that trace, accepts honest partial
failures as ineligible, and rejects correlated false-nonfinite, false-OOM,
false-success, optimizer-failure, and raw-nonfinite stories. The deterministic
evidence generator runs the full semantic validator before binding any input.

The complete fail-closed bounded pilot reran on the NVIDIA GeForce RTX 4050
Laptop GPU with 6,438,780,928 bytes of VRAM. All candidates 32, 64, 128, and
256 completed and were resource-eligible. Their full-precision median
throughputs were approximately 1,632.650, 2,307.514, 4,844.706, and
10,190.858 windows/second. The 95% threshold was approximately 9,681.315
windows/second; only batch 256 entered the tied set, so the fresh run again
selected HTGNN production batch size **256** under the frozen rule. Its
execution identity is `e9-20260930t174847z`, and the observation-file physical
SHA-256 is
`a092efe258bdc54598f7dc2ca7c9e03771216d14c9fc7b4bd644b2bf30d4f3c2`.

Validation passed 151 focused E.9 tests, 44 focused E.7 tests, 85
D6/full-split tests, 267 frozen E.2-E.8 regressions, and 1,113 complete-suite
tests, all without failures or skips. The evidence path additionally runs 12
real baseline checkpoint/JSONL smokes and verifies Stage C 33/33 and raw CSV
10/10 integrity.

The independent audit verified canonical contract SHA-256
`adef7d23584b204a02af0aed46e1e40178347d90a079dc74f557492dd5e35755`,
physical contract SHA-256
`bd55f28c1f96b063f8c53787646a6b71b0ed55f9a9c1d34a96a6d5ad7d46ab1b`,
and observation-file SHA-256
`a092efe258bdc54598f7dc2ca7c9e03771216d14c9fc7b4bd644b2bf30d4f3c2`.
Stage E.9 is frozen at annotated checkpoint tag
`v0.12-stage-e9-model-binding-resource-pilot`. The `3e-4` learning rate remains
a resource-pilot-only engineering value; neither E.8 nor E.9 finalized a
learning rate. The E.9 checkpoint itself did not authorize E.10; later owner-authorized completion is recorded below.

The post-documentation E.9 validator's committed canonical contract SHA-256 is
`5c26c8b55069af09009f8bf716bf64ec3e0b82e19a6f7ec4f773374d6b688b0b`;
its physical SHA-256 is
`e569712855d068b97e5d5b1717faf7602e1a083b1016654b8b47ed003f0d2254`.
The preceding audit-time hashes are historical, not the final committed report's
hashes. The observation-file hash and frozen methodology remain unchanged.

## Authoritative Working-Tree Experimental State

### Git boundary and provenance preservation

Reconciled on 2026-10-06 in the authoritative checkout
`C:\Project\ComputerNetworks`. HEAD intentionally remains at the frozen E.9
commit `2fa823943ba318fe32b2867bc8e2b1f799d78ac1`, with annotated tag
`v0.12-stage-e9-model-binding-resource-pilot`. Local `origin/main`, live remote
`main`, and the remote peeled tag resolve to that same commit.

E.10/E.11/E.12A/E.12B evidence exists in the authoritative working tree and is not
represented by later Git commits. Later implementation, tests, documentation,
logs, checkpoints, and evidence were deliberately preserved outside committed
history to avoid disturbing provenance-sensitive execution. This is intentional
provenance preservation, not an indication that those stages are unauthorized
or absent. Preserve all pre-existing untracked/ignored files. E.9 remains the
latest Git release checkpoint; E.10/E.11 report PASS denotes completed execution,
not a newly asserted independent release freeze. E.12A's artifact freeze is
represented by its separate attestation, not a Git tag or rewritten originals.

### E.10 COMPLETE — official train/validation study

Study: `e10-official-20261001-v1`; final report:
`results/stage_e10/e10-official-20261001-v1/stage_e10_report.json`.

Phase A completed 27/27 LR-selection runs; Phase B completed validation-only LR
selection for all nine family/task pairs; Phase C completed 18/18 additional
seed runs. There are 45 unique official training runs and 27 final
validation-selected checkpoints. The selected seed-2025 Phase A winners are
reused with seeds 2026/2027. Final aggregation is PASS: its pre/postflight record
`training_executed=false`; E.10 records `predictive_test_array_access=false`
and each run's configuration records `test_predictive_array_access=false`.
No test metric, threshold selection, or pruning occurred in E.10.

| Selected learning rate | E1 | E2 | E3 |
|---|---:|---:|---:|
| HTGNN | 0.001 | 0.001 | 0.001 |
| final_flow_mlp | 0.001 | 0.001 | 0.001 |
| flattened_five_flow_mlp | 0.0001 | 0.0003 | 0.0001 |

### E.11 COMPLETE through reviewed recovery — first-test evaluation

Original run: `e11-official-20261006-v1`; reviewed continuation:
`e11-recovery-20261006-v1`. Combined final report:
`results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/stage_e11_report.json`.

The combined report is PASS for 27/27 frozen checkpoints: six original HTGNN
E1/E2 results were reused in place and 21 checkpoints were evaluated during
reviewed continuation. Original FAILED status, six results, manifest, and
test-consumption evidence remain preserved. The original
`logs/stage_e11/e10-official-20261001-v1/test_consumption.json` remains
`CONSUMED_OR_RESERVED`, physical SHA-256
`f1b528868e61c9650acb75692ae933bab1552aedb8b7fbffb6c53a25b58b0a0a`.
Do not repeat E.11 or portray the test partition as unused.

The original evaluator incorrectly required long targets for E3; the frozen
adapter legitimately emits float32 `[B]` targets. The correction repaired this
target-dtype check without changing model, split, metric, or decision methodology.
The interrupted checkpoint restarted within the reviewed continuation; the
original failure was not erased. No training, pruning, LR selection, or threshold
tuning occurred. Corrected evaluator physical SHA-256:
`ce41f85f285a0557ed9898c5c45deb6ba534e42804dab39e3cea521cf4c61da3`.

Primary E2 eligible-class test macro-F1 across seeds 2025/2026/2027:

| Family | Mean | Sample standard deviation (ddof=1) |
|---|---:|---:|
| HTGNN | 0.8759879709008183 | 0.02264023047925946 |
| final_flow_mlp | 0.7241750943498745 | 0.0029248034408850486 |

These are descriptive seed summaries, not confidence intervals. E1 remains a
strict chronological multiclass diagnostic with limited support. E3 strict
chronological binary test generalization was poor across all three families:
mean average-precision PR-AUC was approximately 0.08903136 (HTGNN), 0.09291219
(final-flow MLP), and 0.09658414 (flattened-five-flow MLP). Preserve this limitation.

### E.12A COMPLETE and artifact-frozen — validation-only pruning selection

Run: `e12a-20261006-v1`; evidence:
`results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/`.
Training attention ranked features on the same bounded 8,192 E2-train examples
for each of three seeds; complete E2 validation evaluated five declared
candidates: 0%, 10%, 25%, 50%, 75%. The largest candidate satisfying EVERY E2
seed's pruned eligible-class validation macro-F1 >= 0.95 times its own unpruned
value was requested per-category ratio **0.25**. Passing candidates: 0%, 10%,
25%; failing candidates: 50%, 75%. No test predictive arrays were opened,
test support was not consulted, and model states remained unchanged.

The frozen actual inference graph retains 61/78 feature nodes and removes 17,
retains all eight category nodes and the sample node, and reduces total nodes
87 -> 70 and directed structural edges 86 -> 69. Requested ratio 0.25 is distinct
from actual feature removal 17/78. The input adapter still constructs the full
graph; compaction removes inference computation, not learned parameters.

The original proposed policy and COMPLETE_PENDING_REVIEW evidence remain
immutable. The separate
`results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1-freeze/freeze_attestation.json`
records review PASS, owner-approved freeze and 5% relative macro-F1 tolerance,
the exact mask, three E2 checkpoint identities, and original artifact/source
hashes. It explicitly records `e12b_execution_authorized=false`.

### E.12B COMPLETE — final frozen-mask comparison; independent audit PASS

Run: `e12b-20261006-v1`; final report:
`results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/stage_e12b_report.json`.
All nine pruned HTGNN evaluations completed: E2 primary, E1 diagnostic and E3
binary, each at seeds 2025/2026/2027, with the same frozen E2-derived mask.
Existing E.11 unpruned results were reused. No retraining, model-parameter change,
policy change, threshold tuning or repeated unpruned test evaluation occurred.

The execution report and terminal status retain `COMPLETE_PENDING_REVIEW`.
The subsequent independent final audit is PASS, as confirmed in the owner's
2026-10-06 closure instruction; original execution artifacts remain untouched.
This closure checks persisted evidence and records that verdict, without claiming
a new independent execution audit or rewriting the execution-time status.

Primary E2 eligible-class test macro-F1 mean/sample SD changes from
0.8759879709 / 0.0226402305 to 0.8896826984 / 0.0090459554. Mean absolute
change is +0.0136947275; mean per-seed relative improvement is +1.617299%.
Seed 2025 improves by +5.393129%; seeds 2026/2027 change by -0.130371% and
-0.410860%. Every seed remains within the frozen 5% relative-loss rule.

The final graph retains 61/78 features: 87 -> 70 nodes and 86 -> 69 directed
edges. All eight category nodes and category-to-sample edges remain. Temporal
encoding and first GAT work decrease; full input-adapter construction/transfer,
second GAT topology and parameter counts remain unchanged. Validation
resident-forward paired median speed ratios are approximately 1.035x-1.117x
on the RTX 4050 Laptop GPU, not universal deployment speedups. Transient
allocated peak decreases approximately 2.931 MiB / 5.91%; peak reserved remains
56 MiB. E1's Benign-only restricted mean is 0.9307991287 -> 0.9307918724.
E3 mean PR-AUC is 0.0890313615 -> 0.0890480581, still below attack prevalence
0.1293481937: chronological generalization remains poor.

The completed execution binds the corrected four-file bundle SHA-256
`75931dbcba5ec9d3bcd7d4e173ae9c25c96ac917dff00a6f8574b4f8ef0a4c80`.
Its consumption ledger remains `CONSUMED_OR_RESERVED`. Do not rerun E.11,
E.12A or E.12B. The exact next safe boundary is independent final release audit
and final report/release/documentation, with no further ML experiment authorized
or required. Detailed measurements, scientific limitations and exact evidence
bindings are in `docs/final_experimental_results.md`.

### Exact artifact provenance and reconciliation checks

| Artifact | Canonical SHA-256 | Physical SHA-256 |
|---|---|---|
| E.10 final report | `a898e76698c6e8f1cbda38fe25792d73322880fcdf3ae94499709dfb8f0c49d7` | `8ff3f89c4aac3b2948936ea794e710059a7fe441c161b8c20cae0b8e51f6639c` |
| Recovered E.11 final report | `4ee7e6a9d7e28231e6ba6765decfdbdc29db624ab5d188d6f8950df8d9bdeba4` | `ace0d585fbb89b0f3df4376b2f69e35bd3172ab663cc66b369b769e84a964aac` |
| E.12A report | `ccc9be16dee2dea3567bad48f0b9e563cc760914311d0bad4d1611a755d6241d` | `97cc371d78d11d54587a64aadecea26829eb2ec2cd2201428876e9bfcb291b0d` |
| E.12A policy | `f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa` | `28727d3d7bfe94f0d624639859d5a3c965dbda1ccbbab760a2ec7bc091f91317` |
| E.12A freeze attestation | `6b58ab9133b95a8096ab731288f340ac498243c33c52c7c659d60fd770c29a35` | `7f92187689993ea049cf4a364746301f1175d18ae25402e5a0518f2094032e51` |
| E.12B final report | `5c07ba4183a619f648118af6415abe61a97ebaed14a2ed93e9b68d265ae1b3c5` | `1b76a810bdcb887e22bb077a99754aa04f79ab950626ababb38a1893d8c031da` |
| E.12B validation benchmarks | No embedded content digest | `e20065f502cb6cd0f6bd009d506c58b4356c3c1f64d81bb6fa5511b9bfa3652f` |

The earlier read-only reconciliation checked canonical report content/physical sidecars,
45 COMPLETE E.10 lifecycles and 225 bound artifacts including checkpoint bytes,
nine original E.11 bindings, 23 recovery-source bindings, all 27 result/checkpoint
memberships and confusion-matrix totals, and rederived nine seed summaries.
It verified the E.12A freeze's seven original artifacts and four source files,
recomputed all 15 validation macro-F1 values and every-seed acceptance, and
confirmed ranking coverage and the four-file E.12B digest. No predictive array
was opened and no training, pruning sweep, E.11 rerun, or E.12B command was run.
The 1,113-test E.9 full-suite count remains a historical baseline, not a new
later-stage full-suite result.

### Historical handoff-hash blocker and completed-execution authority binding

The frozen E.9 contract's `e9_source_sha256["PROJECT_HANDOFF.md"]` pins the
pre-reconciliation physical bytes:
`0eb4f09d989a8c641736be9637f7677cdccf3e675d73425bbd0e7cf22a5df9f6`.
Those historical bytes remain recoverable from the E.9 Git commit. This
authorized factual handoff update necessarily changes the live handoff hash.

At the earlier reconciliation, `stage_e12b.prepare` called
`stage_e12.prepare`, whose frozen-source loop compared that historical pin with
the live handoff and rejected the updated file. E.10/E.11 preflight loops also contain that inherited live-handoff check.
This is a historical-versus-current documentation binding conflict, not changed
experimental artifacts or permission to weaken their integrity checks.
During that documentation reconciliation, frozen evidence, source, checkpoints
and the historical four-file bundle were unchanged; no pin was rewritten and
no rejection was bypassed.

That earlier blocker was resolved for E.12B by the separately authorized
historical/live authority reconciliation described in
`results/provenance/authority/handoff-reconciliation-20261006-v1/correction_report.md`
and its separate attestation, without changing frozen E.9/E.10/E.11/E.12A
artifacts. The completed E.12B execution binds that attestation, the corrected
bundle and the reconciled live handoff physical hash
`58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5`.
E.12A's earlier freeze attestation records `e12b_execution_authorized=false`;
the later E.12B owner authorization/consumption ledger records the separately
authorized one-time execution. Both historical records remain untouched.

This owner-authorized final closure documentation changes the live handoff
again, after experimental completion. The historical execution pin and original
authority attestation remain unchanged. The completed-run preflight's exact
live-handoff check will reject these new closure bytes; no executable-preflight
readiness or rerun is claimed or required. Preserve historical bindings during
final release audit; no source-level repair, ML execution or Git release action
is authorized by this documentation task.

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

## Frozen Stage D Graph-Input Contract

One logical graph corresponds to one Stage C sample. Its type-local hierarchy is
`78 feature -> 8 category -> 1 sample`: 87 nodes and 86 directed edges. Each
feature belongs to exactly one of eight categories with counts
`2/23/24/5/12/6/4/2`; every category contributes to the sample node. There are
no reverse, same-level, temporal, correlation, or structural self-loop edges.

Each feature node exposes the ordered five-flow value sequence. D4 reads one
sample lazily from read-only Stage C memory maps, D6 constructs only requested
batch-scale logical arrays, and D8 covers every split sequentially without
shuffle or dropped final batches. D8's plan authority is tuple-backed and its
iterator verifies the bound dataset, plan, experiment, split, artifacts,
provenance, memmaps, and topology before and after each D6 construction.

Approved D1-D8 hashes are recorded in `docs/STAGE_D_VERIFICATION.md`. The
current D8 hash is
`d1a757459062fe8ca08f0be5951a5a728898a20d28eafbb12f10ce4646047b1f`.

## Frozen Stage E.1-E.2 PyG Contract

The verified environment is Python 3.13.0, PyTorch 2.14.0+cu130, PyG
2.8.0.post1, and CUDA runtime 13.0 on the RTX 4050 Laptop GPU. The E.1
environment report SHA-256 is
`10af77a05a28d760a9cbf52916e8b3205f5590646af5608882538966e2e36f6a`.

Stage E.2 exposes
`logical_batch_to_heterodata(batch) -> HeteroData` in
`src/graph/pyg_adapter.py`. Its only production input is an approved D6
`LogicalGraphBatch`; it creates one bounded, already-batched, CPU-owned
`HeteroData` with the exact `feature -> category -> sample` topology and no
learned parameters. The canonical E.2 report hash is
`370048a60c7a7b25643d820730bb82438e031eee5f6e8a7c07b49efaf17e4fec`.

The approved E.3 temporal encoder is a shared causal TCN over each feature's
ordered five values: explicit left-pad 2, `Conv1d(1,8,kernel_size=3)`, GELU,
explicit left-pad 2, `Conv1d(8,16,kernel_size=3)`, GELU, then select temporal
position 5. It uses no right padding or dilation and has receptive field 5.
The 16-dimensional temporal result is concatenated with
`Embedding(78,8)`, then fused by
`Linear(24,32) -> LayerNorm(32) -> GELU -> Dropout(0.10)`. This architecture
is implemented and frozen in the Stage E.3 Git checkpoint.

## Frozen Stage E.3 Contract

`src/graph/feature_encoder.py` exposes
`CausalTemporalEncoder.forward_sequence([N,5]) -> [N,16,5]`,
`CausalTemporalEncoder([N,5]) -> [N,16]`, and
`FeatureNodeEncoder([N,5], [N]) -> [N,32]`. The implementation has exactly
1,920 learned parameters: 432 temporal-convolution, 624 feature-ID embedding,
800 fusion-linear, and 64 LayerNorm parameters.

The E.3 validation covers explicit left-only padding, preserved sequence
length, prefix invariance and zero future gradients at all five positions,
position-1 reachability from position 5, position-4 isolation from position 5,
deterministic temporal-order sensitivity, feature-ID distinction, evaluation
determinism, CPU/CUDA forward and backward, and bounded real D4-to-D6-to-E.2
integration for all nine E1/E2/E3 split pairs. Identical E1/E3 inputs produce
identical embeddings under the same model, and all source inputs remain
unmodified. The canonical E.3 report SHA-256 is
`b3e838c90a7488eca2c7b9083744386a6340234549318c4709b1176b87ae9adb`.

Stage E.3 deliberately contains no category embeddings, sample token, GATv2,
classifier, loss, optimizer, training, pruning, or E.4 implementation.

## Frozen Stage E.4 Contract

`src/graph/hierarchical_encoder.py` consumes only frozen Stage E.3 feature
embeddings and the exact graph-major Stage E.2 relations. It initializes eight
learned 32-dimensional category identities and one shared 32-dimensional sample
token, then applies separate feature-to-category and category-to-sample
`GATv2Conv` blocks. Each block uses four concatenated 8-channel heads,
attention dropout 0.10, no self-loops, no shared projections, no edge features,
and no internal residual, followed by external message dropout 0.10, the
original target residual, `LayerNorm(32)`, and GELU.

The module has exactly 4,768 learned parameters; frozen E.3 plus E.4 has 6,688.
Both relation tensors must exactly equal the frozen schema authority used by
E.2, including shape, `torch.long` dtype, source and destination indices, and
edge order, before either GATv2 module runs. In PyG 2.8.0.post1 returned
attention is after internal attention dropout; evaluation mode disables that
dropout and is mandatory for future attention collection. Attention is a model
weighting signal, not causal feature importance. E.4 contains no classifier,
loss, optimizer, training loop, metric, pruning policy, or E.5 implementation.

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

Group windows by final-flow class, preserve chronology within each class, split
each class 70/10/20, and recombine partitions. E2 is not globally
chronological. Validation-selection eligibility is train ≥100 and validation
≥30, with no test information consulted. This set alone governs E2 validation
macro F1, early stopping, best-checkpoint selection, and E.8 learning-rate
ranking. Final locked-test reporting eligibility remains separately defined as
train ≥100, validation ≥30, and test ≥30. E2 preprocessing is fitted on
E2-train only.

### E3 — Primary Strict Global Capture-Order Binary Experiment

Use the global E1 boundaries with `Benign → 0` and every attack label `→ 1`. E1 and E3 may share train-fitted feature state because their feature windows and boundaries are identical.

## Verification Evidence

- Frozen E.7/E.8 E2 selection-leakage correction and independent release audit:
  **44 focused E.7 tests**, **20 focused E.8 tests**, **203 frozen E.2-E.6
  regressions**, **247 E.2-E.7 regressions**, and **962 complete-suite tests
  passed**, with no failures or skips; Stage C/raw integrity remained 33/33 and
  10/10. The combined selection-methodology/E.7/E.8 suite passed 116 tests.
- Stage E.7 implementation, log-ownership correction, and independent release
  re-audit: **42 focused tests passed**, **203 frozen E.2-E.6 regression tests
  passed**, and **936 complete-suite tests passed**, with no failures or skips.
- Frozen E.5 checkpoint complete suite: **857 passed, 0 failed, 0 skipped**.
- Stage E.6 implementation and independent release-audit verification: **17
  focused tests passed**, **166
  frozen E.2-E.5 regression tests passed**, and **874 complete-suite tests
  passed**, with no failures or skips.
- Stage E.5 focused suite: **67 passed**; frozen Stage E.2-E.4 regression
  suite: **99 passed**.
- Stage E.4 focused suite: **31 passed**; Stage E.2-E.3 regression suite:
  **68 passed**; Stage D regression suite: **231 passed**.
- Stage E.3 focused suite: **30 passed**; Stage E.2 suite: **38 passed**;
  combined Stage E.2-E.3 suite: **68 passed**; Stage D regression suite:
  **231 passed**.
- Stage E.4 deterministic report: canonical SHA-256
  `8de906e6adc61c6f0f32f7b4a2c2e1a2b32a39f5fd0b1c9f60f26ebe7e655db9`;
  physical file SHA-256
  `707ed190e0e9e7c36ae5161b5c192ae54f27789515de93c0ef26f2c77397463b`.
- Stage E.5 deterministic report: canonical SHA-256
  `9be79532fdc8a685cb5ad8d7b5f7cf88f752e4ae6cac1646b20a28f63f2c9e1f`;
  physical file SHA-256
  `edb54f1501de17b4a73f92b6457113fceee014bc2ad7344e4bf0c81ad2628c88`.
- Stage E.3 deterministic report: canonical SHA-256
  `b3e838c90a7488eca2c7b9083744386a6340234549318c4709b1176b87ae9adb`;
  physical file SHA-256
  `790afb645080f7fd0ae70b7d5c7cb6c1ad4dcab9da8fd28ec9e21ab7b3e6b956`.
- Two complete corrected full-corpus runs: 33 canonical artifacts each.
- Relative file sets identical; all 33 corresponding SHA-256 values identical.
- Comparison result: missing = 0, extra = 0, differing = 0.
- All 10 raw CSV files remained byte-identical.
- Independent re-audit verdict: **PASS — Stage C can be frozen**.
- Stage D release-audit verdict: **PASS — D1-D8 graph input can be frozen**.
- Stage E.3 release-audit verdict: **PASS — causal feature encoder can be frozen**.
- Stage E.4 release-audit verdict: **PASS — hierarchical GATv2 encoder can be frozen**.

Large arrays and local recovery copies are deliberately excluded from Git. Compact validation evidence is in `results/stage_c_full_validation.json`, `results/stage_d1/` through `results/stage_d8/`, and `results/stage_e1/` through `results/stage_e8/`.

## Frozen Stage E.5 Contract

Stage E.5 is the complete untrained HTGNN model contract. Its construction-time task/output contract is immutable through
the supported public model API. Independent E.5 audits should reproduce public
task-mutation rejection, ordinary and currently detected head corruption,
task/output shapes, trainability, stable `state_dict()` structure, final-sample
readout, target/metadata independence, exact E.4 attention passthrough,
topology enforcement, multi-graph isolation, bounded nine-pair integration,
and frozen-data integrity.

E.5 audits must not require resilience to arbitrary reflective Python mutation
such as reassigning `__class__`, rewriting `nn.Module._modules`, monkeypatching
methods or module globals, mutating class dictionaries, or replacing imported
modules/functions. A caller deliberately exercising arbitrary same-process
Python code execution is outside the supported API and the model-integrity
threat model. Existing detection of some isolated internal head mutations is
defense in depth and does not expand this boundary. This policy does not weaken
architecture, leakage, topology, provenance, or data-integrity checks, and it
does not authorize E.6.

## Known Methodological Limitations

1. Capture order is deterministic but not guaranteed physical chronology because timestamps lack AM/PM.
2. T=5 windows are not guaranteed individual conversations.
3. Repeated temporal/network states occur across partitions. Indices do not overlap, but evaluation is not exclusively on previously unseen states.
4. E2 preserves within-class chronology rather than one global timeline.
5. The chosen T=5 window length has not been established as optimal; completed model-family comparisons do not isolate a causal effect of window length.
6. Original report performance claims are not verified results.
7. Trained checkpoints, first-test results, and an artifact-frozen inference pruning policy now exist as recorded above. No prediction calibration or E3 threshold tuning was performed. E.12B pruned-test results and validation resident-forward efficiency observations exist; they do not establish universal or end-to-end deployment gains.
8. The bounded E.8 E3 validation prefix contains 4,095 attacks and one benign example; its observed PR-AUC is highly composition-dependent.
9. The Stage E.9 batch-size selection is a bounded hardware- and environment-specific engineering result, not a predictive-performance result or portability guarantee.
10. E1 strict chronological multiclass diagnostic reporting has limited class support; E3 strict chronological binary test generalization was poor for every model family.
11. Attention is a pruning-ranking heuristic, not a causal feature explanation. The bounded E.12A ranking sample did not contain DDOS attack-LOIC-UDP (class code 5).
12. The E.12A 5% relative eligible-class validation macro-F1 rule is owner-approved, not a statistical noninferiority guarantee or a guarantee of test performance.
13. Exact inference graph reduction does not automatically imply end-to-end speedup or input-adapter memory savings; model parameter count is unchanged by inference-time pruning.
14. Historical E.11 CUDA arithmetic is not certified bitwise equivalent to the strict E.12 deterministic context; do not repeat unpruned test evaluation to resolve that limitation.

## Current Boundary and Exact Next Step

The committed historical boundary remains frozen E.9. The authoritative
working-tree experimental boundary is E.10 COMPLETE, E.11 COMPLETE through
reviewed recovery, E.12A COMPLETE and artifact-frozen, and E.12B COMPLETE and
independently audited PASS as confirmed by the owner. The approved experimental
scope through E.12B is complete.

The exact next safe boundary is **independent final release audit of the completed
experimental scope and final report/documentation**. Final report/release/
documentation is now the active boundary. No further ML experiment is currently
authorized or required. Review the final measured results and frozen-evidence
bindings in `docs/final_experimental_results.md`; preserve historical Git,
execution status, failure/recovery and authority-hash provenance.

This documentation closure authorizes no experiment, predictive-array access,
E.11/E.12A/E.12B repetition, source/artifact repair, checkpoint change or Git
commit/push/tag/reset/clean/stash. Release actions require separate authorization
after final review. Stop at final release-audit/documentation readiness.
