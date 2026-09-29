# Project Handoff — Stage E.8 Frozen; Stage E.9 Fail-Closed Correction Awaiting Re-Audit

## Project Goal

The project studies automated feature extraction and dynamic heterogeneous temporal graph optimization for efficient network intrusion detection using CSE-CIC-IDS2018. The frozen checkpoint includes the Stage E.6 bounded forward/resource pilot, the independently audited E.2 provenance-cache correction, the corrected Stage E.7 training harness, and the independently audited Stage E.8 bounded optimization pilot. Stage E.8 provides bounded train/validation learning-rate observations only: no heavy/full model training, finalized learning rate, locked-test evaluation, pruning, or project-level predictive-performance result exists yet. The Stage E.9 engineering contract is frozen below. Its first implementation and its production-integration correction were each independently audited and blocked. The authorized fail-closed correction has now made checkpoint loading transactional, strengthened raw execution invariants, and rerun the complete bounded pilot. E.9 remains unfrozen pending another independent read-only re-audit.

## Authority and Source Hierarchy

Use sources in this order when continuing the work:

1. The current task's explicit requirements and stage boundary.
2. `results/stage_e9/stage_e9_contract.json`,
   `results/stage_e9/htgnn_batch_size_pilot_observations.json`, and
   `docs/stage_e9_baselines_batch_size_pilot.md` for the current, unfrozen
   Stage E.9 correction evidence and its explicit limitations.
3. `results/stage_e8/htgnn_optimization_pilot_contract.json`,
   `results/stage_e8/htgnn_optimization_pilot_observations.json`, and
   `docs/stage_e8_optimization_pilot.md` for the frozen Stage E.8 bounded
   optimization-pilot contract and its observational output.
4. `results/stage_e7/htgnn_training_harness_validation.json` and `docs/stage_e7_training_harness.md` for the corrected and frozen Stage E.7 training-harness contract.
5. `results/stage_e5/htgnn_classifier_validation.json` and `docs/stage_e5_htgnn_classifier.md` for the frozen Stage E.5 untrained classifier contract.
6. `results/stage_e4/hierarchical_encoder_validation.json` and `docs/stage_e4_hierarchical_encoder.md` for the frozen Stage E.4 hierarchical graph encoder.
7. `results/stage_e3/feature_encoder_validation.json` and `docs/stage_e3_feature_encoder.md` for the frozen Stage E.3 causal feature-node encoder.
8. `results/stage_e1/`, `results/stage_e2/`,
   `docs/stage_e2_heterodata_adapter.md`, and
   `docs/stage_e2_provenance_cache.md` for the verified PyG environment,
   D6-to-`HeteroData` adapter, and independently audited and frozen
   provenance-cache correction.
9. `docs/STAGE_D_VERIFICATION.md` and `results/stage_d1/` through `results/stage_d8/` for the independently verified Stage D graph-input milestone.
10. `docs/STAGE_C_VERIFICATION.md` and `results/stage_c_full_validation.json` for the independently verified Stage C milestone.
11. Frozen implementation and tests in `src/preprocessing/`, `src/graph/`, `src/training/`, `scripts/`, and `tests/`.
12. Compact Stage B and temporal-analysis evidence in `results/data_profile/` and `results/temporal_analysis/`.
13. The original report and extracted text as background only; their experimental performance claims are not verified authority.

If a later document conflicts with verified Stage C facts, stop and resolve the conflict before changing code or claims.

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

E.10 remains unauthorized.

## Stage E.9 Implementation, Two Blocked Audits, and Corrections

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
throughputs were approximately 1,061.324, 2,210.558, 6,226.861, and
11,598.067 windows/second. The 95% threshold was approximately 11,018.163
windows/second; only batch 256 entered the tied set, so the fresh run again
selected HTGNN production batch size **256** under the frozen rule. Its
execution identity is `e9-20260929t155457z`, and the observation-file physical
SHA-256 is
`6e9b484d0f0d4219d2141a2bcec9ae3b7b52a4114005319ccfc9f1bde6140d6d`.

Validation passed 121 focused E.9 tests, 44 focused E.7 tests, 85
D6/full-split tests, 267 frozen E.2-E.8 regressions, and 1,083 complete-suite
tests, all without failures or skips. The evidence path additionally runs 12
real baseline checkpoint/JSONL smokes and verifies Stage C 33/33 and raw CSV
10/10 integrity.

The `3e-4` learning rate remains a resource-pilot-only engineering value. E.8
did not finalize a learning rate. E.9 remains **unfrozen** pending an independent
read-only re-audit. No E.9 checkpoint tag exists, and E.10 remains unauthorized.

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
5. The predictive value of T=5 has not been measured.
6. Original report performance claims are not verified results.
7. The task-specific training harness and learned GATv2 attention exist, but no full trained-model checkpoint, calibrated prediction, performance result, pruning policy, or physical node removal has been established.
8. The bounded E.8 E3 validation prefix contains 4,095 attacks and one benign example; its observed PR-AUC is highly composition-dependent.
9. The Stage E.9 batch-size selection is a bounded hardware- and environment-specific engineering result, not a predictive-performance result or portability guarantee.

## Current Boundary and Exact Next Step

Stop after the Stage E.9 fail-closed correction and fresh bounded execution.
Do not launch full model training, a three-seed study,
weighted-loss ablations, locked-test evaluation, pruning, or E.10 and later
work.

The exact next safe boundary is an independent, read-only Stage E.9 correction
re-audit. A checkpoint/tag may be created only after an independent PASS and
separate explicit freeze authorization. E.10 remains unauthorized.
