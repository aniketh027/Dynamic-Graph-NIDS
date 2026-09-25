# Project Handoff — Frozen Through E.2 Provenance Cache Correction

## Project Goal

The project studies automated feature extraction and dynamic heterogeneous temporal graph optimization for efficient network intrusion detection using CSE-CIC-IDS2018. The frozen checkpoint includes the Stage E.6 bounded forward/resource pilot, tagged `v0.9-stage-e6-resource-pilot`, and the independently audited E.2 provenance-cache correction checkpoint, tagged `v0.9.1-e2-provenance-cache`. No training, pruning, or predictive-performance result exists yet.

## Authority and Source Hierarchy

Use sources in this order when continuing the work:

1. The current task's explicit requirements and stage boundary.
2. `results/stage_e5/htgnn_classifier_validation.json` and `docs/stage_e5_htgnn_classifier.md` for the frozen Stage E.5 untrained classifier contract.
3. `results/stage_e4/hierarchical_encoder_validation.json` and `docs/stage_e4_hierarchical_encoder.md` for the frozen Stage E.4 hierarchical graph encoder.
4. `results/stage_e3/feature_encoder_validation.json` and `docs/stage_e3_feature_encoder.md` for the frozen Stage E.3 causal feature-node encoder.
5. `results/stage_e1/`, `results/stage_e2/`,
   `docs/stage_e2_heterodata_adapter.md`, and
   `docs/stage_e2_provenance_cache.md` for the verified PyG environment,
   D6-to-`HeteroData` adapter, and independently audited and frozen
   provenance-cache correction.
6. `docs/STAGE_D_VERIFICATION.md` and `results/stage_d1/` through `results/stage_d8/` for the independently verified Stage D graph-input milestone.
7. `docs/STAGE_C_VERIFICATION.md` and `results/stage_c_full_validation.json` for the independently verified Stage C milestone.
8. Frozen implementation and tests in `src/preprocessing/`, `src/graph/`, `scripts/`, and `tests/`.
9. Compact Stage B and temporal-analysis evidence in `results/data_profile/` and `results/temporal_analysis/`.
10. The original report and extracted text as background only; their experimental performance claims are not verified authority.

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

Group windows by final-flow class, preserve chronology within each class, split each class 70/10/20, and recombine partitions. E2 is not globally chronological. Reporting eligibility is train ≥ 100, validation ≥ 30, and test ≥ 30. E2 preprocessing is fitted on E2-train only.

### E3 — Primary Strict Global Capture-Order Binary Experiment

Use the global E1 boundaries with `Benign → 0` and every attack label `→ 1`. E1 and E3 may share train-fitted feature state because their feature windows and boundaries are identical.

## Verification Evidence

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

Large arrays and local recovery copies are deliberately excluded from Git. Compact validation evidence is in `results/stage_c_full_validation.json`, `results/stage_d1/` through `results/stage_d8/`, and `results/stage_e1/` through `results/stage_e5/`.

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
7. The untrained task-specific classifier and learned GATv2 attention exist, but no training, calibrated prediction, performance result, pruning policy, or physical node removal is implemented.

## Current Boundary and Exact Next Step

Stop after the independently audited and frozen E.2 provenance-validation
performance correction built on the frozen Stage E.6 checkpoint. E.7 remains
unimplemented. Do not train a model, implement pruning, optimize training batch
size, extend the E.2 correction, or begin E.7 or later stages under this
scope.

The exact next safe numbered boundary is Stage E.7, which requires separate
explicit authorization. This correction remains an engineering-project
performance correction and does not authorize or constitute a new research
experiment.
