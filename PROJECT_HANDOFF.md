# Project Handoff — Frozen Through Stage E.4

## Project Goal

The project studies automated feature extraction and dynamic heterogeneous temporal graph optimization for efficient network intrusion detection using CSE-CIC-IDS2018. The frozen checkpoint stops after the Stage E.4 hierarchical GATv2 encoder, tagged `v0.7-stage-e4-hierarchical-encoder`. No classifier, training, pruning, or performance result exists yet.

## Authority and Source Hierarchy

Use sources in this order when continuing the work:

1. The current task's explicit requirements and stage boundary.
2. `results/stage_e4/hierarchical_encoder_validation.json` and `docs/stage_e4_hierarchical_encoder.md` for the frozen Stage E.4 hierarchical graph encoder.
3. `results/stage_e3/feature_encoder_validation.json` and `docs/stage_e3_feature_encoder.md` for the frozen Stage E.3 causal feature-node encoder.
4. `results/stage_e1/`, `results/stage_e2/`, and `docs/stage_e2_heterodata_adapter.md` for the verified PyG environment and D6-to-`HeteroData` adapter.
5. `docs/STAGE_D_VERIFICATION.md` and `results/stage_d1/` through `results/stage_d8/` for the independently verified Stage D graph-input milestone.
6. `docs/STAGE_C_VERIFICATION.md` and `results/stage_c_full_validation.json` for the independently verified Stage C milestone.
7. Frozen implementation and tests in `src/preprocessing/`, `src/graph/`, `scripts/`, and `tests/`.
8. Compact Stage B and temporal-analysis evidence in `results/data_profile/` and `results/temporal_analysis/`.
9. The original report and extracted text as background only; their experimental performance claims are not verified authority.

If a later document conflicts with verified Stage C facts, stop and resolve the conflict before changing code or claims.

## Frozen Status

Stage C passed an independent adversarial re-audit and is frozen. Stage D D1-D8 also passed an independent adversarial release audit and is frozen. Stage E.1-E.2 passed independent environment, conversion, CUDA-transfer, storage-ownership, provenance, and integrity checks. Stage E.3 passed an independent causal-contract, architecture, integration, device, evidence, and regression audit and is frozen. Stage E.4 passed an independent exact-topology, attention-semantics, architecture, isolation, leakage, device, integration, evidence, and regression audit and is frozen. Do not change Stage C cleaning, transforms, windows, splits, the Stage D graph-input layer, or the Stage E.1-E.4 environment, adapter, encoders, verified counts, hashes, or tests without a separately approved methodology change and a new audit.

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

- Current complete suite: **790 passed, 0 failed**.
- Stage E.4 focused suite: **31 passed**; Stage E.2-E.3 regression suite:
  **68 passed**; Stage D regression suite: **231 passed**.
- Stage E.3 focused suite: **30 passed**; Stage E.2 suite: **38 passed**;
  combined Stage E.2-E.3 suite: **68 passed**; Stage D regression suite:
  **231 passed**.
- Stage E.4 deterministic report: canonical SHA-256
  `8de906e6adc61c6f0f32f7b4a2c2e1a2b32a39f5fd0b1c9f60f26ebe7e655db9`;
  physical file SHA-256
  `707ed190e0e9e7c36ae5161b5c192ae54f27789515de93c0ef26f2c77397463b`.
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

Large arrays and local recovery copies are deliberately excluded from Git. Compact validation evidence is in `results/stage_c_full_validation.json`, `results/stage_d1/` through `results/stage_d8/`, and `results/stage_e1/` through `results/stage_e4/`.

## Known Methodological Limitations

1. Capture order is deterministic but not guaranteed physical chronology because timestamps lack AM/PM.
2. T=5 windows are not guaranteed individual conversations.
3. Repeated temporal/network states occur across partitions. Indices do not overlap, but evaluation is not exclusively on previously unseen states.
4. E2 preserves within-class chronology rather than one global timeline.
5. The predictive value of T=5 has not been measured.
6. Original report performance claims are not verified results.
7. Learned GATv2 attention now exists, but no classifier, training, calibrated prediction, performance result, pruning policy, or physical node removal is implemented.

## Current Boundary and Exact Next Step

Stop after the frozen Stage E.4 hierarchical GATv2 encoder. Do not train a model, implement pruning, or begin later model stages under this scope.

The exact next safe boundary is **Stage E.5**, beginning only after a separate explicit Stage E.5 contract and authorization. It must not be inferred from the Stage E.4 checkpoint authorization.
