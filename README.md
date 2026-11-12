# Dynamic Graph Optimization and Automated Feature Extraction for Efficient Intrusion Detection in Network Systems

This research repository implements a CSE-CIC-IDS2018 intrusion-detection pipeline: train-fitted preprocessing, five-flow temporal inputs, a heterogeneous temporal graph neural network (HTGNN), and attention-guided graph pruning at inference time.

The approved experimental scope through E.12B is complete and frozen at [`v1.0-experimental-complete`](https://github.com/aniketh027/Dynamic-Graph-NIDS/tree/v1.0-experimental-complete). The release includes E.10 training/validation, reviewed E.11 recovery, frozen E.12A policy selection, and E.12B comparison. E.9 remains an earlier historical checkpoint. Further experiments and test-set reuse require separate authorization. [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) records the current authority; [final experimental results](docs/final_experimental_results.md) contains measured results and limitations, and [release provenance](docs/release_provenance.md) documents the external evidence and integrity contract.

## Documentation map

| Start here | Purpose |
| --- | --- |
| [Project handoff](PROJECT_HANDOFF.md) | Current authority, frozen state, and authorized boundaries |
| [Final experimental results](docs/final_experimental_results.md) | Measured results, experiment roles, and limitations |
| [Release provenance](docs/release_provenance.md) | External evidence scope and byte-integrity requirements |
| [Stage D.1 taxonomy](docs/STAGE_D1_TAXONOMY.md) | Approved feature-to-category mapping [Manifest](results/stage_d1/feature_taxonomy.json). |
| [Stage D.2 graph schema](docs/STAGE_D2_GRAPH_SCHEMA.md) | Logical nodes, structural edges, and temporal semantics [Schema JSON](results/stage_d2/graph_schema.json). |
| [Stage D.3 structural validation](docs/STAGE_D3_STRUCTURAL_VALIDATION.md) | Endpoint, connectivity, and acyclic hierarchy checks [Validation JSON](results/stage_d3/structural_validation.json). |
| [Stage D.4 lazy adapter](docs/STAGE_D4_LAZY_ADAPTER.md) | Read-only sample access and feature tensor orientation |
| [Stage D.5 experiment boundaries](docs/STAGE_D5_EXPERIMENT_BOUNDARIES.md) | Split membership and experiment boundaries |
| [Stage D.6 batching](docs/STAGE_D6_TEMPLATE_BATCHING.md) | Bounded batch assembly and index requirements |
| [Stage D.7 construction sanity](docs/STAGE_D7_CONSTRUCTION_SANITY.md) | Deterministic sample and batch checks across experiment splits |
| [Stage D.8 lazy integration](docs/STAGE_D8_FULL_SPLIT_INTEGRATION.md) | Sequential batch planning and complete index coverage |
| [Environment report](results/stage_e1/environment-report.json) | Recorded runtime and build provenance |
| [E.6 resource pilot](docs/stage_e6_resource_pilot.md) | Bounded forward-path resource measurements |
| [E.10 training record](docs/stage_e10_full_training.md) | Training and validation execution contract |
| [E.11 evaluation record](docs/stage_e11_test_evaluation.md) | Test-evaluation contract and execution history |
| [E.11 recovery record](docs/stage_e11_failure_recovery.md) | Interrupted evaluation and reviewed recovery |
| [E.12 policy record](docs/stage_e12_attention_pruning.md) | Attention ranking and validation-based selection |
| [Raw-data setup](data/raw/README.md) | Expected source CSVs and local data placement |

For pipeline design, start with [Stage C verification](docs/STAGE_C_VERIFICATION.md) for preprocessing and [Stage D verification](docs/STAGE_D_VERIFICATION.md) for graph input construction.

Reading these records does not require running training or test evaluation.

## Implemented pipeline

1. **Preprocessing and temporal input.** Canonical cleaning produces windows of five consecutive clean flow records (`T=5`) in deterministic capture order. Each input has 78 features, excluding Timestamp and Label; the fifth/final flow supplies the target. A window need not represent one conversation. Preprocessing is fitted only on the applicable training partition, with experiment-specific membership and provenance retained.
2. **Heterogeneous graph.** The frozen full graph is `78 feature -> 8 category -> 1 sample`: 87 type-local nodes and 86 directed structural edges. Lazy graph input and bounded batching feed the PyTorch Geometric `HeteroData` adapter. Each feature node carries its five ordered values.
3. **Causal temporal encoding.** A shared two-layer causal Conv1d encoder uses left padding and takes the fifth/final temporal representation. It combines that representation with learned feature identity before graph aggregation.
4. **GATv2-based HTGNN.** Bipartite GATv2 blocks aggregate feature nodes into category nodes, then category nodes into the sample node. The sample embedding feeds the task-specific classification head. Final-flow and flattened-five-flow MLPs provide graph-free baselines.
5. **Inference-time pruning.** E.12A ranks features using evaluation-mode attention from E2 training examples, then selects a per-category mask using E2 validation only. The frozen 61-feature graph retains all eight category nodes and the sample node: 70 nodes and 69 edges. E.12B transfers the same mask to E1/E2/E3 and compares pruned inference with persisted unpruned results. Pruning changes graph computation; it uses no retraining, fine-tuning, parameter pruning, or parameter-count reduction.

The D.4 adapter exposes a sample as `(78, 5)`, with feature rows and ordered flow
columns. D.6 stacks samples as `(B, 78, 5)`. Use the adapter and batching records
when checking tensor orientation at a model-input boundary.

Implementation is in [preprocessing](src/preprocessing/), [graph adapters and encoders](src/graph/), [training](src/training/), [evaluation](src/evaluation/), and [pruning](src/pruning/). Historical stage documents remain evidence of their original preparation/execution boundary; the current handoff supersedes their stale status statements without rewriting provenance-bound bytes.

- For PyG conversion and provenance checks, see the [E.2 adapter](docs/stage_e2_heterodata_adapter.md) and [provenance cache](docs/stage_e2_provenance_cache.md).
- For model components, see the [E.3 temporal encoder](docs/stage_e3_feature_encoder.md), [E.4 graph encoder](docs/stage_e4_hierarchical_encoder.md), and [E.5 classifier](docs/stage_e5_htgnn_classifier.md).
- For training and resource pilots, see the [E.7 training harness](docs/stage_e7_training_harness.md), [E.8 optimization pilot](docs/stage_e8_optimization_pilot.md), and [E.9 baselines and batch-size pilot](docs/stage_e9_baselines_batch_size_pilot.md).

For graph-input details, read D.1 taxonomy and D.2 topology first, then D.3
structural validation. D.4 introduces sample access, D.6 batches samples, and
D.7/D.8 document construction checks and sequential split coverage. The links
in the documentation map follow those responsibilities.

## Experiment roles

| Experiment | Membership and task | Reporting role |
| --- | --- | --- |
| E1 | Strict global capture-order 70/10/20 split; multiclass | Diagnostic; restricted class support limits interpretation |
| E2 | Final-flow class groups split 70/10/20 chronologically within each class, then recombined; multiclass | Primary supervised multiclass experiment; not one global timeline |
| E3 | E1 global capture-order membership; Benign = 0, attacks = 1 | Strict chronological binary generalization experiment |

E2 validation selection uses training and validation support only. Final test reporting additionally uses the declared test-support eligibility rule. E1/E3 share preprocessing state because their feature membership and boundaries match; E2 has its own train fit. Their reporting roles remain separate.

For the split and experiment boundary definitions behind this table, see [Stage D.5 experiment boundaries](docs/STAGE_D5_EXPERIMENT_BOUNDARIES.md).

Detailed predictive and hardware-specific efficiency measurements belong in [docs/final_experimental_results.md](docs/final_experimental_results.md). Attention is a ranking heuristic, not causal feature importance. Observed resident-forward speed and allocated-memory changes do not establish universal deployment performance or statistical significance. E3 chronological generalization remains poor.

For the comparison between the original and pruned graphs, see the [E.12B comparison record](docs/stage_e12b_final_comparison.md). Read the [E.12A policy record](docs/stage_e12_attention_pruning.md) for the preceding mask selection.

## Data and environment

Raw CSE-CIC-IDS2018 data, generated predictive arrays, checkpoints, and runtime logs are external to Git. See [raw-data setup](data/raw/README.md) for the ten expected CSVs and [release provenance](docs/release_provenance.md) for the exact external evidence archive and per-file integrity contract. No dataset or checkpoint download is supplied here.

Use the raw-data setup when locating source CSVs. Use release provenance when verifying an evidence archive, including its inventory and per-file hashes.

The verified modelling environment is the isolated `.venv`, with CPython 3.13 on Windows. Use [requirements-gnn-lock.txt](requirements-gnn-lock.txt) and the recorded setup commands/build provenance in [the Stage E.1 environment report](results/stage_e1/environment-report.json). The earlier base-only `requirements.txt` is insufficient for the completed graph model. Environment recreation must follow the recorded lock and platform requirements.

For training decisions, consult the E.10 training record. For pruning-mask
selection, consult the E.12 policy record and its E2 validation boundary. The
final-results document brings the completed comparison together with its
reporting limitations.

## Safe verification

The complete test suite includes real-data integration and training paths. For the reviewed release subset, use the isolated environment and exclude the exact real CUDA resource/training test:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider `
  tests/test_stage_e10_full_training.py `
  tests/test_stage_e11_test_evaluation.py `
  tests/test_stage_e11_failure_recovery.py `
  tests/test_stage_e12_pruning.py `
  tests/test_stage_e12b.py `
  tests/test_stage_e9_batch_size_pilot.py `
  tests/test_release_authority.py `
  -k "not test_run_candidate_short_real_cuda_timing_and_memory_path"
```

This subset uses synthetic/temp fixtures and compact provenance. Release verification also runs under a guard that rejects real-checkout writes, real predictive-array opens, backward calls, training, and optimizer steps. A generic `not slow` filter does not establish safety. The separate release fixture authenticates the historical execution handoff and current closure handoff; it leaves frozen execution preflight pins unchanged.

When comparing multiclass results, record the experiment ID alongside the metric.
E1 uses the global capture-order split, while E2 recombines per-class chronological
splits. Consult the experiment-role table and D.5 record when describing either
result in a report.

## Interpretation and release boundary

Capture order is a deterministic proxy rather than guaranteed physical chronology because source timestamps lack AM/PM. Repeated temporal/network states can occur across partitions even though split indices do not overlap. E2 preserves chronology within each class, and its results must not be presented as strict global chronological evaluation. The experiment does not establish an optimal window length or universal deployment efficiency.

[Stage C verification](docs/STAGE_C_VERIFICATION.md), [Stage D verification](docs/STAGE_D_VERIFICATION.md), and the later stage documents preserve the methodology and evidence. Original project/literature write-ups are background, not measured-results authority. Stop at final release-audit/documentation readiness; preserve the completed experiments and their failure/recovery history.
