# Dynamic Graph Optimization and Automated Feature Extraction for Efficient Intrusion Detection in Network Systems

This student research project investigates an intrusion-detection pipeline that combines automated feature extraction, heterogeneous temporal graphs, graph neural networks, and attention-guided graph optimization. The repository currently contains the verified data/preprocessing foundation and framework-independent Stage D graph-input layer; it does not yet contain a graph-framework object or trained detection model.

## Current Status

Stage C preprocessing and the Stage D graph-input layer are independently verified and frozen. Stage D defines the approved `78 feature -> 8 category -> 1 sample` hierarchy (87 nodes and 86 directed edges) through D1-D8. The next project activity is **model/framework design** against that frozen contract.

No PyG/HeteroData construction, HTGNN, GATv2, temporal encoder, pruning, model-training, or performance result is claimed in this checkpoint.

## Project Pipeline

| Pipeline stage | Status |
| --- | --- |
| CSE-CIC-IDS2018 dataset inspection and Stage B profiling | Completed |
| Canonical cleaning, T=5 windows, E1/E2/E3 splits, train-only preprocessing | Completed, independently verified, frozen |
| Framework-independent Stage D graph-input schema, adapters, batching, and full-split integration | Completed, independently verified, frozen |
| Graph-framework and model design | Planned; design review is next |
| HTGNN / GATv2 detection | Planned |
| Attention-guided graph optimization and pruning | Planned |
| Model evaluation | Planned |

In sequence: dataset → preprocessing → feature engineering → heterogeneous temporal graph → HTGNN / GATv2 → detection → attention-guided graph optimization/pruning → evaluation.

## Dataset

The project uses **CSE-CIC-IDS2018**. The dataset is not redistributed by this repository. Obtain it from its authorized source and place these 10 CSV files under `data/raw/csv/`:

```text
02-14-2018.csv
02-15-2018.csv
02-16-2018.csv
02-20-2018.csv
02-21-2018.csv
02-22-2018.csv
02-23-2018.csv
02-28-2018.csv
03-01-2018.csv
03-02-2018.csv
```

The optional Parquet comparison copy belongs under `data/raw/parquet/`. Both raw-data directories are ignored by Git. See [`data/raw/README.md`](data/raw/README.md) for setup details.

## Verified Dataset Facts

| Measurement | Verified value |
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

The model-input convention is exactly 78 features. `Timestamp` and `Label` are metadata/target fields, not model-input features.

## Temporal Representation

Stage C constructs a **contiguous capture-order temporal window** from five consecutive clean flow records (`T=5`). Files are ordered by source-file capture date, then by synthetic `Original_Index`. `Original_Index` is assigned before embedded-header filtering. The label of the fifth/final flow is the window target.

These windows are not guaranteed to represent individual network conversations. `T=5` is a reasonable engineering choice, not a mathematically optimal value, and its predictive value has not yet been measured.

## Evaluation Configurations

- **E1 — strict global capture-order multiclass diagnostic:** global 70/10/20 split, with no shuffle or stratification. It is diagnostic only and is not the primary multiclass performance evaluation.
- **E2 — primary multiclass supervised experiment:** group windows by final-flow class, preserve chronology within each class, split each class 70/10/20, then recombine partitions. E2 is not globally chronological. A class is reportable only when train ≥ 100, validation ≥ 30, and test ≥ 30.
- **E3 — primary strict global capture-order binary experiment:** the E1 global 70/10/20 ordering with `Benign → 0` and every attack label `→ 1`.

All preprocessing transforms are fitted on the applicable training partition only. E2 has its own E2-train fit; E1 and E3 share state because their feature membership and split boundaries are identical.

## Validation

- **691 tests passed; 0 failed.** Stage D contributes 231 passing tests, including 44 D8 tests.
- Two corrected full-corpus runs produced identical relative sets of 33 canonical artifacts.
- All 33 corresponding SHA-256 values were identical; missing = 0, extra = 0, differing = 0.
- All 10 raw CSV files remained byte-identical.
- Stage C passed independent adversarial re-audit and is frozen.
- Stage D D1-D8 passed independent adversarial release audit; see [`docs/STAGE_D_VERIFICATION.md`](docs/STAGE_D_VERIFICATION.md).

Compact machine-readable evidence is retained in [`results/stage_c_full_validation.json`](results/stage_c_full_validation.json). Multi-gigabyte arrays are deliberately not stored in Git.

## Project Structure

```text
configs/                       portable run configuration
data/raw/                      local-only CIC-IDS2018 input and setup guide
data/processed/                local-only generated Stage C artifacts
docs/                          verification and freeze documentation
results/data_profile/          compact Stage B reports and metadata
results/inspection/            compact inspection report
results/temporal_analysis/     compact T=5 and T-sweep reports
scripts/                       inspection, profiling, preprocessing, validation CLIs
src/analysis/                  temporal continuity and robustness analysis
src/preprocessing/             frozen Stage C implementation
src/graph/                     frozen framework-independent Stage D graph-input layer
src/utils/                     logging and reproducibility helpers
tests/                         setup, profiling, analysis, and Stage C tests
PROJECT_HANDOFF.md             verified milestone handoff and next boundary
```

`extracted_report.txt` and `graph optimization papers (1).docx` preserve the original project write-up and literature material. Their performance statements are background material, not verified experimental results.

## Installation / Environment

The project targets Python 3.10–3.13 and was verified on Python 3.13. Create an environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively:

```powershell
conda env create -f environment.yml
conda activate nids-graph
```

`requirements-gnn.txt` documents the later modelling stack and is not required for Stage C.

## Running Tests

From the repository root:

```powershell
python -m pytest
```

Real-corpus tests expect the 10 CSV files under `data/raw/csv/`.

## Stage C Reproduction

Full-corpus reproduction is disk- and time-intensive. It generates arrays locally and does not require downloading generated arrays from GitHub.

```powershell
python scripts/run_preprocessing.py
python scripts/run_preprocessing.py --config configs/run2_repro.yaml
python scripts/validate_stage_c.py `
  --output-dir data/processed/preprocessed `
  --compare-dir data/processed/preprocessed_run2 `
  --raw-dir data/raw/csv `
  --report results/stage_c_full_validation.json
```

For a bounded pipeline smoke run, use `python scripts/run_preprocessing.py --debug`. Generated output directories remain Git-ignored.

## Limitations

1. Capture order is a deterministic proxy, not guaranteed physical chronology, because dataset timestamps lack AM/PM.
2. T=5 windows are not guaranteed to represent individual conversations.
3. Repeated temporal/network states occur across evaluation partitions. Split indices do not overlap, but results must not be described as performance exclusively on previously unseen states.
4. E2 preserves chronology within each class, not one global timeline.
5. T=5 predictive value remains unmeasured because no model has been trained.
6. Performance claims in the original project report are not verified results.
7. No PyG/HeteroData object, temporal encoder, GATv2/HTGNN model, training, learned attention, or pruning has yet been implemented.

## Next Stage

The exact next step is **model/framework design** against the frozen Stage D graph-input contract. Stage C algorithms and Stage D D1-D8 contracts, counts, and evidence remain frozen during that review.
