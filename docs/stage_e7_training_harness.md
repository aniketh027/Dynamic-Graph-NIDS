# Stage E.7 HTGNN Training Harness

Stage E.7 adds the engineering subsystem that trains and validates the frozen
E.5 classifier. It does not change E.2 conversion, E.3/E.4 architecture,
E.5 heads, topology, or attention behavior. It does not run the later full
training study, learning-rate selection, imbalance ablations, test evaluation,
or pruning.

## Run contract

`TrainingConfig` requires one task (`E1`, `E2`, or `E3`), one approved seed
(`2025`, `2026`, or `2027`), an explicit learning rate (`1e-4`, `3e-4`, or
`1e-3`), and a bounded batch size. The default production ceiling is 30
epochs. Training batches use a seeded, constant-memory affine permutation;
validation batches remain sequential. Only the current D4/D6 batch is
converted through E.2 and moved to the selected device.

Exact E3 average precision retains only one CPU score and label per validation
example so that the global ranking is correct; input windows, graph batches,
and GPU tensors still remain batch-bounded and are released after use.

The task contracts are:

| Task | Raw output | Loss | Validation selection |
| --- | --- | --- | --- |
| E1 | `[B, 15]` | unweighted `CrossEntropyLoss` | macro F1 restricted to classes supported by the processed train and validation batches |
| E2 | `[B, 15]` | unweighted `CrossEntropyLoss` | macro F1 over `validation_selection_class_codes`, derived only from E2 train support ≥100 and validation support ≥30 |
| E3 | `[B]` | unweighted `BCEWithLogitsLoss` | PR-AUC represented by average precision on validation raw-logit rankings |

Softmax and sigmoid are not applied before the losses. Ordinary training does
not request attention and contains no pruning path. Weighted E2/E3 loss
ablations are intentionally deferred because the current handoff does not put
them inside E.7.

E2 validation selection is deliberately separate from final reporting. The
selection class set is derived in bounded chunks directly from the
authoritative E2 train and validation target memory maps. Its API has no test
input, and E.7 does not read the three-split reporting-eligibility field for
selection. The later locked-test reporting policy remains train ≥100,
validation ≥30, and test ≥30; that policy is not consulted by training, early
stopping, checkpoint selection, or validation metrics.

## Optimization and stopping

The optimizer is AdamW with explicit learning rate, weight decay `1e-4`, and
no scheduler. Each train batch performs zero-grad, forward, loss, backward,
finite-gradient validation, and optimizer step. Nonfinite loss, gradients, or
parameters stop the run with a diagnostic error; no gradient clipping is
applied.

Early stopping maximizes the task validation objective with minimum 3 epochs,
patience 5, and `min_delta=1e-4`. The saved checkpoint is the best qualifying
validation checkpoint, never merely the last epoch. Test data is not accepted
by the E.7 iterator and is absent from optimization, weighting, stopping,
checkpoint selection, and threshold decisions.

## Checkpoints and logs

Checkpoints contain state dictionaries and plain metadata: model and optimizer
state, epoch, task, seed, learning rate, validation value/loss, frozen source
hashes, model version, parameter count, architecture-state schema hash, config,
and Torch RNG state. They do not pickle a model object. Loads reject task,
configuration, model-version, parameter-count, and architecture-schema
mismatches before applying state. The checkpoint format carries the state
needed by a future authorized resume interface, but this stage does not expose
resume execution because the handoff does not require it.

Checkpoint and log directories remain ignored by Git. JSONL logs record epoch,
train/validation loss, selection value, batch and example counts, and traversal
digests. One JSONL log path belongs to exactly one training run. Before model,
data, or optimizer setup, E.7 normalizes the checkpoint and log paths, rejects
resolved checkpoint/log aliases, and atomically reserves the log with exclusive
creation. Every existing log path is rejected, including an empty precreated
file; E.7 never truncates or appends a new run to prior log evidence. Resume and
log-append semantics remain unsupported. Smoke values are engineering checks,
not model-quality results.

Compact deterministic contract and smoke evidence is written by
`scripts/validate_stage_e7.py`; it stores pass/fail facts and shapes, never
checkpoint weights, loss values, predictive scores, timing, or memory claims.

The authorized E2 methodology correction regenerated this evidence after 44
focused E.7 tests, 203 frozen E.2-E.6 regression tests, and 962 complete-suite
tests passed with zero failures or skips. Stage C integrity remained 33/33 and
raw CSV integrity remained 10/10. The corrected canonical evidence SHA-256 is
`9b6669e35344757dde130b7ffc6f089192aae56494a009ba3e76994e7920062a`;
the physical file SHA-256 is
`3ea99b2bb39e9f8a5e6e1b04ef46ba837037d0797cde38dadfd9bb9a1b00e67b`.
This correction is ready for independent re-audit and is not frozen.

## Device and determinism

`--device auto` selects CUDA when available, otherwise CPU. Python, NumPy,
PyTorch CPU, and all visible CUDA devices receive the selected seed. cuDNN
deterministic mode is enabled and benchmarking disabled. This is best-effort
reproducibility, not a bitwise CUDA guarantee: PyG/CUDA scatter and aggregation
kernels can remain nondeterministic.

## Manual full-run example

Run this later when a real full training run is authorized and supervised:

```powershell
.\.venv\Scripts\python.exe scripts\train_htgnn.py --task E2 --learning-rate 0.0003 --seed 2025 --batch-size 32 --max-epochs 30 --device cuda --checkpoint checkpoints\stage_e7\e2_seed2025_lr3e-4.pt --log logs\stage_e7\e2_seed2025_lr3e-4.jsonl
```

Omitting `--max-train-batches` and `--max-validation-batches` traverses the
complete train and validation splits lazily. Do not use this E.7 command for
test evaluation, learning-rate grids, multi-seed studies, weighted-loss
ablations, batch-size selection, baselines, or pruning.
