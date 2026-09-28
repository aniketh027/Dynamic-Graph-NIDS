# Stage E.9 Baselines and HTGNN Batch-Size Pilot

Stage E.9 implements the two frozen non-graph MLP baselines and executes the
bounded HTGNN E2 training-step resource pilot. It is project engineering
evidence, not full model training, a model-quality comparison, locked-test
evaluation, learning-rate selection, weighted-loss work, or pruning.

## Baseline contracts

The final-flow MLP receives only temporal position five as `float32[B,78]`.
It cannot observe flows 1–4. Its hidden path is exactly
`Linear(78,64) -> LayerNorm(64) -> GELU -> Dropout(0.10) -> Linear(64,32)
-> LayerNorm(32) -> GELU -> Dropout(0.10)`.

The flattened five-flow MLP receives the full ordered `float32[B,5,78]`
window and reshapes it in temporal-major order to `[B,390]`. Its hidden path
is the same except for `Linear(390,64)` at the input. Neither model imports or
executes graph, TCN, attention, pooling, or pruning code.

Both use `Linear(32,15)` heads for E1/E2 and `Linear(32,1)` for E3. Forward
returns raw logits, with canonical E3 output shape `[B]`.

| Baseline | E1 parameters | E2 parameters | E3 parameters |
| --- | ---: | ---: | ---: |
| Final-flow MLP | 7,823 | 7,823 | 7,361 |
| Flattened five-flow MLP | 27,791 | 27,791 | 27,329 |

The thin integration adapter consumes only the frozen E.2 feature values and
passes dense tensors to the graph-free models. Bounded training and validation
smoke reuse the Stage E.7 loss factory, AdamW factory, train/validation epoch
helpers, validation objectives, and early-stopping implementation. The smoke
does not write checkpoints or JSONL logs. State-dictionary schema identity and
the E.7 exclusive log-reservation/alias-rejection contract are reused and
tested. No competing preprocessing, split, label, eligibility, optimizer,
stopping, or training-loop authority was added.

## HTGNN E2 resource pilot

The pilot used only the authoritative E2 train stream. Every candidate
restarted the seed-2025 affine traversal, used the unchanged HTGNN, unweighted
cross-entropy, AdamW with weight decay `1e-4`, and the resource-only learning
rate `3e-4`. Each candidate ran five unmeasured warmup optimizer steps and 20
measured optimizer steps. Each measured interval included host-to-device
transfer, forward, loss, backward, and optimizer step, with CUDA synchronization
before and after timing. D4/D6/E.2 CPU construction was outside the timed
interval. Pilot-trained weights were discarded.

The live device was an NVIDIA GeForce RTX 4050 Laptop GPU with 6,438,780,928
bytes of VRAM. Values below are runtime observations, not canonical replay
targets.

Baseline allocated memory was 17,186,816 bytes for every candidate. Baseline
reserved memory was 25,165,824 bytes for batches 32, 64, and 256, and
23,068,672 bytes for batch 128. These nonzero baselines are reported separately
from the absolute peaks and peak deltas in the observational JSON.

| Batch | Median step (ms) | Nearest-rank p95 (ms) | Median windows/s | Peak allocated | Peak reserved | Reserved / VRAM | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 32 | 15.320 | 19.568 | 2,089.209 | 22,591,488 B | 27,262,976 B | 0.4234% | yes |
| 64 | 17.265 | 22.458 | 3,708.789 | 28,013,056 B | 52,428,800 B | 0.8143% | yes |
| 128 | 17.217 | 23.995 | 7,434.589 | 39,496,704 B | 48,234,496 B | 0.7491% | yes |
| 256 | 16.090 | 24.150 | 15,913.454 | 61,809,152 B | 71,303,168 B | 1.1074% | yes |

All candidates completed with finite losses and gradients, successful optimizer
steps, and absolute peak reserved memory strictly below the 80% eligibility
threshold. The maximum median throughput was 15,913.454 windows/s; its 95%
threshold was 15,117.782 windows/s. Only batch size 256 entered the tied set, so
the frozen smallest-in-tied-set rule selected **HTGNN training batch size 256**.

## Scope and limitations

The selected batch size is a hardware- and environment-specific engineering
configuration supported by this bounded pilot. Stage E.9 does not finalize a
learning rate, weighted-loss configuration, task performance, final model
checkpoint, or test performance. No validation or test data contributed to
the resource measurements or selection. The baseline smoke is not a model
comparison, and the timing observations should be remeasured on materially
different hardware or software environments.

Compact deterministic contract evidence is stored in
`results/stage_e9/stage_e9_contract.json`. Individual noisy timing, throughput,
loss, and CUDA-memory observations remain separate in
`results/stage_e9/htgnn_batch_size_pilot_observations.json`.

Stage E.9 stops here. E.10, full training, three-seed studies, locked-test
evaluation, learning-rate finalization, weighted-loss experiments, and pruning
remain unauthorized.
