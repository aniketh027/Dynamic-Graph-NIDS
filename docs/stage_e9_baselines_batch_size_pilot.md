# Stage E.9 Baselines and HTGNN Batch-Size Pilot

Stage E.9 implements the two frozen graph-free MLP baselines and the bounded
HTGNN E2 training-step resource pilot. It is project engineering evidence, not
full model training, a model-quality comparison, locked-test evaluation,
learning-rate selection, weighted-loss work, or pruning.

The original E.9 implementation commit
`a717a2af00d1f333f462b27ad8dad7f9e05ac31b` was independently audited and
**BLOCKED**. Its resource measurements reproduced, but the selected batch size
256 could not enter the production training path, baseline checkpoint identity
was not first-class, and critical observation fields were not validated
fail-closed. The correction described here does not erase that history. E.9 is
still unfrozen and requires an independent re-audit.

## Production training integration correction

The production batch-size bound is now exactly `1 <= batch_size <= 256` in
both `TrainingConfig` and the authoritative bounded index iterator. The default
remains 32. Split membership, order, shuffle semantics, tensor contents,
topology, preprocessing, and memory ownership are unchanged. Boundary tests
accept 128 and 256 and reject 257 and 512.

The shared E.7 harness now derives an immutable production model contract from
the actual supported model instance. The supported families are `htgnn`,
`final_flow_mlp`, and `flattened_five_flow_mlp`. Checkpoint schema v2 binds the
task, model family, architecture version, input contract, parameter count,
training configuration, epoch, validation-selection state, model state, and
optimizer state. The loader checks identity before applying weights. Existing
valid schema-v1 HTGNN checkpoints remain loadable; schema-v1 baseline
checkpoints are rejected.

Both baseline families now run through the shared production harness, including
the frozen losses, AdamW with weight decay `1e-4`, finite-gradient checking,
train/eval modes, validation metrics, early stopping, best-checkpoint
selection, JSONL accounting, exclusive one-log-per-run ownership, and
checkpoint/log alias rejection. The correction adds no second optimizer loop,
checkpoint system, metric implementation, preprocessing path, split loader, or
class-eligibility authority.

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

| Model family | E1 parameters | E2 parameters | E3 parameters |
| --- | ---: | ---: | ---: |
| HTGNN | 7,183 | 7,183 | 6,721 |
| Final-flow MLP | 7,823 | 7,823 | 7,361 |
| Flattened five-flow MLP | 27,791 | 27,791 | 27,329 |

Bounded production-path smoke covers both baseline families for all E1/E2/E3
tasks on CPU and CUDA. Every run creates one real JSONL epoch record and one
real best checkpoint in a temporary location, verifies task/family/input
metadata, reloads into a fresh matching model, and reproduces the saved model
output. Cross-task, cross-family, baseline-to-HTGNN, and HTGNN-to-baseline
loads fail before weights are applied. No weights or logs are committed.

## Corrected HTGNN E2 resource pilot

The corrected pilot reran from the current source identity on the authoritative
E2 train stream. Every candidate restarted the seed-2025 affine traversal and
used the unchanged HTGNN, unweighted cross-entropy, AdamW with weight decay
`1e-4`, and the resource-pilot-only learning rate `3e-4`. Each candidate ran
five unmeasured warmup optimizer steps and 20 measured optimizer steps. Every
measured interval included host-to-device transfer, forward, loss, backward,
and optimizer step, with CUDA synchronization before and after timing. D4/D6/
E.2 CPU construction remained outside the timed interval. Pilot-trained
weights were discarded.

The corrected run used an NVIDIA GeForce RTX 4050 Laptop GPU with
6,438,780,928 bytes of VRAM. Values below are runtime observations, not
canonical replay targets.

| Batch | Median step (ms) | Nearest-rank p95 (ms) | Median windows/s | Peak reserved | Reserved / VRAM | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 32 | 17.149 | 21.856 | 1,867.347 | 27,262,976 B | 0.4234% | yes |
| 64 | 19.577 | 23.229 | 3,270.384 | 52,428,800 B | 0.8143% | yes |
| 128 | 23.383 | 31.736 | 5,474.137 | 48,234,496 B | 0.7491% | yes |
| 256 | 24.037 | 30.276 | 10,650.330 | 71,303,168 B | 1.1074% | yes |

All candidates completed with finite losses and gradients, successful optimizer
steps, and absolute peak reserved memory strictly below the 80% threshold. The
maximum full-precision median throughput was 10,650.330239 windows/s and the
95% threshold was 10,117.813727 windows/s. Only batch 256 entered the tied set,
so the frozen smallest-in-tied-set rule selected HTGNN training batch size
**256** without hard-coding that outcome.

The post-selection production gate constructed the actual
`TrainingConfig(batch_size=256)`, obtained a 256-example E2 training batch from
`iter_production_batches`, and executed it through `train_one_epoch`. It
produced raw logits `[256,15]`, finite cross-entropy and gradients, and a
successful optimizer update without validation or test access.

## Fail-closed observation evidence

The production validator requires the exact 32/64/128/256 candidate sequence,
the frozen task/seed/optimizer/loss/count configuration, current source hashes,
the frozen E.9 contract identity, the audited device identity, exactly 20 raw
timing and loss samples per completed candidate, and consistent CUDA-memory
relationships. It recomputes per-sample throughput, median latency,
nearest-rank p95, median windows/s, absolute reserved-memory percentage,
resource eligibility, the 95% tied set, and the selected batch. The
deterministic contract evidence binds the exact observation-file bytes by
SHA-256; offline validation does not require CUDA, while an intentional live
validation cross-checks the stored GPU name and VRAM.

Adversarial tests corrupt source identity, device identity, VRAM, raw timings,
summary throughput, p95, memory values, finite status, eligibility, candidate
identity/count/order, selected batch, task, seed, learning rate, step counts,
execution/config identity, and observation-file binding. Each corruption is
rejected by the production validator.

## Scope and next boundary

The selected batch size is a hardware- and environment-specific engineering
configuration supported by this bounded pilot. The learning rate `3e-4`
remains a resource-pilot-only value; Stage E.8 did not select or finalize a
learning rate. No task performance, weighted-loss configuration, final model
checkpoint, test performance, full training, three-seed study, or pruning has
been finalized.

Compact deterministic evidence is stored in
`results/stage_e9/stage_e9_contract.json`; exact noisy observations remain in
`results/stage_e9/htgnn_batch_size_pilot_observations.json` and are bound by
the deterministic evidence hash. E.9 remains unfrozen pending independent
re-audit. E.10 and later work remain unauthorized.
