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
fail-closed. Production-integration correction commit
`5f945149859b9b7d2002e712525b7b34ed82e1be` fixed those defects, but a second
independent audit remained **BLOCKED**: a rejected checkpoint could already
have mutated the target model, and correlated edits could forge candidate
status while retaining a successful raw measurement record. The fail-closed
correction was recorded in
`43f4582063e32e68bde411dc1c46c19c52afe7a8`. A third independent audit
verified both transactionality and the observation correction, but remained
**BLOCKED** because five semantically invalid AdamW payloads were still
accepted: an optimizer LR contradicting checkpoint metadata, a non-frozen
weight decay, incomplete parameter-state coverage, a nonfinite first moment,
and a nonfinite step counter. The optimizer-state correction described here
preserves every earlier fix. E.9 is still unfrozen and requires another
independent re-audit.

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

Checkpoint loading is now transactional. Before touching the target, the one
production loader validates the exact model-state keys, tensor shapes, dtypes,
layouts, model contract, optimizer fields, parameter-group membership, and
AdamW per-parameter state structure, including a disposable optimizer-load
probe. Immediately before the first mutating load it snapshots the exact model
and optimizer states. Any later exception restores both snapshots, including
optimizer state and parameter-group configuration, before the original load
failure is re-raised. Missing optimizer state, missing or unexpected model
keys, wrong tensor shapes, malformed optimizer state, and task/family/
architecture/input mismatches are all tested to leave both targets byte-for-
byte tensor-equivalent to their pre-call state. Matching schema-v2 checkpoints
for every task and family, plus valid schema-v1 HTGNN checkpoints, still load.

Optimizer preflight now treats AdamW state as a semantic contract, including
model-only reloads. The single serialized group LR must equal the checkpoint
`training_config` LR exactly, and its weight decay must equal the frozen
`1e-4`. The other PyTorch 2.14 group fields are bound to a freshly constructed
production AdamW instance: `betas`, `eps`, `amsgrad`, `maximize`, `foreach`,
`capturable`, `differentiable`, `fused`, and `decoupled_weight_decay`. Parameter
IDs must preserve the production mapping, and every grouped parameter must
have exactly one state entry with exactly `step`, `exp_avg`, and `exp_avg_sq`
under the frozen `amsgrad=False` configuration. Step is the authoritative
scalar float32 representation and must be finite, nonnegative, and integral.
Both moments must match parameter shape/dtype/layout and be finite;
`exp_avg_sq` must also be nonnegative. Unknown, incomplete, extra, malformed,
or nonfinite state fails before target mutation. The production lifecycle saves
best checkpoints only after a train epoch has performed optimizer updates, and
bounded real HTGNN and baseline paths confirm complete state for every
trainable parameter. Legacy schema-v1 remains restricted to HTGNN and carries
the same training-configuration and complete-state proof; no weaker baseline
or metadata-free compatibility path was added.

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
| 32 | 55.205 | 69.610 | 579.668 | 27,262,976 B | 0.4234% | yes |
| 64 | 55.803 | 65.803 | 1,147.166 | 52,428,800 B | 0.8143% | yes |
| 128 | 45.011 | 155.810 | 2,844.003 | 48,234,496 B | 0.7491% | yes |
| 256 | 65.095 | 178.717 | 3,934.513 | 71,303,168 B | 1.1074% | yes |

All candidates completed with finite losses and gradients, successful optimizer
steps, and absolute peak reserved memory strictly below the 80% threshold. The
maximum full-precision median throughput was 3,934.512947 windows/s and the
95% threshold was 3,737.787300 windows/s. Only batch 256 entered the tied set,
so the frozen smallest-in-tied-set rule selected HTGNN training batch size
**256** without hard-coding that outcome. The fresh execution identity is
`e9-20260930t163056z`; the exact observation-file physical SHA-256 is
`85d693fe3ed3eeb5f2f7715c31528e7ccfbbd8524f655ed2463f3a437390892d`.

The post-selection production gate constructed the actual
`TrainingConfig(batch_size=256)`, obtained a 256-example E2 training batch from
`iter_production_batches`, and executed it through `train_one_epoch`. It
produced raw logits `[256,15]`, finite cross-entropy and gradients, and a
successful optimizer update without validation or test access.

## Fail-closed observation evidence

The production validator requires the exact 32/64/128/256 candidate sequence,
the frozen task/seed/optimizer/loss/count configuration, current source hashes,
the frozen E.9 contract identity, and the audited device identity. Each
candidate now includes a compact ordered raw trace for every attempted warmup
or measured step: phase and index, completion, loss/gradient finite facts,
optimizer-step success, CUDA-OOM fact and failure stage, loss value, and the
measured latency where applicable. Legal states are complete, CUDA OOM,
nonfinite loss, nonfinite gradient, optimizer failure, and the bounded skip
after a smaller OOM. Attempted/completed counts, finite summaries, optimizer
summary, timing/loss arrays, candidate status, and resource eligibility are
derived from that trace and must match it exactly. Honest partial failures are
valid but ineligible; contradictory failure or success stories are rejected.

For completed candidates the validator still recomputes per-sample throughput,
median latency, nearest-rank p95, median windows/s, absolute reserved-memory
percentage, resource eligibility, the 95% tied set, and the selected batch.
The deterministic evidence generator performs this full semantic validation
before any binding work and binds the exact observation-file bytes by SHA-256.
Offline validation does not require CUDA, while an intentional live validation
cross-checks the stored GPU name and VRAM.

Adversarial tests corrupt source identity, device identity, VRAM, raw timings,
summary throughput, p95, memory values, finite status, eligibility, candidate
identity/count/order, selected batch, task, seed, learning rate, step counts,
execution/config identity, and observation-file binding. Correlated false-
nonfinite, false-OOM, false-success, optimizer-failure, and raw-nonfinite
stories are also rejected. A forged correlated observation cannot be rebound
through the official deterministic evidence path, while an internally
consistent partial CUDA-OOM fixture remains valid and ineligible.

Implementation validation passed 141 focused E.9 tests, 44 focused E.7 tests,
85 D6/full-split tests, 267 frozen E.2-E.8 regressions, and 1,103 complete-suite
tests, with zero failures or skips. The deterministic evidence run also
executes all 12 real baseline production artifact smokes and independently
checks Stage C 33/33 and raw CSV 10/10 integrity.

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
