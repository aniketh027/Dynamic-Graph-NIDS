# E.12 attention-guided inference-time feature pruning

This is a preparation contract and harness, not a pruning result. Neither real
E.12A nor E.12B was executed during implementation. E.10 weights, E.11 evidence,
the test-consumption marker and `PROJECT_HANDOFF.md` remain unchanged. The owner
explicitly authorized preparation despite the stale E.9 handoff on 2026-10-06.
No commit, push, tag, training, fine-tuning or new epochs are authorized here.

## Audit and architecture

`src/pruning/__init__.py` was a placeholder. No implemented graph-pruning policy,
executor or pruning tests were found. E.4 already exposes raw attention through
E.5 `HTGNNClassifier(data, return_attention=True)`. Frozen E.4 tests cover edge
identity, destination normalization and the installed PyG dropout semantics.

For B graphs the returned dictionary contains, in this order:

| Relation | Edge index | Attention | Batched row mapping |
| --- | --- | --- | --- |
| feature to category | `[2, B*78]` | float32 `[B*78, 4]` | row `g*78+f`: source `g*78+f`, target `g*8+category[f]` |
| category to sample | `[2, B*8]` | float32 `[B*8, 4]` | row `g*8+c`: source `g*8+c`, target `g` |

Both modules concatenate four 8-channel head outputs into 32 channels. Returned
coefficients are **post-softmax and post internal attention dropout**. In eval
mode dropout is disabled and each destination/head's incoming coefficients sum
approximately to one. Training-mode values cannot be used for this policy.
The installed source's `GATv2Conv.edge_update` applies destination softmax,
then `F.dropout(..., training=self.training)`. The [official PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)
also describes head concatenation, normalized-coefficient dropout and return
shapes; installed-source/synthetic checks determine this repository's semantics.

There are only the original 78+8 directed edges. There are no stored reverse,
same-level, temporal or structural self-loop edges. T=5 remains five flow
records per feature, with the fifth flow's label as graph target. The sample
node remains the sole classifier readout.

## A. Pruning unit

Remove a **whole feature node and its sole feature-to-category edge**. Keep all
eight categories, all eight category-to-sample edges and the sample node. Keep
at least one feature in every category. Pruning category-to-sample edges would
discard whole semantic groups; category identities/residuals also remain active,
so feature attention and category attention are not interchangeable units.
Hierarchical removal adds another policy axis without a frozen acceptance rule.

The original E.2 input remains the complete approved graph. E.12 validates it
with the frozen E.4 validator, selects retained feature rows, preserves their
original 0..77 embedding IDs, remaps only compact source positions, and invokes
the original feature encoder, two GATv2 modules, residuals, norms, activations
and classifier. No parameters are copied, replaced, trained or edited. Softmax
is recomputed by the original GATv2 on the retained neighborhood. Zero pruning
calls the original model directly for exact logits. Original E.2/E.4/E.5 code and
their complete-topology contract remain frozen.

CUDA's ordinary scatter reductions are not bitwise deterministic in the frozen
training contract. The initial synthetic run reproduced that limitation even
between two original unpruned forwards. E.12A therefore uses strict deterministic
PyTorch algorithms and cuDNN settings for **both** baseline and pruning, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` set by the CLI before importing Torch. These
settings are recorded, and unsupported deterministic operations fail loudly.
Exact zero-pruning equality is tested on CPU and CUDA under this shared execution
configuration, not claimed against historical nondeterministic CUDA executions.
Timing consequently describes this deterministic configuration. The context
restores prior PyTorch settings on exit and does not change training or weights.
Historical E.11 best-effort CUDA predictions are not certified bitwise equivalent
to this configuration. A future E.12B review must state this arithmetic limitation
when reusing the original unpruned metrics; it does not authorize re-evaluation.

Graph-size reporting distinguishes full input-adapter allocation (87 nodes)
from the reduced inference computation (K+9 nodes, K+8 edges). Removed features
do not enter the temporal encoder or feature GAT projections. This does not
reduce classifier/model parameter count or the full input adapter's allocation.

## B. Attention aggregation and ranking

Use unpruned, eval-mode attention from each frozen **E2 HTGNN** seed. Ranking is
estimated on at most 32 batches of 256 E2 training samples (8,192), using the
existing E.7 O(1)-state affine index permutation with fixed seed 12012. Use
exactly the same sample membership/order for seeds 2025/2026/2027. No labels
select scores; observed training support is recorded as a sampling diagnostic.
This is a bounded population-oriented ranking sample, not a guarantee of
rare-class representation or independent statistical validation.

For original feature f, define:

`score[f] = mean_seed(mean_train_sample(mean_four_heads(alpha_feature[f])))`.

Accumulate in float64 and record all 78x4 per-seed means, 8x4 category means
(diagnostic only), observed support, sample-order hashes and raw returned
edge/attention stream hashes. Do not save per-sample predictive arrays.

Mean heads is an explicitly chosen equal-weight, head-permutation-invariant
ranking heuristic. The actual network concatenates heads; this mean does not
reproduce its message vector or establish causal importance. A head maximum
would systematically favor a single concentrated head; no frozen method calls
for that choice. Do not multiply attention across levels: independently learned
head spaces, projection vectors, target residuals, LayerNorm and GELU prevent
interpreting that product as a classifier contribution.

Rank **within each category**, decreasing score, then increasing original
feature ID for exact ties. Softmax normalization neighborhoods have sizes
2/23/24/5/12/6/4/2, so direct cross-category coefficient ranking would favor
smaller neighborhoods. For a declared ratio r retain
`max(1, n_category - floor(r*n_category))` features per category. Finally restore
original feature order for inference. The mean across seeds defines one common
mask, avoiding seed-specific degrees of freedom and simplifying comparison.

## C. E.12A predeclared validation-only sweep

Training ranks masks; full sequential **E2 validation** evaluates candidates.
No test membership, support, arrays, metrics or E.11 reports enter selection.
Each candidate is evaluated with all three frozen E2 HTGNN checkpoints.

| Requested per-category r | Removed features | Retained features | Inference nodes | Inference edges | Actual feature removal |
| --- | --- | --- | --- | --- | --- |
| 0% | 0 | 78 | 87 | 86 | 0% |
| 10% | 5 | 73 | 82 | 81 | 6.41% |
| 25% | 17 | 61 | 70 | 69 | 21.79% |
| 50% | 38 | 40 | 49 | 48 | 48.72% |
| 75% | 56 | 22 | 31 | 30 | 71.79% |

The actual ratios differ because per-category quotas are integer-valued. Record
feature, total-node and total-edge removal separately. This gives 15 validation
passes and three bounded training-attention passes. The last ratio is a declared
stress candidate, not a promised result. No adaptive expansion/stopping is used.

## D. Selection and tolerance approval

Use the frozen E.7 validation-selection eligibility (train >=100 and validation
>=30 only) and its multiclass confusion/F1 calculation, including false positives
from all validation rows. Do not substitute E.11's test-reporting eligibility or
true-row filter. The validation and final-reporting metrics have their separate
frozen roles.

For seed s let B_s be its unpruned validation macro F1. An owner-approved
**relative** tolerance delta admits a candidate only if, for **every** seed:

`F1_pruned[s] >= (1 - delta) * B_s`.

Require positive B_s, complete validation support and all three seeds. Select
the admitted candidate removing the most feature nodes; duplicate graph-size
ties select the smaller requested ratio. Zero pruning is the fallback. Latency
and throughput are measured and reported, not a noisy threshold that changes
the predeclared predictive rule. Owner review must assess whether measured
efficiency actually improved before freezing or making an efficiency claim.

No repository stage contract defines a numeric eligible-class macro-F1
noninferiority margin. Original report section 3.2.4 proposes <=50% original
nodes and no more than **5% detection accuracy loss**, ambiguously mentioning
test and validation. That background also describes an obsolete graph/model.
It does not freeze a relative-vs-absolute convention or authorize macro F1 to
replace accuracy automatically.

**Proposal for owner approval before any real sweep:** delta=0.05 (5% relative
macro-F1 loss per seed), explicitly translating the report's 5% accuracy budget
to the primary metric. This is a scientific choice requiring owner review; it
is not a measured or already-approved tolerance. The CLI has no default margin
and requires the numeric margin, rationale and owner acknowledgement. The
report's <=50% node target is recorded as background, not enforced to select a
damaging ratio. Approving a tighter tolerance requires no change to the ranking.

## E. Seeds and tasks

One E2 training-derived ranking and one E2 validation-selected mask apply to
all three E2 seeds. E1/E3 do not supply rankings, tolerances or selection metrics.
After a separately reviewed E.12A freeze, a future E.12B may transfer this exact
mask to their task-specific frozen HTGNNs for diagnostic/transfer reporting,
without retuning. This estimates E2-driven policy transfer, not E1/E3 optimal
pruning. E3 test distribution-shift observations cannot change this procedure.
This version is a fixed mask, not sample-adaptive online pruning; it avoids
paying for a full unpruned attention pass at every inference. An adaptive policy
would require another predeclared contract and measured ranking overhead.

## F-I. Implementation, tests, safeguards and OWNER command

Created files:

- `src/pruning/stage_e12.py`: policies, guarded inference, authenticated checkpoint
  loading, train scoring, validation selection and resident-input measurement.
- `scripts/run_stage_e12_pruning.py`: safe `dry-run` and owner-only E.12A CLI.
- `tests/test_stage_e12_pruning.py`: unit/synthetic contracts and negative cases.
- This document. Frozen source, checkpoint/test evidence and handoff are unchanged.

The inference loader authenticates model bytes/metadata to the pinned completed
E.10 report and reuses E.7 model-contract/schema/tensor preflight. Unlike E.7's
training loader it creates no AdamW and never restores optimizer state. Historical
optimizer validation remains recorded E.10 provenance; this inference adapter
does not claim a new optimizer-resume audit. Model contents are hashed before
and after scoring/evaluation, and checkpoint/source/input hashes are checked
again before report publication. The execution environment is matched to the
frozen Python/Torch/PyG/CUDA contract, with hardware reported for interpretation.

The process-wide audit guard blocks all real test arrays, non-E2 predictive data,
E.11 reports and writes outside `logs/stage_e12` or `results/stage_e12`. It covers
ordinary Python/NumPy opens and Python worker threads. It is an accidental-access
boundary, not a sandbox against arbitrary native/external code or hard links.
All output publication is exclusive; existing E.12A run IDs cannot be reused.
Failed/partial E.12 outputs do not authorize retry, selection or E.12B.

Preparation validation on 2026-10-06:

- **78 E.12 unit/synthetic tests passed**, including CPU/CUDA bitwise zero-pruning
  equality under the shared deterministic settings, retained encoder-row counts,
  graph isolation/mapping/connectivity, unchanged weights/checkpoint bytes,
  invalid policies, access lockout, and a fully mocked 15-comparison workflow.
- **187 E.10/E.11 unit/synthetic regressions passed** (E.10 result/provenance,
  E.11 evaluator and failure recovery).
- **80 frozen E.4/E.5 and E.9 provenance regressions passed**, with 21 backward
  or real-data integration cases deliberately deselected. No full project suite
  was run because this task explicitly permits only cheap unit/synthetic tests.
- Each run's outer guard recorded **zero forbidden accesses**. No optimizer step,
  backward pass, epoch, training, real validation sweep or test evaluation ran.
  E.12's own tests additionally reject optimizer construction.
- The CLI metadata-only `dry-run` matched all five graph sizes. Existing PyG
  annotation and Torch deprecation warnings remain; the final tests had no
  failures or skips.
- Metadata-only checks confirmed canonical and physical-sidecar hashes for
  the E.10 PASS report (45 jobs) and recovered E.11 PASS report (27 results).
  E.10 physical report SHA256 is
  `8ff3f89c4aac3b2948936ea794e710059a7fe441c161b8c20cae0b8e51f6639c`;
  recovered E.11 physical report SHA256 is
  `ace0d585fbb89b0f3df4376b2f69e35bd3172ab663cc66b369b769e84a964aac`.
  These are existing report checks, not new predictive evaluation or release audits.
- Final SHA256 comparison preserved **372 protected files byte-for-byte**, including
  E.10 checkpoints/evidence, E.11 evidence/marker, frozen model sources, dependency
  lock and handoff. No real `logs/stage_e12` or `results/stage_e12` namespace exists.
  Syntax/source review and tracked/new-file whitespace checks passed. All four
  added files remain untracked alongside the pre-existing owner files. Local HEAD
  and `origin/main` remain `2fa823943ba318fe32b2867bc8e2b1f799d78ac1`; no Git mutation
  was performed.

Preparation may run the metadata-only CLI:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B scripts\run_stage_e12_pruning.py dry-run
```

**OWNER E.12A only, after reviewing and approving the proposed relative margin**:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B scripts\run_stage_e12_pruning.py select-validation --run-id e12a-20261006-v1 --device cuda --max-relative-f1-drop 0.05 --tolerance-rationale "Owner-approved translation of original report 3.2.4 five-percent accuracy budget to per-seed relative eligible-class macro-F1 loss" --owner-confirmed-validation-contract
```

This documents a prospective command, not authorization to execute it during
preparation. The owner must choose/approve the margin before running it. Run ID
is identity only and may be changed to a fresh date-labelled ID.

## J. Expected E.12A outputs

Under `logs/stage_e12/e10-official-20261001-v1/<run-id>/`:

- `manifest.json`: frozen sweep, margin/rationale, ranking membership rule,
  seeds, source/environment/input/checkpoint provenance and canonical hash.
- `status.json`: `COMPLETE_PENDING_REVIEW`; or `failure.json` on a run failure.

Under `results/stage_e12/e10-official-20261001-v1/<run-id>/`:

- `attention_ranking.json`: per-seed/head training means, pooled ranking,
  attention/order hashes, original feature names and every mask/edge identity.
- `validation_sweep.json`: 15 seed/candidate measurements, confusion matrices,
  complete validation counts, eligible codes and traversal hashes.
- `proposed_policy.json`: selected mask, ratios, sizes and canonical policy hash;
  explicitly `PROPOSED_REQUIRES_OWNER_REVIEW_AND_FREEZE`.
- `stage_e12a_report.json` and `.sha256`: canonical/content and physical-byte
  hashes, unchanged state evidence, zero forbidden accesses, no E.12B enablement.

Measurements use a resident first validation batch: five warmups and 20 timed
forwards with CUDA synchronization. Contract checks and compaction are included;
dataset loading, H2D and offline ranking are excluded. Report median/p95 batch
latency, throughput and GPU allocated/peak memory (model/full input included).
CPU allocator memory is null with a reason. These are one-batch hardware-specific
observations, not full end-to-end or statistically certified speedups. Full graph
adapter allocation and model parameter count remain unchanged.

## K. E.12B procedure — unavailable and unauthorized

There is no E.12B CLI/executor in this implementation. First independently review
E.12A evidence, verify every-seed tolerance and unchanged weights, review timing
quality and approve/freeze the policy hash and input/model identities. Then obtain
separate explicit authorization and an audited E.12B implementation/protocol.

A future final comparison uses the same frozen checkpoint and fixed policy for
each seed, sequential test membership, frozen E.11 reporting definitions and
unchanged decision rules. Reuse the 27 existing official E.11 unpruned predictive
results after metadata/provenance validation; do not repeat their evaluation.
Only the new, separately authorized pruned predictions would access test arrays.
Measure unpruned/pruned timing on a separately predeclared benchmark without
repeating official unpruned predictive evaluation. No policy ranking, tuning or
threshold changes may follow those results. Report E2 primary and any E1/E3
transfer observations separately. Preserve the existing consumption marker and
interrupted/recovered E.11 history; do not repurpose E.11's one-time executor.

## L-M. Risks and recommended handoff reconciliation

Remaining choices requiring review are the metric-margin translation, the
bounded ranking sample's rare-class coverage, and transfer of one E2 mask to
different task checkpoints. Global masks make no online-adaptivity claim. The
linear projection/residual architecture prevents a causal attention explanation.
Removing features can reduce graph arithmetic without reliably speeding up small
CUDA workloads; benchmarking must establish that empirically. No result exists
from this preparation.

Append a separately authorized reconciliation to `PROJECT_HANDOFF.md`:

1. Authoritative checkout and live Git HEAD, preserving E.9 as the latest audited
   checkpoint/tag and distinguishing later owner execution from release freeze.
2. Completed E.10 study `e10-official-20261001-v1`: 45 completed training jobs,
   validation-only selected LRs, 27 final checkpoints, report physical/canonical
   hashes, checkpoint/source/env identities and reviewed dirty-source provenance.
3. E.11's failed six-checkpoint prefix plus reviewed 21-checkpoint continuation,
   combined 27-checkpoint completion and final recovery-report hash. Keep original
   prefix, FAILED status and `CONSUMED_OR_RESERVED` marker immutable. Explicitly
   forbid repeating E.11. Report completion is not an independent release PASS.
4. E.12 code-preparation status, synthetic validation and preservation evidence;
   original-vs-proposed tolerance distinction, fixed feature/mask contract and
   no actual pruning results, no execution or test access during preparation.
5. Next boundary: owner review/approval of E.12A contract and margin, then owner
   validation-only execution; separate review/freeze and explicit authorization
   before any E.12B. Supersede obsolete "no trained model/test result" statements
   without rewriting historical evidence.

The historical handoff's literal next boundary is separately authorized E.10.
The current task and owner's explicit stale-handoff exception authorize only
this E.12 preparation. The handoff itself has deliberately not been edited.
