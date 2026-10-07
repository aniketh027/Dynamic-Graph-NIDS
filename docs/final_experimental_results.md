# Final experimental results through E.12B

Closure date: 2026-10-06. Authoritative checkout: `C:\Project\ComputerNetworks`.
The approved experimental scope is complete. The active boundary is final
report/release/documentation and independent final release audit. No further ML
experiment is currently authorized or required.

## 1. Experimental scope

The project studies **Dynamic Graph Optimization and Automated Feature Extraction
for Efficient Intrusion Detection in Network Systems** using CSE-CIC-IDS2018.
`T=5` denotes five consecutive clean flow records in deterministic capture order;
the graph target is the fifth/final flow label. Windows need not be conversations,
and capture order is not guaranteed physical chronology because timestamps lack
AM/PM. Exactly 78 features exclude Timestamp and Label.

E1 is a strict global capture-order multiclass diagnostic. E2 is the primary
supervised multiclass experiment, preserving chronology within each class rather
than one global timeline. E3 is strict global capture-order binary evaluation.
Their partition membership, preprocessing provenance and reporting roles remain
separate. Train-fitted preprocessing is frozen. E2 validation selection uses
train support >=100 and validation support >=30 without test information; final
test reporting additionally requires test support >=30. The E2 primary reporting
classes are codes `[0,1,4,5,6,7,8,9,10,11,12,14]`, with true rows restricted to those
eligible classes. All-test descriptive macro-F1 is a different measure.

E.10 is COMPLETE; E.11 is COMPLETE through reviewed recovery; E.12A is COMPLETE
and artifact-frozen; E.12B is COMPLETE and independently audited PASS, as confirmed
by the owner's final closure instruction. Frozen E.12B execution report/status
files retain `COMPLETE_PENDING_REVIEW`: that execution-time state precedes the
owner-confirmed independent PASS and is not rewritten. This documentation task
performed compact persisted-evidence checks, not a new independent execution audit.
No separate final E.12B audit attestation was found in the inspected persisted
result/provenance namespaces; the PASS source is the owner's closure instruction.

## 2. E.10 model-selection summary

Official study: `e10-official-20261001-v1`. Phase A completed 27 LR-selection runs
at seed 2025, Phase B selected the LR for nine family/task pairs using validation
only, and Phase C completed 18 additional runs at seeds 2026/2027. There are 45
unique official runs and 27 final validation-selected checkpoints; the seed-2025
winner is reused. The final report is PASS. No predictive test access, pruning,
test metrics or E3 threshold selection occurred in E.10.

| Family | E1 selected LR | E2 selected LR | E3 selected LR |
|---|---:|---:|---:|
| HTGNN | 0.001 | 0.001 | 0.001 |
| Final-flow MLP | 0.001 | 0.001 | 0.001 |
| Flattened-five-flow MLP | 0.0001 | 0.0003 | 0.0001 |

The frozen batch size is 256. Training used the frozen unweighted task losses,
maximum 30 epochs, minimum 3 epochs, patience 5 and min_delta 0.0001. LR selection
does not establish an optimal architecture, window length or deployment setting.

## 3. E.11 unpruned baseline summary

Original first-test attempt: `e11-official-20261006-v1`; reviewed continuation:
`e11-recovery-20261006-v1`. The original evaluator rejected legitimate float32
E3 targets. Reviewed recovery preserved the failure/consumption evidence and
six completed original HTGNN E1/E2 results, then evaluated 21 remaining
checkpoints. The combined report is PASS for 27/27 checkpoints. The original
test partition was consumed/reserved; E.11 must not be repeated.

Primary E2 eligible-class test macro-F1, three seeds:

| Family | Mean | Sample SD (ddof=1) |
|---|---:|---:|
| htgnn | 0.8759879709 | 0.0226402305 |
| final_flow_mlp | 0.7241750943 | 0.0029248034 |
| flattened_five_flow_mlp | 0.6531317526 | 0.0025948334 |

These are descriptive seed summaries. They are not confidence intervals or
statistical-significance evidence. E.12B reused the existing nine unpruned HTGNN
results and did not re-evaluate unpruned test data or run MLP comparisons again.

## 4. E.12A pruning-policy selection

Run: `e12a-20261006-v1`. Evaluation-mode attention from the three E2 HTGNN
checkpoints ranked features on the same bounded 8,192 E2-train examples per seed.
Full E2 validation compared declared requested per-category ratios 0%, 10%, 25%,
50% and 75%. The largest ratio satisfying every seed's pruned validation
eligible-class macro-F1 >=95% of its own unpruned value was 25%. Candidates
0%, 10% and 25% passed; 50% and 75% failed. The 25% requested ratio removed
17/78 features after the category constraints; actual removal is 21.794872%.

The separate `e12a-20261006-v1-freeze/freeze_attestation.json` records independent
review PASS and owner-approved policy freeze/5% relative-loss tolerance. No
predictive test array or test support was consulted during selection. Attention
is a ranking heuristic, not causal importance. The bounded ranking sample did
not contain DDOS attack-LOIC-UDP (class code 5).

## 5. E.12B final comparison

Run: `e12b-20261006-v1`. One frozen E2-derived mask was transferred unchanged to
E1/E2/E3 at seeds 2025/2026/2027. Nine pruned HTGNN checkpoint evaluations
completed. No retraining, fine-tuning, backpropagation, optimizer, parameter
change, mask reselection, calibration or threshold tuning occurred.

| Primary E2 eligible-class macro-F1 | Unpruned | Pruned |
|---|---:|---:|
| Mean | 0.8759879709 | 0.8896826984 |
| Sample SD (ddof=1) | 0.0226402305 | 0.0090459554 |

Mean absolute change is **+0.0136947275** (about +1.369473 percentage points).
Mean relative improvement is **+1.617299%**, computed as the mean of the three
per-seed relative changes `(pruned - unpruned) / unpruned`; it is not the relative
change obtained by dividing the aggregate means. Every E2 seed remained within
the frozen 5% relative-loss rule. This is an observed test comparison, not a new
test-selected policy or a statistical noninferiority guarantee.

## 6. Per-seed E2 results

| Seed | Unpruned macro-F1 | Pruned macro-F1 | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| 2025 | 0.8513260520 | 0.8972391626 | +0.0459131106 | +5.393129% |
| 2026 | 0.8808074488 | 0.8796591323 | -0.0011483165 | -0.130371% |
| 2027 | 0.8958304119 | 0.8921498002 | -0.0036806117 | -0.410860% |

One seed improved and two declined slightly. The increase in mean and smaller
observed sample SD do not establish significance or guaranteed robustness.

## 7. Graph-size comparison

| Inference structure per graph | Unpruned | Pruned | Removed | Reduction |
|---|---:|---:|---:|---:|
| Feature nodes | 78 | 61 | 17 | 21.794872% |
| Category nodes | 8 | 8 | 0 | 0% |
| Sample nodes | 1 | 1 | 0 | 0% |
| Total nodes | 87 | 70 | 17 | 19.540230% |
| Feature-to-category edges | 78 | 61 | 17 | 21.794872% |
| Category-to-sample edges | 8 | 8 | 0 | 0% |
| Total directed structural edges | 86 | 69 | 17 | 19.767442% |

The removed original feature IDs are `[6,10,29,32,34,35,40,41,42,50,52,57,64,70,71,72,73]`.
All eight categories and all eight category-to-sample edges remain. Retained
features preserve their original feature identities. The input adapter still
constructs/transfers the complete graph. Temporal encoding processes 61 instead
of 78 features and first GAT message-passing work decreases. Second GAT topology
is unchanged. Parameter counts remain 7,183 for E1/E2 and 6,721 for E3. This is
inference-time structural pruning, not model-weight pruning.

## 8. Latency and throughput

Persisted validation resident-forward benchmark on NVIDIA GeForce RTX 4050
Laptop GPU (6,438,780,928 bytes VRAM), Windows, Python 3.13.0, PyTorch
2.14.0+cu130, CUDA runtime 13.0 and PyG 2.8.0.post1. These are recorded execution
identities, not a newly probed environment. Each task/seed used four sequential
validation batches of 256 at starts 0,256,512,768, five warmups per path per
batch and 20 measured pairs per batch (80 pairs). Float32 and strict
deterministic algorithms were enabled; pair execution order alternated.

The forward scope includes resident full-input checks, feature compaction,
encoders and head; it excludes lazy loading/conversion and host-to-device
transfer. Those costs were recorded separately and are not reduced by the mask.
Observed paired median speed ratios range from **1.034540x to 1.117286x**
(approximately 1.035x-1.117x).

| Task | Seed | Unpruned median ms | Pruned median ms | Unpruned windows/s | Pruned windows/s | Median paired speed ratio |
|---|---:|---:|---:|---:|---:|---:|
| E2 | 2025 | 14.85610 | 13.57400 | 17231.98 | 18859.58 | 1.084082x |
| E2 | 2026 | 14.87360 | 14.65905 | 17211.70 | 17463.61 | 1.034540x |
| E2 | 2027 | 16.48570 | 15.78475 | 15528.61 | 16218.19 | 1.073488x |
| E1 | 2025 | 12.64560 | 11.45810 | 20244.20 | 22342.27 | 1.115104x |
| E1 | 2026 | 15.60865 | 14.51075 | 16401.16 | 17642.09 | 1.077162x |
| E1 | 2027 | 15.06300 | 14.36080 | 16995.29 | 17826.30 | 1.079333x |
| E3 | 2025 | 13.63435 | 12.59325 | 18776.11 | 20328.35 | 1.117286x |
| E3 | 2026 | 13.20910 | 12.76115 | 19380.58 | 20060.89 | 1.087943x |
| E3 | 2027 | 13.15570 | 11.14135 | 19459.25 | 22977.47 | 1.111840x |

Throughput is `256,000 / median_ms`. The paired ratio is
`median(unpruned_pair_ms / pruned_pair_ms)`; it need not equal the ratio of the
two separately summarized latency medians. Raw samples and spreads remain in
`validation_benchmarks.json`. These modest observed forward improvements do not
establish end-to-end deployment, training, other-hardware or universal speedup.

## 9. Memory summary

| Measurement | Unpruned | Pruned |
|---|---:|---:|
| E1/E2 peak PyTorch allocated bytes | 51,959,296 | 48,886,272 |
| E1/E2 peak allocated MiB | 49.552 | 46.622 |
| E3 peak PyTorch allocated bytes | 51,956,736 | 48,883,712 |
| E3 peak allocated MiB | 49.550 | 46.619 |
| Peak PyTorch reserved bytes (all pairs) | 58,720,256 | 58,720,256 |
| Peak reserved MiB | 56 | 56 |

Transient allocated peak decreases by 3,073,024 bytes = 2.930664 MiB
(approximately 5.91%). The reported memory scope includes the model and resident
full adapter input and measures the PyTorch caching allocator. Peak reservation
does not decrease. No CPU-memory reduction, total device-memory reduction,
checkpoint-size reduction, parameter reduction or initial adapter-memory
reduction is established.

## 10. E1 diagnostic result

Mean support-restricted macro-F1 is **0.9307991287 -> 0.9307918724**,
essentially unchanged. The reporting subset contains **only Benign**. This high
restricted score is not evidence of strong multiclass attack detection under
chronological evaluation. All-test descriptive macro-F1 is only
0.1551331881 -> 0.1551319787; several test attack classes were unseen in training.

## 11. E3 chronological limitation

Mean average-precision PR-AUC is **0.0890313615 -> 0.0890480581**,
essentially unchanged and below attack prevalence **0.1293481937** (80,694 of
623,851 records). Strict capture-order chronological generalization remains
poor. Unpruned mean PR-AUC is 0.0929121944 for final-flow MLP and 0.0965841419
for flattened-five-flow MLP, so the limitation extends across the three families.

At the fixed 0.5 probability / 0.0 logit threshold, mean HTGNN attack recall
is 0.0010740162 -> 0.0011896795 and mean F1 is 0.0021344147 -> 0.0023595621.
The approximately 0.87 accuracy largely reflects the Benign majority. No E3
threshold tuning was performed. Pruning did not solve temporal
distribution-shift/generalization failure; this negative result remains part
of the final project conclusion.

## 12. Scientific interpretation

A fixed attention-derived mask can remove some inference feature/edge work while
preserving this primary E2 metric under the declared tolerance. The measured
mean improvement is descriptive and does not establish that attention reveals
causal feature relevance. The unchanged second GAT and full input adapter limit
the computational savings. Hardware-specific forward gains and allocator peak
changes must remain distinct from graph-size reduction.

Historical E.11 CUDA arithmetic is not certified bitwise equivalent to the strict
E.12 deterministic context. The frozen E.11 unpruned results were reused by
design; no repeated unpruned test evaluation is warranted to remove this caveat.
Repeated temporal/network states can occur across partitions even though sample
indices do not overlap. E2's within-class split does not establish strict global
chronological deployment performance. T=5 has not been established as optimal.

## 13. Supported claims

- One frozen mask retained 61/78 features and reduced the inference graph from
  87 to 70 nodes and 86 to 69 directed edges for all three tasks and seeds.
- No retraining or model-parameter change accompanied E.12B pruning.
- Mean E2 eligible-class macro-F1 increased from 0.8759879709 to 0.8896826984;
  all three observed seed changes stayed within the declared 5% loss tolerance.
- This validation resident-forward benchmark observed paired median speed
  ratios around 1.035x-1.117x and about 2.931 MiB less transient allocated peak.
- E1's restricted diagnostic metric stayed essentially unchanged, while E3's
  strict chronological PR-AUC remained poor and below attack prevalence.

## 14. Unsupported claims

- Statistical significance, invented confidence intervals, guaranteed
  noninferiority, guaranteed seed stability or universal accuracy improvement.
- Causal feature importance, optimal window length or optimal graph policy.
- More than 50% overhead reduction, universal deployment speedup, measured
  end-to-end speedup or training acceleration from these inference observations.
- Reduced parameter count, checkpoint size, peak GPU reservation, total GPU
  usage, CPU memory or initial input-adapter construction/transfer memory.
- Strong chronological attack detection from E1's Benign-only restricted metric,
  or a solution to E3 temporal shift/generalization failure.
- A new Git release freeze: later working-tree evidence has not been committed
  or tagged, and this document grants no release or experimental authorization.

## 15. Final project conclusion and release boundary

The approved experimental scope through E.12B is complete. A fixed
attention-derived inference-time mask retained 61/78 features and reduced the
graph from 87 to 70 nodes and 86 to 69 edges without retraining or changing
parameters. On primary E2, mean eligible-class macro-F1 increased from about
0.8760 to 0.8897. Validation resident-forward measurements showed modest observed
speed gains and less transient allocated GPU memory. E1 stayed essentially
unchanged; E3 remained poor, so pruning did not resolve chronological
generalization failure.

The next safe boundary is **independent final release audit of the completed
experimental scope and final report/documentation**, followed by separately
authorized release actions. No additional ML experiment is currently authorized
or required. Audit readiness does not mean a clean or release-approved Git tree.
Historical HEAD remains the E.9 commit
`2fa823943ba318fe32b2867bc8e2b1f799d78ac1` with tag
`v0.12-stage-e9-model-binding-resource-pilot`. Later untracked/ignored source,
tests, logs, checkpoints and compact evidence must be inventoried for release;
datasets, arrays, logs, environments and checkpoints remain excluded from Git.

This closure changes the live handoff after E.12B completion. Its prior execution
binding remains a historical pin; no attestation, source constant, result or
metric is rewritten. The completed-run preflight's old exact-live-handoff check
will reject the new closure bytes. This is a documentation-binding limitation,
not a need or authorization to rerun E.12B. Final release review should preserve
the historical/current authority distinction and verify persisted evidence.

## 16. Exact provenance and evidence hashes

Study: `e10-official-20261001-v1`; original E.11 run:
`e11-official-20261006-v1`; reviewed recovery: `e11-recovery-20261006-v1`;
E.12A selection: `e12a-20261006-v1`; E.12A freeze namespace:
`e12a-20261006-v1-freeze`; E.12B: `e12b-20261006-v1`.

Canonical content hashes exclude the object's own digest field and use UTF-8
JSON with sorted keys, compact separators and ensure_ascii=False. Physical
hashes cover exact file bytes; these hash types are distinct.

| Artifact and repository-relative path | Canonical content SHA-256 | Physical SHA-256 |
|---|---|---|
| E.10 final report: `results/stage_e10/e10-official-20261001-v1/stage_e10_report.json` | `a898e76698c6e8f1cbda38fe25792d73322880fcdf3ae94499709dfb8f0c49d7` | `8ff3f89c4aac3b2948936ea794e710059a7fe441c161b8c20cae0b8e51f6639c` |
| E.11 recovered report: `results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/stage_e11_report.json` | `4ee7e6a9d7e28231e6ba6765decfdbdc29db624ab5d188d6f8950df8d9bdeba4` | `ace0d585fbb89b0f3df4376b2f69e35bd3172ab663cc66b369b769e84a964aac` |
| E.12A report: `results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/stage_e12a_report.json` | `ccc9be16dee2dea3567bad48f0b9e563cc760914311d0bad4d1611a755d6241d` | `97cc371d78d11d54587a64aadecea26829eb2ec2cd2201428876e9bfcb291b0d` |
| E.12A proposed policy: `results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/proposed_policy.json` | `f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa` | `28727d3d7bfe94f0d624639859d5a3c965dbda1ccbbab760a2ec7bc091f91317` |
| E.12A freeze attestation: `results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1-freeze/freeze_attestation.json` | `6b58ab9133b95a8096ab731288f340ac498243c33c52c7c659d60fd770c29a35` | `7f92187689993ea049cf4a364746301f1175d18ae25402e5a0518f2094032e51` |
| Historical/live authority attestation: `results/provenance/authority/handoff-reconciliation-20261006-v1/attestation.json` | `4e338a1582834aea9a66a77a6ba2de4517ee28c6c61654701f1405ae01549d4b` | `af4b21ea289932dfad648c6affd13a4cc9f91944bcc8d4eca7a94397d3422895` |
| E.12B execution manifest: `logs/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/manifest.json` | `c6717590a5de2413243f4a6a62cecc22bc3851ceef102d6aac3419426f0f10f5` | `b25c15db18880890e7de026a7225c545458f41dd2ddc033c9d4faa27facdf10e` |
| E.12B consumption/authorization ledger: `logs/stage_e12b/e10-official-20261001-v1/test_consumption.json` | `ee9e4b7294e90fa1690eeba7a6b1e6888efdf492406d7909a67baa0daa7afb5a` | `de5e7444f05f31c1dd92901306bc70222d4f5d2408d5418601fc28c0e9f6ba48` |
| E.12B final report: `results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/stage_e12b_report.json` | `5c07ba4183a619f648118af6415abe61a97ebaed14a2ed93e9b68d265ae1b3c5` | `1b76a810bdcb887e22bb077a99754aa04f79ab950626ababb38a1893d8c031da` |
| E.12B validation benchmarks: `results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/validation_benchmarks.json` | No separate embedded content digest | `e20065f502cb6cd0f6bd009d506c58b4356c3c1f64d81bb6fa5511b9bfa3652f` |

The E.12B four-file implementation bundle is
`75931dbcba5ec9d3bcd7d4e173ae9c25c96ac917dff00a6f8574b4f8ef0a4c80`, the SHA-256
of the canonical map of these paths to their physical hashes:

- `docs/stage_e12b_final_comparison.md`: `e90d5214cbfb8190d511a3202752df0fe8990da225e5e1cfe214d4b7aab71771`
- `scripts/run_stage_e12b.py`: `8153c71d87c6aee9afd2e3fc34e287bda8e40872da66acb92d40be9b03df65de`
- `src/pruning/stage_e12b.py`: `9ce95ce70d288de32f3b74cdccef6a37c56a4590fe95ca4b3e1c4ccf7f88afce`
- `tests/test_stage_e12b.py`: `af949fd0d61754c99b28a3b1620c615370241cdd2317e4f6ac844f9e14847d4c`

Historical E.9 handoff physical pin:
`0eb4f09d989a8c641736be9637f7677cdccf3e675d73425bbd0e7cf22a5df9f6`.
Reconciled handoff physical pin used for completed E.12B execution:
`58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5`.
Both remain unchanged in their original bindings. The new documentation hashes
are reported separately at closure, avoiding a self-referential file hash.

Individual final pruned results are `<candidate_id>.json` under
`results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/`.
Checkpoint paths are recoverable from each candidate's bound E.10 artifact record;
checkpoint bytes were hashed without deserializing model state.

| Task | Seed | Candidate ID | Checkpoint physical SHA-256 | Pruned result canonical SHA-256 | Pruned result physical SHA-256 |
|---|---:|---|---|---|---|
| E2 | 2025 | `e10-htgnn-e2-lr1e-3-s2025-b256-4c0dbe73d0e4` | `7a371915bc358363afe910029a5b9b44bbf7c7b7bccb52bc22e1e4e0c15c3710` | `8d1595005d85aef980ee04ee9a1752aa3edf748f63baeae8782e50da38384f49` | `7dc439f7a547a59dd37283923ddb6afb5f58b12fcde484761e234e0ead25ce90` |
| E2 | 2026 | `e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42` | `ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f` | `60229f6c6810293e41cba16fa6ffd102bc4d0c432bd9cf4b8206e7d4b5670d19` | `bd70727dddeafbc9f5da95fbfcfe8a642e66459557bd0c0cb1cafb27a295d33e` |
| E2 | 2027 | `e10-htgnn-e2-lr1e-3-s2027-b256-3df7694884f0` | `e40b15fb581016b20c8bf8d66ba249959d3ab529ba7f9420b37cb5f8bd28fcb5` | `e0b099a6c4d2d6a4abeaadcd0605429b94128ef256bfe5102706c8040b6ef7ba` | `51c68db0b564a6dcf8d2f3e4be92f97168cce21ab0b9e5c468089efa19c765d8` |
| E1 | 2025 | `e10-htgnn-e1-lr1e-3-s2025-b256-c2fd73b62871` | `91215dca8c9ac153d74fb3c150a41ba0890677f978376f6ad0345a4517b62051` | `982e2e5f77a0c518082acc8aa52801bf688f9cfe5aa17b853f9ee6ad09f49543` | `a82d3b073cfd0abaf5d0bf592a290d54be00922c7331ecda5e7d36e292f416ac` |
| E1 | 2026 | `e10-htgnn-e1-lr1e-3-s2026-b256-07fde048fbfd` | `4456c9fc088ac22ef3cc6cd7200e1ad62f0eae29861260f687d3ffbde47f42b4` | `ce131f7f6e61491f687c5537a8a0a10db6b3359a7eae0f0a5993746bd54ff095` | `7680cf910211f1ef9eb5393e1d99ecc6b33b3badb620c0f68ecc22d0da61c538` |
| E1 | 2027 | `e10-htgnn-e1-lr1e-3-s2027-b256-79771d0a8e2c` | `92bdb621f8a566accf77f9fba179709e91e95e41f9436982af4b4028b7e7112c` | `1ecbc8926c269eaf20e13826af96bd6eaa57f270e40c1052596fede3512d0d22` | `b1cbe88aa0fc88a1226d9f8c9faa9ff859a2db721298e80a81776d03b3c46f3e` |
| E3 | 2025 | `e10-htgnn-e3-lr1e-3-s2025-b256-b0ed07a45fa6` | `a26bfded6f661e5d73b673ddff7fb32e23d6e58b82a8ce071fe56a00e1f62e39` | `1086ebcfb385e3a75b5b6b04b2959c610b907b904dd76d27290a6f0493826aad` | `5564fe99120e6aea5cff4b3821ee443dcd841f005695285a11d5c9c77702c0e7` |
| E3 | 2026 | `e10-htgnn-e3-lr1e-3-s2026-b256-3d991d8ab94c` | `0807c4b3818bf61817dc297942853ee1e655d521e39041ac49875f07e5e575cb` | `362b5d70e534d1e0c70c1389bbdfc87a37b3cd1127d405877a3d06c8de139f31` | `20a32c2e1ceaaa7ff5d15bf24afd3765518eb405a05a812099464ddc653ae682` |
| E3 | 2027 | `e10-htgnn-e3-lr1e-3-s2027-b256-63b7a22cd5e9` | `187536855ba976710ad392885f096d9f7ea2cd28f1d8522fee2f7beee02de40c` | `c45c996dbbc3631ca74891b827c1b699601693e28aa3ca4445335949ec42dd3a` | `41868f1b22cf2600ef0c77f0471bc0361fa27dea8b629aa9fcc67600d36df122` |

The exact existing unpruned result bindings reused by E.12B are:

| Task | Seed | Baseline result path | Canonical SHA-256 | Physical SHA-256 |
|---|---:|---|---|---|
| E2 | 2025 | `results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e2-lr1e-3-s2025-b256-4c0dbe73d0e4.json` | `620ecd85049738873b5c75632bd8d3b2f8435cf03276aae397fabddb4e5e8301` | `9be44707b105403e784a9fb55cfa4397dfeb22db53d9d6c39f57e7ac36520b50` |
| E2 | 2026 | `results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42.json` | `92eaaaf497dc2ef41ae8826240edabcc9646118f1b926ce92313afa6c5caafec` | `6890e8c2a2c34fd9c4996810cc90ad2a7ecef1616cf410f6e1950fdca656baf1` |
| E2 | 2027 | `results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e2-lr1e-3-s2027-b256-3df7694884f0.json` | `8b90174a1de3d54418e32d2d48031b2a28e6db7aa8ff5c4ede9d8a1014235c76` | `2b6091a02d9fc5baaa7b0d27e8d66ff0a5c165a298ce7ee378be4ffa28e5bacd` |
| E1 | 2025 | `results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e1-lr1e-3-s2025-b256-c2fd73b62871.json` | `d415ebb27efde080bf4ca2a6f1c09f5a48b37f05b26a16a151365e16e893c85f` | `44f3bed7240d909a630d2425aab5726d20c764cd159ed0f5339fd0dc5e26ba4d` |
| E1 | 2026 | `results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e1-lr1e-3-s2026-b256-07fde048fbfd.json` | `08a9a8f2deaacca5d64968f8c85176725b9f774a8813118383449e3555f43bfe` | `95686ffca2582c4022368c2421c0ff8231793f0426d6bde70318f06bd7ee2aef` |
| E1 | 2027 | `results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e1-lr1e-3-s2027-b256-79771d0a8e2c.json` | `5d62ed2ef3fef8a627b76ddccc3da5d1e79228f7a974187f75e63fed7e97d3f1` | `5e0f0e7b014e9fe4371d608d59d34b4ce840aaff7d75d16880ad64f5d8c03ca8` |
| E3 | 2025 | `results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/e10-htgnn-e3-lr1e-3-s2025-b256-b0ed07a45fa6.json` | `198b5aa326cbd45f49a63fdfefd95e6f39a92e4185ac5505e010dff16610b676` | `b0a69a4205328e31c003786479dd4972b63a2c79bd1bb728802e9d31e23f5f2e` |
| E3 | 2026 | `results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/e10-htgnn-e3-lr1e-3-s2026-b256-3d991d8ab94c.json` | `3f3ed412a06d3017d0858346e60fbe858e2a96cf2dbb0bd1482c6f6ed887befd` | `c8e1a3b6e21ac32a3ad641dc6de50faf3dd4b573997f671578a98e92b7d11d83` |
| E3 | 2027 | `results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/e10-htgnn-e3-lr1e-3-s2027-b256-63b7a22cd5e9.json` | `933f967aaf215c8266ed38843f5f1e544a10ec0a8973acbad7c851427097e848` | `6c105d38a0e8b3da364d2183a640d63fd537091f8f87c63e8f27a867016664bf` |

Read-only closure checks reproduced canonical report hashes, physical report
sidecars, 225 E.10 artifact bindings (including all 45 official checkpoints),
27 E.11 result/checkpoint memberships, original E.11 preservation bindings,
E.12A freeze artifact/source pins and 15 validation macro-F1/acceptance values,
nine E.12B result/baseline/checkpoint/policy bindings and seed summaries, and
720 persisted benchmark forward pairs. Macro-F1 was rederived from persisted
confusion matrices. E3 PR-AUC was checked as a persisted value and seed summary;
it cannot be reconstructed from confusion counts alone without prediction scores.
No test predictive array was opened or rehashed, no ML command ran, and no suite
containing real-data/training tests was invoked. Dataset digests stored in the
original manifests remain historical bindings, not newly verified array hashes.
