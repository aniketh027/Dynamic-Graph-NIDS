# Stage E.8 Bounded HTGNN Optimization Pilot

Stage E.8 evaluates the three already approved learning rates by orchestrating
the corrected Stage E.7 training harness. It is a small train/validation project
engineering pilot, not full training, a multi-seed study, test evaluation, a
batch-size study, an imbalance ablation, a baseline comparison, or pruning.
The current frozen checkpoint remains the historical
`v0.10-stage-e7-training-harness`. An independent E.8 audit found that the
original E2 objective used a D5 class set derived from train, validation, and
test support. E.8 remained unfrozen. This correction removes that indirect
test leakage; both the E.7 correction and regenerated E.8 package require a
new independent audit and explicit release authorization.

## Pilot contract

The handoff identified E.8 as the next safe numbered boundary but did not
freeze exact pilot bounds or assign the weighted-loss ablations to E.8. This
implementation therefore uses only the unweighted losses already frozen in
E.7 and varies only learning rate within each task.

| Field | E.8 value |
| --- | --- |
| Tasks | E1, E2, E3 |
| Learning rates | `1e-4`, `3e-4`, `1e-3` |
| Loss mode | frozen unweighted task loss |
| Seed | `2025` |
| Batch size | `32` |
| Epochs | exactly 3 maximum |
| Train bound | 16 batches / 512 examples per epoch |
| Validation bound | 128 sequential batches / 4,096 examples per epoch |
| Candidates | 9 |
| Maximum batch passes | 3,888 |

Three epochs are an explicit pilot override to the E.7 production ceiling of
30 epochs. The frozen E.7 minimum-epoch, patience, and `min_delta` values
remain unchanged at 3, 5, and `1e-4`; this short pilot is not long enough to
exercise patience-based termination. Candidate initialization, training
membership/order, validation membership/order, batch size, epoch budget,
architecture, preprocessing, loss mode, and objective are identical within a
task. The same seed is used for all learning rates.

The 128-batch validation bound was selected because a 128-example prefix was
observed to contain only one class for each task. The 4,096-example prefix
contains the two validation-supported E1 classes that also occur in E1 train
and both E3 binary labels. It remains incomplete for the E2 validation-selection
class set, so E2 records `validation_selection_support_complete=false`. The E2
macro F1 is calculated over the test-blind class set derived from train support
≥100 and validation support ≥30. Test labels, support, membership, metadata,
and the separate three-split final-reporting eligibility are not consulted.
Absent selection classes contribute zero within this bounded prefix. This
limitation is a reason not to treat the pilot ranking as a finalized
configuration.

## Data boundary and execution

The E.8 API accepts no split argument and delegates each candidate to
`src.training.harness.run_training`. The corrected E.7 path constructs only the
train and validation datasets and rejects test traversal. E.8 candidate and
summary records explicitly require both `test_accessed=false` and
`test_metrics_recorded=false`. Test data and labels do not contribute to
candidate generation, loss, learning-rate comparison, checkpoint selection,
ranking, or any other E.8 decision.

Codex executed the bounded pilot on CUDA with:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_e8_optimization_pilot.py --device cuda --executor Codex
```

The completed run ID is `pilot-20260927t183642z-40eba2b1`. All nine candidates
completed. Each candidate used a unique checkpoint, JSONL log, and compact
candidate-result path. The runner rejected pre-existing paths, verified the
best checkpoint against the exact candidate configuration, compared the JSONL
history with the returned E.7 history, and checked cross-candidate traversal
digests before writing the observational report.

## Bounded observations

Higher is better for every frozen validation objective.

| Task | LR | Best epoch | Best validation objective | Support complete |
| --- | ---: | ---: | ---: | --- |
| E1 | `1e-4` | 1 | 0 | yes |
| E1 | `3e-4` | 3 | 0.000244081034903588 | yes |
| E1 | `1e-3` | 1 | 0.000244081034903588 | yes |
| E2 | `1e-4` | 1 | 0.02813852813852814 | no |
| E2 | `3e-4` | 1 | 0.02813852813852814 | no |
| E2 | `1e-3` | 1 | 0 | no |
| E3 | `1e-4` | 1 | 0.9980720183208773 | yes |
| E3 | `3e-4` | 1 | 0.9980720170907134 | yes |
| E3 | `1e-3` | 3 | 0.9998418353237147 | yes |

For observational ordering only, ties are resolved by the predeclared approved
LR order: `1e-4`, then `3e-4`, then `1e-3`. Under the bounded Stage E.8 pilot
configuration, `3e-4` produced the strongest E1 observation, `1e-4` produced
the strongest E2 observation, and `1e-3` produced the strongest E3
observation. The E1 and E2 leaders are numerical ties resolved by that rule.
No result establishes multi-seed robustness or final model quality, and no
production configuration is frozen or selected.

The corrected E2 candidate values are numerically unchanged from the rejected
pre-correction package because the three real classes failing the test-support
threshold also fail the train/validation thresholds. The equality is a corpus
coincidence, not permission to use test support for selection. The E3 bounded
prefix remains 4,095 attacks and one benign example, so its near-one PR-AUC is
highly composition-dependent and is not a general performance conclusion.

## Artifacts, resumability, and evidence

Checkpoints and JSONL logs remain under ignored `checkpoints/stage_e8/` and
`logs/stage_e8/` paths. They are not committed. The compact observational
report contains the candidate configurations, histories, identities, artifact
hashes, failure status, and elapsed runtime. Deterministic contract evidence
separately records the candidate definitions, corrected E.7 source hashes,
frozen model-source hashes, data boundary, completed identities, integrity
results, and verification results.

The runner never overwrites a prior run. A manual multi-command execution must
use one fresh run ID. Its results can be aggregated only when all nine exact
candidate-result files and their matching checkpoint/log hashes validate:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_e8_optimization_pilot.py --aggregate-only --run-id <fresh-run-id> --output <fresh-observations-path> --executor User
```

Canonical contract evidence is in
`results/stage_e8/htgnn_optimization_pilot_contract.json`; noisy losses,
validation metrics, epoch counts, and runtime observations are deliberately
separate in
`results/stage_e8/htgnn_optimization_pilot_observations.json`.

Corrected implementation validation reproduced 20 focused E.8 tests, 247
E.2-E.7 regression tests, and 962 complete-suite tests, with zero failures or skips.
Stage C integrity remained 33/33 and raw CSV integrity remained 10/10. The
canonical contract SHA-256 is
`2c34c71c99c2aee4dba69a909e7094eadd4aa12b1cd7627e318b43c4813746ee`;
the physical contract-file SHA-256 is
`2015a4b71c7e70f6a9600f76b34388a953b27b33bca356e539e1c1570da1ee62`.

Stage E.8 stops here. The next safe action is an independent read-only audit of
the E2 selection-leakage correction and regenerated E.8 package. E.9 and later
work remain unauthorized.
