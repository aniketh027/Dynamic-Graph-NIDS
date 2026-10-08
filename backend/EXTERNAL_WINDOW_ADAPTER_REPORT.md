# External application window adapter completion — 2026-10-07

**PASS: the missing post-transform application adapter contract is implemented.**
The version is `external-application-graph-adapter-v1`. This is application
infrastructure on `app-demo`, separate from the frozen scientific release.
Graph, unpruned-logit and frozen-mask pruned-logit equivalence passed with exact
`torch.equal` comparisons on deterministic CPU execution for synthetic B=1,2,3.
No numerical tolerance was needed. No official data sample was accessed.

## Files and scope

Seven files were created:

- `backend/app/ml/__init__.py`
- `backend/app/ml/external_windows.py`
- `backend/app/ml/application_graph.py`
- `backend/app/ml/application_inference.py`
- `backend/tests/test_external_application.py`
- `backend/tests/run_application_safe.py`
- `backend/EXTERNAL_WINDOW_ADAPTER_REPORT.md` (this report)

No pre-existing file was modified. In particular, no frozen `src/` file,
scientific test, frontend file, dependency declaration, checkpoint, policy or
evidence artifact changed. The approved pre-existing unstaged
`PROJECT_HANDOFF.md` reconciliation was preserved. No endpoint, CSV ingestion,
raw preprocessing, file replay, PCAP parsing or live capture was implemented.
The C1 routes, capabilities and closed `FrozenInferenceService.predict_window`
boundary remain as they were; the new adapter is a separate internal Python API.

## Window and provenance contract

`ApplicationInferenceAdapter.windows(...)` accepts only a NumPy `float32`
array with shape `(B,5,78)`, finite values and `0 < B <= max_windows`.
`max_windows` is an explicit positive integer, revalidated during graph creation
and inference. Noncontiguous and Fortran-order arrays are normalized by a
bounded owned copy, preserving logical graph/flow/feature order. No scaling,
fitting, replacement of nonfinite values, sentinel modification or column
reordering occurs.

`ExternalWindows` owns immutable C-order payload bytes. Its NumPy views are
read-only and cannot be made writable. Graph conversion creates independent
Torch-owned storage. Caller mutation after construction cannot change the
snapshot or graph; graph mutation cannot change the snapshot or a later graph.
The payload SHA-256 binds the bytes plus canonical shape, dtype, order and axes.

The frozen dataclasses `TransformIdentity`, `ApplicationIdentity` and
`ApplicationProvenance` form the typed, immutable application provenance schema.
`ApplicationProvenance` contains:

- `source_kind = external_application`, source ID, analysis ID, replay-order
  semantics, batch size and transformed-payload SHA-256;
- nested application identity with adapter version, feature-schema/order
  hashes, expected transform-state ID/hash, T=5, clean-flow/window and final-flow
  target semantics;
- taxonomy/schema versions and canonical hashes, frozen policy artifact hash
  and policy-record hash, checkpoint ID/hash, deployment seed, model task and
  authoritative class-mapping hash.

There is no research split or source experiment field. `model_task=E2` describes
the deployed classifier only. Audit metadata occupies the global
`application_audit` namespace. Research audits, `e2_audit`, fake labels, extra
stores, extra predictive fields and extra relations are rejected.

The expected transform identity is supplied by trusted application configuration
at adapter initialization and matched exactly thereafter. This version records
and checks that identity; it does not authenticate scaler-file bytes or prove
that a producer cleaned/transformed consecutive flow records. The synthetic
tests deliberately use a synthetic transform identity. Establishing the actual
frozen transform identity and correct raw-to-window provenance belongs to C2A.

## Graph construction and validation

One direct already-batched `HeteroData` is created per bounded batch. No
per-sample graph collection is created. Input `(B,5,78)` becomes graph-major
`(B,78,5)` and then `feature.x` of shape `(B*78,5)` without value changes.

Scientific authorities reused directly:

- `src.preprocessing.schema.MODEL_FEATURE_COLUMNS`;
- Stage D `taxonomy` manifest builder and category membership;
- Stage D `schema` node kinds/counts/IDs, relation types, source/destination
  edge templates, temporal metadata and canonical serialization;
- `d4.STATIC_GRAPH_METADATA` approved taxonomy/schema identities;
- `src.preprocessing.labels.INT_TO_LABEL` for the E2 output mapping;
- C1 `discover` and `load_frozen`, E.12 `FeaturePolicy`, `compact_inputs`,
  `state_sha256`, deterministic inference context and selected model loader;
- the frozen E.7 `model_contract` and E.5 `_validated_head` contract checks.

No feature list, class vocabulary or category membership mapping was copied.
Edges are derived by adding graph offsets to the authoritative Stage D templates.

Before compaction, the independent application validator checks exact node and
relation stores; B*78/B*8/B counts; graph-major original feature/category IDs;
all feature/category/sample `batch` and `ptr` tensors; CPU device, dtype, shape,
contiguity and disabled tensor gradients; finite float32 feature values and their
payload identity; and exact ordered relation tensors. Exact template equality
enforces B*86 edges, valid type-local endpoints, no cross-sample edges, and the
complete frozen feature-category/category-sample mappings. PyG endpoint
validation also runs. No labels, reverse edges, self-loops, additional node types
or convenience relations are attached.

## Authentication and inference

Initialization uses the unchanged C1 discovery path: pinned E.10 report,
physically pinned E.12A freeze attestation, canonical checks, original artifact
and source bindings, policy bytes, persisted seed/checkpoint bindings and
checkpoint byte hashes. Missing or corrupt artifacts fail closed.

Frozen policy canonical SHA-256:
`f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa`.
Policy physical SHA-256:
`28727d3d7bfe94f0d624639859d5a3c965dbda1ccbbab760a2ec7bc091f91317`.
The artifact supplies the retained IDs; the adapter does not hardcode them.
Tests assert the expected 61 retained features, 17 specified removals, eight
categories, one sample, 70 inference nodes and 69 edges.

The selected checkpoint remains seed **2026**, ID
`e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42`, physical SHA-256
`ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f`.
Loading uses C1 `load_frozen` -> E.12 `load_bound_model`, including the frozen
metadata/source/tensor preflight and strict state loading. No application
`torch.load`, optimizer construction or optimizer restoration was introduced.
The model has **7,183 unchanged parameters**. Its loaded-state digest was
`e2aa39731f862e542cf790d03ccd77aa23d22f429b14596d4a9a4c094409b178`;
equivalence tests verify unchanged state after all forwards.

`logits(graph, max_windows=...)` validates model task/head/parameter contract,
CPU float32 state, all-module eval mode, disabled/absent parameter gradients,
the authenticated model-state digest, application graph/provenance and exact
frozen policy-record identity. The caller must disable gradients using
`torch.inference_mode()` or `torch.no_grad()`; gradient-enabled entry is rejected.

**E.12 `compact_inputs` is reused directly** after this independent application
validation. It has no split assumptions. Neither research `_validate_batch` nor
research `pruned_logits` is called by the application path. The small forward
wrapper invokes the same feature encoder, feature-category GATv2,
residual/dropout/norm/activation, category-sample GATv2,
residual/dropout/norm/activation and validated E2 head in the same order.
It returns finite raw logits `[B,15]`. No softmax, threshold or additional
operation changes the model math. CPU deterministic settings are restored on
exit. All eight category IDs and category-sample edges remain unchanged.

Example internal usage, with already transformed values:

```python
from pathlib import Path
import torch
from backend.app.ml.application_inference import ApplicationInferenceAdapter
from backend.app.ml.external_windows import TransformIdentity

adapter = ApplicationInferenceAdapter(
    Path(r"C:\Project\ComputerNetworks"),
    transform_state=TransformIdentity(expected_transform_id, expected_transform_sha256),
)
owned = adapter.windows(
    transformed_windows,  # np.float32 [B,5,78]; supplied by an authorized producer
    source_id=source_id,
    analysis_id=analysis_id,
    replay_order_semantics=replay_order_semantics,
    max_windows=batch_limit,
)
with torch.inference_mode():
    raw_logits = adapter.infer(owned, max_windows=batch_limit)
```

## Reproduced acceptance evidence

The existing `tests/test_pyg_adapter.py` helper creates a tiny synthetic dataset
in pytest's temporary directory. Its authorized noncanonical D4/D6 fixture
mechanism feeds the unchanged E.2 converter. It supplies actual synthetic fixture
membership/targets on the research side; no identity or label is attached to the
application side. The same numeric values, with nontrivial ordered indices,
are used for both paths at B=1,2,3.

| Acceptance check | Result |
| --- | --- |
| Graph equivalence | Exact for all three batch sizes |
| Compared tensors | `feature.x`, `feature_id`, `category_id`; feature/category/sample `batch` and `ptr`; both `edge_index` tensors |
| Compared structure | Node/relation order, every type-local node count, total nodes/edges and graph/feature/temporal order |
| Compaction equivalence | Exact values, original retained IDs and compacted edges |
| Unpruned real-checkpoint logits | Exact `torch.equal` on CPU |
| Pruned real-checkpoint logits | Exact `torch.equal` against unchanged E.12 research fixture path |
| Deterministic repeats | Exact, including repeated `logits` and the `infer` path |
| Research-call isolation | Application still passes while research `pruned_logits` and `_validate_batch` are patched to fail |
| Ownership | Caller, immutable snapshot and Torch graph storage are independent |
| Model immutability | State digest unchanged; all parameter gradients remain absent |

Adversarial coverage rejects malformed dimensions, empty/over-bound batches,
float64/float16/integer/object/foreign-endian arrays, NaN and both infinities;
mutated IDs/counts/batch/ptr, wrong tensor shape/dtype/device, missing/extra/
redirected/reordered/cross-sample/negative/out-of-range edges; extra nodes,
relations, labels and fields; changed feature values/payloads; missing or fake
research provenance; altered schema/transform/policy/checkpoint/task/seed identity
and wrong identity-field types; a valid-quota but different retained-feature
mask; corrupt artifact bytes; model state mutation, training mode, enabled or
present gradients, and gradient-enabled callers. Rejection-before-encoder hooks
verify that malformed inference inputs do not reach predictive computation.
Research E.12 still rejects test, unknown/application splits and application
graphs. Frozen research source and guard behavior were not changed.

Run from the repository's existing isolated `.venv`, using `-B`:

```powershell
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe --regressions
```

| Final validation | Result |
| --- | --- |
| Focused application adapter | **143 passed**, no failure/skip, 0 guard violations |
| Safe regression preset | **257 passed**, 78 intentionally deselected, no failure/skip, 0 guard violations |
| Regression composition | 219 frozen D6/E.2/E.3/E.4/E.5/E.12 checks plus 38 C1 backend checks |
| Python syntax | All six new Python files parse |
| Tracked `git diff --check` | PASS |
| New-file whitespace/diff review | PASS |

The preset excludes real-data integrations, report regeneration that reads
partitions, backward/gradient tests, training/dropout-mode checks, ranking,
evaluation, benchmark, policy selection and scientific orchestration. The full
scientific suite was not run because it requires unauthorized data/training.
Existing Torch/PyG and Starlette/httpx deprecation warnings were retained.
No new dependency was installed.

The Python audit guard blocks official data, CSV/NumPy arrays in the repository,
scaler-state access, E.11/E.12B evidence, scientific writes, protected E.10 writes,
optimizers, backward/autograd, ranking, evaluation and benchmarking. Only the
two exact E.12A compact manifest/status paths under `logs/` are readable through
the additional application guard. The guard is an accidental Python-access
boundary, not a native-code security sandbox.

Initial validation attempts were handled explicitly: the new guard first
overblocked those two permitted compact provenance records; its exceptions were
narrowed to their exact paths. An initial regression selection included the D6
saved-report regeneration test; the guard rejected its official split-metadata
open before any read. That test was excluded, and the source-selected safe preset
above passed. No blocked read was bypassed and no scientific validator was
weakened. Final successful runs have zero guard violations.

## Preservation, Git and next boundary

SHA-256 comparison preserved **408/408 pre-existing tracked/application files**
byte-for-byte. This includes all frozen tracked files, existing backend/frontend
files and the approved handoff. The handoff hash remains
`b4d30d66785121ce4b27db5023bf124125225e53b559cee3207ba3a77ea557e1`.

All **20 protected local E.10 root entries** remain present. Their complete
**58/58 file/directory metadata records**, including **46 files**, have unchanged
inventory, size, modification time and attributes. Protected local recovery
payloads were not opened to create new byte hashes. This is metadata-preservation
evidence plus the scientific-write guard, not a new cryptographic content audit
of recovery archives/logs. All adapter writes were confined to the seven new
backend files; a preservation baseline was saved outside the repository.

Branch and HEAD remain `app-demo` /
`2854d153d65c93689875884f3ad45f0b3b468ec2`; the peeled final release tag is the
same commit. Tracked `git diff` still contains only the pre-existing handoff
reconciliation: **90 insertions, 24 deletions**. Staged diff remains empty.
The new files remain within the already-untracked `backend/` directory, so
tracked diff statistics do not include them. Scoped source/new-file review was
performed separately. `git status --short` is identical to the initial snapshot:

```text
 M PROJECT_HANDOFF.md
?? backend/
?? e10-interrupted-attempt-20261003-140737/
?? e10-overnight-20261002-042253/
?? e10-overnight-20261002-042528/
?? e10-phase-a-final-resume-20261003-183405/
?? e10-phase-a-final-resume-verification-20261003-183405.zip
?? e10-phase-a-final-two-20261003-190112/
?? e10-phase-a-final-two-verification-20261003-190112.zip
?? e10-phase-a-overnight-20261002-043636/
?? e10-phase-a-resume-20261003-140507/
?? e10-phase-a-resume-20261003-140810/
?? e10-phase-a-resume-verification-20261003-140507.zip
?? e10-phase-a-resume-verification-20261003-140810.zip
?? e10-verification-20261002-042253.zip
?? e10-verification-20261002-042253/
?? e10-verification-20261002-042528.zip
?? e10-verification-20261002-042528/
?? frontend/
?? run_e10_phase_a_final_two.ps1
?? run_e10_phase_a_overnight.ps1
?? run_e10_phase_a_resume.ps1
?? run_e10_phase_a_resume_after_job1.ps1
```

No stage, commit, push, tag, stash, reset or clean occurred. No official sample,
training, scientific evaluation, attention ranking, pruning selection or latency
benchmark occurred. Synthetic logits validate execution equivalence only; they
carry no scientific prediction-quality interpretation.

Limitations: this version supports bounded CPU float32 inference only; transform
and replay-order assertions come from a trusted producer; it does not verify raw
cleanliness/consecutiveness or implement network-facing inference. Artifact bytes
are authenticated on initialization; each forward validates the loaded model
state and policy against that authenticated binding. No deployment, calibration,
throughput, end-to-end memory or real-data accuracy claim is made.

**The adapter acceptance checks pass and close the missing post-transform path.**
It is technically ready to serve as C2A's graph/inference boundary. The exact next
safe boundary from the unchanged handoff is **owner review and approval of this
application-specific external-window/provenance contract before C2A can proceed**.
CSV ingestion and C2A require their own authorization and were not started.
This task stops here.
