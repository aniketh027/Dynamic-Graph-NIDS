# Phase C1 completion report - 2026-10-07

C1 application foundation is complete and remains unstaged. The scientific
release remains `2854d153d65c93689875884f3ad45f0b3b468ec2` on `app-demo`.
No C2 CSV replay implementation, deployment publication or Git release occurred.

## Application deployment decision

The owner's application-only rule was applied to exactly three persisted E2
HTGNN pruned-validation results under the existing frozen E.12A policy. Higher
eligible-class macro-F1 wins; exact ties use the lowest numeric seed.

| Seed | Already-recorded pruned-validation eligible-class macro-F1 |
| --- | ---: |
| 2025 | 0.8598611221399799 |
| **2026** | **0.8682839242491506** |
| 2027 | 0.8546017058225671 |

Exact evidence for all three values:
`results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/validation_sweep.json`,
`rows[2].seed_results[0..2]`, matched by seed and the exact attested policy.
This is ratio 0.25 and full E2 validation: 311,916 examples per seed, identical
membership/order SHA-256
`f801622add9b529c156299da5179c239fb9a8b8a19159a49e68ba6c95cdf615b`.
Eligibility codes are `[0,1,4,5,6,7,8,9,10,11,12,14]`, matching the E2 checkpoint
selection provenance. Recorded confusion-matrix totals and their eligible-class
macro-F1 values were independently reconciled during inspection without loading
predictive arrays or running evaluation.

Sweep physical SHA-256:
`ebb76b88b3f08e2b27a027a93ab8878d32e1ff5ebf2af220cc584f1a2c2ddf95`.
Binding attestation:
`results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1-freeze/freeze_attestation.json`.
Its physical SHA-256 is
`7f92187689993ea049cf4a364746301f1175d18ae25402e5a0518f2094032e51`.

Selected deployment seed: **2026**.
Checkpoint ID: `e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42`.
Checkpoint physical SHA-256:
`ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f`.
Exact checkpoint path:
`checkpoints/stage_e10/e10-official-20261001-v1/final_seed/e2/htgnn/e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42/best.pt`.
Official membership/configuration/metadata/bytes were checked against
`results/stage_e10/e10-official-20261001-v1/stage_e10_report.json` and the freeze.
All three checkpoint byte hashes match their frozen records. The learning rate
0.001 is the existing E.10 validation-only selection at seed 2025.

**Selection used zero test evidence. No new ML study/evaluation run occurred.**
No E.10 unpruned validation score, E.11/E.12B metric, test latency or ensemble
was substituted for the frozen pruned-validation scores.

## Files and backend structure

Created backend application area:

```text
backend/
  __init__.py
  .gitignore
  .env.example
  requirements.txt
  requirements-dev.txt
  README.md
  C1_REPORT.md
  app/
    __init__.py
    main.py
    api/       __init__.py, routes.py
    core/      __init__.py, config.py, errors.py
    schemas/   __init__.py, contracts.py
    services/  __init__.py, artifacts.py, inference.py
  tests/
    __init__.py
    test_foundation.py
    safety.py
    run_safe.py
    serve_safe.py
```

Created frontend files:

- `frontend/.env.example`
- `frontend/src/components/backend-provider.tsx`
- `frontend/src/lib/api/client.ts`
- `frontend/src/lib/api/context.ts`
- `frontend/tests/api-client.test.mjs`

Modified frontend files:

- `frontend/package.json` - native Node API-client test command only
- `frontend/README.md` - backend setup, state semantics and C1 limitations
- `frontend/src/App.tsx` - shared backend provider
- `frontend/src/components/app-shell.tsx` - connection/adapter status, honest notice,
  system information and neutral footer
- `frontend/src/components/demo-controls.tsx` - demo copy independent of connection
- `frontend/src/components/traffic-file-input.tsx` - honest disabled-analysis copy
- `frontend/src/pages/workspace-pages.tsx` - connection/adapter/capability state

An ignored browser QA image was created at `frontend/qa/c1-connected.png`.
Frontend dependency declarations/versions and `package-lock.json` were not changed.
No root, scientific, handoff, research-result or release file was modified.

## Dependencies and environment

Application-only additions were installed into the existing repository `.venv`
after a constrained dry run. Every pre-existing installed package kept its exact
version. Python remains 3.13.0, Torch 2.14.0+cu130 and PyG 2.8.0.post1.
No global installation or scientific lockfile change occurred. `pip check` passes.

| New package | Exact version | Declaration |
| --- | --- | --- |
| fastapi | 0.142.2 | backend/requirements.txt |
| uvicorn | 0.54.0 | backend/requirements.txt |
| pydantic | 2.13.5 | backend/requirements.txt |
| pydantic_core | 2.46.5 | backend/requirements.txt |
| starlette | 1.7.0 | backend/requirements.txt |
| anyio | 4.15.1 | backend/requirements.txt |
| annotated-doc | 0.0.5 | backend/requirements.txt |
| annotated-types | 0.8.0 | backend/requirements.txt |
| typing-inspection | 0.4.4 | backend/requirements.txt |
| click | 8.5.0 | backend/requirements.txt |
| h11 | 0.16.0 | backend/requirements.txt |
| opentelemetry-api | 1.45.1 | backend/requirements.txt |
| httpx | 0.28.1 | backend/requirements-dev.txt |
| httpcore | 1.0.9 | backend/requirements-dev.txt |

## API, adapter and frontend behavior

Five GET routes are implemented: `/api/v1/health`, `/status`, `/model`, `/schema`
and `/capabilities`, all under `/api/v1`.
Pydantic schemas are `HealthResponse`, `StatusResponse`, `ModelResponse`,
`SchemaResponse`, `CapabilitiesResponse`, `APIError` and `ErrorResponse`.
Public errors are sanitized `{error:{code,message}}` envelopes. Missing metadata,
unimplemented operations, invalid requests and internal exceptions have typed
responses without local paths or tracebacks.

`FrozenInferenceService` owns authenticated discovery, startup loading, status,
metadata, numeric shape/order validation and the closed C1 prediction boundary.
States are uninitialized/loading/ready/error. Loading occurs during FastAPI
lifespan; health/status/schema survive an initialization failure. Configuration
supports explicit CORS origins and metadata-only startup. Wildcards are rejected.

Authoritative ML APIs reused:

- `src.preprocessing.schema.MODEL_FEATURE_COLUMNS` for ordered input features
- `src.graph.schema` for canonical serialization and temporal/topology semantics
- `src.preprocessing.labels.LABEL_VOCABULARY` for class count
- `src.pruning.stage_e12.load_bound_model` for E2 inference-only checkpoint loading
- `src.pruning.stage_e12.FeaturePolicy` for exact frozen-mask compatibility
- the loader's existing E.7 model contract, metadata/tensor preflight and strict
  model-state loading; no application `torch.load` or alternative state loader

Discovery reads the pinned E.10 report, E.12A attestation, policy, sweep, original
compact provenance/source bindings and three checkpoint hashes. It has no E.11,
E.12B or dataset reader. Compact JSON reads are size-bounded, and artifact paths
must stay inside the repository. The selected model is CPU/eval with gradients
disabled, exactly 7,183 parameters and no optimizer construction/restoration.
Historical optimizer validation stays in E.10 provenance; this is not a new
optimizer-resume audit.

Pruning remains the shared frozen E.12A mask: 78 model-input features, 61 retained
feature nodes, 17 removed, eight categories retained, 87 -> 70 total nodes and
86 -> 69 edges. Model parameters are unchanged. T=5 is five consecutive clean
flow records; the prediction target is the fifth/final flow. The policy canonical
hash remains `f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa`.

`ready` means the frozen model/policy adapter loaded. `inference_ready` remains
false because application replay/prediction is not implemented. There is no
prediction/upload endpoint. `predict_window` rejects prediction even for valid
synthetic dimensions. Shape checks do not certify CSV cleanliness or provenance.

Frontend polling is centralized, uses `VITE_API_BASE_URL` with local fallback,
has a three-second timeout and checks every ten seconds. It distinguishes backend
connected, adapter ready/unloaded/loading/error, replay pending and live capture
unsupported. Demo mode remains frontend-only. Analyze stays disabled; no file
contents are read, parsed, uploaded or processed. The footer says
**Application preview**, with no visible implementation-phase branding.

## Verification results and limits

| Check | Result |
| --- | --- |
| Focused backend tests | 38 passed, 0 failed, 0 skipped |
| Selected frozen D2/E.12 schema/policy/loader regressions | 35 passed, 61 deselected |
| Python safety guard | 0 violations on focused, regression and real-loader checks |
| Real selected-checkpoint/policy loading | PASS; CPU/eval/frozen/7,183 parameters; no forward |
| Native Uvicorn startup | PASS; local listener at 127.0.0.1:8000 |
| Native HTTP endpoints | All five returned 200 with expected bodies |
| Frontend client tests | 6 passed, 0 failed |
| npm run build | PASS |
| npm run lint | PASS; 44 existing upstream registry warnings retained |
| npm run typecheck | PASS |
| pip check | PASS |
| git diff --check and scoped whitespace/source review | PASS |
| Tracked/staged diffs | Both empty |
| Baseline SHA-256 comparison | 331/331 identical, including 46 protected local files |

The regression command selects static D2 graph tests and E.12 synthetic policy,
compaction and loader tests. Its pruned forward smoke uses only newly constructed
synthetic graphs and temporary synthetic checkpoints, fully detached from all
official partitions. No official train/validation/test array was opened.
The full research suite was deliberately not run: it contains real-data/training
checks outside the owner's C1-safe authorization.

Browser QA verified connected dashboard/Analyze Traffic, model-adapter readiness,
disabled Analyze, neutral footer, no horizontal overflow at the existing desktop
viewport, and synthetic demo start/reset. Screenshot: `frontend/qa/c1-connected.png`.
After stopping the smoke backend, an in-app browser resource/load error
(`ERR_NO_BUFFER_SPACE`) prevented native disconnected-screen verification.
Offline/abort/recovery behavior passed the deterministic API-client tests. No
responsive-matrix claim or browser-offline claim is made.

Remaining warnings: existing >500 kB frontend chunk advisory and registry lint
warnings, Torch/PyG deprecations on selected synthetic tests, and Starlette's
httpx TestClient deprecation. No failure was suppressed. The native smoke backend
was stopped after verification; the pre-existing Vite process was preserved.

## Git, preservation and next boundary

Final branch/HEAD remain `app-demo` / `2854d153d65c93689875884f3ad45f0b3b468ec2`.
`git diff` and `git diff --cached` are empty. `backend/` is newly untracked;
`frontend/` remains untracked; all 20 pre-existing protected root entries remain
unchanged. No staging, commit, push, tag, reset, stash or clean occurred.
All 285 pre-existing tracked files and 46 protected E.10 local files were verified
byte-identical. Scientific methodology, results, evidence, release records,
checkpoints, pruning mask and root dependencies were preserved.

The exact next application boundary is a **separately authorized C2 bounded CSV
replay contract and implementation** using frozen E2 train-fitted transforms,
canonical ordering, five-flow semantics and authoritative graph/model/pruning APIs.
External-request provenance must be designed explicitly: current D4/D6/E.2 and
E.12 validators bind scientific split membership. Do not spoof user windows as
scientific samples, duplicate topology or weaken frozen checks. C1 ends here;
C2, official evaluation, calibration, training and live capture were not started.
