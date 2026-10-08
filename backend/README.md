# GraphNIDS local backend foundation

FastAPI wraps the frozen scientific implementation in `src/`. This local
application preview provides metadata, health, adapter status and bounded C2A
CSV replay inference. It is not a production deployment. C2B frontend replay
integration and C2B.1 UI polish are complete; demo mode remains synthetic
frontend behavior.

## Setup and local workflow

Run from `C:\Project\ComputerNetworks` using the existing Python 3.13 environment:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -m pip install -c requirements-gnn-lock.txt -r backend/requirements-dev.txt
```

`backend/requirements.txt` pins application runtime additions;
`backend/requirements-dev.txt` adds TestClient dependencies. The frozen root
requirements and scientific lock are unchanged. Existing scientific package
versions were preserved. No global installation or multipart dependency is used.

Terminal 1, backend:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, frontend:

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks\frontend'
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Frontend: http://127.0.0.1:5173. Backend: http://127.0.0.1:8000.
API documentation: http://127.0.0.1:8000/docs. Run one backend worker locally.
Restart after changing server settings. Checkpoint initialization happens during
lifespan startup, never at module import. Health remains available after a failed
initialization; status reports `error` and the details are logged locally.

Optional environment variables (examples are declarations, not auto-loaded):

```powershell
# Authenticate metadata without initializing the model:
$env:GRAPHNIDS_LOAD_MODEL = 'false'
# Explicit origins; wildcard origins are rejected:
$env:GRAPHNIDS_CORS_ORIGINS = '["http://127.0.0.1:5173","http://localhost:5173"]'
```

For a different backend URL, set `VITE_API_BASE_URL` in the frontend process or
its local Vite environment. The fallback is `http://127.0.0.1:8000`.
The frontend checks health/status with a three-second timeout and polls every
ten seconds. Backend shutdown produces a graceful disconnected state.

## API and states

| GET route | Contract |
| --- | --- |
| `/api/v1/health` | Application liveness only: `ok`, `graphnids-backend` |
| `/api/v1/status` | Connection, adapter state, loaded flag and honest capabilities |
| `/api/v1/model` | Authenticated E2 HTGNN checkpoint, selection and graph metadata |
| `/api/v1/schema` | Canonical ordered 78-feature names; five-flow input contract |
| `/api/v1/capabilities` | Strict application CSV replay available; live capture, PCAP and arbitrary CICFlowMeter CSV unsupported |

Adapter states: `uninitialized`, `loading`, `ready`, `error`.
`ready` means the checkpoint and policy are authenticated and the frozen model
is loaded on CPU in evaluation mode with gradients disabled. The C2A adapter
loads lazily on the first valid analyze request and authenticates its own binding.
`inference_ready` is true once that adapter is initialized; it is cleared on an
inference failure or shutdown. A false value before the first analysis does not
mean that the replay endpoint is unsupported. `GRAPHNIDS_LOAD_MODEL=false`
disables analysis model loading while validation remains available. Model
metadata can remain available if model loading fails after metadata validation.
Missing/corrupted evidence instead makes `/model` return a typed 503 error.

All errors use `{"error":{"code":"...","message":"..."}}`. Codes include
`BACKEND_NOT_READY`, `MODEL_NOT_LOADED`, `SCHEMA_MISMATCH`, `INVALID_REQUEST`,
`CAPABILITY_NOT_IMPLEMENTED` and `INTERNAL_ERROR`. No traceback or local
checkpoint path is returned. Default CORS allows only the two explicit Vite
origins and GET/POST requests, with credentials disabled.

## Owner-authorized application checkpoint selection

Application V1 selects the largest **already recorded E.12A pruned-validation
eligible-class macro-F1**, considering exactly seeds 2025/2026/2027 under the
frozen shared policy. Exact score ties select the lowest numeric seed. No E.11,
E.12B or other test evidence enters this rule. Nothing is re-evaluated.

Exact evidence:
`results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/validation_sweep.json`,
`rows[2].seed_results`, ratio 0.25, full E2 validation with 311,916 examples per
seed, identical membership/order and the checkpoint-bound eligibility codes.

| Seed | Pruned-validation eligible-class macro-F1 |
| --- | ---: |
| 2025 | 0.8598611221399799 |
| **2026** | **0.8682839242491506** |
| 2027 | 0.8546017058225671 |

Selected checkpoint ID: `e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42`.
Physical SHA-256: `ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f`.
Exact authenticated local path:
`checkpoints/stage_e10/e10-official-20261001-v1/final_seed/e2/htgnn/e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42/best.pt`.
Learning rate 0.001 comes from E.10 validation-only LR selection at seed 2025.
Choosing seed 2026 for the application changes no research methodology/result.

Discovery authenticates the pinned E.10 report and E.12A freeze attestation,
canonical JSON hashes, all seven original E.12A artifacts and four frozen E.12A
source bindings. It matches the selected policy to the sweep row, all three E2
checkpoint identities/configurations, eligibility/membership and physical
checkpoint bytes. Checkpoints and original official logs are local-only;
the release archive excludes model weights. Missing dependencies/artifacts fail
closed. No arbitrary recursive checkpoint discovery or public path setting exists.

## Scientific reuse and approved adapter

`FrozenInferenceService` owns discovery, loading, status, metadata and bounded
numeric contract validation. The existing
`src.pruning.stage_e12.load_bound_model` loads the selected checkpoint using
the frozen E.7 model contract/tensor preflight, exact checkpoint metadata,
source bindings, optimizer-name provenance and strict model state loading.
It constructs/restores no optimizer. Historical optimizer-resume validation
remains the E.10 record; C1 makes no new optimizer-resume certification claim.
`stage_e12.FeaturePolicy` reconstructs and validates the attested shared mask.
No attention ranking or mask selection is executed.

Canonical feature names come from `src.preprocessing.schema.MODEL_FEATURE_COLUMNS`;
temporal/topology metadata comes from `src.graph.schema`; class count comes from
`src.preprocessing.labels`. No feature list, class mapping, graph builder,
preprocessor, split reader or ML implementation is duplicated.

The research contract uses 78 features per flow and five consecutive clean flow
records in its deterministic capture-order proxy. Application replay instead
preserves uploaded CSV row order, targeting the fifth/final flow. The mask retains 61
feature nodes, removes 17 and keeps all eight categories: 87 -> 70 total nodes,
86 -> 69 edges. Model parameters stay unchanged. Arbitrary 61-column input is
unsupported. Shape validation alone certifies no cleaning/preprocessing/provenance.

The frozen D4/D6/E.2 graph APIs and E.12 inference validators are bound to
scientific split membership/provenance. The legacy `predict_window` still rejects
application prediction even after successful loading; it never labels a user
window as a training/validation/test sample or bypasses those validators.
The owner-approved `external-application-graph-adapter-v1` supplies the separate
post-transform path. C2A now binds strict CSV parsing and frozen E2 transforms
to that unchanged adapter. It reuses E.12 `compact_inputs` and frozen model math,
with independent application provenance. No research guard is bypassed.

## C2A Accepted CSV Format

Use an **uncompressed UTF-8 CSV** (an optional UTF-8 BOM is accepted), with exactly
the 78 names in `src.preprocessing.schema.MODEL_FEATURE_COLUMNS`. Get the ordered
list from `GET /api/v1/schema`. All 78 names must appear exactly once. Header
names are case-sensitive and are not trimmed or silently renamed. Columns may
appear in any order; the backend reorders by name to canonical feature order
and reports `columns_reordered`.

Every row is one **flow record** of pre-transform numeric model features.
`Dst Port` is a valid canonical feature. Missing columns, unknown columns,
duplicate headers, `Timestamp`, `Label`, IPs, identifiers, noncanonical ports
and all extra metadata are rejected. Arbitrary numeric CSVs and 61-column
pruned inputs are unsupported. This is an application restriction; it does
not imply that the research corpus lacked Timestamp or Label.

**Application replay preserves uploaded CSV row order.** It does not sort,
infer capture chronology, deduplicate, recreate research corpus cleaning or
assign research partition membership. The adapter's inherited scientific
window descriptor is not a claim about uploaded chronology; its request-level
`replay_order_semantics` is explicitly `csv-row-order`.

Five consecutive flow records form each **non-overlapping** window. The fifth
and final flow is its prediction target. For N rows, there are `N // 5`
predictions and `N % 5` trailing flows excluded from inference, without padding.
API indices are zero-based: target flow indices are `4, 9, 14, ...`. These are
flow indices, not timestamps or packet numbers. Validation accepts 1-4 rows and
reports zero windows; analysis requires at least five rows. Empty files and
header-only files are rejected. Trailing rows must still pass numeric validation.

Cells accept signed decimal/exponent numeric notation and explicit IEEE `NaN`,
`inf`/`infinity`, `+inf`, `-inf` (case-insensitive). Surrounding value whitespace
is allowed. Blank cells, `N/A`, `unknown`, arbitrary text, formulas and decimal
values outside float32 range are rejected. Nothing is executed or coerced from
arbitrary text to NaN.

The backend authenticates the exact compact state at
`data/processed/preprocessed/exp2/scaler_state.json`, SHA-256
`9d9fc971b6a2f2d4bda3b31805a08742a26037a436f2c8130483769b800f7fe2`,
checks its canonical feature order, fields and E2 train-fit provenance, and
reconstructs it only with `FitTransformState.from_dict`. `apply_transforms`
replaces NaN with the train median, +inf with the train maximum finite value,
and -inf with the train minimum finite value, then applies the frozen robust
scaling. The mask-and-restore path preserves -1 only in `Init Fwd Win Byts` and
`Init Bwd Win Byts`; ordinary -1 elsewhere is transformed normally. No fitting,
refitting or upload-derived statistics occur. Transformed `(B,5,78)` float32
windows must be entirely finite.

Inference uses the authenticated seed-2026 E2 HTGNN checkpoint and frozen E.12A
policy through the approved adapter on CPU, with eval mode, disabled gradients
and `torch.inference_mode()`. Batches contain at most 256 windows. The policy
retains 61 feature nodes, eight categories and one sample: 70 nodes, 69 edges,
and 7,183 unchanged parameters. This batch size is an application bound, not a
benchmark result. There is no CUDA support, live capture or raw PCAP parsing.

Predictions use canonical `INT_TO_LABEL` and `argmax` over all 15 E2 logits.
`model_score = softmax(logits)[predicted_class]` is an **uncalibrated normalized
model score**, defined at the application presentation layer. It is not a
calibrated measure. Only exact canonical `Benign` gives `is_attack=false`.
No thresholds or severity levels are introduced.

### Upload transport and limits

Both POST routes receive the file's **raw bytes**, `Content-Type: text/csv`, and
a basename-only `X-Filename` header. They do not accept JSON paths or multipart
forms. This uses bounded streaming and `SpooledTemporaryFile` as the equivalent
spooled upload strategy; **no python-multipart or other dependency was added**.
Content-Length exceeding 10 MiB is rejected before reading the body. The streamed
byte count enforces that cap even when Content-Length is absent. Compressed
transport and archives are unsupported; filename extension alone cannot pass
the actual schema/numeric parser.

Limits: **10 MiB**, **25,000 flow rows**, **5,000 predictions**, **one active
analysis per process**. Run exactly one backend worker. A second analysis
immediately returns `409 ANALYSIS_BUSY`; no analysis queue is created. Validation
does not acquire that lock. Spooling uses at most 1 MiB in memory before spilling
into the OS temporary directory, with a total upload cap of 10 MiB. Temporary
storage must be outside the repository and closes on success, failure or upload
disconnect. No upload is retained. The raw float32 array holds at most 25,000
rows, while transformed arrays and graphs are processed batch by batch.

PowerShell (the local path is used by the client to send bytes; the server never
receives that filesystem path):

```powershell
$csv = 'C:\ApplicationInputs\synthetic-features.csv'
$headers = @{ 'X-Filename' = 'synthetic-features.csv' }
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/replay/validate' `
  -ContentType 'text/csv' -Headers $headers -InFile $csv
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/replay/analyze' `
  -ContentType 'text/csv' -Headers $headers -InFile $csv
```

### Responses, provenance and errors

`/validate` returns a validation ID, filename, source byte count/SHA-256, row and
window counts, trailing count, feature count, reordered flag, schema validity,
T=5, replay order, authenticated transform hash and
`model_inference_performed=false`. The request does not load/run a model or
transform rows. The separate C1 server startup may already have loaded its model.

`/analyze` returns source identity, analysis ID, counts, class histogram, compact
predictions, model/preprocessing/graph summaries and authenticated provenance.
Each prediction contains window index, target flow index, canonical class ID and
label, model score and attack boolean. No raw rows, logits vectors, transformed
arrays, graphs, attention, embeddings, preprocessing statistics or scientific
performance metrics are returned.

The unchanged adapter binds the upload hash as `source_id=sha256:<file hash>`
and an API-version-prefixed analysis ID to each batch. Its immutable provenance
binds feature order/schema, transform identity, T=5, explicit CSV replay order,
taxonomy/graph schema, frozen policy, checkpoint, seed and class mapping. The
response makes those identities and `api_version=v1` explicit in a typed
`source_kind=external_application` provenance record, accompanied by its
canonical SHA-256. `model_task=E2` is a model identity; no upload is assigned an
experiment or research split. Analysis IDs vary between calls; identical CSVs
produce identical prediction fields under unchanged artifacts and batching.

Errors retain the C1 `{error:{code,message}}` envelope. C2A uses
`INVALID_FILE_TYPE` (415), `FILE_TOO_LARGE`/`TOO_MANY_ROWS` (413),
`ANALYSIS_BUSY` (409), `EMPTY_FILE`, `CSV_PARSE_ERROR`, `DUPLICATE_COLUMN`,
`UNEXPECTED_COLUMN`, `MISSING_COLUMN`, `NON_NUMERIC_FEATURE`,
`INSUFFICIENT_FLOWS`, `TRANSFORM_FAILED` (422), and
`TRANSFORM_STATE_INVALID`, `MODEL_NOT_READY`, `INFERENCE_FAILED` (503).
Transport errors use `INVALID_REQUEST` (400). Messages contain no stack traces
or machine-local artifact paths. An inference failure returns no partial results.

The status client continues health/status polling. The C2B frontend integrates
`POST /api/v1/replay/validate` and `POST /api/v1/replay/analyze` for validation,
analysis and session-only results, with C2B.1 UI polish complete. There is no
persistence or server-side progress percentage; demo mode remains synthetic
frontend behavior.

## Application-safe verification

```powershell
Set-Location -LiteralPath 'C:\Project\ComputerNetworks'
.\.venv\Scripts\python.exe -B -m backend.tests.run_safe
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe --replay
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe --regressions
# Local real-loader/server smoke with the same guard, no forward/evaluation:
.\.venv\Scripts\python.exe -B -m backend.tests.serve_safe
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/status
Invoke-RestMethod http://127.0.0.1:8000/api/v1/model
Invoke-RestMethod http://127.0.0.1:8000/api/v1/schema
Invoke-RestMethod http://127.0.0.1:8000/api/v1/capabilities
Set-Location -LiteralPath 'C:\Project\ComputerNetworks\frontend'
npm run test:api
npm run build
npm run lint
npm run typecheck
```

The Python guard rejects official dataset access, E.11/E.12B evidence access,
scientific writes, optimizer creation and backward passes. It is an accidental
Python-access guard, not a sandbox against arbitrary native code. The full
scientific suite includes real-data/training checks and must not run under this
application-only authorization. No commit, stage, push or tag is authorized.

Only `--replay` permits reading the exact E2 scaler JSON; other files under
`data/` remain blocked, as do all writes there. The application runner also blocks
transform fitting, ranking, evaluation and benchmarking. The real endpoint smoke
uses TestClient and small newly created synthetic CSVs; it is implementation
validation, not scientific evaluation. The older `serve_safe` command retains
its C1-only guard and must not be used for replay smoke testing.
