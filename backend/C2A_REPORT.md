# C2A bounded CSV replay completion — 2026-10-07

C2A implementation validation passes. Both replay endpoints execute the strict
application CSV contract; real frozen seed-2026 HTGNN inference was validated
only with small synthetic inputs. C2B was not started. Nothing was staged,
committed, pushed, tagged, stashed, reset or cleaned.

## Files

Created:

- `backend/app/api/replay.py`: raw-body upload endpoints, early byte checks,
  bounded request-scoped spooling and immediate busy rejection.
- `backend/app/schemas/replay.py`: typed validation, prediction, summary and
  application-provenance responses.
- `backend/app/services/csv_replay.py`: strict CSV parser, canonical column
  reordering, frozen-state authentication and authoritative transform reuse.
- `backend/app/services/replay.py`: bounded inference orchestration and scores.
- `backend/tests/test_replay.py`: 75 synthetic contract/adversarial checks.
- `backend/C2A_REPORT.md`: this report.

Modified:

- `backend/app/main.py`: register replay service/routes, close service at shutdown,
  permit POST and X-Filename for the existing explicit CORS origins.
- `backend/app/api/routes.py`: report lazy replay-adapter readiness.
- `backend/app/schemas/contracts.py`: extend typed errors and honest capabilities.
- `backend/app/services/inference.py`: distinguish application row-order semantics
  from research chronology in the schema response.
- `backend/tests/run_application_safe.py`: additive `--replay` mode with the exact
  E2 scaler-read exception and explicit transform-fitting prohibition.
- `backend/tests/safety.py`: exact read-exception support; block all data writes.
- `backend/tests/test_foundation.py`: update capability/CORS assertions for C2A.
- `backend/README.md`: accepted format, transport, responses and commands.
- `frontend/src/lib/api/client.ts`: accept boolean replay/readiness flags.
- `frontend/tests/api-client.test.mjs`: regression for those flags and malformed
  values, while retaining health/status-only polling.

The frontend correction is necessary because its original literal-false validator
would classify the required C2A capability response as malformed/disconnected.
No upload client, Analyze Traffic integration, visual change or results UI was
added. Of the seven approved adapter files, only the safe runner received the
small additive integration change above. The four ML implementation files,
adapter tests and adapter report remain byte-identical.

## Accepted input and temporal behavior

The sole feature authority is
`src.preprocessing.schema.MODEL_FEATURE_COLUMNS`. Exactly those 78 case-sensitive
header names must appear once each. Headers are inspected with strict
`csv.reader` before any renaming or coercion. A shuffled canonical header is
accepted and reordered by name, with a reported flag. Missing, unknown or
duplicate columns fail. Timestamp, Label, IP addresses, identifiers,
noncanonical ports and other metadata fail; canonical `Dst Port` is accepted.
Arbitrary numeric CSV, 61-column input, PCAP and arbitrary CICFlowMeter output
are unsupported.

The input is uncompressed UTF-8 CSV, optionally with a BOM. Each row is a flow
record. Cells accept decimal/exponent numeric text and explicit case-insensitive
IEEE NaN/inf/infinity with optional signs. Value whitespace is stripped. Blank,
N/A, arbitrary text, formulas and finite decimal values outside float32 range
fail; malformed quoting and encoding failures are typed errors. No contents
are executed or silently converted from arbitrary text to NaN.

**Application replay preserves uploaded CSV row order.** No sorting, chronology
inference, deduplication or research corpus cleaning is recreated. Windows are
five consecutive rows, non-overlapping, with the fifth/final flow as target.
N rows yield `floor(N/5)` windows and `N%5` trailing rows. Trailing rows pass
input validation but are excluded from inference without padding. Target flow
indices are zero-based `4,9,14,...`. Validation permits 1-4 rows and reports
zero windows; analysis requires five. Empty/header-only files are rejected.

## Transform, graph, checkpoint and output identities

The only data-area artifact read is the compact frozen state:

`data/processed/preprocessed/exp2/scaler_state.json`

Physical SHA-256:
`9d9fc971b6a2f2d4bda3b31805a08742a26037a436f2c8130483769b800f7fe2`.

Every accepted request authenticates its bytes and checks expected fields,
canonical feature order, sentinel metadata, finite statistics, positive scales
and E2 train-fit provenance. Reconstruction uses `FitTransformState.from_dict`.
Only analyze calls `apply_transforms`; nothing fits or updates statistics.
NaN uses the train median, +inf the train maximum finite value, and -inf the
train minimum finite value, followed by frozen robust scaling. The authoritative
mask-and-restore path preserves -1 in `Init Fwd Win Byts` and `Init Bwd Win Byts`
only. Ordinary -1 elsewhere follows ordinary scaling.

Raw storage is bounded float32 in canonical feature order. Each inference batch
is `(B,5,78)`, transformed through the frozen API and checked for finite float32
output. The unchanged `external-application-graph-adapter-v1` then constructs
the graph and reuses frozen E.12 `compact_inputs` and the validated model path.

Selected model: E2 HTGNN, deployment seed **2026**.
Checkpoint ID: `e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42`.
Physical SHA-256:
`ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f`.

Frozen E.12A policy canonical SHA-256:
`f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa`.
Its physical artifact SHA-256 remains
`28727d3d7bfe94f0d624639859d5a3c965dbda1ccbbab760a2ec7bc091f91317`.
The existing approved adapter authenticates these artifacts through its
unchanged discovery/loader. No new seed comparison or checkpoint selection
procedure was implemented or executed separately from that approved loader.
No official data or test result informed application behavior.

The policy retains 61 of 78 input feature nodes and all eight categories plus
one sample: **70 nodes, 69 edges, 7,183 unchanged parameters**. Execution is
CPU-only, with model eval state and absent/disabled gradients checked by the
adapter, inside `torch.inference_mode()`. The maximum batch is **256 windows**;
there is no full-upload graph collection, CUDA change or benchmark.

Canonical classes come from `src.preprocessing.labels.INT_TO_LABEL`.
Scientific inference uses argmax but exposes no reusable normalized class-score
extraction helper. The application therefore defines
`softmax(logits)[argmax(logits)]` as an **uncalibrated normalized model score**.
Only exact canonical Benign gives `is_attack=false`; every other class gives
true. No thresholds, severity levels or calibration were added.

## API, bounds and provenance

- `POST /api/v1/replay/validate`: source identity/hash/size, validation ID,
  row/window/trailing counts, canonical feature count, reorder flag, schema
  validity, T=5, CSV replay order, transform hash and
  `model_inference_performed=false`. This request loads/runs no model and returns
  no transformed values. C1 lifespan initialization is separate from the request.
- `POST /api/v1/replay/analyze`: source/analysis identity, counts, canonical class
  histogram, compact predictions, model/preprocessing/graph summary and typed
  authenticated application provenance plus its canonical SHA-256.
- Each prediction contains window index, target flow index, class ID, canonical
  label, model score and attack boolean. Maximum response: 5,000 predictions.
  No uploaded rows, raw logits vectors, tensors, graphs, attention, embeddings,
  preprocessing statistics or E.11/E.12B scientific metrics are returned.

Transport is raw CSV bytes with `Content-Type: text/csv` and a basename-only
`X-Filename` header. No request filesystem path or multipart form is accepted.
This is the permitted equivalent bounded/spooled upload strategy. Content-Length
over **10 MiB** fails before body reading; streamed byte counting enforces the
same maximum when no length is supplied. **25,000 rows** is the row limit.
The spool spills after 1 MiB into OS temporary storage outside the repository,
is request-scoped, and closes on success, failure or disconnect. Uploads are
never retained in the repository. No dependency or lockfile was changed;
python-multipart is unnecessary for this transport.

One process-wide nonblocking lock allows **one active analysis**; another gets
`409 ANALYSIS_BUSY` immediately. Validation is independent. Run one server
worker; multi-process/global scheduling is outside C2A. Processing runs in a
non-abandoning worker so the lock and spool outlive its work on cancellation.

The adapter binds `source_kind=external_application`, the upload hash in
`source_id`, API-version-prefixed analysis ID, CSV replay order, feature
schema/order, transform identity, T=5, graph taxonomy/schema, frozen policy,
checkpoint, seed, E2 model task and canonical class mapping. The response exposes
those bindings and `api_version=v1` explicitly in a typed provenance record and
hash. No fake experiment or split is assigned. Different analysis IDs are
expected on repeat requests; the prediction fields repeat exactly.

Errors extend the C1 envelope with INVALID_FILE_TYPE, FILE_TOO_LARGE, EMPTY_FILE,
CSV_PARSE_ERROR, DUPLICATE_COLUMN, UNEXPECTED_COLUMN, MISSING_COLUMN,
NON_NUMERIC_FEATURE, TOO_MANY_ROWS, INSUFFICIENT_FLOWS,
TRANSFORM_STATE_INVALID, TRANSFORM_FAILED, MODEL_NOT_READY, ANALYSIS_BUSY and
INFERENCE_FAILED. Existing SCHEMA_MISMATCH/INVALID_REQUEST remain available.
All public messages are sanitized; failures return no partial predictions.

Capabilities now report CSV replay and strict 78-feature input available;
live capture, PCAP, arbitrary CICFlowMeter CSV and frontend replay integration
remain false. C1 model loading remains intact. The separate C2A adapter loads
lazily after valid parsing and the first finite transformed batch. Readiness
becomes true when that adapter exists and clears on inference failure/shutdown.

## Verification

| Check | Final result |
| --- | --- |
| `python -B -m backend.tests.run_application_safe --replay` | **75 passed**, no failures/skips, 0 unplanned guard violations |
| Approved external-adapter regression runner | **143 passed**, no failures/skips, 0 guard violations |
| Application-safe frozen regression preset | **257 passed**, 78 intentionally deselected, 0 guard violations |
| Preset composition | 219 frozen synthetic checks + 38 C1 checks |
| Separate `python -B -m backend.tests.run_safe` | **38 passed**, 0 guard violations |
| Real frozen endpoint smoke | PASS on small synthetic CSVs, including lazy model initialization |
| Direct adapter versus endpoint | Exact classes and logits-derived scores; same frozen transform and graph path |
| Deterministic repeat | Exact prediction fields for identical CSVs |
| Shuffled header and trailing-row invariance | Exact prediction agreement |
| Model mapping/batch boundary | All 15 classes and 256+1 batches tested with explicitly stubbed logits |
| Upload bounds | Actual 25,000-row acceptance; 25,001-row rejection; 10 MiB overflow rejected with/without Content-Length |
| Concurrency and cleanup | Busy rejection while active; validation independent; spool closure on success/failures; disconnect releases lock |
| Guard adversarial self-test | 7 intentional denials in isolated child process, before forbidden read/write |
| Frontend client tests | **12 passed** |
| Frontend build / lint / TypeScript | PASS / PASS / PASS |
| Python AST / scoped whitespace / `git diff --check` | PASS |

Tests additionally cover exact/missing/unknown/Timestamp/Label/duplicate headers,
malformed CSV/encoding, empty and short inputs, strict numeric conversion,
NaN/+inf/-inf replacements, both sentinels and ordinary -1, state byte and
semantic authentication, source hash, no model access on validate, canonical
provenance without split/experiment, sanitized model errors, count/index
semantics and continued C1 health/status behavior.

The guard permits only the exact E2 scaler file in replay mode, denies official
X/y arrays and raw CSVs, blocks data/scientific/protected-artifact writes, and
disables fitting, optimizers, backward/autograd, ranking, evaluation, policy
selection and benchmarking. Its seven expected negative probes run in an
isolated subprocess and are distinct from the **zero unplanned violations** in
each successful runner. No official dataset sample was accessed. The guard is
an accidental Python-access boundary, not a native-code security sandbox.

No full scientific suite was run because it includes prohibited real-data and
training/evaluation tests. No native browser/visual QA or native socket endpoint
smoke is claimed; endpoint evidence uses FastAPI TestClient and actual frozen
inference. Existing Torch/PyG/Starlette deprecation warnings, frontend chunk-size
advisory and upstream registry lint warnings remain. An initial float-overflow
comparison warning in a negative test was corrected using a Python-float bound;
the final C2A run has no new warning from that path.

## Preservation, Git and next boundary

Baseline preservation: **405 of 415 pre-existing tracked/application files**
remain byte-identical; the other **10** are exactly the intended modifications
listed above. All frozen tracked source, scientific evidence, lockfiles and
release files are unchanged. The approved handoff's SHA-256 remains
`b4d30d66785121ce4b27db5023bf124125225e53b559cee3207ba3a77ea557e1`.

All **20 protected E.10 root entries** and **58 file/directory metadata records**
(including 46 files) retain inventory, size, mtime and attributes. Their recovery
payloads were not read for new hashes. This establishes metadata preservation,
supported by write guards; it is not a new cryptographic audit of those archives.

Branch remains `app-demo`; HEAD remains
`2854d153d65c93689875884f3ad45f0b3b468ec2`. The staged diff is empty. Tracked diff
still contains only the approved pre-existing handoff reconciliation:
**90 insertions, 24 deletions**. Application edits remain inside already-untracked
backend/frontend directories and therefore require the separate file review
above. Final `git status --short` matches the starting status:

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

No official train/validation/test data, training, scientific evaluation, new
attention ranking, pruning selection or CPU benchmark was performed. No frozen
src change, handoff edit or protected E.10 artifact mutation occurred.

Remaining limits: local single-worker CPU execution; strict model-feature CSV
only; uncalibrated model scores; no chronology/real-world input suitability or
scientific accuracy claim; no live capture, PCAP, CUDA or frontend results UI.

The unchanged handoff records owner approval of the external-window contract as
its next boundary. The current explicit request supplies that approval and
authorizes C2A without changing the handoff. After owner review of this C2A
implementation, the exact next application boundary is **separately authorized
C2B frontend integration of the validate/analyze endpoints, typed errors,
bounded prediction responses and honest model-score/replay semantics**.
That integration must preserve the same frozen model/transform/policy and
external-application provenance. C2B is not authorized or started here.
