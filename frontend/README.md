# GraphNIDS application — C3B exact-layout CSV preparation

React / Vite / TypeScript console connected to the approved C2A bounded CSV
replay API. Scientific implementation and release evidence remain frozen.

## Run locally

From the repository root, start the application backend using the isolated
`.venv` and the instructions in [backend/README.md](../backend/README.md).
For synthetic-only browser QA with the existing application guards:

```powershell
.\.venv\Scripts\python.exe -B -m frontend.tests.serve_c2b
```

From `frontend/`:

```powershell
npm ci
npm run dev
```

`VITE_API_BASE_URL` defaults to `http://127.0.0.1:8000`. Vite normally serves
`http://127.0.0.1:5173`. Use one backend worker. Production hosting needs an SPA
fallback. Dependencies are local and locked; no C2B dependency was added.

## Real replay workflow

1. Browse for or drop one uncompressed `.csv` file, at most 10 MiB. A dedicated
   browser worker inspects its strict UTF-8 snapshot without uploading it.
2. Canonical 78-feature files, including shuffled columns, use **Validate file**.
   Exact ordered standard 80-column and extended 84-column files first require
   **Prepare locally**. Review **Prepared locally — backend validation required**,
   then select **Validate prepared file**.
3. The client sends canonical CSV bytes with `Content-Type: text/csv` and a
   basename-only `X-Filename` to the unchanged `/api/v1/replay/validate`. Review
   backend row/feature/window/trailing counts and file identity. Validation
   does not run inference. Only the backend validates the scientific schema.
4. Select **Analyze traffic** to send the same file to `/api/v1/replay/analyze`.
   The display is indeterminate: no server progress phases or percentages exist.
5. Inspect the returned summary, all-class distribution and paginated predictions.
   Dashboard and Intrusion Alerts use the same latest session result.

The backend still requires exactly 78 canonical model features. Canonical
columns can be reordered by name by the backend. Browser preparation accepts
only exact ordered `CSV_COLUMNS` (80 columns) or `CSV_COLUMNS_EXTENDED` (84)
from `src.preprocessing.schema`. Standard preparation removes only `Timestamp`
and `Label`; extended preparation removes only `Flow ID`, `Src IP`, `Src Port`,
`Dst IP`, `Timestamp` and `Label`. Removed values stay local. Only the generated
`graphnids-prepared.csv` is uploaded, never the original 80/84 source or its
filename. Retained model features remain in the upload, so this is not complete
anonymization. No metadata values are included in preparation errors or provenance.

Reordered 80/84 headers, aliases, arbitrary CICFlowMeter versions, Zeek,
NetFlow/IPFIX, Suricata, raw PCAP and arbitrary 61-feature inputs are unsupported.
Header matching does not authenticate exporter identity/version, units or
measurement equivalence. Browser preparation performs no sorting, filtering,
numeric reformatting, imputation, normalization, feature engineering or model
computation. CSV row order is
preserved: each non-overlapping group of five flow records yields one prediction
for its fifth/final flow. Incomplete trailing rows are not inferred. UI indices
are explicitly zero-based, matching C2A.

Source and prepared output each have an independent **10 MiB** limit. Preparation
preserves all records, including incomplete tails, and accepts at most **25,000**
records (**5,000** complete windows). The existing inference batch remains **256**
windows. Records are processed one at a time without a full JavaScript table;
source bytes, decoded text, serialization, output and hash buffers still require
more memory than the file-size limit. No browser latency or memory guarantee is
made. Retained features obey the descriptor's current C2A CSV field-size limit;
the verified Python 3.13 environment uses **131,072 decoded characters**.
Larger discarded metadata can be accepted within the source byte limit.

The worker preserves decoded feature text and matches C3A quoted-field UTF-8/LF
output. Python numeric grammar, float32 bound and `str.strip` code points come
from the generated manifest, rather than JavaScript `parseFloat` or `trim`.
Before each prepared-file validation or analysis, the client fetches
`/api/v1/schema` and compares the complete `csv_compatibility` descriptor and its
identity with the bundled manifest. Missing, malformed or mismatched descriptors
block upload; unknown versions are not accepted. Canonical uploads retain their
existing backend flow without this descriptor check.

The manifest is generated from backend/Python authority. From the repository root:

```powershell
.\.venv\Scripts\python.exe -B frontend/scripts/generate-csv-schema.py
.\.venv\Scripts\python.exe -B frontend/scripts/generate-csv-schema.py --check
```

Check mode never rewrites the target. Descriptor identity includes the current
process `csv.field_size_limit()` and Python whitespace behavior; differing server
environments fail closed. The descriptor getter does not change global parser
settings. See [C3B_REPORT.md](C3B_REPORT.md) for parity evidence and limits.

E2 HTGNN seed 2026 uses the frozen attention-pruned graph: 61/78 feature nodes,
eight categories, 70 total nodes, 69 edges and 7,183 unchanged parameters.
`Model score` is an uncalibrated normalized score, not a performance measure.
No accuracy/F1/precision/recall is computed for uploads. Real intrusion details
contain only returned predictions and source/model/graph identities; there are
no invented timestamps, endpoints, protocol, severity or response actions.

## State, requests and demo

`ReplayProvider` owns one `ReplaySession` through React context and
`useSyncExternalStore`. This is memory-only state, retained across navigation
and cleared by a page refresh or file replacement/removal. Original source,
prepared upload, local preparation record, backend validation and analysis are
separate state. No database, localStorage or IndexedDB is used. Replacement,
removal and provider disposal cancel active worker/request work and invalidate
late responses. Leaving Analyze Traffic cancels pending inspection, preparation,
descriptor checks and uploads; completed state and results remain available on
return. Duplicate submissions are ignored and Analyze requires validation
with at least one complete window. Prediction tables show 50 rows per page;
Dashboard shows at most the latest 50 intrusion predictions in CSV order.

Validation has a 30-second timeout; CPU analysis has a separate 10-minute timeout.
Cancel, timeout, replacement and unmount abort the client request and release
timers/listeners. Server work may continue after client cancellation; the UI
says so. `ANALYSIS_BUSY` is a clear retryable message, without discarding the
validated file. Other backend codes have concise safe messages; raw exception
text is never displayed. Runtime decoders reject malformed/inconsistent results.

Health/status polling retains C1.5's three-second timeout and 10-second interval,
with 10/20/40/60-second offline backoff. Its sibling-request cancellation remains
intact. Offline file selection, inspection, explicit local preparation and demo
activation continue to work; validation and analysis require the backend.

Preparation provenance contains source/output SHA-256 and sizes, profile,
descriptor identity/version, removed field names and record/window/tail counts.
It contains no removed values or absolute paths and is not persisted/downloaded.
The original-source hash is calculated locally, not backend-authenticated.
Existing backend responses authenticate the prepared filename, byte size, digest
and flow count; mismatches reject the result. C3A sidecars are not consumed.

Selecting a real file resets demo state. Switching to demo explicitly clears the
real result. Demo values remain labeled DEMO / SIMULATED / DEMO EVENT, with the
existing pause/resume/stop/reset controls. Real results take precedence in the
result views. Network Monitor always says live capture is not connected.
Model Performance displays the frozen scientific evidence described below.

## Frozen Model Performance evidence

Model Performance presents the released E.11/E.12B E2 eligible-class test
macro-F1 comparison, three-seed sample SD and individual seed results, graph
counts, validation-only RTX 4050 forward/allocator observations, visible E1/E3
limitations and exact repository-relative sources. It remains available offline.
Uploaded CSV predictions, demo activity, mock fixtures and backend status cannot
update these historical measurements. The full 78-feature CSV remains required.

`src/lib/model-performance-evidence.ts` holds only a compact immutable display
snapshot. `project-facts.ts` remains the graph presentation authority. There is
no new API, scientific evaluator, dependency or model behavior. Focused tests
compare values and scopes with the compact reports and verify every displayed
source against its Git blob at the frozen scientific release:

```powershell
node --test tests/model-performance.test.mjs
```

See [MODEL_PERFORMANCE_REPORT.md](MODEL_PERFORMANCE_REPORT.md) for metrics,
validation, browser QA, preservation and the owner visual-review boundary.

## Verification and visual fixtures

```powershell
npm test
npm run test:api
npm run test:csv
npm run schema:check
npm run build
npm run lint
npm run typecheck
```

The tests use Node 24's TypeScript support plus the installed TypeScript compiler
for test-only TSX rendering. No new test framework or runtime dependency is needed.
`tests/synthetic-wire-response.json` was captured from actual C2A endpoints using
12 all-zero synthetic rows. It contains no official dataset sample. Unit tests
also create explicitly mocked intrusion responses.

`/tests/ui-fixture.html` is a Vite development-only visual fixture with a permanent
MOCKED UI VALIDATION banner. Its attack predictions and delayed response are
mocked, never model output. It is not imported by the production app or included
in the production build. Its canonical synthetic CSV derives names from the
generated manifest and waits for actual worker inspection before mocked
validation. Use it for drawer, pagination and processing-state QA.

See [C2B_REPORT.md](C2B_REPORT.md) for the acceptance report and screenshot index.
Earlier phase reports retain their historical scope. The original application
screenshots and vendor UI sources are unchanged.

## Historical C2B owner clarification

The owner's C2B clarification supersedes only the stale application-boundary
wording in PROJECT_HANDOFF.md. Its approved bytes are untouched. No C2A/backend,
frozen scientific source, model, methodology, release or protected E.10 artifact
was changed. No official dataset sample was accessed. Nothing was staged,
committed, pushed or tagged. The next boundary is owner visual review of C2B;
further implementation and release work require separate authorization.

## C3B owner authorization and boundary

The owner separately authorized browser CSV preparation from application
checkpoint `06145e9af61d57e48a282d5c99c08c1314f8a06a`, and specifically authorized
the development-only `tests/ui-fixture.tsx` correction after its async-inspection
regression was found. No runtime dependency or lockfile change was needed.
Scientific evidence, Model Performance, C2A replay and C3A conversion stay frozen.
C3B implementation is awaiting independent owner audit; nothing is staged,
committed, pushed or tagged. No later application stage or release is authorized.
