# GraphNIDS application — C2B CSV replay

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

1. Browse for or drop one uncompressed `.csv` file, at most 10 MiB.
2. Select **Validate file**. The client sends raw CSV bytes with `Content-Type:
   text/csv` and `X-Filename` to `/api/v1/replay/validate`.
3. Review backend row/feature/window/trailing counts and file identity. Validation
   does not run inference. Only the backend validates the scientific schema.
4. Select **Analyze traffic** to send the same file to `/api/v1/replay/analyze`.
   The display is indeterminate: no server progress phases or percentages exist.
5. Inspect the returned summary, all-class distribution and paginated predictions.
   Dashboard and Intrusion Alerts use the same latest session result.

Exactly 78 canonical model features are required. Label, Timestamp, extra
metadata, arbitrary CICFlowMeter CSV, PCAP and 61-feature input are unsupported.
Canonical columns can be reordered by name by the backend. CSV row order is
preserved: each non-overlapping group of five flow records yields one prediction
for its fifth/final flow. Incomplete trailing rows are not inferred. UI indices
are explicitly zero-based, matching C2A.

E2 HTGNN seed 2026 uses the frozen attention-pruned graph: 61/78 feature nodes,
eight categories, 70 total nodes, 69 edges and 7,183 unchanged parameters.
`Model score` is an uncalibrated normalized score, not a performance measure.
No accuracy/F1/precision/recall is computed for uploads. Real intrusion details
contain only returned predictions and source/model/graph identities; there are
no invented timestamps, endpoints, protocol, severity or response actions.

## State, requests and demo

`ReplayProvider` owns one `ReplaySession` through React context and
`useSyncExternalStore`. This is memory-only state, retained across navigation
and cleared by a page refresh or file replacement. No database or browser
storage is used. Each replacement cancels the request and invalidates late
responses. Duplicate submissions are ignored and Analyze requires validation
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
intact. Offline file selection and explicit demo activation continue to work.

Selecting a real file resets demo state. Switching to demo explicitly clears the
real result. Demo values remain labeled DEMO / SIMULATED / DEMO EVENT, with the
existing pause/resume/stop/reset controls. Real results take precedence in the
result views. Network Monitor always says live capture is not connected;
Model Performance remains an evidence-integration placeholder.

## Verification and visual fixtures

```powershell
npm test
npm run test:api
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
in the production build. Use it for drawer, pagination and processing-state QA.

See [C2B_REPORT.md](C2B_REPORT.md) for the acceptance report and screenshot index.
Earlier phase reports retain their historical scope. The original application
screenshots and vendor UI sources are unchanged.

## Owner clarification and boundary

The owner's C2B clarification supersedes only the stale application-boundary
wording in PROJECT_HANDOFF.md. Its approved bytes are untouched. No C2A/backend,
frozen scientific source, model, methodology, release or protected E.10 artifact
was changed. No official dataset sample was accessed. Nothing was staged,
committed, pushed or tagged. The next boundary is owner visual review of C2B;
further implementation and release work require separate authorization.
