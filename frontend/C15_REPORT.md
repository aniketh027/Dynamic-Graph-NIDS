# C1.5 browser runtime / offline-state correction

## Root cause and evidence boundary

The historical C1 in-app navigation failure (`ERR_NO_BUFFER_SPACE`) did **not**
reproduce. Before any edits, a fresh in-app browser tab loaded and reloaded the
unchanged frontend with FastAPI stopped. It rendered the correct disconnected
state without a captured uncaught console error. Connected operation also worked
before edits. The exact cause of that historical browser error remains
**unconfirmed**; this report does not claim that a React exception, CORS defect,
model/status mismatch or the correction below caused it.

A separate, reproducible client resource-lifetime defect was found: health and
status were requested with `Promise.all`. On an early fetch or JSON parse failure,
the poll returned while its sibling operation could remain pending. The provider
then cleared the deadline timer, leaving that sibling without cancellation.
A synthetic regression reproduced the uncancelled request before correction.
This establishes a client defect, not a causal link to the historical browser
navigation error. No attempt was made to exhaust OS/browser resources.

The prior client also erased the distinction between network, deadline, HTTP and
malformed-response failures. These diagnostics are now typed and sanitized.

## Narrow correction and files

Four existing frontend files were modified:

- `src/lib/api/client.ts`: each poll owns a cancellation controller, relays caller
  cancellation, removes the relay and cancels outstanding fetch/body operations
  on every exit. Failure reasons distinguish network, timeout, non-2xx HTTP,
  malformed response, cancellation and URL configuration. Connection loss while
  reading JSON remains a network failure; a syntax/schema error is malformed.
- `src/components/backend-provider.tsx`: the three-second deadline has an explicit
  timeout reason. Requests remain sequential between polls; offline retry delays
  are 10, 20, 40 and then at most 60 seconds. Recovery resets the delay to ten
  seconds. Cleanup cancels the active poll and suppresses stale state updates.
- `tests/api-client.test.mjs`: regressions for outstanding siblings, typed failure
  reasons, network loss during body parsing, timeout versus cleanup cancellation,
  success/disconnect/recovery and bounded retry reset.
- `README.md`: documents the corrected lifecycle and offline retry behavior.

This report is new. Ignored browser QA output is under `qa/`. No backend source,
dependencies, lockfiles, scientific files or original C1 report was edited.
Product layout/copy and existing demo implementation were preserved. No generic
error toast or console retry logger was added. Typed backend errors are unchanged.

## Native browser validation

The existing in-app Chromium browser was used through its supported automation
API. Local Vite ran at `http://127.0.0.1:5173`; the guarded C1 FastAPI server ran
at `http://127.0.0.1:8000`. Server startup and HTTP logs confirmed successful
health/status requests. Detailed console inspection used the browser's captured
warning/error log API; those collections were empty throughout the checks.

| Scenario | Observed result |
| --- | --- |
| Backend online | Backend connected; actual model adapter ready; replay pending; Analyze disabled |
| Backend stopped before loading | Application loads; backend disconnected; model adapter unavailable |
| Backend stopped after frontend loaded | Automatic transition to disconnected/unavailable; no reload required |
| Backend restarted | Automatic recovery to connected/adapter ready; no reload required |
| Offline demo | Synthetic chart/counter/events work; start, pause, resume, stop and reset pass |
| Browser console | No captured warning/error or uncaught React/fetch rejection |
| Error toasts | No offline error-toast loop; only expected user-triggered demo notices |

All six routes were navigated **and reloaded** with FastAPI stopped on final code:

| Route | Cold offline load |
| --- | --- |
| `/` | PASS: Dynamic Graph NIDS |
| `/network-monitor` | PASS: Network Monitor |
| `/analyze-traffic` | PASS: Analyze Traffic |
| `/intrusion-alerts` | PASS: Intrusion Alerts |
| `/graph-optimization` | PASS: Graph Optimization |
| `/model-performance` | PASS: Model Performance |

Each route had the expected heading, disconnected footer and unavailable adapter.
No blank screen, infinite checking state, uncaught error or offline error toast
was observed. Browser evidence is `qa/c15-offline-routes.json`,
`qa/c15-recovered.png`, `qa/c15-disconnected.png` and `qa/c15-offline-demo.png`.
No responsive-matrix claim is made for this narrow correction.

## Tests and checks

| Check | Result |
| --- | --- |
| `python -B -m backend.tests.run_safe` | 38 passed; 0 failed; 0 skipped; 0 guard violations |
| `npm run test:api` | 11 passed; 0 failed |
| `npm run build` | PASS |
| `npm run lint` | PASS; 44 pre-existing registry warnings; no warnings in corrected files |
| `npm run typecheck` | PASS |
| Tracked `git diff --check` | PASS |
| Separate whitespace review of untracked correction files | PASS |

The outstanding-sibling regression first failed with `released=false`, then passed
after ownership/cancellation was added. A body-read connection-loss classification
regression also failed before its correction and passed afterward. Tests use
synthetic responses and cancellation signals, with no inference or dataset access.
The backend tests use metadata/discovery checks and stub model loaders. The native
server loaded the unchanged checkpoint but ran no model forward/prediction.

Existing warnings remain: the frontend chunk size advisory, registry lint warnings,
Starlette's httpx TestClient deprecation and a Torch import deprecation. None was
suppressed. The scientific suite was not run because it is outside this
frontend/runtime-only authorization.

## Preservation, Git and next boundary

Branch remains `app-demo`; HEAD remains
`2854d153d65c93689875884f3ad45f0b3b468ec2`.
Tracked and staged diffs are empty because the application directories remain
untracked. The scoped C1-to-C1.5 comparison contains exactly the four frontend
edits above plus this report. `backend/`, `frontend/` and all 20 original protected
root entries remain untracked. No stage, commit, push, tag, stash, reset or clean
occurred.

SHA-256 comparison verifies 331/331 baseline files identical: 285 tracked files
and 46 protected local E.10 files. Every pre-existing backend file is unchanged.
Frozen ML architecture, checkpoint selection, shared pruning mask, preprocessing,
graph construction, evidence, results and `PROJECT_HANDOFF.md` were untouched.
Application selection remains seed 2026, checkpoint
`e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42`, owner-pinned SHA-256
`ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f`.
Selection was not reconsidered. No ML inference, dataset access, training,
evaluation, preprocessing execution or new pruning occurred.

The application offline/recovery readiness checks for proceeding to C2 pass.
C2 still needs separate authorization and its explicit external-input/provenance
design from the C1 report. The historical `ERR_NO_BUFFER_SPACE` is an unresolved
reproduction/root-cause limit, not a claimed fixed incident. C2 was not started.
The temporary backend and frontend smoke servers were stopped after verification.
