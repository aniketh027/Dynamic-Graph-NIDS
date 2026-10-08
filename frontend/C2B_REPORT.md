# C2B completion report — 2026-10-07

C2B frontend integration is implemented and ready for owner visual review.
The existing C2A backend was consumed without changes. No release action was
performed. Browser inference used only a fixed, small synthetic application CSV;
its returned values provide integration evidence only, not scientific evidence.

## Owner clarification and preservation

The explicit OWNER CLARIFICATION resolved the stale application-boundary wording
in PROJECT_HANDOFF.md. It authorized C2B after the approved external adapter and
completed C2A, while preserving every scientific/release/provenance boundary.
PROJECT_HANDOFF.md remains byte-identical at SHA-256:

`b4d30d66785121ce4b27db5023bf124125225e53b559cee3207ba3a77ea557e1`

Branch remains `app-demo`; HEAD remains
`2854d153d65c93689875884f3ad45f0b3b468ec2`. Nothing is staged. No commit, push,
tag, stash, reset or clean was run. The only tracked diff remains the approved
pre-existing handoff change: 90 insertions, 24 deletions. Application changes
are in the already-untracked frontend directory, so Git diff alone does not
represent this implementation; see the file inventory below and preservation
record. All backend files and the package lock retain their baseline hashes.

## Files

Modified existing files, all under frontend/:

- `package.json`: additive complete frontend test command.
- `README.md`: current C2B usage and limits; earlier phase reports are preserved.
- `src/App.tsx`: session provider and scoped replay stylesheet.
- `src/lib/api/client.ts`: typed raw-upload operations and request lifetime.
- `src/components/traffic-file-input.tsx`: actual validate/analyze workflow.
- `src/components/demo-controls.tsx`: explicit switch clears real session state.
- `src/components/app-shell.tsx`: accurate replay/status/notification language.
- `src/components/intrusion-drawer.tsx`: route real results to the real inspector.
- `src/pages/dashboard.tsx`: current-session real results and pending state.
- `src/pages/workspace-pages.tsx`: Analyze, Alerts and capability copy integration.

Created source files:

- `src/lib/api/replay.ts`: C2A response interfaces, runtime decoding and errors.
- `src/lib/replay-session.ts`: bounded session lifecycle and request ownership.
- `src/lib/replay-context.ts`: shared React context.
- `src/components/replay-provider.tsx`: provider using useSyncExternalStore.
- `src/components/replay-results.tsx`: summary, class counts and prediction table.
- `src/components/real-intrusion-drawer.tsx`: returned prediction details.
- `src/styles/replay.css`: responsive replay styling and table containment.

Created verification/delivery files:

- `tests/replay.test.mjs`, `tests/replay-components.test.mjs`,
  `tests/register-tsx.mjs`.
- `tests/synthetic-wire-response.json`: actual endpoint response from synthetic
  data only; no uploaded feature values or official sample.
- `tests/serve_c2b.py`: local browser QA entry using the existing C2A guard.
- `tests/ui-fixture.html`, `tests/ui-fixture.tsx`: explicitly marked development
  mock fixture; not imported by the production application/build.
- `C2B_REPORT.md`, `C2B_PRESERVATION.json` and `screenshots-c2b/` artifacts.

## Acceptance report

| Item | Result |
| --- | --- |
| 1. Files created | Listed above; all within frontend/. |
| 2. Files modified | Ten existing frontend files, listed above. |
| 3. API client | validateReplay and analyzeReplay use actual response schemas, raw File body, text/csv and X-Filename. Existing VITE_API_BASE_URL is reused. |
| 4. State architecture | One memory-only ReplaySession exposed by React context/useSyncExternalStore. Navigation retains it; refresh clears it. No persistence. |
| 5. Analyze workflow | Idle, selected, validating, validated, validation error, analyzing, complete, analysis error and backend-offline handling. Explicit validation avoids repeated automatic uploads. |
| 6. Validation UX | Filename/size precede validity; backend counts, reorder flag, SHA prefix and row-order semantics appear after validation. The validated state explicitly says inference has not run. Zero-window files cannot analyze. |
| 7. Analysis UX | Green indeterminate scanner; duplicate calls disabled/ignored; no measured phase or percentage claimed. |
| 8. Results | REAL ANALYSIS / FROZEN MODEL, input rows/windows/trailing rows, benign/intrusion counts and returned model/graph facts. |
| 9. Table | Zero-based window and target-flow indices, prediction, Model score and BENIGN/INTRUSION. Fifty rows per page; horizontal containment on small screens. |
| 10. Distribution | Native compact meters/list from all 15 returned class counts. No class eligibility filtering or chart dependency. |
| 11. Demo separation | Selecting a real file resets demo. Explicit switch to demo clears real state. Real result views take precedence; demo labels are absent from real prediction details. |
| 12. Event mapping | Only is_attack=true predictions become real intrusion rows; no invented event fields. |
| 13. Drawer | Attack class, Model score, window/target indices, five-flow semantics, retained features/categories/nodes/edges, parameters, E2 HTGNN seed, filename, analysis ID and checkpoint identity. |
| 14. Omitted fields | No severity, timestamp, IPs, protocol, port, packet count, geolocation, latency, blocking/isolation/remediation result or server-stage timestamps. |
| 15. Dashboard | Latest session summary, counts, distribution and most recent 50 intrusion predictions in CSV order; honest pending and idle states. |
| 16. Alerts | Same session result, attack-only pagination and real drawer. All-Benign output shows no intrusion predictions. Demo-off/no-result remains empty. |
| 17. Monitor | Live capture not connected; static capabilities and replay link. No animated real live traffic. |
| 18. Performance | Existing evidence-integration placeholder preserved. Uploads never update research performance metrics. |
| 19. Offline | Local selection and demo work. Validate/analyze require connected backend; status polling recovers. Returned session results remain inspectable after disconnect. |
| 20. Busy | ANALYSIS_BUSY has a specific retry message and retains validated file state. |
| 21. Typed errors | All current C2A codes mapped, including schema/duplicates/missing/extra/nonnumeric/size/row/transform/model/inference errors. Raw backend message/stack content is never rendered. Malformed response fields/counts/provenance links are rejected. |
| 22. Lifetime | Validation 30 seconds; analysis 10 minutes. Abort on cancel/replacement/provider disposal; timers/listeners released in finally; generations reject late/stale responses. Cancellation notes server work may continue. C1.5 polling cancellation preserved. |
| 23. Responsive/accessibility | Desktop 1440x900; laptop 1280x800; tablet 820x1180; mobile 390x844. Mock intrusion table exercised at all sizes; mobile Analyze/Dashboard and drawer checked. Mobile hidden-header overflow reproduced and fixed. Native buttons, focusable table/drawer regions, status/error announcements, picker and existing drop handler retained. Existing reduced-motion CSS/Motion policy preserved. OS reduced-motion toggle, assistive-technology session and physical OS file drag were not exercised. |
| 24. Frontend tests | 30 passed: 12 existing API regressions plus 18 new client/session/component tests. Includes real wire decoding, errors, timeout, cancellation, stale replacement, identity mismatch, busy retry, mapping, 5000-row bound, summary, propagation and nonfabricated drawer fields. |
| 25. C2A regressions | 75 passed, no failures/skips, zero unplanned guard violations. |
| 26. External adapter | 143 passed, no failures/skips, zero guard violations. |
| 27. C1 / safe regressions | Separate C1: 38 passed, zero violations. Application-safe regression preset: 257 passed, 78 intentionally deselected, zero violations. |
| 28. Real browser E2E | Real native HTTP/model path accepted a fixed 12-row all-zero strict CSV. Rendered 12 input rows, 2 complete windows, 2 trailing flows and 2 returned Benign predictions; dashboard/alerts propagation, replacement and typed invalid-schema error verified. No attack claim is made. |
| 29. Demo regression | Explicit demo after backend disconnect, pause, inspector and reset verified in browser. Labels remain DEMO / SIMULATED / DEMO EVENT. |
| 30. Build | PASS. Main JS 629.76 kB / gzip 197.85 kB; lazy chart 92.69 kB / gzip 32.70 kB; CSS 98.69 kB / gzip 19.59 kB. |
| 31. Lint | PASS, exit 0. |
| 32. TypeScript | PASS. |
| 33. Authored warnings | Zero final lint warnings/errors in authored application and test code. No browser errors/warnings in the final connected result capture. |
| 34. Vendor warnings | 44 existing BKLIT/shadcn lint warnings unchanged; existing Vite >500 kB advisory. Backend Torch/PyG/Starlette deprecation warnings remain. |
| 35. Screenshots | Index below. Genuine benign results and explicitly mocked intrusion/processing views are kept distinct. |
| 36. Dataset boundary | No official train/validation/test sample accessed. Only approved compact state/checkpoint/provenance needed by the frozen application path; synthetic values fixed before inference. |
| 37. Methodology | No training, evaluation, policy selection, graph/model/class-map/transform change. |
| 38. Frozen source | All baseline tracked/source/scientific files retain bytes; backend also unchanged. |
| 39. Handoff | Owner clarification used; exact approved SHA-256 unchanged, quoted above. |
| 40. E.10 | All 20 protected roots and 58 inventory/size/mtime/attribute records match the initial baseline. No recovery payloads read for new hashes; this is metadata preservation, not a new cryptographic archive audit. |
| 41. Whitespace | git diff --check passes; authored text reviewed separately because frontend is untracked. |
| 42. Git diff | Only pre-existing handoff reconciliation: 90 insertions, 24 deletions. Application changes remain untracked. |
| 43. Git status | Same root status as start: modified handoff, untracked backend/frontend and protected E.10 entries; empty index. Exact status is in C2B_PRESERVATION.json. |
| 44. Limits | Session-only results; CPU-only backend; client cancellation does not terminate server inference; ten-minute bounded request policy; no persistence/live capture/PCAP. No full scientific suite run because it would access prohibited official data and training/evaluation paths. Real all-Benign output cannot verify a genuine attack drawer; mocked tests cover it. |
| 45. Owner review | Ready for owner visual review, with genuine/mocked evidence and QA limitations explicitly stated. This is implementation validation, not an independent scientific/release audit. |
| 46. Next boundary | Owner visual review and acceptance of C2B. Any subsequent application feature, backend capability update or Git/release action requires separate authorization. No downstream phase started. |

## Screenshot index

Primary PNG deliverables in `frontend/screenshots-c2b/`:

| File | Evidence |
| --- | --- |
| analyze-validated.png | Genuine C2A validation, 1440x900. |
| analyze-results.png | Genuine frozen-model synthetic result, 1440x900. |
| dashboard-real-analysis.png | Genuine result in session dashboard, 1440x900. |
| intrusion-alerts-real.png | Genuine all-Benign analysis; correctly zero intrusions, 1440x900. |
| network-monitor-replay-state.png | Live capture remains disconnected, 1440x900. |
| analyze-running-MOCKED-UI-VALIDATION.png | Delayed mock response to inspect the indeterminate scanner. Not backend timing/progress evidence. |
| intrusion-alerts-MOCKED-UI-VALIDATION.png | Explicit 105-prediction mock, 50 rows/page. |
| intrusion-drawer-MOCKED-UI-VALIDATION.png | Mock attack inspector; filename and on-screen banner identify the fixture. |
| intrusion-drawer-demo.png | Existing separate demo inspector. |
| intrusion-alerts-empty.png, dashboard-idle.png | Honest empty states after reset. |
| analyze-offline.png, analyze-error.png | Offline and typed schema-error views. Error capture uses the browser's smaller physical viewport. |
| alerts-1280-MOCKED-UI-VALIDATION.png | Laptop table containment. |
| alerts-820-MOCKED-UI-VALIDATION.png | Tablet table containment. |
| alerts-390-MOCKED-UI-VALIDATION.png | Mobile after hidden-header overflow correction. |
| drawer-mobile-MOCKED-UI-VALIDATION.png | Mobile inspector. |
| analyze-mobile-MOCKED-UI-VALIDATION.png, dashboard-mobile-MOCKED-UI-VALIDATION.png | Mobile result layouts. |

Browser captures are natively JPEG; PNG copies are format conversions with
unchanged decoded pixels. Original captures are retained beside them. The real
small request completed before a stable processing screenshot could be captured,
so there is no falsely labeled `analyze-running.png`. Likewise, there is no
`intrusion-drawer-real.png`: the real synthetic result was all Benign. The named
MOCKED UI VALIDATION alternatives document those visual checks honestly.

## Reproduction

Frontend: `npm test`, `npm run build`, `npm run lint`, `npm run typecheck`.
Application-safe Python checks, from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe --replay
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe
.\.venv\Scripts\python.exe -B -m backend.tests.run_application_safe --regressions
.\.venv\Scripts\python.exe -B -m backend.tests.run_safe
```

The native browser server uses `frontend.tests.serve_c2b`, which installs the
unchanged C2A runner guards before starting Uvicorn. It was deliberately stopped
for the disconnect check. Test-suite guard counts above are completed-run
measurements; the interrupted server does not claim a final guard-count report.
No performance or scientific accuracy conclusion follows from synthetic output.
