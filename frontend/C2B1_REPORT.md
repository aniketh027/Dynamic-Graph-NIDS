# C2B.1 final UI polish

Date: 2026-10-07. Verdict: **READY FOR APPLICATION CHECKPOINT COMMIT**.
No checkpoint commit was started. This is a local frontend correction report, not a scientific release audit.

## Preconditions and authority

Local branch `app-demo`, HEAD `2854d153d65c93689875884f3ad45f0b3b468ec2`.
Read repository instructions, both repository skill descriptions, the applicable
stage-implementation skill, handoff, C1.5/B/B.5/C2B reports and preservation record,
and relevant live source/tests. The local application remains untracked and uncommitted.
Remote main was not used as the application source of truth.

The owner's explicit clarification was used to resolve the stale handoff statements
that C2A was incomplete/blocked. Approved adapter/C2A source and the new explicit
C2B.1 authorization establish this application boundary. Scientific authority was
not superseded. The handoff remains byte-identical at SHA-256
`b4d30d66785121ce4b27db5023bf124125225e53b559cee3207ba3a77ea557e1`.
Its pre-existing tracked diff (90 insertions, 24 deletions) was preserved.
All 20 original E.10/recovery/provenance root entries were protected.

## Modified files

Exactly nine existing frontend files changed:

- `src/lib/replay-context.ts`: explicit QA presentation flag and shared labels.
- `src/components/app-shell.tsx`: header consumes shared replay labels.
- `src/components/replay-results.tsx`: mobile cards, shared labels, empty pagination guard.
- `src/components/real-intrusion-drawer.tsx`: shared badge, description and accessible labels.
- `src/components/traffic-file-input.tsx`: phase-derived file-row styling and text.
- `src/pages/workspace-pages.tsx`: mode-correct Alerts labels and demo-only severity UI.
- `src/styles/replay.css`: cards below 601px; neutral/amber/green/red file-row treatments.
- `tests/replay-components.test.mjs`: five new focused semantic regression cases.
- `tests/ui-fixture.tsx`: explicit mock provider flag and bounded validation delay for QA.

New evidence: this report, `C2B1_PRESERVATION.json`, and nine JPEG captures in
`screenshots-c2b1/`. Previous phase reports/evidence and all dependencies are unchanged.
Original line-ending conventions were retained. No API, session state machine,
backend validation, inference, scientific model or preprocessing contract changed.

## Five fixes and their verification

| Issue | Root cause | Correction | Verification |
| --- | --- | --- | --- |
| Mobile clipping | A minimum-width desktop table was the only replay presentation. | Below 601px, compact cards show Prediction, Status, Window, Target flow, Model score and Inspect. Same bounded page and inspection handler; desktop/tablet table retained. | Component test bounds both views to 50 predictions; native 390px browser inspection and pagination; 820px tablet table. |
| Mock authenticity | The fixture supplied normal replay context and components hardcoded real labels. | Only the QA provider opts into `presentation: 'mock'`; shared helper supplies labels for header/results/Alerts/drawer. No payload/filename heuristics. | Tests cover summary, dashboard, Alerts and drawer content; native desktop and mobile drawers show MOCKED UI VALIDATION / NO MODEL OUTPUT; no REAL ANALYSIS, Real intrusion predictions or FROZEN MODEL in fixture results. Genuine results retain real labels. |
| Green unvalidated file | Selected-file row inherited unconditional green background/border. | Neutral selection, amber validating, green only when validation exists, red for failed/cancelled/offline validation. Written status accompanies color. | Tests cover selection, pending, success, failure, offline and completed analysis. Native invalid and valid CSV uploads verify transitions; mock delay exposes amber pending state. |
| Severity legend in replay | Alerts rendered the demo legend unconditionally and idle demo event filters implied severity. | Legend and demo EventsPanel render only with demo enabled. Real prediction status stays BENIGN/INTRUSION. | Tests cover idle, demo and real empty results. Native real result has no legend; native demo retains INFO/SAFE/WARNING/HIGH/CRITICAL and its filters. |
| Empty pagination | Pagination rendered despite a zero-length prediction list. | Omit pagination when the relevant count is zero. Nonzero controls/page size unchanged. | Tests check zero, one, 105 and 5000 predictions. Browser real-zero state has no pagination; mock pages show 50/50/5 records. |

## Validation

Commands executed from `frontend/` unless noted:

| Check | Result |
| --- | --- |
| `node --test tests/replay-components.test.mjs` (focused first) | 11 passed, 0 failed |
| `npm test` (complete frontend suite) | 35 passed, 0 failed |
| `npm run build` | PASS: TypeScript and Vite production build |
| `npm run lint` | PASS: 0 errors; 44 existing vendor warnings, no corrected-file warnings |
| `npm run typecheck` | PASS |
| `git diff --check` (repository root) | PASS |
| Scoped original-to-current frontend diff and whitespace review | PASS |

The existing >500kB bundle advisory remains (main JS 631.49kB, gzip 198.25kB).
No dependency or lint-rule changes. The broader scientific suite was not run:
this task explicitly authorizes frontend validation and forbids scientific evaluation.
The full frontend suite includes API cancellation, decoding, session reset/stale
response suppression, bounded pagination and existing C1.5 connection regressions.
Prior backend test counts are not represented as newly rerun tests.

## Direct browser QA

Used the existing explicit development fixture and unchanged guarded
`python -B -m frontend.tests.serve_c2b` localhost server. No external upload occurred.

- **390px Alerts:** cards readable; all six fields/actions exposed; document
  clientWidth/scrollWidth both 375px (scrollbar excluded). Inspect works. Page 2
  starts at window 50 / target flow 254; page 3 has five cards (100-104).
- **Desktop mock:** banner stays visible; mock-specific prediction heading;
  DOM check found no real-authenticity label leak. Table remains visible and cards hidden.
- **Mock drawer:** MOCKED UI VALIDATION / NO MODEL OUTPUT, mock description,
  mock accessible region and footer. Mobile drawer uses the viewport width.
- **820px tablet:** table visible, cards hidden, clientWidth/scrollWidth both 820px.
- **Genuine real zero-intrusion:** fixed pre-existing synthetic 12-row canonical
  CSV completed backend inference. Zero intrusion predictions; REAL ANALYSIS /
  FROZEN MODEL retained. DOM counts: pagination 0, severity legend 0. Inputs were
  not searched or modified to force an intrusion. All attack UI evidence is mock.
- **Invalid file:** backend returned UNEXPECTED_COLUMN; neutral selected row
  (`rgb(25,32,43)`) became red (`rgb(41,28,35)`), never green.
- **Validated file:** selecting replacement returned neutral; successful backend
  validation became green (`rgb(23,32,30)`). Mock pending validation was amber
  (`rgb(41,36,25)`).
- **Demo:** explicit switch clears replay; DEMO / SIMULATED header, severity legend,
  CRITICAL / DEMO EVENT drawer with DDoS (simulated) and synthetic fields preserved.
- Both fresh QA tabs reported no captured browser warnings/errors.

Captures: `mock-mobile-cards.jpg`, `mock-desktop.jpg`, `mock-drawer-mobile.jpg`,
`mock-drawer-desktop.jpg`, `real-zero-intrusions.jpg`, `file-unvalidated.jpg`,
`file-invalid.jpg`, `file-validated.jpg`, and `demo-drawer.jpg`.
These are UI evidence, not scientific detection/performance evidence.
Some clipped captures included scroll offsets; the mobile drawer was recaptured
using the standard viewport screenshot after verifying its dimensions.
Temporary browser tabs closed and viewport override reset; task-owned servers stopped.

## Preservation and final Git state

SHA-256 comparison: 472 baseline tracked/application files, exactly the nine
frontend changes above, 463 byte-identical files. All baseline backend, frozen
source, `.agents/skills/`, prior reports and handoff files are unchanged.
All 20 protected E.10 roots and 58 recursive metadata records remain unchanged;
payloads were not read for this comparison. No official dataset or test arrays
were accessed. Guarded inference used the existing fixed synthetic CSV and frozen
application artifacts, including the C2A-authorized compact scaler state.

Branch, HEAD, refs and root Git status are unchanged from preflight. The staged
diff is empty. The only tracked diff remains the pre-existing handoff change.
See `C2B1_PRESERVATION.json` for exact status, path inventory, and file hashes.
No staging, commit, push, tag, stash, reset, clean, move or deletion of protected
artifacts occurred. No changes outside the frontend deliverables were made.

## Concerns and stop boundary

No unresolved C2B.1 blocker. Existing vendor warnings and bundle advisory remain.
Synthetic-only browser checks do not establish model accuracy or production performance.
This work stops at C2B.1. The next application boundary is owner review/checkpoint
authorization; this report does not execute a commit or authorize another stage.
The scientific release and its audit/provenance work are complete and frozen at
commit `2854d153d65c93689875884f3ad45f0b3b468ec2`, tagged
`v1.0-experimental-complete`. The remaining boundary is application checkpoint
review/commit work under owner authorization.

**READY FOR APPLICATION CHECKPOINT COMMIT**
