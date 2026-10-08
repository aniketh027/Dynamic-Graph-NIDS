# Model Performance — frozen scientific evidence integration

2026-10-08. Verdict: **READY FOR MODEL PERFORMANCE VISUAL REVIEW**.
Frontend presentation only; no new scientific result or release action.

## Preflight and authority

Initial branch: `app-demo`. HEAD, local `origin/main` and live remote main:
`876526e327c336b3155e4356e630dc8e2a6c8dd9`.
The annotated `v1.0-experimental-complete` tag peels locally and remotely to
`2854d153d65c93689875884f3ad45f0b3b468ec2`.
Initial tracked and staged diffs were empty; `git diff --check` passed.
Initial status contained only 20 protected E.10 recovery/provenance roots
and four excluded existing screenshot directories (`screenshots/`,
`screenshots-b5/`, `screenshots-c2b/`, `screenshots-c2b1/`).

Read AGENTS.md, both custom skill descriptions, both repository SKILL.md files,
PROJECT_HANDOFF.md, the application reports/source/tests, final scientific
results, release provenance and E.11/E.12A/E.12B documentation and compact reports.
Applied `nids-stage-implementation`; this is not a release audit. The owner's
explicit completed-application clarification resolves the handoff's historical
C2A/C2B wording. The task's frontend-only tests and no-Git-mutation instructions
override broader skill/repository defaults. No new material authority conflict.
Historical pending/proposed status fields remain intact; the separate E.12A
freeze attestation and completed final results establish their later chronology.

## Exact scientific evidence displayed

The following repository-relative references are shown in the page:

- `docs/final_experimental_results.md`
- `results/stage_e11/e10-official-20261001-v1/recovery/e11-recovery-20261006-v1/stage_e11_report.json`
- `results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/stage_e12a_report.json`
- `results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/proposed_policy.json`
- `results/stage_e12/e10-official-20261001-v1/e12a-20261006-v1-freeze/freeze_attestation.json`
- `results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/stage_e12b_report.json`
- `results/stage_e12b/e10-official-20261001-v1/e12b-20261006-v1/validation_benchmarks.json`
- `docs/release_provenance.md`

All eight source files were compared byte-for-byte with their blobs at the
scientific release. Five report/policy/attestation canonical digests and all
nine E.12B baseline physical/canonical digests and task/seed bindings passed.
These were compact-JSON checks, without predictive-array access or evaluation.

Numbers below are the rendered precision; source values are retained at their
persisted numeric precision in the immutable presentation module.

| Displayed measure | Unpruned | Attention-pruned | Scope |
| --- | ---: | ---: | --- |
| E2 mean eligible-class macro-F1 | 0.875988 | 0.889683 | Frozen test, three HTGNN seeds |
| E2 sample SD, ddof=1 | 0.022640 | 0.009046 | Across seeds 2025/2026/2027 |
| Seed 2025 macro-F1 | 0.851326 | 0.897239 | E2 eligible true-row test filter |
| Seed 2026 macro-F1 | 0.880807 | 0.879659 | Same definition |
| Seed 2027 macro-F1 | 0.895830 | 0.892150 | Same definition |
| Feature nodes | 78 | 61 | Inference graph per window |
| Category nodes | 8 | 8 | Same graph |
| Sample nodes | 1 | 1 | Same graph |
| Total nodes | 87 | 70 | Same graph |
| Directed edges | 86 | 69 | Same graph |
| E2 HTGNN parameters | 7,183 | 7,183 | Learned weights unchanged |
| Peak PyTorch allocated | 49.552 MiB | 46.622 MiB | E1/E2 GPU validation |
| Peak PyTorch reserved | 56 MiB | 56 MiB | All nine GPU validation pairs |
| E1 support-restricted macro-F1 | 0.930799 | 0.930792 | Three-seed test mean; Benign-only reporting subset |
| E3 average-precision PR-AUC | 0.089031 | 0.089048 | Three-seed chronological binary test mean |

Per-seed E2 absolute changes shown: +0.045913, -0.001148, -0.003681.
Eligible-class F1 equally averages per-class F1 over classes with train support
>=100, validation >=30 and test >=30; true test rows are restricted to eligible
classes. All 15 output classes remain possible predictions. This is separate
from full-test descriptive metrics and the E.12A validation selection metric.
E2's within-class chronology is explicitly distinguished from a global timeline.

Paired median speed-ratio range: **1.034540x–1.117286x**, across E1/E2/E3 and
three seeds on NVIDIA GeForce RTX 4050 Laptop GPU. Four validation batches of
256 windows per task/seed, five warmups per path/batch, twenty measured pairs
per batch (80 pairs), alternating path order, float32 and strict deterministic
algorithms. Resident forward includes checks, feature compaction, encoders/head;
loading/conversion/H2D are excluded. The ratio is a median of paired ratios.

Allocated peak reduction is approximately **2.931 MiB / 5.91%** for E1/E2,
derived only by unit conversion/subtraction of the persisted allocator bytes
(51,959,296 -> 48,886,272). Reservation is unchanged at 58,720,256 bytes.
The allocator scope includes the model and resident full adapter input.
Neither metric measures current CPU CSV upload/inference latency.

## Delivered UI and files

Created:

- `src/components/model-performance.tsx`: dedicated evidence view.
- `src/lib/model-performance-evidence.ts`: deeply frozen, typed compact snapshot.
- `src/styles/model-performance.css`: scoped console styling and responsive layout.
- `tests/model-performance.test.mjs`: 11 focused evidence/state/provenance tests.
- `MODEL_PERFORMANCE_REPORT.md`: this report.

Modified:

- `src/pages/workspace-pages.tsx`: replaces only the placeholder with a component export; other route functions unchanged.
- `src/index.css`: imports the scoped stylesheet.
- `README.md`: documents the evidence page and focused check.

Sections: scientific overview; four factual summary cards; accessible 0–1 E2
bar comparison and seed table; graph table; GPU validation observations and
protocol; amber scientific limitations; release identity and eight exact sources.
Section anchors, restrained hover/focus feedback and existing cards/badges
preserve the GRAPHNIDS console design. No new dependency or API.

The page has no replay/demo/backend state input. Changing upload results, fixture
presentation, demo counts, session phase or connection cannot change its markup.
Scientific weaknesses remain visible without a tooltip or collapsed disclosure.
The full canonical 78-feature input remains required; the input adapter still
builds the full graph. No retraining, weight pruning or parameter reduction.

## Validation

| Command, frontend directory | Result |
| --- | --- |
| Initial `npm test` | 35 passed |
| `node --test tests/model-performance.test.mjs` | 11 passed, 0 failed/skipped |
| Final `npm test` | 46 passed, 0 failed/skipped |
| `npm run build` | PASS; TypeScript and Vite production build |
| `npm run lint` | PASS; 44 existing vendor UI warnings, no authored warnings |
| `npm run typecheck` | PASS |
| Root `git diff --check` | PASS |

Focused coverage checks exact E2 values/scopes/seeds/SD, independently recomputes
the three-seed mean/sample SD from persisted scalar results, checks structural counts,
unchanged parameters, GPU validation protocol and allocator scope, visible E1/E3
limitations, immutable data, full markup equality under adversarial session
states, exact release sources and absence of unsupported claims/blank cards.
The complete frontend suite retains REAL/DEMO/MOCK, wire-decoding, cancellation,
stale-response, pagination and connection regressions.
An initial visibility assertion incorrectly matched a decorative SVG's
`aria-hidden`; the assertion now checks hiding of content elements, and passes.

Existing Vite >500 kB main-bundle advisory remains (645.61 kB, gzip 202.34 kB).
No dependency, lint rule or unrelated vendor change. Scientific/full Python
suite, official evaluation, attention ranking and benchmarks were not run.

## Browser QA

Used the local Vite application and unchanged development-only UI fixture in
the in-app browser. Backend remained offline for production-page checks.
No checkpoint was loaded and no new inference was executed.

- Desktop 1440x1000: two-column comparison/graph, legible labels; document client/scroll widths both 1425px when scrollbar present; no clipped metric cells.
- Tablet 820x1180: stacked comparison/graph, two-column summary cards; client/scroll widths both 805px when scrollbar present; no clipped values.
- Mobile 390x844: readable seed table and chart without horizontal overflow; client/scroll widths equal (375px with scrollbar or 390px with overlay scrollbar). Limitations and sources anchors expose visible content under the sticky header. Exact source paths wrap.
- All six routes loaded. Dashboard idle/demo, Analyze offline controls, demo Alerts/event inspector, Graph Optimization and honest disconnected Network Monitor were verified.
- Unchanged mock fixture completed validation and 105 mocked predictions; Alerts page 2 and window-50 drawer worked, showing target flow 254 with MOCKED UI VALIDATION / NO MODEL OUTPUT.
- Model Performance rendered identical text before/after active demo and completed mock analysis. Mock fixture banner/header stayed separate from scientific content.
- Both browser tabs recorded no warning/error logs during checks.

QA JPEGs are outside the repository under the task's temporary review directory,
including desktop/tablet/mobile full captures, mobile comparison/limitations/
sources and mock-performance. No QA capture is a source deliverable or scientific
measurement; none was staged. Existing screenshot directories were preserved.
This browser pass did not repeat real backend inference; existing real-wire
unit regressions passed. Assistive technology and physical touch were not tested.

## Preservation, final Git state and stop boundary

Final scope is exactly the eight frontend files listed above. Initial tracked
file SHA-256 baseline covered 420 source/document/compact-evidence files; only
the three intended existing frontend files changed. All remaining 417 are
byte-identical, including scientific source/tests/evidence, backend, handoff,
skills, model contracts and dependency locks. Six tracked summary CSVs were
excluded from payload hashing; none appears in the Git diff.
All 107 initial untracked file size/mtime records were preserved, covering the
protected recovery/provenance roots and four existing screenshot directories.
Historical recovery payloads were not opened for this preservation check.

HEAD, local origin/main and scientific tag are unchanged. Staged diff is empty.
Root status has the three intended modified files, five intended new frontend
files, and all original protected entries. Full tracked diff and all new files
were reviewed; whitespace checks passed. Nothing staged, committed, pushed or
tagged; no reset, clean, stash, rebase or history rewrite.

Precision/recall/accuracy summary cards, confusion matrix and application latency
were intentionally omitted to keep one explicit evaluation scope and avoid
invented current measurements. No statistical significance, noninferiority,
universal/end-to-end/training speedup, checkpoint-size, total GPU-memory or
CPU-memory reduction claim. No unresolved implementation blocker.

The handoff's literal adapter-approval boundary is historical and superseded
for completed application work by the owner's explicit clarification. The next
safe boundary for this authorized substage is **owner Model Performance visual
review**. No downstream stage or Git publication follows automatically.

**READY FOR MODEL PERFORMANCE VISUAL REVIEW**
