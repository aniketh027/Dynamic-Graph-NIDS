# C3B browser CSV compatibility implementation report

Owner date: 2026-10-11. Implementation validation is complete; independent owner
audit is pending. This report records application/synthetic evidence only.

## A. Repository identity and authorization

- Repository: `C:\Project\ComputerNetworks`; branch: `app-demo`.
- Local HEAD, `origin/main` and independently queried live remote main:
  `06145e9af61d57e48a282d5c99c08c1314f8a06a`.
- Scientific release: `v1.0-experimental-complete`; tag object
  `3e6e54057f26585cd7d3b2443afbcc668ea96dbb`; peeled commit
  `2854d153d65c93689875884f3ad45f0b3b468ec2`. Live remote identities matched.
- Initial tracked/staged diff was empty. There were 107 protected untracked
  files: 45 E.10 recovery/provenance files and 62 screenshots.
- Final implementation inventory: 15 modified tracked paths and 10 added paths,
  all within the 24-path task allowlist plus the separately authorized fixture.
  The 107 original untracked files remain. No staged changes.
- No stage, commit, push, tag, history rewrite or next-stage work was performed.
  Codex's automatic `refs/codex/turn-diffs/*` captures can change between turns;
  branch/remote/release refs and the raw Git index remained unchanged.

Preflight read the actual handoff, root instructions, both repository skills,
complete C3B feasibility review, C3A implementation/correction/audit evidence,
existing frontend reports, C1/C2A contracts and frozen provenance. Frozen source
and relevant tests were inspected before implementation/testing. Historical
handoff and report boundaries were preserved; this owner's task separately
authorized C3B. The historical C3A audit defects were checked against the corrected
current implementation rather than treated as current blockers.

Browser QA found an existing development-only mock fixture regression: its
all-spaces placeholder could not pass mandatory local inspection and its
immediate validate call raced inspection. Work stopped, the owner explicitly
authorized `frontend/tests/ui-fixture.tsx`, and only then was that path corrected.
No production inspection bypass or additional dependency was introduced.

## B. Exact implementation inventory

Paths below are repository-relative. M means modified; A means added.

| State | Path | Purpose and scope |
| --- | --- | --- |
| M | `backend/app/schemas/contracts.py` | Add required read-only `csv_compatibility` field to existing schema response |
| M | `backend/app/services/inference.py` | Include descriptor in existing metadata construction; no inference change |
| A | `backend/app/services/csv_schema.py` | Derive schema, projections, numeric semantics and limits from Python authority |
| A | `backend/tests/test_csv_ui_schema.py` | Descriptor, generator and inert route verification |
| A | `backend/tests/run_csv_ui_safe.py` | Fixed guarded schema tests and bounded synthetic C3A differential oracle |
| M | `backend/README.md` | Add descriptor/browser workflow, limits and guarded verification documentation |
| M | `frontend/package.json` | Add CSV test and Windows manifest-check scripts; dependency set unchanged |
| M | `frontend/src/components/traffic-file-input.tsx` | Explicit local preparation, safe errors and separate validation/provenance UI |
| M | `frontend/src/lib/api/client.ts` | Abortable read-only schema GET and strict descriptor rejection |
| M | `frontend/src/lib/api/replay.ts` | Fixed safe local/schema error messages |
| M | `frontend/src/lib/replay-session.ts` | Separate source/prepared state, upload gating, linkage and lifecycle protection |
| M | `frontend/src/pages/workspace-pages.tsx` | Cancel pending session work when leaving Analyze Traffic |
| M | `frontend/src/styles/replay.css` | Amber preparation section and wrapping provenance hashes |
| A | `frontend/src/lib/csv-compatibility.ts` | Typed preparation record, full descriptor comparison and ephemeral worker transport |
| A | `frontend/src/lib/csv-preparation.worker.ts` | Strict lexical CSV parsing, numeric checks, projection, output and hashes |
| A | `frontend/src/lib/csv-schema.generated.ts` | Deterministic manifest generated from backend authority |
| A | `frontend/scripts/generate-csv-schema.py` | Reproducible generation and read-only drift check |
| M | `frontend/tests/api-client.test.mjs` | Schema GET, malformed/drift rejection, deadline and abort coverage |
| M | `frontend/tests/replay.test.mjs` | Source privacy, prepared linkage, lifecycle and canonical regressions |
| M | `frontend/tests/replay-components.test.mjs` | Explicit confirmation, preparation/validation separation and local status |
| A | `frontend/tests/csv-compatibility.test.mjs` | Parser/worker/limits/schema tests and Python byte differential parity |
| A | `frontend/tests/fixtures/csv-compatibility-cases.json` | Synthetic numeric strings; no copied feature schema or scientific sample |
| M | `frontend/tests/ui-fixture.tsx` | Separately authorized canonical synthetic fixture and awaited inspection |
| M | `frontend/README.md` | Current exact-format workflow, privacy, limits, cancellation and audit boundary |
| A | `frontend/C3B_REPORT.md` | Implementation evidence, preservation and limits |

## C. Browser worker architecture

Each inspection/preparation operation creates one native module Worker with a
cloned File. The main application thread handles UI/session state. The worker
reads a bounded source ArrayBuffer, decodes strict UTF-8 with an optional initial
BOM, and iterates logical records without constructing a full table of cells.
Inspection recognizes the header; canonical record validation remains C2A's
existing backend responsibility. Preparation validates the entire 80/84 source
before publishing any prepared File to the session.

The lexical parser supports quoted cells, doubled quotes, multiline quoted
cells, LF, CRLF and lone CR record endings. It rejects malformed quoting even
in discarded metadata, invalid record widths and embedded headers. Standard
and extended headers must match exact ordered Python constants. Canonical names
must be unique and complete, with their existing arbitrary-order support.

Retained cells are validated against the generated C2A full-token decimal/IEEE
grammar, Python strip code points and binary64-to-float32 finite acceptance
bound. No `parseFloat`, numeric typing or broad JavaScript `trim` is used.
Signed zero, exponent text, signed NaN/infinity, sentinels, allowed whitespace
and very small finite numbers remain exactly as decoded. No cells are reformatted,
imputed or transformed. Explicit special IEEE values remain allowed by C2A.

Output uses authoritative model order, quotes every field, doubles embedded
quotes, encodes UTF-8 without BOM and ends records with LF, matching C3A. All
records/order/duplicate-looking rows/tails remain; no sorting or row filtering
occurs. For N records, windows are `N // 5`, tails are `N % 5` and target indices
are `4,9,14,...`. These are flow records, not packets or guaranteed conversations.

| Resource | Enforced maximum |
| --- | ---: |
| Source bytes | 10,485,760 |
| Prepared bytes, independently | 10,485,760 |
| Flow records | 25,000 |
| Complete windows | 5,000 |
| Existing inference batch | 256 windows |
| Retained feature field | 131,072 decoded characters in verified environment |

Output uses a fixed bounded byte buffer plus a final exact-length copy. A single
row and serialization buffers are transient. Source bytes, decoded UTF-16 text,
metadata decoding, output copies, hashing, File/worker transport and browser
overhead can coexist: the 10 MiB file limit is not a 10 MiB memory guarantee.
No latency, peak-memory, browser-wide performance or benchmark claim is made.

## D. Schema authority and generation

`csv_descriptor()` derives canonical/source arrays and projection/removed names
from `MODEL_FEATURE_COLUMNS`, `CSV_COLUMNS` and `CSV_COLUMNS_EXTENDED`. It uses
C2A's NUMBER pattern, float32 bound, byte/row/window/batch constants, Python
`str.isspace()` points and the current `csv.field_size_limit()` getter. It never
changes process-global CSV settings and loads no scaler/model state.

- Contract version: `graphnids-c3b-browser-csv-v1`.
- Feature-order SHA-256:
  `7ae8c123332c05b1136e1093c53d710b69b5c397ad15309e207667b104f5fe4c`.
- Descriptor SHA-256:
  `6461ebd6ca91cf2222209705b9f2c60afdeab41ca2c2cced233d318b745360f1`.
- Identity uses repository `canonical_sha256` over the descriptor excluding its
  own digest. Finite bound text is `3.4028234663852886e+38`.
- Generator output is deterministic UTF-8/LF TypeScript with sorted JSON keys.
  `--check` fails on missing/drifted output without rewriting it. Tests mutate
  only a disposable external target to prove that behavior.

The existing `/api/v1/schema` response adds the descriptor. No new endpoint,
capability, parser setting or replay response change is introduced. Schema
construction itself needs no discovery or model initialization; ordinary backend
lifespan behavior remains separately configured and was not started for this task.

Before each prepared validation and analysis, a read-only no-store GET checks
all nested keys, types, ordered arrays, values and the bundled identity. Missing,
malformed, unknown-version or mismatched content rejects upload. A copied digest
with altered structure fails. Nothing is regenerated or accepted opportunistically.
Canonical uploads skip this additional check and retain their backend flow.

Environment assumption: C2A currently inherits Python's process CSV field limit,
131,072 in the isolated Python 3.13 environment. A changed process limit or
whitespace table changes descriptor identity and fails closed. This is not a
cross-version certification. Valid numeric tokens/strip points are BMP; oversized
astral cells cannot exclude an otherwise valid numeric token. Discarded metadata
can exceed the retained field limit within the source bound.

## E. UI integration and session behavior

- Canonical 78/shuffled files: local header recognition, existing Validate/Analyze
  flow, no preparation confirmation, original File used for upload.
- Exact 80/84: source recognition names the two/six removed fields and requires
  **Prepare locally**. Neither source inspection nor pre-confirmation uploads.
- Prepared: amber **Prepared locally — backend validation required** state;
  backend success alone sets validated presentation. Offline preparation remains
  available, while validation/analysis controls require connection.
- Original File, prepared File, local metadata, backend validation and analysis
  are distinct state. Prepared operations cannot select the original 80/84 File.
- Replacement/removal invalidates the generation, aborts active work and clears
  old metadata/validation/results/drawer state. Worker success/error/abort/crash
  terminates the worker and clears handlers. Late callbacks cannot publish old
  state. Duplicate preparation/request submissions are suppressed.
- Analyze Traffic effect cleanup cancels pending inspection/preparation/schema
  checks/uploads. Settled preparation, validation and results survive navigation.
  Provider disposal uses the same cancellation path. Cancelled inspection is
  retryable. Client cancellation does not prove server processing stopped.
- Source, prepared bytes and results stay memory-only. No localStorage,
  IndexedDB, database, downloaded sidecar or session persistence was added.
- DEMO/MOCK results retain existing explicit labels and frozen Model Performance
  remains independent. The corrected development fixture uses generated names,
  525 all-zero synthetic records and real worker inspection, followed by delayed
  mocked responses. Its source byte size/hash linkage matches its synthetic File;
  attack predictions remain invented mock data, not model output.

## F. Privacy and provenance

For 80/84 inputs, only prepared canonical bytes reach upload calls. The safe
basename is `graphnids-prepared.csv`; original filename/metadata cells are not
uploaded. Worker code has no fetch, telemetry, logging, filesystem or persistent
storage writes. Errors use fixed codes/messages, never exception/source text.

Preparation records hold source/output SHA-256, sizes, profile, descriptor
identity/version, removed names and row/window/tail counts with `rows_removed=0`,
`units_changed=false`, `values_imputed=false`, `transforms_applied=false`. No removed
values or absolute paths are recorded. Local source hashes are not authenticated
by the backend. Existing validation/analysis response linkage authenticates
prepared filename, byte size, SHA-256 and record count; mismatches reject state.
The existing API still does not consume C3A sidecars.

Retained features can be sensitive. Metadata removal is not complete anonymization,
and hashes are not signatures of exporter identity, physical plausibility,
measurement equivalence, chronology or research membership. Backend upload
identity refers to the prepared file. Matching headers cannot prove origin.

## G. Differential and adversarial evidence

The Node differential test invokes the guarded Python runner with bounded
base64 synthetic bytes and no caller-provided paths. C3A publishes to fresh
disposable external local NTFS paths; the oracle returns prepared bytes/hashes
and counts without reading scientific artifacts.

There are **110 differential cases: 62 accepted and 48 rejected**. Accepted cases
match exact C3A output bytes, source/prepared SHA-256 and record counts. Composition:

- 16 profile x BOM x LF/CRLF x minimal/quoted serialization combinations;
  multiline/escaped-quote metadata, 17 distinct ordered records and two tails.
- 23 accepted numeric strings x two source profiles = 46 accepted cases.
- 21 rejected numeric strings x two profiles = 42 rejected cases.
- Unterminated quoting, embedded headers and reversed headers x two profiles =
  six further rejected cases.

Additional worker tests cover duplicate-looking rows, fifth-flow correspondence,
exact decoded-cell preservation, strict UTF-8 failures, malformed removed cells,
missing/extra/duplicate columns, aliases, 61 columns, unknown exporter headers,
source exactly/above 10 MiB, output below/at/above 10 MiB, retained fields at/above
131,072, discarded metadata above that field bound, and 24,999/25,000/25,001 rows.
All generated Python strip points are checked. Worker transport and session tests
exercise abort-ignoring/late responses, disposal, replacement and duplicate clicks.
Boundary-limit cases are additional worker/C3A regression coverage rather than
all being included in the 110-case differential matrix. No universal parser
equivalence or exhaustive float-conversion proof is claimed.

## H. Commands and actual results

Frontend commands run from `frontend/`, using installed locked tooling. Python
commands run from the repository root with `.venv\Scripts\python.exe -B`.

| Command/check | Actual result |
| --- | --- |
| `node --test tests/csv-compatibility.test.mjs` / complete-suite CSV portion | 26 passed |
| API client portion | 14 passed |
| Replay-session portion | 25 passed |
| Replay-component portion | 14 passed |
| Frozen Model Performance portion | 11 passed |
| `npm test` | 90 passed; zero failed/cancelled/skipped/todo |
| `npm run typecheck` | PASS |
| `npm run lint` | Exit 0; 44 existing vendor chart warnings, no authored-file warning |
| `npm run build` | PASS; emitted dedicated worker 8.72 kB; main 659.46 kB / gzip 206.37 kB; 500 kB advisory |
| `npm run schema:check` | PASS after Windows executable-path correction |
| `python -B frontend/scripts/generate-csv-schema.py --check` | PASS; no rewrite |
| `python -B -m backend.tests.run_csv_ui_safe` | 6 passed; one existing Starlette warning; zero unexpected guard violations |
| `python -B -m backend.tests.run_csv_compatibility_safe` | 106 passed; zero unexpected guard violations |
| `python -B -m backend.tests.run_csv_compatibility_safe --parser-regressions` | 27 passed; zero unexpected guard violations |
| Differential oracle inside CSV suite | 110 cases; zero unexpected guard violations |
| `git diff --check` | PASS |
| `git diff --cached` and raw index comparison | Empty staged diff; index unchanged |

During development, one legacy async-selection test and one route-introspection
test failed and were corrected before their passing runs. The first npm manifest
command failed under Windows cmd syntax; the quoted backslash path now passes.
The mock browser regression was corrected only after separate owner authorization.
No unresolved final test failure or skipped mandatory check remains. Existing
import/build warnings are not scientific or performance results.

The new runner installs C3A's corrected read/write/traversal guard before
application imports, self-tests denials including directory enumeration, blocks
scientific/model imports and transform/model entry points, and confines native
C3A regular-file handles to the disposable arena. Route testing uses an inert
service initialization and forbidden discovery/loading. It checks additive
schema metadata, health/status/capabilities and unchanged replay route presence.
It asserts no torch/training/pruning/backend ML module was loaded.

Not run: broad backend replay/scientific suite, actual backend/server/model
startup, real upload validation through scaler authentication, inference,
training, evaluation, pruning, ranking, CUDA or benchmarking. No scientific
dataset, predictive array, scaler payload or checkpoint was read by C3B checks.
The complete frontend suite reads compact frozen reports/Git blobs solely for
the existing Model Performance tests; that is presentation integrity evidence.

### Actual browser runtime QA

Local Vite browser QA used native Workers and newly generated synthetic CSVs
outside the repository. No backend server ran; application health/status polling
reported offline. Actual browser evidence, separate from simulated transports:

| Browser case | Observed result |
| --- | --- |
| Extended 84, 17 records | Explicit local preparation; 7,076 prepared bytes; 3 windows; 2 tails; offline upload disabled |
| Standard 80, 12 records | Replacement cleared old metadata; explicit preparation; 5,126 bytes; 2 windows; 2 tails |
| Settled prepared navigation | Dashboard and return retained same prepared identity/counts |
| Shuffled canonical 78 | Worker reached selected/unvalidated state without preparation prompt; offline validation disabled |
| Unsupported exporter header | Safe `UNSUPPORTED_CSV_FORMAT`; no upload workflow |
| Corrected mock fixture | Awaited real canonical worker inspection; mocked validation showed 525 rows/105 windows |
| Mock pending analysis navigation | Leaving route set cancellation; late delayed response did not restore results; validation retained |
| Mock retry and settled navigation | 105 explicitly mocked results completed and survived Dashboard/return; no REAL ANALYSIS label |
| Browser console | No errors/warnings in application and mock QA tabs |

Extended browser source SHA-256:
`aa2a4c3a2023cda346811c23b492d5954e924c98b70bbd4ba3c4856433e9f6e2`;
prepared SHA-256:
`59be6257f83fdda825760638fa4bdf5bc7f5f403fefab9837bd62017876d6155`.
Standard source SHA-256:
`06c2e0ccfbc3b92b1371fb736e6a350f5b8bf776fc4226a9f8685b1bcbda7ca1`;
prepared SHA-256:
`e57ab5b468dd22483b84cb545fce50f67cece4e739bbd6c160f8853f6ef5c267`.
The corrected mock canonical File hash was
`8777f3bb257cf8a476dc06055bcb4c683618e4176cd053e020cc5985604b6088`.

The actual prepared UI was visually inspected, including mobile hash wrapping
and amber local-preparation separation. No cross-browser/viewport matrix,
real-backend upload, native-worker cancellation stress matrix or performance
measurement was performed. Deterministic worker cancellation/disposal/schema
mismatch/upload lifecycle evidence is unit coverage; browser navigation cancellation
uses the explicitly mocked delayed transport. Temporary QA tabs/server are closed.

## I. Preservation and integrity

Pre/post inventory covers 437 tracked paths plus 107 original untracked paths.
Baseline SHA-256 covers 431 tracked source/compact evidence files, excluding six
CSV/large-payload-extension paths. After excluding the 15 authorized modified
tracked paths, **416 non-allowlisted tracked hashes** remain identical. The other
**113 baseline paths** retain path, size, timestamp and attributes, including all
107 recovery/screenshot files. No new unapproved repository path was found.
Metadata preservation is not presented as a new cryptographic payload audit.
Ignored datasets/checkpoints/predictive arrays were not traversed or freshly hashed.

Preserved handoff SHA-256:
`87658fc128fabe515d9409c9a7f4715f09ece02316d4d259f0f1edb9dc396cb9`.
Historical closure fixture SHA-256:
`cdbdf14742b9f7a7a7b982305d97e2b39d53aa652672c8a6f03ba657d44df230`.
Execution-era fixture SHA-256:
`58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5`.
Release handoff binding SHA-256:
`ddfe421dacbc9397ae6c7b2cdccdedbf2f0d2a8cd925343eab13d7a2ec36fa8f`.
These epochs remain separate; no handoff bytes or attestation was replaced.

Scientific `src/`, frozen E2 transforms/checkpoint/policies/validators, results,
provenance, C2A parser/service/routes, C3A implementation/tests and Model
Performance evidence source remain unchanged. Descriptor addition does not alter
78-feature or five-flow semantics. Existing dependency/lockfile, fixture JSON,
vendor sources and all protected E.10/screenshot paths remain unchanged. Required
Vite/typecheck generated build outputs are ignored tooling products, not staged
application/scientific evidence. No package install or scientific output occurred.

## J. Limitations and exact next boundary

No unresolved implementation blocker remains. Evidence supports the tested
bounded formats and cases, not all exporters or all cross-language inputs.
Descriptor identities detect configured drift, not malicious same-origin code,
trusted-server impersonation or authenticity of exporter/source data. Worker
isolation is execution separation, not a security sandbox against hostile page
code. Cancellation prevents client publication; server cancellation is unproven.

Native Worker, TextDecoder, File and Web Crypto support are required. Unsupported
browser environments fail safely without uploading the source. Source/output
caps do not guarantee browser memory/latency. Field-size/whitespace semantics
depend on the verified Python environment and intentionally fail closed on drift.
Scores remain uncalibrated and no uploaded-input accuracy/efficiency certification
or improvement to frozen scientific performance is claimed.

The next safe boundary is **independent owner C3B CSV UI integration audit** of
this unstaged work. This implementation report is not an independent audit or
checkpoint freeze. Do not stage, commit, push, tag, update PROJECT_HANDOFF, start
another application stage or execute scientific work under this authorization.
