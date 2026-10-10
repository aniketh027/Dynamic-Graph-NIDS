# C3A exact-schema offline CSV preparation — 2026-10-10

Implementation validation is complete; independent owner review remains pending.
This milestone prepares formats only. It creates no scientific result, release,
API conversion contract, frontend integration or deployment claim.

## Authority and scope

Read AGENTS.md, both custom skills, current PROJECT_HANDOFF.md including its
2026-10-09 addendum, application reports, Stage C schema/loader/cleaning/transforms,
the C2A production parser/orchestration/tests/guards, and the complete previous
CSV compatibility feasibility findings. Applied nids-stage-implementation.
The explicit six-file allowlist, no publication and guarded-only testing override
the skill/repository's broader defaults. The dated handoff snapshot at
94423244 is the parent of the documentation-only d847db1 reconciliation; it is
not a new authority conflict. The owner authorized C3A only.

Initial and final branch: `app-demo`. Initial and final HEAD, origin/main and
independently queried live remote main:
`d847db1c5a05d31e0ad001ce766d934438003488`.
Local and live remote `v1.0-experimental-complete` tag object:
`3e6e54057f26585cd7d3b2443afbcc668ea96dbb`; peeled release commit:
`2854d153d65c93689875884f3ad45f0b3b468ec2`.
Initial tracked/staged diffs were empty. The 24 pre-existing untracked status
entries comprise 20 E.10 recovery/archive/script entries and four screenshot
directories; they remain protected, present and unchanged.

## Exactly six implementation files

| Path | Change |
| --- | --- |
| backend/app/services/csv_compatibility.py | New exact-profile preview, projection, safe publication and linkage verification |
| backend/prepare_flow_csv.py | New offline CLI with explicit source/output and optional unverified declarations |
| backend/tests/test_csv_compatibility.py | New synthetic positive, negative and filesystem adversarial tests |
| backend/tests/run_csv_compatibility_safe.py | New fixed-scope no-data/no-model regression runner |
| backend/CSV_COMPATIBILITY_REPORT.md | New contract, evidence and limitations record |
| backend/README.md | Added offline preparation workflow and limits |

Public Python entry points are `preview(source)`,
`prepare(source, output, exporter=None, exporter_version=None)` and
`verify(output)`. Safe failures use `CompatibilityError`. The CLI prints compact
JSON on success and safe actionable diagnostics with exit code 2 on failure.
It never includes removed cells in diagnostics or exposes native exception paths.

## Format and preservation contract

| Profile | Exact header authority | Removed fields |
| --- | --- | --- |
| standard_80 | src.preprocessing.schema.CSV_COLUMNS | Timestamp, Label |
| extended_84 | src.preprocessing.schema.CSV_COLUMNS_EXTENDED | Flow ID, Src IP, Src Port, Dst IP, Timestamp, Label |

Only exact ordered decoded headers are recognized. Canonical 78-column CSVs
already work with C2A and are not a conversion profile. No approximate matches,
reordered source headers, aliases (including CWR Flag Count), duplicates, missing
or extra headers are accepted. Names/order come from authoritative constants.
Header recognition identifies a documented layout; it does not authenticate an
exporter or its feature definitions/configuration.

The converter projects decoded text cells directly into MODEL_FEATURE_COLUMNS
order. It never passes cells through float formatting. Canonical serialization
is UTF-8 without BOM, all cells quoted, LF record endings. The decoded feature
text and record order/count remain identical; quoting/newline bytes may differ.
Signed zero, exponents, explicit signed non-finite values, surrounding whitespace
and -1 sentinels survive. Duplicate-looking records and incomplete tails remain.
There is no sorting, filtering, padding, feature computation, unit conversion,
imputation, fitting or transform application.

The unchanged production `csv_replay.parse_csv` validates the provisional
canonical bytes, including blank/nonnumeric/formula rejection and finite float32
range. Strict lexical CSV validation also rejects malformed quotes in discarded
metadata, which Python csv.reader alone can tolerate. Embedded headers and wrong
record widths fail explicitly. Metadata values are never logged or serialized.
No loader cleaning or scientific duplicate cascade is reused.

Marker-valued tests prove source-column selection and exact record correspondence.
Both layouts, BOM, LF/CRLF and quoted inputs yield the expected canonical header
and decoded rows. C2A parses them identically to directly prepared canonical
inputs. Thirteen-record tests retain all three trailing rows and map complete
window targets to original records 4 and 9. One-record numeric cases preserve
signed-zero sign bits, NaN/infinity and both negative sentinels. No inference is
needed to establish this correspondence.

## Sidecar and linkage

`<output>.provenance.json` is sorted compact canonical JSON plus one final LF.
It contains contract_version, source_profile, original_headers, source/output
byte_count and SHA-256, feature_order_sha256, ordered source_to_model_columns
(zero-based source_index/model_index/name), removed_metadata_fields,
source/output_record_count, explicit zero-based record_order_semantics,
rows_removed=0, units_changed=false, values_imputed=false,
transforms_applied=false, and exporter_declaration marked
user-declared-unverified. No declaration is required. A manifest_sha256 covers
the canonical record without that checksum field.

There are no removed IPs, Timestamp values, labels, raw rows, absolute input
paths, partition assignments, research identities, credentials or secrets.
Metadata privacy tests use distinctive comma/quote/multiline markers and prove
their absence from CSV, sidecar and error diagnostics.

`verify` checks exact canonical sidecar bytes, complete expected schema/mapping,
checksum, output size/hash/count/order and the production C2A parser. It rejects
changed CSV bytes, ordinary sidecar edits, duplicate JSON keys, extra nested
metadata, wrong types, altered mappings/flags/counts and noncanonical headers.
Determinism tests prove equal output/sidecar bytes for equal input/declarations
at different output paths. The CSV digest hashes exactly the bytes C2A uploads.

This is integrity linkage, not origin authentication. A test deliberately
changes the self-reported source digest and recomputes the checksum: verification
then passes, documenting the absence of a trusted external signature/digest.
Coordinated replacement of CSV/sidecar can likewise pass. The current replay API
does not consume the sidecar and is unchanged.

## Resource and filesystem behavior

Both source and canonical output are independently capped at 10 MiB; count is
capped at 25,000 records. There is no truncation or splitting. Source bytes are
read once into a locked bounded snapshot; decoding/row serialization remain
bounded by that cap. Projection writes one record at a time to temporary files.
Final validation uses the existing bounded 25,000-by-78 float32 parser allocation.
No corpus arrays/graphs are built. Sidecars are capped at 64 KiB. CSV field-size
limits are temporarily bounded by the source limit and restored. The CLI is a
single-invocation offline workflow, not a concurrent server service.

Only existing owner-controlled local Windows NTFS directories are supported;
the native drive/filesystem check rejects unsupported/mapped drives. Entire
repository input/output is rejected, covering its data/artifact/provenance/
screenshot/environment/recovery roots. UNC/device/ADS/reserved/trailing-dot or
space names, symlinks, junctions, reparse ancestors, hard-linked inputs, aliases,
existing outputs and existing sidecars are rejected. Parent components are
pinned top down with native handles denying write/delete sharing. Input and
temporary-file handles deny writes/deletion throughout processing/publication.
Actual junction and lock-substitution tests exercised Windows behavior.

Both temporaries are fsynced and validated before exclusive hard-link publication.
The sidecar is published first and CSV last. On ordinary processing/publication
failure, only new links/temporaries with this invocation's file identity are
removed; pre-existing/concurrent competitor destinations survive. Injected
failures before and after each link, during parse/metadata/fsync, and unsupported
filesystem checks leave no usable partial CSV or sidecar. Lock tests reject
source/provisional replacement and parent rename during publication.

This is not an atomic two-file crash transaction or a sandbox against arbitrary
same-user native code. Forced termination, power loss or OS-level rollback
failure may leave invocation temporaries/orphan sidecars and requires manual
inspection. Arbitrary reflective in-process mutation and hostile directory
owners are outside the supported trust boundary. No power-loss, second-user,
cross-platform or non-NTFS certification is claimed.

## Reproduced validation

Run using the existing Python 3.13.0 `.venv` and locked installed dependencies:

```powershell
.\.venv\Scripts\python.exe -B -m backend.tests.run_csv_compatibility_safe
.\.venv\Scripts\python.exe -B -m backend.tests.run_csv_compatibility_safe --parser-regressions
```

| Check | Final outcome |
| --- | --- |
| Focused C3A positive/negative/adversarial suite | 106 passed; no failures or skips |
| Fixed existing C2A rejection regressions | 27 passed; no failures or skips |
| Guard self-checks | Three planned filesystem denials before OS access per run |
| Unexpected guard violations | Zero on both final successful runs |
| Actual source byte limit | 10 MiB accepted; +1 byte rejected |
| Actual output byte overflow | Sub-10-MiB source yielding >10-MiB quoted output rejected |
| Actual record limit | 25,000 accepted; 25,001 rejected |
| Determinism, linkage and source/value preservation | Passed in focused suite |
| CLI preview/prepare/verify and safe error output | Passed in focused suite |
| Whitespace, Python AST syntax and scoped/full new-file review | Passed |

The fixed C2A subset invokes only test_rejected_headers_before_inference,
test_empty_malformed_and_archive_content, and
test_strict_numeric_rejects_text_and_overflow. Their synthetic TestClient
fixtures inject inert discovery/loaders. Every selected request rejects before
transform-state access or model initialization. Module definitions/imports were
inspected; the static local import closure includes 48 Python modules. Optional
regression imports define model/training code but execute no model operations.

The runner disables external pytest plugin autoload, bytecode/cache writes and
unrelated conftests; pytest files/logs remain in one external synthetic arena.
Audit hooks reject protected reads, all writes outside that arena and subprocess
execution. Native compatibility opens are separately confined. Transform loading,
application and fitting are patched to reject. Focused mode rejects scientific/
model/server imports; regression mode additionally rejects discovery, loading,
adapter construction/forward, Torch loading, optimizers, backward and loaded
training/pruning functions. This is an accidental-access guard, not a native
security sandbox.

Development failures were not concealed: initial runner setup overblocked pytest
temporary capture/log files, which were redirected into the arena. The first
complete suite had 96 passes/four failures: one fixture size arithmetic error,
two trailing-dot/space path normalization failures and guard removal of its own
repository-target junction. Corrections validate lexical names before Windows
normalization and permit removal of the junction entry without following its
target. The failed external arena was removed under the corrected write guard.
Later verifier/schema and explicit NTFS checks received new rejection coverage.
All final runs passed. Existing Starlette/httpx and Torch import deprecation
warnings remain; no dependency was installed or warning suppressed.

The full scientific suite, ordinary full replay suite, real-loader tests, server
startup, checkpoint loading, inference, preprocessing state reads/fitting,
training, pruning/ranking, official dataset traversal, evaluation and benchmarks
were deliberately excluded by the owner's explicit scope.

## Preservation, Git and next boundary

Compared an external pre-edit baseline against final working files: 431 of 432
pre-existing tracked files retain identical SHA-256/size/modification time; the
sole intended exception is backend/README.md. All 107 original untracked
protected files retain inventory, size, modification time and attributes.
Those local recovery/screenshot checks are metadata evidence, not fresh payload
hashes. No official dataset or ignored model checkpoint was opened for a new
content audit. Frozen tracked scientific evidence, release bindings/attestations,
closure/execution fixtures, current handoff, source, replay endpoints/parser,
adapters, frontend and dependency locks are byte-identical. Model artifacts and
E.12A policy were untouched; no model/data operation was performed.

The current authority files independently retain these physical SHA-256 values:

| File | SHA-256 |
| --- | --- |
| PROJECT_HANDOFF.md | 87658fc128fabe515d9409c9a7f4715f09ece02316d4d259f0f1edb9dc396cb9 |
| tests/fixtures/scientific_closure_handoff.md | cdbdf14742b9f7a7a7b982305d97e2b39d53aa652672c8a6f03ba657d44df230 |
| tests/fixtures/reconciled_execution_handoff.md | 58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5 |

Index bytes and branch/tag refs are unchanged. Staged diff is empty. Final
changed/new paths equal the six-file allowlist exactly; all untracked additions
were reviewed in full. Git diff --check and separate new-file whitespace checks
pass. The final worktree has the intended unstaged README edit and five new
unstaged files, alongside exactly the original 24 protected status entries.
No stage, commit, push, tag, reset, clean, stash or ignore-rule edit occurred.

PROJECT_HANDOFF.md remains unchanged. Its owner-decision boundary authorized
only the expressly supplied C3A milestone. The exact next task boundary is
independent owner review/audit of C3A. Any handoff update, frontend/API conversion,
other exporter, PCAP or live monitoring requires separate owner authorization.
No accuracy, calibration, predictive suitability or production-readiness claim
follows from compatibility. Stop here for independent owner review.

## Independent-audit correction validation — 2026-10-10

Only the safe runner and this report changed for the two authorized corrections.
The runner now guards os.listdir/os.scandir audit events with lexical/resolved
protected-root checks. Omitted/None paths mean cwd; relative, bytes and Path
inputs are checked. Unsupported forms, failed resolution and Windows device/
extended namespaces fail closed. Discovery of the repository root remains allowed.

Both corrected commands above passed: 106 focused tests and 27 fixed C2A
rejections, with no failures/skips or unexpected violations and no outer guard.
Each run recorded 42 planned enumeration denials (including real junction aliases)
and 12 permitted synthetic traversals, plus the original three read/write denials.
Patched scientific entry points recorded two focused/three regression denials.
Separate emitted repository-policy probes recorded 40 denials and allowed four
discovery events, without OS traversal. Native Windows integer directory arguments
raise TypeError before auditing; synthetic integer audit events were denied.

An initial self-check failed on bytes-versus-text filename comparison; the helper
was corrected before pytest. An additional synthetic namespace probe exposed the
retained Windows prefix, prompting explicit rejection before the final reruns.
The corrected handoff digest above authenticates unchanged working bytes exactly
equal to the HEAD blob. The other four C3A files remain byte-identical. Python-only
accidental-access limits remain; no scientific execution or Git publication occurred.
Stop for independent correction re-audit and owner review.
