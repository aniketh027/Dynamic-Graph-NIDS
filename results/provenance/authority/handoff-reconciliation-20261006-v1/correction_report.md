# E.12B handoff authority/provenance correction

The historical-to-current handoff authority gate is technically resolvable through a narrowly pinned reconciliation attestation. Live authority validation and semantic E.9 observation validation passed. This is implementation/correction evidence, not an independent E.12B audit PASS or an owner execution approval. E.12B remains unexecuted.

## A. Exact original E.9 handoff SHA-256

`0eb4f09d989a8c641736be9637f7677cdccf3e675d73425bbd0e7cf22a5df9f6`

This value was read from `results/stage_e9/stage_e9_contract.json`, field `e9_source_sha256["PROJECT_HANDOFF.md"]`. Read-only `git cat-file blob 2fa823943ba318fe32b2867bc8e2b1f799d78ac1:PROJECT_HANDOFF.md` reproduces the exact physical bytes/hash. No newline normalization was used. The older observation contract-commit handoff binding remains separately enforced by the unchanged E.9 observation validator; it was not replaced with this later E.9 report pin.

The frozen E.9 contract retains physical SHA `e569712855d068b97e5d5b1717faf7602e1a083b1016654b8b47ed003f0d2254` and canonical SHA `5c26c8b55069af09009f8bf716bf64ec3e0b82e19a6f7ec4f773374d6b688b0b`.

## B. Root cause and reproduction

Before this correction, `src/pruning/stage_e12b.py:prepare` invoked `stage_e12.prepare(root)`. In `src/pruning/stage_e12.py`, the loop over `frozen_upstream_source_sha256` and `e9_source_sha256` checked every recorded E.9 source digest against the current physical file, including the historical handoff pin. Therefore historical E.9 evidence integrity and current dynamic authority documentation were conflated for that one path.

The authorized current handoff was rejected with `PruningError: frozen upstream source changed`. The rejection was reproduced inside E.12B's test-locked file guard before changing source. Test arrays opened: zero; guard violations: zero. E.10/E.11 contain similar historical execution-path checks; their frozen source is untouched by this E.12B correction.

## C. Narrow reconciliation design

A separate immutable-by-physical-pin attestation binds the original handoff digest, exact current digest, historical commit, original E.9 contract, scope limited to `PROJECT_HANDOFF.md`, reconciliation reason, UTC timestamp/run ID, preserved owner request, and authority/review evidence hashes. It states that no frozen E.9 experimental artifact changed and that the current handoff supersedes the historical handoff ONLY as the live authority/status record.

The attestation explicitly records `e12b_execution_authorized=false`, `git_release_authorized=false` and `independent_review_required=true`. Owner authorization here means the exact current request for documentation/provenance reconciliation. It is not an execution approval or cryptographic owner signature.

The namespace is `results/provenance/authority/handoff-reconciliation-20261006-v1/`, outside the frozen experimental-stage namespaces. Its attestation was created exclusively, and remains physically pinned by E.12B source. Replacing both JSON and its sidecar does not update the independent source pin. An arbitrary future handoff edit requires a separately authorized correction and independent review.

## D. Attestation path and hashes

Path: `results/provenance/authority/handoff-reconciliation-20261006-v1/attestation.json`

- Physical SHA: `af4b21ea289932dfad648c6affd13a4cc9f91944bcc8d4eca7a94397d3422895`
- Canonical content SHA, excluding `attestation_sha256`: `4e338a1582834aea9a66a77a6ba2de4517ee28c6c61654701f1405ae01549d4b`
- Exact current handoff physical SHA: `58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5`
- Exact owner request physical SHA: `37be8f723992df529544714655b13d22b8f614dd323cedf7b6ef52a0122dc136`
- Run ID: `handoff-reconciliation-20261006-v1`
- Reconciliation timestamp UTC: `2026-10-06T12:53:37.691707+00:00`
- Validation physical SHA: `ef4e3144cdad85c4d026b7f9c5bad843ac903242737c016be55a5a8d19d6015f`
- Validation canonical content SHA: `935c3c1c65e3fdde3a0b328655f391c6f4274e1876c53c1b32e08deee6d402d0`
- Scoped correction diff SHA: `3f0d3fd74e81371ecbd965774319e646f0ca45840b314a9007cbd5828856192a`
- Guarded verification script physical SHA: `98723299625a1b3b2ba51886f82cf71e2b1f0babec5ca27feccf07c914528457`

The attestation lists exact physical hashes of the frozen E.9 contract/sidecar/observations, E.10 final report, recovered E.11 final report, E.12A report/freeze, `AGENTS.md` and the release-audit skill. Each is rehashed during authority validation.

## E. Exact file allowlist

Modified existing files only:

- `src/pruning/stage_e12b.py`
- `tests/test_stage_e12b.py`
- `docs/stage_e12b_final_comparison.md`

Added in the new authority namespace only:

- `attestation.json`, `attestation.sha256`, `owner_authorization.txt`
- `validation.json`, `validation.sha256`
- `correction.diff`, `guarded_validation.py`, `correction_report.md`

The E.12B CLI source is unchanged. `PROJECT_HANDOFF.md`, all E.9 evidence, E.10/E.11/E.12A source/documents/tests/artifacts, checkpoints, authority instructions and Git metadata remain unchanged. The existing ignored-results convention applies to the new namespace; no Git ignore rules were changed. Machine-local initial snapshots and diagnostic diffs are outside the repository.

## F. Exact provenance logic changed

`verify_authority_reconciliation(root)` accepts no caller-supplied hashes or fallback mode. It checks:

1. Independently pinned attestation physical bytes, canonical digest/serialization, and physical sidecar.
2. Explicit original/current hashes, exact one-file scope, original frozen commit/Git object, reason/supersession, owner request, and no-execution/no-release requirements.
3. Authority/review evidence physical hashes, original E.9 report physical/canonical identity, and the original historical handoff pin.
4. Historical handoff bytes read through `git cat-file blob`, against the original historical hash.
5. Current live `PROJECT_HANDOFF.md` against the exact new hash.
6. Every other frozen E.9/upstream source against its original hash. Only the handoff has the separate historical/live binding; no unrelated source exception exists.

To preserve frozen E.12A source, E.12B now calls its own compact `_prepare_upstream` replay of E.12A's preflight. It reuses existing E.9/E.10 APIs, retaining E.10 physical/canonical report checks, reviewed execution-source pins, 45 completed-record reconciliation, exact E2 seed membership, semantic E.9 observation validation, upstream evidence, environment, Stage C and train/validation/split integrity checks. It substitutes only the historical/live handoff comparison. A second authority validation after upstream preflight catches changes during that preflight.

The return contract adds `authority_reconciliation` with attestation path/physical/canonical hashes and both handoff hashes. Existing E.12B manifest/preparation revalidation binds that evidence. The rest of E.12B preparation, policy/freeze/checkpoint/baseline validation, reservation, test-locking, inference and benchmark contracts are unchanged. No global monkeypatch, provenance-disable flag, alternate loader/split/topology, policy change or frozen-source edit was introduced.

## G. Tests and allowed integrity validation

Final guarded combined run: **416 passed, 0 failed, 0 skipped, 1 deselected**. E.12B contributed **115 passing cases**, including **33 new correction regressions**. The earlier focused authority/upstream selection passed 31 cases before adding the final two regressions; the final combined run covers all 33.

Exact test arguments, using the approved `.venv` and `-B`:

```text
-m pytest -q -p no:cacheprovider
 tests/test_stage_e10_full_training.py
 tests/test_stage_e11_test_evaluation.py
 tests/test_stage_e11_failure_recovery.py
 tests/test_stage_e12_pruning.py
 tests/test_stage_e12b.py
 tests/test_stage_e9_batch_size_pilot.py
 -k "not test_run_candidate_short_real_cuda_timing_and_memory_path"
```

The excluded E.9 test performs real CUDA training/resource execution. The complete project suite was deliberately not run because it includes real predictive accesses and training forbidden by this request. No complete-suite PASS is claimed.

New regressions cover exact old-to-new acceptance; wrong old/current hash, commit, scope, owner scope and attempted execution/release grants; missing/corrupt/malformed attestation and sidecar; changed live handoff/owner evidence/historical Git object/E.9 evidence/unrelated source/AGENTS/release skill/review evidence; replacement of the old E.9 pin even after rebinding fixture evidence; six locked synthetic test-array access attempts; unchanged upstream-return semantics; retained non-handoff source/environment/report/split/input/evidence rejection; missing authority failure before loading; authority recheck after preflight; and stale implementation-hash rejection before reservation.

The guarded runner forbids writes to the real checkout and any real predictive `.npy` opens, while forbidding actual training, optimizer steps and backward calls. The combined run recorded **zero** such attempts. Synthetic tests operate on temporary fixtures; synthetic compaction/forward checks are permitted and were performed. No real pruning selection, benchmark or stage execution occurred.

Live read-only `verify_authority_reconciliation` and the existing `_validate_frozen_e9_evidence` both passed, including the original E.9 semantic observation validator. Actual test arrays were not hashed, loaded or opened. E.12B result/log namespaces and its test-consumption ledger remain absent. Full live `stage_e12b.prepare` was not run; complete executable-preflight readiness remains for independent review.

Only existing PyTorch/PyG deprecation warnings occurred. Syntax parsing and direct trailing-whitespace checks passed. `git diff --check` passed. Scoped no-index checks produced no whitespace findings; their exit code 1 indicates the expected content differences.

## H. Byte preservation and Git evidence

`validation.json` contains initial-and-final matching physical SHA values for **713 unchanged pre-existing files** and **108 unchanged Git metadata files**. Only the three allowlisted existing E.12B files changed.

| Preserved namespace | Files |
|---|---:|
| E.9 results | 3 |
| E.10 logs | 277 |
| E.10 results | 4 |
| E.11 logs | 5 |
| E.11 results | 29 |
| E.12A logs | 2 |
| E.12A results/freeze | 7 |
| Checkpoints, including historical upstream checkpoints | 75 |

The manifest also covers frozen source/tests/documents, authority files and other compact upstream evidence. It verifies the current handoff is still exactly `58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5`. The E.9 contract is still `e569712855d068b97e5d5b1717faf7602e1a083b1016654b8b47ed003f0d2254`.

Local HEAD and `origin/main` remain `2fa823943ba318fe32b2867bc8e2b1f799d78ac1`. Git refs, index, config, reflogs and other existing metadata are byte-identical to the initial snapshot. No commit/push/tag/reset/clean/stash/staging or history rewrite occurred. `git status` remains the pre-existing dirty/untracked authoritative checkout: tracked handoff modification predates this task; E.10-E.12B files were already untracked. New authority evidence is ignored by existing results rules. All other baseline file bytes and reported status entries were preserved.

## I-J. Four-file implementation-bundle identity

The bundle changed. It is SHA-256 of the canonical JSON map of the four source paths to their physical hashes.

- OLD: `0fd1177ef4128a7384f5ef4c5b8cecdf0e1469e0448499d6f5df9d279e930002`
- NEW: `75931dbcba5ec9d3bcd7d4e173ae9c25c96ac917dff00a6f8574b4f8ef0a4c80`

The old owner command hash must not be reused. The new digest identifies this correction for independent review; it is not approval to execute. Exact final per-file digests are in `validation.json`. The unchanged live handoff retains its earlier bundle/blocker description as a historical preparation record; this separate correction evidence supplements it without changing its pinned bytes.

## K. Gate disposition and remaining limits

**Technically resolvable: YES.** The exact original and current authority bindings verify on the live authoritative checkout, and synthetic preflight regressions retain every non-handoff rejection. The previous forced live comparison is removed only from E.12B's local upstream-preparation path and replaced by verified historical plus current bindings. Frozen E.10/E.11/E.12A paths remain unchanged.

This correction is not an independent review, release freeze or readiness approval. Independent review must still inspect the local compact-preflight replay and retained guards, reproduce provenance evidence, validate the updated bundle, and assess the entire E.12B implementation/protocol. Historical E.11 CUDA arithmetic compatibility and the scientific limitations already documented in E.12B remain applicable. No new predictive/efficiency result is claimed.

## L. Exact restart instructions for independent E.12B review

Start a fresh independent pre-execution reviewer in `C:\Project\ComputerNetworks`. Scope: read-only E.12B implementation/protocol review, incorporating this separately authorized correction. Preserve handoff/frozen evidence and perform no Git mutations or real test-array access.

1. Read `AGENTS.md`, `.agents/skills/nids-release-audit/SKILL.md`, pinned `PROJECT_HANDOFF.md`, E.12B documentation, this report, `attestation.json`, and `owner_authorization.txt`. Treat stored PASS claims as claims.
2. Review `correction.diff` and actual implementation/tests. Confirm the one-path authority exception, unchanged frozen E.12A source, preserved original E.9 pin and both explicit handoff hashes. Inspect retained source/data/provenance/reservation guards and scientific/reporting limitations.
3. Reproduce the guarded synthetic and live compact-integrity checks from the authoritative checkout:

```powershell
.\.venv\Scripts\python.exe -B results\provenance\authority\handoff-reconciliation-20261006-v1\guarded_validation.py
```

The runner emits results in a newly created system temporary directory and prints their path. It cannot write into the actual checkout or open actual predictive arrays. It performs no E.12B execution.

4. Independently recompute the four-file canonical bundle and compare with NEW above. Rehash `validation.json` against its sidecar and reproduce its preservation manifest. Confirm the handoff, frozen evidence/checkpoints, refs/index and absent E.12B consumption namespace.
5. Review the full preparation path separately if assessing executable-preflight readiness; this task verified the live authority gate and semantic E.9 evidence, not full live E.12B preparation. Keep real test predictive arrays inaccessible, and do not train, rerun E.11, select/prune a real policy or execute E.12B.
6. Report independent defects/verdict. After an independent PASS, the next boundary remains separately explicit owner authorization of one-time pruned-test execution. This correction supplies no such approval or execution command.
