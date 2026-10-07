# Release provenance and byte preservation

The completed CSE-CIC-IDS2018 study is `e10-official-20261001-v1`. Its official execution provenance is supplied as the external sibling directory `ComputerNetworks-release-archive/v1.0-experimental-complete/`. The owner-created local archive is separate from Git; no public hosting/download location is claimed.

## Exact scope and integrity

The archive contains exactly 280 files: {'E.10': 270, 'E.11': 5, 'E.12A': 2, 'E.12B': 3}. [release/provenance_manifest.json](../release/provenance_manifest.json) enumerates every repository-relative logical path, archived path, size, physical SHA-256, stage and record category. Its bytes are identical to the external manifest. Selection is exactly the independent final release audit's `external-provenance-paths.txt`, whose physical hash is recorded in the manifest. Runtime logs stay Git-excluded; the 60 compact later result/authority artifacts are separate release candidates.

Manifest physical SHA-256: `f14f946271ed96fbef3a34a4b4e4bacba9a0a014061cd06208681fd801d2f3b7`.

[release/provenance_manifest.sha256](../release/provenance_manifest.sha256) repeats that binding. Verify the manifest hash first, then every archived file's physical bytes/size against its entry. The archived file set must equal the manifest set with no extras or omissions. There is no container/ZIP hash because no container was created. Dataset arrays and model checkpoints are excluded and require separate owner-supplied acquisition for any future authorized replay; this archive supports compact evidence/provenance inspection.

Preserve original logical paths on restore to a separate inspection checkout. Reject existing differing files; never convert text line endings. Acquisition/restoration grants no experimental execution, test-set reuse, or release approval.

## Historical, execution and closure authority

The original E.9 handoff remains authenticated by its historical Git blob. The unchanged reconciliation attestation binds that hash to the execution-era handoff. The current handoff is the later closure authority record. [release/handoff_closure_binding.json](../release/handoff_closure_binding.json) binds all three epochs without replacing any historical hash or granting execution authority.

The exact execution-era snapshot is retained in `tests/fixtures/reconciled_execution_handoff.md`; its physical hash equals the frozen attestation's execution pin. `tests/conftest.py` substitutes only that authenticated temporary handoff for the original E.12B authority fixture. The original four-file E.12B bundle, all validator constants, and experimental evidence stay unchanged. Separate release regressions reject corrupted closure/snapshot/attestation/sidecar/binding bytes and confirm that live closure documentation still fails the frozen execution gate.

## Git attributes

Hash-bound files use `-text -filter -ident -working-tree-encoding` in `.gitattributes`. Exact source/document/test paths follow recorded physical-hash dependencies; compact evidence namespaces, the audited official log scopes, and release manifest/snapshot bindings are protected as well. Ordinary unbound source/docs remain normal text. See the correction verification report for the exhaustive effective path list and clean/smudge filter comparisons. The index remains untouched; release staging/commit/tag/push requires a fresh independent audit and separate authorization.
