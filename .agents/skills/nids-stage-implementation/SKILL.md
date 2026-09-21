---
name: nids-stage-implementation
description: Implement an explicitly requested numbered or subnumbered stage in this NIDS repository while preserving frozen upstream contracts and stopping before downstream work.
---

# NIDS stage implementation

Use this workflow only when the user asks to implement a project stage or substage. Authorization for one stage does not authorize the next stage, training, evaluation, release, or changes to frozen methodology.

## Establish the boundary

Before editing:

1. Read the root `AGENTS.md` and `PROJECT_HANDOFF.md`. Stop if either is missing or if the handoff materially conflicts with Git, source, tests, or evidence.
2. Read the requested stage documents and their acceptance contract. Inspect the actual upstream implementation and tests; do not rely on summaries alone.
3. Identify and record:
   - the frozen baseline and authoritative upstream API;
   - the exact files and behavior allowed by the requested stage;
   - frozen or downstream behavior that is forbidden;
   - focused tests, regression suites, integrity checks, and expected evidence.
4. Resolve the next safe implementation boundary from the handoff. If the requested stage is ambiguous, beyond that boundary, or not authorized, stop and report the discrepancy.

## Implement

- Make the smallest defensible implementation of only the requested contract.
- Consume authoritative upstream objects and provenance directly. Do not add competing loaders, feature/category mappings, topology, split logic, or batching authorities.
- Keep allocations bounded to the requested sample or batch and preserve lazy, read-only, ownership, ordering, dtype, and device contracts.
- Reject malformed, stale, cross-experiment, or provenance-inconsistent inputs explicitly.
- Do not smuggle later-stage layers, training, metrics, pruning, or evaluation into the change.

## Validate

Run, in order appropriate to the change:

1. focused positive contract tests;
2. meaningful adversarial and negative tests, including mutation, corruption, boundary, and provenance cases where applicable;
3. relevant frozen upstream regression tests;
4. the complete project suite;
5. repository integrity checks for frozen artifacts and raw inputs when the contract requires them;
6. deterministic evidence validation when the stage produces a report;
7. `git diff --check`, scoped diff review, and `git status`.

Do not accept a test count alone as proof. Verify the semantics claimed by the stage.

## Evidence

- Prefer compact deterministic machine-readable evidence when it materially supports the contract.
- Do not store large arrays, datasets, per-sample graphs, secrets, or absolute machine-specific paths.
- Hash the correct canonical content and accurately label canonical content hashes versus physical file-byte hashes.
- Include provenance sufficient to identify the frozen inputs without duplicating volatile handoff history.

## Git and completion

Do not commit, tag, or push unless the current user request explicitly authorizes that action. Implementation authorization alone is insufficient.

Report:

- files created or modified;
- the API and contract delivered;
- focused, adversarial, regression, and full-suite results;
- integrity and deterministic evidence results;
- Git diff/status and any pre-existing changes;
- unresolved concerns;
- the exact next safe boundary from `PROJECT_HANDOFF.md`.

Stop after the requested stage.
