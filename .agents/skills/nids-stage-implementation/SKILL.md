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

An explicit user authorization for the stage or substage activates the repository's standing authorization to commit and push that successfully validated implementation. It does not authorize any other stage or any stage/checkpoint tag. A later explicit task instruction such as `do not commit` or `do not push` overrides the standing default.

If any required implementation or validation step fails, do not commit, push, or tag. Report the blocker and leave the work available for correction. Never commit partially validated work merely because files changed.

After all required validation passes:

1. Inspect the current `HEAD`, `git status`, and full diff. Confirm that only intended files belonging to the authorized work are present and that no unrelated concurrent changes appeared. Two Codex chats must not simultaneously mutate or commit the same working tree; stop on unrelated changes rather than including or altering them.
2. Stage only the inspected intended paths explicitly. Never use `git add .`. Exclude datasets, secrets, temporary or debug files, caches, virtual environments, large unexpected artifacts, and model weights or checkpoints unless the authorized stage explicitly permits them.
3. Inspect the staged diff and confirm it is a natural, complete engineering boundary. Do not create empty, no-op, artificial, misleading, or partially validated commits.
4. Create one meaningful conventional commit using an appropriate prefix such as `feat:`, `fix:`, `test:`, `docs:`, `chore:`, or `refactor:`.
5. Push the resulting commit to `origin/main` without force, then verify that local `HEAD` equals `origin/main` and inspect the final worktree status.

The implementation commit means only that the authorized implementation boundary completed and passed its implementation validation. It does not mean the stage was independently audited or frozen. Do not create a stage/checkpoint tag in this workflow.

If an independent audit later finds a defect, do not amend or squash the implementation commit and do not rewrite history. Return to this workflow for only the authorized correction; after its validation passes, create and push a new meaningful correction commit.

Report:

- files created or modified;
- the API and contract delivered;
- focused, adversarial, regression, and full-suite results;
- integrity and deterministic evidence results;
- Git diff/status and any pre-existing changes;
- the commit message, local commit SHA, `origin/main` SHA, and final worktree status when the standing Git workflow ran;
- unresolved concerns;
- the exact next safe boundary from `PROJECT_HANDOFF.md`.

Stop after the requested stage.
