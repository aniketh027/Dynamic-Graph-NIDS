---
name: nids-release-audit
description: Independently audit a completed NIDS stage and, only when it passes and release actions are explicitly authorized, freeze, commit, tag, and push its checkpoint.
---

# NIDS release and checkpoint audit

Use this as an adversarial workflow when the user asks to audit, independently review, freeze, checkpoint, or tag a completed stage. A read-only audit does not authorize a stage/checkpoint tag. An explicit request to freeze, checkpoint, or tag the named stage authorizes the release actions only after an independent PASS; the repository's standing Git authorization covers a meaningful commit and push for legitimate required audit-time file changes. It does not authorize implementation of another stage.

## Establish authority and scope

1. Read the root `AGENTS.md` and `PROJECT_HANDOFF.md`. Stop if either is missing or materially inconsistent with Git, source, tests, or evidence.
2. Identify the exact stage, frozen predecessor, claimed deliverables, intended release allowlist, expected tests and integrity checks, and whether the user explicitly authorized freezing or checkpointing the named stage. A stage/checkpoint tag is not authorized by a read-only audit request.
3. Treat implementation summaries, saved PASS fields, hashes, and passing test counts as claims—not proof.

## Audit independently

Inspect the actual source, tests, compact evidence, stage documentation, and Git diff. Reproduce the important invariants independently rather than merely rerunning the implementation's happy path.

Verify, as applicable:

- architecture, API, shapes, dtypes, ordering, topology, and stage boundaries;
- train/validation/test isolation and absence of validation/test fitting or leakage;
- exact source/value preservation, ownership, immutability, and experiment separation;
- provenance binding, canonical serialization, and accurate canonical versus physical hashes;
- lazy/bounded construction and absence of hidden full-corpus materialization;
- meaningful positive, negative, adversarial, and regression test quality;
- frozen artifact and raw-input integrity;
- repository hygiene, intended-file scope, and absence of downstream scope creep.

Attempt relevant corruption and mutation cases. When immutability or authority binding is claimed, test bypasses such as `object.__setattr__`, nested payload mutation, copied impostors, and runtime source rebinding where those attacks are meaningful.

A passing suite or matching stored hash is necessary evidence only when relevant; neither is sufficient for an audit PASS.

## Decide before release

- If an IMPORTANT or BLOCKING defect exists, do not commit, tag, or push. Report the exact defect, evidence, affected contract, and smallest safe fix boundary.
- If the audit passes but freezing or checkpointing the named stage is not explicitly authorized, report PASS and stop without Git mutations.
- Only after an independent PASS and explicit authorization to freeze or checkpoint the named stage may release work proceed. Configuration-only changes never receive a stage tag.
- If the audit finds a defect, do not amend, squash, or rewrite the original implementation commit. Return the defect to the implementation workflow; an authorized, validated correction must be a new meaningful commit before a fresh independent audit.

## Authorized checkpoint

When and only when authorized:

1. Update `PROJECT_HANDOFF.md` and necessary stage documentation minimally with reproduced dynamic facts.
2. Rerun focused, regression, complete-suite, integrity, and deterministic-evidence verification after the documentation update.
3. Inspect the current `HEAD`, worktree, and full diff. Ensure that no unrelated or concurrent changes appeared; two Codex chats must not simultaneously mutate or commit the same working tree. Stop rather than include another task's changes.
4. If legitimate audit-time documentation, handoff, or compact evidence changes are actually required, stage only the intended allowlist explicitly, never use `git add .`, inspect the staged diff, create one meaningful conventional commit, push it to `origin/main` without force, and verify that local `HEAD` equals `origin/main`.
5. If the audit produced no actual file change, do not create an empty, no-op, or audit-pass-only commit. Use the already verified implementation commit as the frozen commit.
6. Create the approved annotated stage/checkpoint tag at the verified frozen commit, then push the tag without force. Never create a tag after a failed audit.
7. Verify that remote `main` resolves to the local frozen commit and that the remote peeled annotated-tag ref resolves to that same commit.
8. Verify the final worktree is clean. If pre-existing unrelated changes prevent that, report them precisely rather than altering them.

## Report

Return the audit verdict; defects or concerns; files and contracts audited; reproduced tests, integrity checks, and hashes; release actions actually performed; local and remote refs when applicable; final Git status; and the next safe boundary. Stop after the authorized checkpoint.
