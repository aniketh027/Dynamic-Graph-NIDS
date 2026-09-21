---
name: nids-release-audit
description: Independently audit a completed NIDS stage and, only when it passes and release actions are explicitly authorized, freeze, commit, tag, and push its checkpoint.
---

# NIDS release and checkpoint audit

Use this as an adversarial workflow when the user asks to audit, independently review, freeze, checkpoint, commit, tag, or push a completed stage. Audit authorization does not by itself authorize Git mutations.

## Establish authority and scope

1. Read the root `AGENTS.md` and `PROJECT_HANDOFF.md`. Stop if either is missing or materially inconsistent with Git, source, tests, or evidence.
2. Identify the exact stage, frozen predecessor, claimed deliverables, intended release allowlist, expected tests and integrity checks, and whether commit, annotated tag, and push are explicitly authorized.
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
- If the audit passes but checkpoint actions are not explicitly authorized, report PASS and stop without Git mutations.
- Only after an independent PASS and explicit authorization for the requested checkpoint actions may release work proceed.

## Authorized checkpoint

When and only when authorized:

1. Update `PROJECT_HANDOFF.md` and necessary stage documentation minimally with reproduced dynamic facts.
2. Rerun focused, regression, complete-suite, integrity, and deterministic-evidence verification after the documentation update.
3. Review the full diff and stage only the intended allowlist; never use `git add .`.
4. Inspect the staged diff, then create the authorized commit and annotated tag.
5. Push only the authorized refs without force.
6. Verify remote branch and peeled annotated-tag refs resolve to the intended commit.
7. Verify the final worktree is clean. If pre-existing unrelated changes prevent that, report them precisely rather than altering them.

## Report

Return the audit verdict; defects or concerns; files and contracts audited; reproduced tests, integrity checks, and hashes; release actions actually performed; local and remote refs when applicable; final Git status; and the next safe boundary. Stop after the authorized checkpoint.
