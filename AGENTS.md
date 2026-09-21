# Repository instructions

These instructions apply to the entire repository.

## Project and authority

- This repository implements **Dynamic Graph Optimization and Automated Feature Extraction for Efficient Intrusion Detection in Network Systems** using CSE-CIC-IDS2018.
- Use each source according to its purpose rather than treating them as one linear authority chain:
  - The current user's explicit scope and approvals define the work that is authorized.
  - `PROJECT_HANDOFF.md` defines the current verified project state and next safe boundary.
  - Applicable stage documentation defines the approved or intended contract.
  - Actual source and tests are the implementation truth.
  - Compact validation evidence records measurements and provenance; important claims must remain independently reproducible.
  - The original project report is background for intended terminology and architecture only.
- `PROJECT_HANDOFF.md` is the authority for dynamic state: the current frozen commit/tag, completed stages, test baseline, hashes, environment versions, approved architecture, and next safe implementation boundary. Do not copy those volatile values into this file or skills.
- Before changing code, inspect the live Git state, the handoff, the requested stage documents, the authoritative upstream implementation, and relevant tests. Treat summaries and prior PASS statements as claims until supported by the repository.
- If the handoff is missing or materially inconsistent with Git, source, tests, or evidence, stop and report the inconsistency. Do not guess which state is authoritative.

## Stage and scope control

- Previously checkpointed stages are frozen. If a requested change would alter a frozen contract, stop unless the user has explicitly reported a defect or approved a methodology change and re-audit.
- Implement only the requested stage or substage. Do not add downstream APIs, model components, training, evaluation, or release work as a convenience.
- Never begin the next stage automatically. End at the requested boundary and state the exact next safe boundary from the handoff.
- Avoid scope creep. Prefer the smallest defensible change that satisfies the approved contract.
- Never fabricate performance, efficiency, accuracy, or completion results. Distinguish proposed values and estimates from reproduced measurements.

## Stable methodology

- `T=5` means five consecutive clean **flow records** in deterministic capture order. It does not mean packets, and a window is not guaranteed to represent one conversation.
- The graph target is the label of the fifth/final flow record.
- The frozen logical graph contract is `78 feature -> 8 category -> 1 sample`: 87 type-local nodes and 86 directed structural edges.
- E1, E2, and E3 are distinct experiments. Preserve their separate membership, labels, preprocessing provenance, and reporting roles as defined by the handoff and stage documents.
- Preserve train/validation/test boundaries. Fit preprocessing, feature selection, calibration, thresholds, or other learned state only on the authorized training partition; never fit on validation or test data.
- The intended pruning direction is inference-time, attention-guided graph node/edge pruning. It is not scheduled global unstructured L1 pruning of model weights during training.

## Implementation rules

- Use the repository-approved isolated environment and locked dependencies identified by the handoff. Do not install packages globally.
- Reuse authoritative repository APIs for data access, split membership, mappings, topology, batching, provenance, and integrity. Do not create parallel loaders, mappings, topology definitions, or split logic.
- Preserve lazy and bounded behavior. Do not materialize full-corpus arrays or per-sample graph collections when a bounded batch or memory-mapped path exists.
- Validate inputs and provenance at trust boundaries and fail loudly on malformed, stale, cross-experiment, or inconsistent inputs.
- Preserve source/value order, ownership, dtypes, shapes, read-only guarantees, and integrity checks required by the relevant frozen contract.
- Do not hide functionality from a later stage inside an earlier-stage implementation.

## Verification

- Test the focused contract first with positive cases, then meaningful adversarial and negative cases.
- Run relevant frozen-stage regression tests and the complete project suite for implementation changes, using the repository's documented commands and environment.
- Reproduce applicable integrity checks. Do not replace semantic verification with a matching stored hash or passing test count.
- For deterministic reports, validate content, provenance, canonical serialization, and hashes before accepting or writing evidence.
- Finish with `git diff --check`, a scoped diff review, and `git status`.

## Git and repository hygiene

- Do not commit, tag, push, force-push, or rewrite history unless the user explicitly authorizes the specific release action.
- Never use broad staging such as `git add .`; stage only an inspected allowlist when checkpointing is explicitly authorized.
- Preserve unrelated user changes and stop if they overlap the requested work.
- Do not track raw or processed datasets, large arrays, generated graph collections, virtual environments, caches, logs, checkpoints, secrets, credentials, or machine-local recovery files.

## Completion report

Report the files created or modified, the implemented API or contract, focused and full-suite results, integrity results, Git diff/status, unresolved concerns, and the exact next safe boundary. Then stop.
