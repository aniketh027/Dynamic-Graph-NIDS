# Stage E.2 provenance-validation performance correction

This correction changes only how `logical_batch_to_heterodata` reuses a
successful validation of the compact frozen D8 and E.1 provenance reports.
The first conversion in a process still reads, hashes, parses, and validates
the same artifacts as the frozen Stage E.2 implementation. No logical batch,
split content, label, graph, `HeteroData`, model tensor, or predictive value is
cached.

The process-local cache has exactly one entry. Its key binds the validation to
the resolved project root, approved provenance constants, approved report
hashes, and each artifact's resolved path, device and file identity, mode,
link count, size, nanosecond modification time, and nanosecond change/creation
time. A normal write, replacement, relink, resize, or timestamp change causes
a miss and full validation. The key is measured before and after cold
validation; a change during validation is rejected instead of cached.

The cache value is an immutable tuple containing only the validated identity
strings. Each caller receives a new dictionary, so mutation of returned audit
metadata cannot mutate cached authority. An `RLock` covers lookup and cold
validation, making concurrent misses produce one complete entry and preventing
partially initialized state. Replacing the single entry for another project or
artifact identity keeps memory bounded.

Per-batch checks are not cached. Every call still validates the exact D6 type,
provenance mapping, experiment and split, read-only array contract, ownership,
sample bounds, targets, and complete frozen topology before creating new
Torch-owned tensors. E1, E2, and E3 therefore share only artifact validation;
an incompatible experiment, split, provenance mapping, or batch is rejected
even when the artifact cache is warm.

The key detects changes visible through the recorded filesystem identity
fields. A privileged same-process or filesystem adversary that rewrites bytes
and restores every recorded identity field is outside this optimization's
supported cache-coherence boundary; detecting that case would require reading
and hashing the files on every conversion and would eliminate the authorized
optimization. All normal repository edits and replacements are covered, and
changed content is subjected to the original full validation path.

Deterministic evidence is written to
`results/stage_e2/provenance_cache_validation.json`. It records the cache
contract, exact frozen pre-correction output fingerprints for bounded real
batches from all nine experiment/split pairs, corrected cold/warm
fingerprints, Stage E.6 batch-32 contract facts, and frozen-data integrity.
Noisy before/after conversion, H2D, and forward measurements are kept
separately in `results/stage_e2/provenance_cache_observations.json`.

The validation command is:

```powershell
.\.venv\Scripts\python.exe scripts\validate_e2_provenance_cache.py
```

This correction does not implement a `DataLoader`, training, loss, optimizer,
epochs, early stopping, checkpoints, metrics, class weighting, batch-size
selection, pruning, or any Stage E.7 behavior.
