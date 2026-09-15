# Stage D8: Full-Split Lazy Integration

D8 completes the graph-input layer with an immutable sequential batch planner
and a reiterable lazy D6 batch iterator. Full-split integration means complete
arithmetic coverage of every index from `0` through `N-1`; it does not mean
prebuilding graphs or loading a complete split into memory.

`SequentialBatchPlan` is a tuple-backed value type with empty instance slots.
Its authoritative tuple payload contains only dataset length and batch size; it
has no instance `__dict__`, private backing field, or nested mutable coverage
container. Normal assignment, `object.__setattr__`, and tuple item assignment
cannot replace those values. Batch count, final size, and boundaries are derived
from that representation-level immutable authority, so validated no-drop
coverage cannot be silently shortened later.

Every returned `BatchBoundary` is likewise tuple-backed with empty instance
slots. Its batch number, start, and stop reject normal assignment,
`object.__setattr__`, and tuple item assignment. Calling `plan.boundary()` again
always derives a fresh canonical boundary from the plan payload.

`LazyBatchIterator` is also frozen. Its dataset, plan, experiment, and split
cannot be directly rebound. Construction captures an immutable binding snapshot
covering experiment, split, length, relative artifact identities, D1-D7
provenance, resolved numeric sources, shapes/dtypes, memory-map identities, and
topology identity. The current binding is compared with that snapshot before
and after every D6 numeric batch construction, so deeper dataset mutation or a
bypassed attribute replacement fails loudly instead of yielding a mismatched
batch.

Plans use contiguous half-open ranges, never shuffle, never sample with
replacement, and never drop the final partial batch. Batch sizes 1, 32, and 128
validate the arithmetic contract; size 128 is used for bounded real first,
interior, penultimate, final, and adjacent-boundary construction checks across
all nine experiment/split pairs.

Numeric batches are built through D6 only on demand. Previously yielded batches
are not cached, there is no prefetching, and complete coverage is checked by
streaming batch boundaries rather than allocating a full index list. The D2
topology remains static and shared.

D8 introduces no PyG objects, model, temporal encoder, training, balancing
sampler, loss, pruning, or correlation computation. No sample is dropped and no
full-corpus graph collection is materialized.
