# Stage D6: Template-Based Batching

D6 combines requested `LazyGraphDataset` samples into a bounded, NumPy-only
`LogicalGraphBatch`. One batch belongs to exactly one experiment and one split.
Input indices retain caller order; empty, duplicate, negative, out-of-range,
Boolean, and other non-integer indices are rejected.

`feature_x` is a batch-scale `(B, 78, 5)` float32 copy and `targets` is a
batch-scale `(B,)` int16 copy. The frozen Stage C memory maps remain read-only,
and every returned batch array is also read-only, so batch mutation cannot
propagate to its source.

The canonical D2 topology remains shared. D6 expands its 78 feature-to-category
and 8 category-to-sample template edges only for the requested batch, using
graph-local offsets of `b*78`, `b*8`, and `b`. Ownership vectors and validation
enforce exactly `B*86` edges and zero cross-sample edges.

D6 has no PyG objects, model, temporal encoder, training, or pruning behavior.
