# Stage D4 Read-Only Lazy Sample Adapter

Stage D4 opens each frozen Stage C experiment/split `X` and `y` artifact as a
read-only NumPy memory map. Indexing one sample exposes the exact transpose from
`X[i]` shape `(5, 78)` to `feature_x` shape `(78, 5)` without scaling, casting,
reordering, aggregation, or relabeling. The target remains the matching
`int16` value `y[i]`.

Feature rows remain in the frozen D1 order. The five columns remain consecutive
capture-order flow positions: column 0 is earliest and column 4 is the
fifth/final flow that supplies the target. These positions are flow records,
not packets and not necessarily one conversation.

D2 taxonomy and topology tuples are shared immutable metadata rather than
copied per sample. Graphs are not pre-materialized, no graph framework is used,
and no model, temporal encoder, training, or pruning exists at this stage.
