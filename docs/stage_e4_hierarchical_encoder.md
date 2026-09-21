# Stage E.4 Hierarchical Graph Encoder

Stage E.4 consumes frozen Stage E.3 feature embeddings shaped `[B*78,32]`
and the two relations in the frozen Stage E.2 `HeteroData`. It does not
recompute feature embeddings or load Stage C/D data.

Eight learned 32-dimensional category identities are expanded from
`data["category"].category_id`. One learned 32-dimensional sample token is
expanded to `[B,32]`; sample numeric values, targets, indices, experiment/split
identifiers, paths, and provenance are not model inputs.

The feature-to-category and category-to-sample relations use separate
`GATv2Conv` modules with `(32,32)` input channels, 8 output channels per head,
4 concatenated heads, attention dropout 0.10, no self-loops, no shared source
and target projection, no edge features, and no PyG residual. Each block then
applies message dropout 0.10, adds the original target embedding, applies
`LayerNorm(32)`, and applies GELU. The resulting category and sample shapes are
`[B*8,32]` and `[B,32]`.

Normal forward returns only final sample embeddings. Optional attention mode
also returns, for both relations, the exact edge index used and the four
per-head PyG attention coefficients without head averaging.

Before either GATv2 module runs, both supplied relation tensors must equal the
exact graph-major canonical topology derived from the frozen upstream schema
used by E.2. Equality includes shape, `torch.long` dtype, source indices,
destination indices, and edge order. A same-graph redirect, destination swap,
cross-graph redirect, or edge reorder is rejected. E.4 references the frozen
schema tuples and does not define a second feature-to-category taxonomy.

For the installed PyG 2.8.0.post1, `GATv2Conv` returns attention coefficients
after its internal attention-dropout path. In `model.eval()`, attention dropout
is disabled, so returned coefficients are the normalized softmax values
actually used for aggregation; incoming coefficients sum approximately to one
for each destination and head. In `model.train()`, returned coefficients are
post-dropout, scaled values and are not required to sum to one. Future pruning
or attention collection must therefore run in evaluation mode. Attention is a
model weighting and ranking signal, not causal feature importance.

The measured E.4 parameter total is 4,768: 256 category-embedding, 32 sample
token, 2,176 per GATv2 relation, and 128 across two LayerNorms. Together with
the frozen 1,920-parameter E.3 encoder, the measured graph encoder total is
6,688 parameters.

Validation covers exact architecture and operation order, B=1 and B>1 shapes,
CPU/CUDA forward and backward, attention edge identity and destination-wise
evaluation-mode normalization, training-mode attention-dropout behavior,
adversarial exact-topology rejection before message passing, deterministic
evaluation, direct multi-graph isolation, target/metadata independence, source
immutability, all nine bounded real E1/E2/E3 split paths, and frozen Stage
C/raw-input integrity.

Stage E.4 contains no classifier, logits, loss, optimizer, training loop,
metric, pruning policy, physical node removal, or E.5 implementation.
