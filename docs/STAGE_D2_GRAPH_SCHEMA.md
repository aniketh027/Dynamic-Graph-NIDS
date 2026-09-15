# Stage D2 Hierarchical Graph Schema

Stage D2 freezes one deterministic logical hierarchy containing 78 feature
nodes, 8 category nodes, and 1 virtual sample readout node: 87 type-local nodes
in total. The reusable topology has 78 directed `feature -> category` membership
edges derived directly from D1 and 8 directed `category -> sample` edges, for 86
stored structural edges.

The same static template applies to every future sample and experiment; it is
not duplicated per Stage C window. Temporal information remains the ordered
five-value input `[x1, x2, x3, x4, x5]` for each future feature node, representing
five consecutive capture-order flow records. There are no temporal graph edges.

The machine-readable schema is `results/stage_d2/graph_schema.json`. It records
D1 taxonomy provenance plus deterministic SHA-256 protection for topology,
temporal semantics, and graph-target semantics. No graph framework objects,
temporal encoder, GATv2/HTGNN model, training, or pruning are implemented.
