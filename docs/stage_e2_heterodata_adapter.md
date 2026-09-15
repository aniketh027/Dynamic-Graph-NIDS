# Stage E.2 D6-to-HeteroData adapter

Stage E.2 converts one validated D6 `LogicalGraphBatch` directly into one
already-batched CPU `HeteroData` via `logical_batch_to_heterodata`. Conversion
is bounded by the requested D6 batch and remains lazy at split scale. It does
not create a list of per-sample PyG graphs, access Stage C arrays, or introduce
another dataset, loader, or batching authority.

The `feature.x` tensor preserves D6 float32 values and graph-major feature and
temporal order exactly. `feature.feature_id` and `category.category_id` are
categorical `torch.long` identifiers and are not concatenated into numeric
features. The single sample node per graph stores `y`: E1/E2 targets become
`torch.long`, while E3 targets become exact binary `torch.float32` values. No
target is relabelled or one-hot encoded.

The adapter copies only the supplied read-only NumPy batch into Torch-owned
storage. The two Stage D relations and their already-offset endpoints are
copied unchanged; no self-loop, reverse, temporal, or convenience edge is
added. D6 ownership vectors are copied as each store's `batch` authority.
Derived PyG-compatible `ptr` vectors are also exposed and cross-validated to
make direct batching boundaries explicit without replacing D6 ownership.

Audit provenance is stored in the non-predictive global `e2_audit` metadata,
not in node features. Canonical storage is CPU. A downstream caller may use
`data.to("cuda")`, but device transfer is not part of conversion.

No model exists in E.2, and the adapter has zero learned parameters. TCN,
embeddings, GATv2, classifiers, losses, optimizers, and training remain outside
this stage.
