# Stage E.3 Causal Feature-Node Encoder

Stage E.3 consumes the bounded Stage E.2 `feature.x` and
`feature.feature_id` tensors. `CausalTemporalEncoder.forward_sequence` maps
`[N,5]` ordered flow values to the retained `[N,16,5]` causal sequence, while
its normal forward path selects the fifth/final temporal position as `[N,16]`.

Each temporal layer uses an explicit two-position left pad followed by a
padding-zero, kernel-three convolution and GELU. The two layers are shared
across all feature nodes and have a five-position receptive field. Tests cover
every temporal prefix, zero future-input gradients, position-one reachability
from position five, and isolation of position four from changes at position
five.

`FeatureNodeEncoder` concatenates the temporal result with
`Embedding(78,8)`, then applies `Linear(24,32)`, `LayerNorm(32)`, GELU, and
`Dropout(0.10)`. Its measured total is 1,920 learned parameters. In evaluation
mode the same values and model produce deterministic embeddings; feature IDs
can distinguish otherwise identical value sequences.

Validation runs bounded real batches through the frozen D4 to D6 to E.2 path
for all nine E1/E2/E3 split pairs, confirms E1/E3 embedding identity for equal
inputs under the same model, exercises CPU and CUDA forward/backward, preserves
all source inputs, and rechecks frozen Stage C and raw-data hashes.

This stage contains no category embedding, sample token, GATv2 layer,
classifier, loss, optimizer, training, pruning, or E.4 implementation.
