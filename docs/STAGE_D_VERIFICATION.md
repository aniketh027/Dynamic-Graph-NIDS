# Stage D Verification — Graph Input Layer

Stage D was independently audited against the frozen Stage C checkpoint and is
verified through D8. It defines a framework-independent logical graph-input
layer; it does not implement a graph neural network or report model results.

## Frozen Graph Contract

- One logical graph represents one Stage C `T=5` sample: five consecutive clean
  flow records in deterministic capture order. These are not packets and are
  not guaranteed to belong to one conversation.
- Each feature node exposes its ordered `[x1, x2, x3, x4, x5]` values. The graph
  target is the fifth/final flow label.
- The hierarchy is `78 feature -> 8 category -> 1 sample`: 87 type-local nodes
  and 86 directed structural edges (78 feature-to-category and 8
  category-to-sample). There are no reverse, same-level, temporal, correlation,
  or structural self-loop edges.
- The eight category counts are `2/23/24/5/12/6/4/2` in the order Context,
  Time, Packets and size, Rates and ratios, Flags, Bulk, Subflow, Window.

## D1-D8 Responsibilities and Approved Hashes

| Stage | Responsibility | Canonical SHA-256 |
| --- | --- | --- |
| D1 | Deterministic 78-feature taxonomy | `fe1332dc22727cf952dc1d2117fec46439294821e2a4654537d910df901fab7e` |
| D2 | Framework-independent 87-node/86-edge schema | `0950126394099acea7573e14357f8bd01b22eb91c2a9d0b4f9298ea208ecebea` |
| D3 | Structural, graph-theoretic, and provenance validation | `40f4e23e9e0937f5182ebcdb6d6363301012e05c98a90d8b3985968bd1e605bb` |
| D4 | Read-only lazy Stage C sample adapter | `ccd065556c533d4fd1cd2e4d821566611db9d2a5a9e7ea036c0cca3b3f6500d6` |
| D5 | E1/E2/E3 experiment-boundary validation | `1ddfa6caaaf33a04d8b4a973b39b3b815be9ef11904bd0b6e5bf91a95f898b16` |
| D6 | Bounded logical template batching | `5e9808efba9754dc40af9503a7ef7fc70cca8204e2ca24e72eac1620376e07ac` |
| D7 | Real-path bounded construction sanity and repeatability | `cc67f8cf87c93977b9b7e820ea57ee3fe989849a8c3e1222266482373a17a55b` |
| D8 | Deterministic lazy full-split planning and integration | `d1a757459062fe8ca08f0be5951a5a728898a20d28eafbb12f10ce4646047b1f` |

## Experiment and Construction Boundaries

E1 is the strict global capture-order multiclass diagnostic; E2 is the primary
within-class chronological multiclass experiment; E3 is the strict global
capture-order binary experiment with the same feature/window membership as E1.
Split sizes remain E1/E3 `2,183,472 / 311,924 / 623,851` and E2
`2,183,465 / 311,916 / 623,866`. E2 reporting eligibility remains metadata
only, and E2 preprocessing remains fitted on E2-train rather than E1-train.

D4 opens the frozen Stage C arrays as read-only NumPy memory maps and exposes
one exact `(78, 5)` transpose view at a time. D6 materializes only requested
batch-scale arrays, preserves caller index order, and offsets each topology
within its own sample. D8 stores sequential coverage authority in an immutable
tuple payload, retains final partial batches, and constructs D6 batches only on
demand. For canonical `B=128`, all nine splits cover 9,357,741 logical
positions in 73,110 ranges without shuffle, gaps, overlaps, duplicates, or
dropped indices.

## Verification Result

- D8 targeted suite: **44 passed**.
- Complete Stage D suite: **231 passed**.
- Complete project suite: **691 passed, 0 failed**.
- Frozen Stage C artifacts: **33/33** file-set and SHA-256 checks passed.
- Raw CSV corpus: **10/10** file-set and SHA-256 checks passed.

## Boundaries and Limitations

Stage D1-D8 contain no learned graph-input quantity. No validation/test-derived
feature selection, topology, correlation, threshold, normalization, embedding,
or pruning rule is introduced. Stage C remains the only fitted preprocessing
layer and retains its frozen train-only provenance.

Capture order is still a proxy for physical chronology because source
timestamps lack AM/PM. `T=5` is not proven optimal, windows are not guaranteed
conversations, repeated states can occur across partitions, and E2 is not one
global timeline. Predictive quality, latency, throughput, and memory improvement
remain unmeasured.

No PyTorch Geometric object, `HeteroData`, GATv2/HTGNN model, causal temporal
encoder, training loop, learned attention, or graph/weight pruning exists yet.
The next safe boundary is model/framework design against this frozen input
contract, with a separately approved implementation and evaluation plan before
training.
