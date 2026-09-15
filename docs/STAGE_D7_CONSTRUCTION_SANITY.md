# Stage D7: Bounded Graph-Input Construction Sanity

D7 runs the approved Stage C → D4 → D6 graph-input path twice on five
deterministically selected real samples from each of the nine experiment/split
pairs. The selection rule is `[0, N//4, N//2, (3*N)//4, N-1]`, and batch sizes
1, 2, and 5 are exercised without random selection.

Exact feature values, temporal and feature order, int16 targets, the static D2
topology, batch ownership, experiment/split boundaries, and absence of
cross-sample edges are validated. Corresponding E1/E3 inputs and binary target
conversion are checked, while E2 is verified against its independent sources.

Canonical SHA-256 fingerprints cover each selected logical sample and batch.
Both passes reproduce the same selections, fingerprints, and topology identity.
The run keeps source memory maps and batch arrays read-only and allocates at
most five samples of numeric batch payload at once.

No PyG framework, model, encoder, training, loss, pruning, whole-corpus graph
materialization, or individual graph serialization is part of D7.
