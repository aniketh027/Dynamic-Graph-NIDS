# Stage D3 Structural Validation

Stage D3 independently reconstructs and validates the approved D2 endpoint
template. The static hierarchy has 87 type-local logical nodes and 86 unique
directed structural edges arranged as `feature -> category -> sample`.

Endpoint-level validation proves that the hierarchy is one weakly connected
component and a directed acyclic graph. Every feature has exactly one path to
the virtual sample node and that path is exactly two hops; every category has
one direct one-hop path to the sample node. Degree, membership, uniqueness,
level, and endpoint-range checks are also enforced against D1/D2 provenance.

The deterministic report is `results/stage_d3/structural_validation.json`.
This validation does not construct actual Stage C graph samples and does not
implement PyTorch Geometric objects, temporal encoders, GATv2/HTGNN, training,
or pruning.
