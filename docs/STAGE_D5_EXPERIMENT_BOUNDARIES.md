# Stage D5 Experiment Boundaries and Provenance

Stage D5 validates that E1, E2, and E3 remain separate frozen Stage C
experiments while sharing one deterministic D1-D4 graph contract. Numeric X/y
sources are experiment- and split-specific; no competing loader or split logic
is introduced.

E1 and E3 have identical feature membership, confirmed independently for each
split by frozen X SHA-256 identity. Their targets intentionally differ: a
complete chunked read-only comparison confirms E3 is `0` for E1's Benign code
and `1` for every attack code. E2 uses its own split membership, X artifacts,
and independently E2-train-fitted preprocessing state.

Stage D1-D5 contain no learned graph quantity. Category assignment, topology,
and adapter behavior do not derive from validation or test data. Repeated
network states across split boundaries remain a documented corpus limitation;
they are not the same as split-index overlap.
