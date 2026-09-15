# Stage D1 Feature Taxonomy

Stage D1 deterministically maps the frozen 78-feature Stage C model-input order
into the eight owner-approved semantic groups: Context, Time, Packets and size,
Rates and ratios, Flags, Bulk, Subflow, and Window. The Stage C
`feature_names.json` artifact is validated against the matching canonical schema
before the manifest is generated.

The taxonomy uses only approved feature-name semantics. It performs no learned
feature selection, fitting, scaling, correlation analysis, or inspection of
data values, labels, or split membership. The original report's illustrative
28-feature example is not enforced.

`results/stage_d1/feature_taxonomy.json` is the machine-readable manifest. Its
Stage C feature-order identity and complete taxonomy content are protected by
deterministic SHA-256 digests over canonical JSON. This stage does not construct
graph nodes or edges and does not implement any later Stage D work.
