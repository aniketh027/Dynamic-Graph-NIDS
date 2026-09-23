# Stage E.5 HTGNN Classifier

Stage E.5 is the smallest complete inference-time model composition. An
`HTGNNClassifier` consumes one frozen E.2 `HeteroData` batch directly, runs
the unchanged frozen E.3 `FeatureNodeEncoder`, runs the unchanged frozen E.4
`HierarchicalGraphEncoder`, and applies one construction-time task head to the
final E.4 sample embedding only.

E1 and E2 instances use `Linear(32,15)` and return raw logits shaped
`[B,15]`. E3 instances use `Linear(32,1)` and canonically return raw binary
logits shaped `[B]`. Production forward contains no softmax or sigmoid.
Separate E1, E2, and E3 instances are required; the task is fixed when an
instance is constructed, parameters are not shared, and no automatic weight
transfer is implemented.

Construction-time task identity is bound to one of three internal concrete
model types and resolved through an immutable type-to-contract mapping. It is
not read from a mutable instance `_task` string. Injecting task or output-width
attributes into an instance therefore cannot turn an E1 model into E3, or
otherwise change its authorized output contract.

Normal classifier-head replacement is rejected after construction. This is
defense in depth rather than a claim that the entire PyTorch module is
immutable: every production forward independently requires the resolved head
to be the registered module, exactly `nn.Linear`, with the task-authorized
`in_features`, `out_features`, weight shape, bias shape, and returned logit
shape. Low-level structural corruption therefore fails loudly before E.3 or
E.4 runs; the model never silently adapts its task to a replacement head.
Numerical `Parameter` values remain trainable, retain gradients, and can be
updated normally. Standard `state_dict()` keys and head tensor shapes remain
available for future checkpoint work, but checkpoint saving/loading is not
implemented in this stage.

## Supported integrity boundary

The required invariant is construction-time task/output immutability through
the supported public model API. Supported use cannot reassign `model.task` or
replace `model.head` to silently change output width. Injected shadow `_task`
attributes are non-authoritative, and the runtime head validator continues to
reject detectable malformed or wrong-width heads before E.3 or E.4 executes.

This is not a claim that an `nn.Module` is secure against arbitrary Python
code executing in the same process. Deliberate reflective or interpreter-level
mutation is outside the supported API and outside the E.5 integrity threat
model, including:

- assigning or `object.__setattr__`-assigning `__class__`;
- directly rewriting `nn.Module._modules`;
- monkeypatching forward or validation methods;
- rebinding module-level globals, constants, mappings, imported modules, or
  functions;
- mutating class dictionaries; and
- equivalent arbitrary in-process Python object manipulation.

Some isolated internal mutations, such as a head-only `_modules` rewrite, are
still detected by the current runtime validator as defense in depth. That
useful detection does not expand the contract into resilience against combined
`__class__`, `_modules`, method, or global monkeypatching. Such operations
demonstrate arbitrary same-process code execution, not a predictive-model
contract defect, and are not E.5 acceptance requirements.

The private `_encode_sample_embeddings` diagnostic path is the same complete
E.3-to-E.4 hierarchy used by production forward. It exists to verify the
pre-head contract without exposing a route around graph hierarchy. The
classifier head receives only its `[B,32]` result. Raw feature values, feature
or category pooling, targets, source indices, audit provenance, paths, and
experiment or split identity never enter the head or any predictive feature.

With `return_attention=True`, the model returns `(raw_logits, attention)`.
The attention object is passed through unchanged from E.4 and therefore keeps
the exact relation edge indices and raw four-head feature-to-category and
category-to-sample coefficients. No head averaging or new topology processing
is introduced. Evaluation mode is required when collecting attention, as
documented by the frozen E.4 contract.

Measured model sizes are 7,183 parameters for E1 and E2 (6,688 frozen shared
encoder parameters plus a 495-parameter head), and 6,721 for E3 (6,688 plus a
33-parameter head).

Validation covers synthetic B=1, multi-graph, and odd-batch contracts; exact
readout and attention pass-through; target and metadata independence; source
immutability; CPU/CUDA forward and backward; task-instance isolation; E1/E3
pre-head equivalence under one encoder state; all nine bounded real D4 -> D6
-> E.2 -> E.3 -> E.4 -> E.5 paths; frozen regressions; full-suite tests; and
frozen Stage C/raw-input integrity. Adversarial coverage additionally includes
normal and low-level task/head corruption, combined E1-to-E3 corruption,
complete-model semantic topology redirects and edge reordering rejected before
message passing, multi-graph logit isolation, state-dictionary structure, and
numerical parameter trainability.

For multi-graph isolation, the CPU regression requires bitwise equality for
untouched graphs where that behavior is stable. CUDA/PyG float32 message
passing and scatter-style reductions may differ at tiny floating-point scale,
so unchanged CUDA graph logits are compared per graph with the fixed tolerances
`rtol=1e-5` and `atol=1e-6`. The intentionally perturbed graph must separately
show a maximum absolute logit response greater than `1e-3`, and a deliberate
`1e-3` change to untouched logits must fail the tolerance check. The tolerance
is therefore only a numerical comparison rule for unchanged graphs; it is not
permission for cross-graph leakage or a claim of CUDA bitwise determinism.

No model has been trained. Random untrained logits are contract-test outputs,
not meaningful predictions. This stage adds no loss, class weighting,
optimizer, scheduler, data loader, epoch loop, early stopping, checkpoint
selection, metric, threshold, training, or pruning behavior, and it stores no
model weights.
