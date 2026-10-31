# NN foundation examples

These small compositions show how Cellerator's feature and relation modules fit
into ordinary PyTorch models, losses, autograd, and optimizers. They are
engineering examples, not validated biological models or recommended
scientific objectives.

## Expression features and an auxiliary ATAC view

`ExpressionReferenceModel` accepts a batch of registered integer feature IDs,
values, and an explicit input `observed` mask. Its additive bound route accepts
named `feature_identities`, aligns values to a declared `FeatureBinding`, and
returns outputs in that binding's order. It uses `FeatureEncoder` and
`RelationAttention` to reconstruct per-feature measurements. The example makes
no implicit count normalization or feature-order assumption. Its legacy masked
MSE helper uses only the explicitly supplied target mask; a mask that selects
no targets returns `None` so a caller can skip that optimization step.

If the source data is sparse AnnData, keep it sparse during selection and
convert only a deliberately bounded minibatch to dense values. Choose the
layer/raw matrix and feature/cell identities explicitly, and decide what zero,
missingness, and preprocessing mean for the experiment. The example does not
make those scientific choices.

`attach_atac_view` adds a trainable binary ATAC readout to the base model's
pooled feature result. `ModelWithViews.predict(batch, "atac")` evaluates the
base once and only the requested head. With an output binding, the head uses
`FeatureReadout` so a `PredictionQuery` can request selected peak identities;
without one, an ordinary dense linear readout remains available. The head can
be trained with a frozen base or jointly with it using an ordinary PyTorch
optimizer. This demonstrates parameter sharing and a second observation view;
it does not imply that RNA features determine ATAC biology or that the pooled
result is a unique cellular state.

## Event histories

`EventHistoryReferenceModel` accepts flat event IDs plus row offsets. It shifts
each row for teacher forcing, prepends a learned start identity, keeps repeated
events, adds a learned occurrence-order embedding, and uses causal attention
within each row. Empty rows still have a start representation but contribute
no next-event target or supervised loss. Histories are not converted through a
lossy generic event/history adapter. This example omits physical time and
payloads; a model can add them as explicitly aligned inputs.

The model also returns the explicitly named `history_features` readout source:
one pooled learned representation per packed row, including a start-token
representation for an empty row. `attach_history_view` adds a shared auxiliary
head to that source. The same module can be bound to a generic
`Cell(data=event_batch)` handle; `cell.predict("atac")` passes its opaque event
batch through the view wrapper without materializing a separate Cell state.

The example is a next-event prediction task. It does not claim that a discrete
event stream is a complete or identifiable cell state, and it does not
differentiate through discrete sampling.

## Training

Run `PYTHONPATH=python python examples/nn_foundation/train.py` for a deterministic CPU example from a source checkout.
It reports initial and final masked-expression MSE, next-event cross entropy,
and frozen-base/joint-base ATAC BCE for both expression and event-history bases.
It also trains bound expression and event routes using selected `Prediction`
outputs, identity-aligned targets, and explicit cell- or observation-weighted
loss reduction. All values are small synthetic data. The run exercises
ordinary PyTorch optimizer loops; the synthetic losses are a composition check,
not scientific evidence. An all-empty event batch has a start representation
but no targets, so the loop skips its optimizer step.

For native parameter owners, use the corresponding Cellerator guarded update
API. Cellerator's biological modules compose with ordinary layers; these
examples do not introduce a separate trainer system.

## Named biological features and queries

`FeatureBinding` is an optional model-local declaration that separates
biological identity from embedding row and output-column position. For
example, a gene binding can preserve stable gene IDs while mapping them to
noncontiguous token rows such as `(4, 19, 31)`. An event model can use those
rows for input embeddings while its output class columns remain in the
explicit `binding.identities` order. Repeated event occurrences remain
repeated; only the feature axis is unique.

When a native `GeneVocabulary` is available, construct the binding with
`FeatureBinding.from_gene_vocabulary(vocabulary, model_capacity=...)` so its
canonical gene IDs, canonical indices, token IDs, and reference metadata are
snapshotted distinctly. Use `from_ids` for other feature domains or explicitly
declared model-local identities.

```python
from cellerator.biology import FeatureBinding, PredictionQuery
from examples.nn_foundation.reference_models import (
    ExpressionReferenceModel, attach_expression_view,
)

genes = FeatureBinding.from_ids(
    ("gene-A", "gene-B", "gene-C"), (4, 19, 31),
    domain="gene", model_capacity=32,
)
base = ExpressionReferenceModel(32, width=24, heads=3, binding=genes)
model = attach_expression_view(base)
batch = {
    "feature_identities": ("gene-C", "gene-A", "gene-B"),
    "values": measurements_in_the_same_order,
    "observed": observed_mask_in_the_same_order,
    "sample_ids": cell_ids,
}
prediction = model.query(
    batch,
    PredictionQuery("expression", feature_ids=("gene-A", "gene-C"),
                    sample_ids=cell_ids),
)
```

Query feature IDs label the returned output axis in the requested order. Sample
and window labels describe the caller's batch order; queries do not implicitly
select or reorder inputs. A non-`None` `conditioning` value changes model input
only through a conditioner explicitly registered on `ModelWithViews`.
`context_metadata` is descriptive metadata and does not condition the model.

The bound expression route aligns supplied named features to the binding before
encoding. `explicit_dense_anndata_minibatch(..., binding=genes)` can perform
that alignment while the selected AnnData matrix remains sparse; dense values
are made only for the explicitly bounded minibatch. The caller still chooses
layers, preprocessing, masks, and target semantics. No normalization or
interpretation is inferred.

## Event queries and auxiliary observations

The event model accepts biological IDs (the default) or explicitly declared
model/token IDs (`event_id_space="model"`). A binding's model indices address
embedding rows; its ordered biological identities name output classes. Use
`categorical_target_columns(prediction, event_labels)` to map event identities
to the columns actually present in a selected `Prediction`, then ordinary
unreduced cross-entropy and the explicit reduction helper. Never assume a
gene's vocabulary row, token ID, or output column are the same integer.

`attach_history_view` can attach a target-bound `FeatureReadout` to the
explicitly exposed `history_features` result. This supports paired ATAC BCE
training on event histories with either a frozen or trainable shared base.
The base result is evaluated once per query; only requested readout features
are computed by `FeatureReadout`. The example does not serialize history or
claim that the pooled representation is a unique biological state.

The expression reference model also accepts `num_programs>0` as an optional
learned interaction composition. These shared program queries attend to the
supplied feature set and send a separately composed residual update back. The
program identities are model-local and do not imply known pathways, cell types,
hard topology discovery, or molecular mechanisms. The default model uses no
programs.

## Preserving model meaning beside weights

Reconstruct architecture and callables in code, save weights with ordinary
PyTorch, and pair them with a semantic sidecar. Validate it before loading the
state dictionary:

```python
torch.save(model.state_dict(), "model.pt")
save_biological_manifest(model, "model.bio.json", weights_path="model.pt")

reconstructed = build_the_same_model()
manifest = load_biological_manifest("model.bio.json", weights_path="model.pt")
validate_biological_manifest(
    reconstructed, manifest, weights_path="model.pt",
)
reconstructed.load_state_dict(
    torch.load("model.pt", map_location="cpu", weights_only=True)
)
```

The sidecar records declared feature bindings, view names, output axes, and
weight pairing. It does not serialize model code, callables, cell histories,
conditioning values, or learned activations.
