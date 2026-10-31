# NN foundation examples

These small compositions show how Cellerator's feature and relation modules fit
into ordinary PyTorch models, losses, autograd, and optimizers. They are
engineering examples, not validated biological models or recommended
scientific objectives.

## Expression features and an auxiliary ATAC view

`ExpressionReferenceModel` accepts a batch of registered integer feature IDs,
values, and an explicit input `observed` mask. It uses `FeatureEncoder` and
`RelationAttention` to reconstruct a per-feature measurement. The example makes
no implicit count normalization or feature-order assumption. Its masked MSE
uses only the explicitly supplied target mask; a mask that selects no targets
returns `None` so a caller can skip that optimization step.

If the source data is sparse AnnData, keep it sparse during selection and
convert only a deliberately bounded minibatch to dense values. Choose the
layer/raw matrix and feature/cell identities explicitly, and decide what zero,
missingness, and preprocessing mean for the experiment. The example does not
make those scientific choices.

`attach_atac_view` adds a trainable binary ATAC readout to the base model's
pooled feature result. `ModelWithViews.predict(batch, "atac")` evaluates the
base once and only the requested head. The head can be trained with a frozen
base or jointly with it using an ordinary PyTorch optimizer. This demonstrates
parameter sharing and a second observation view; it does not imply that RNA
features determine ATAC biology or that the pooled result is a unique cellular
state.

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
and frozen-base/joint-base ATAC BCE using small synthetic data. It exercises
ordinary PyTorch optimizer loops; the synthetic losses are a composition check,
not scientific evidence. An all-empty event batch has a start representation
but no targets, so the loop skips its optimizer step.

For native parameter owners, use the corresponding Cellerator guarded update
API. Cellerator's biological modules compose with ordinary layers; these
examples do not introduce a separate trainer system.
