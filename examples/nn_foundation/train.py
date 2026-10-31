"""Run small deterministic CPU training examples for Cellerator's NN modules.

This script is a composition demonstration over synthetic values. Its losses
are not biological evidence and do not validate a scientific objective.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from cellerator.biology import FeatureBinding, PredictionQuery
from cellerator.torch.supervision import (
    align_prediction_targets,
    categorical_target_columns,
    reduce_prediction_loss,
)

if __package__:
    from .reference_models import (
        EventHistoryReferenceModel,
        ExpressionReferenceModel,
        atac_binary_loss,
        attach_atac_view,
        attach_expression_view,
        attach_history_view,
        attach_event_view,
        event_next_token_loss,
        masked_expression_mse,
    )
else:
    from reference_models import (
        EventHistoryReferenceModel,
        ExpressionReferenceModel,
        atac_binary_loss,
        attach_atac_view,
        attach_expression_view,
        attach_history_view,
        attach_event_view,
        event_next_token_loss,
        masked_expression_mse,
    )


def _expression_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    patterns = torch.tensor([
        [0.10, 0.20, 0.30, 0.40, 0.50],
        [0.80, 0.70, 0.60, 0.50, 0.40],
        [0.20, 0.60, 0.30, 0.70, 0.50],
        [0.90, 0.10, 0.80, 0.20, 0.70],
    ], dtype=torch.float32)
    values = patterns.repeat(3, 1)
    feature_ids = torch.arange(values.shape[1], dtype=torch.long).expand(values.shape[0], -1)
    target_mask = torch.zeros_like(values, dtype=torch.bool)
    target_mask[:, (1, 4)] = True
    batch = {
        "feature_ids": feature_ids,
        "values": values,
        "observed": ~target_mask,
    }
    return batch, values, target_mask


def _event_batch() -> dict[str, object]:
    rows = [
        [0, 1, 0, 1, 0, 1],
        [],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
    flat: list[int] = []
    offsets = [0]
    for row in rows:
        flat.extend(row)
        offsets.append(len(flat))
    return {
        "event_ids": torch.tensor(flat, dtype=torch.long),
        "event_offsets": offsets,
        "sample_ids": tuple(f"event-cell-{index}" for index in range(len(rows))),
    }


def _train_expression(steps: int = 80) -> tuple[float, float, ExpressionReferenceModel,
                                                  dict[str, torch.Tensor], torch.Tensor]:
    batch, targets, target_mask = _expression_batch()
    model = ExpressionReferenceModel(vocabulary_size=5, width=12, heads=3).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

    def loss_value() -> torch.Tensor:
        prediction = model(batch)["expression"]
        loss = masked_expression_mse(prediction, targets, target_mask)
        if loss is None:
            raise RuntimeError("synthetic expression batch unexpectedly has no targets")
        return loss

    initial = float(loss_value().detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    return initial, final, model, batch, targets


def _train_events(steps: int = 100) -> tuple[float, float, EventHistoryReferenceModel,
                                                dict[str, object]]:
    batch = _event_batch()
    model = EventHistoryReferenceModel(
        event_vocabulary_size=2, width=12, heads=3, max_history=16
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    def loss_value() -> torch.Tensor | None:
        output = model(batch)
        return event_next_token_loss(output["logits"], output["targets"])

    initial_loss = loss_value()
    if initial_loss is None:
        raise RuntimeError("synthetic event batch unexpectedly has no targets")
    initial = float(initial_loss.detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        if loss is None:
            # An all-empty event minibatch has a start representation but no
            # supervised transition, so there is no optimizer update to make.
            continue
        loss.backward()
        optimizer.step()
    final_loss = loss_value()
    if final_loss is None:
        raise RuntimeError("event supervision disappeared during training")
    final = float(final_loss.detach())
    return initial, final, model, batch


def _train_atac(base: ExpressionReferenceModel, batch: dict[str, torch.Tensor],
                targets: torch.Tensor, *, train_base: bool, seed: int,
                steps: int = 80) -> tuple[float, float]:
    model_base = copy.deepcopy(base).cpu()
    torch.manual_seed(seed)
    model = attach_atac_view(model_base, peak_count=3).cpu()
    model.train()
    if not train_base:
        model.base.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.02,
    )

    def loss_value() -> torch.Tensor:
        return atac_binary_loss(model.predict(batch, "atac"), targets)

    initial = float(loss_value().detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    return initial, final


def _train_history_atac(
    base: EventHistoryReferenceModel,
    batch: dict[str, object],
    targets: torch.Tensor,
    *,
    train_base: bool,
    seed: int,
    output_binding: FeatureBinding | None = None,
    steps: int = 80,
) -> tuple[float, float]:
    """Train a paired ATAC observation head on an explicit history readout.

    The head is shared across histories. ``history_features`` is an explicitly
    exposed computational representation, not a required or canonical cell
    state. Each loss call obtains the base result once through ``predict``.
    """
    model_base = copy.deepcopy(base).cpu()
    torch.manual_seed(seed)
    model = attach_history_view(
        model_base, "atac", output_size=targets.shape[-1],
        output_binding=output_binding,
    ).cpu()
    model.train()
    if not train_base:
        model.base.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.02,
    )

    query = (
        None if output_binding is None else PredictionQuery(
            "atac", feature_ids=output_binding.identities,
            sample_ids=batch["sample_ids"],
        )
    )

    def loss_value() -> torch.Tensor:
        prediction = (
            model.predict(batch, "atac") if query is None
            else model.query(batch, query).values
        )
        return atac_binary_loss(prediction, targets)

    initial = float(loss_value().detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    return initial, final


def _train_bound_expression(steps: int = 60) -> tuple[float, float]:
    """Train an expression model through biological IDs and a selected query."""
    binding = FeatureBinding.from_ids(
        ("gene-A", "gene-B", "gene-C", "gene-D", "gene-E"),
        (2, 7, 11, 15, 19), domain="gene", model_capacity=21,
    )
    model = attach_expression_view(
        ExpressionReferenceModel(21, 12, heads=3, binding=binding)
    ).cpu()
    source_ids = ("gene-C", "gene-A", "gene-E", "gene-B", "gene-D")
    source_columns = torch.tensor([2, 0, 4, 1, 3])
    patterns, target_mask = _expression_batch()[1:]
    # Target/input arrays deliberately use the same permuted named axis.
    target_values = patterns.index_select(1, source_columns)
    withheld = torch.zeros_like(target_mask)
    withheld[:, (0, 3)] = True
    input_mask = ~withheld.index_select(1, source_columns)
    sample_ids = tuple(f"bound-cell-{index}" for index in range(patterns.shape[0]))
    batch = {
        "feature_identities": source_ids,
        "values": target_values,
        "observed": input_mask,
        "sample_ids": sample_ids,
    }
    query = PredictionQuery(
        "expression", feature_ids=binding.identities, sample_ids=sample_ids
    )
    alignment = binding.prepare(source_ids, query.feature_ids)
    aligned_mask = alignment.gather(withheld.index_select(1, source_columns), axis=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    def loss_value() -> tuple[torch.Tensor, torch.Tensor]:
        prediction = model.query(batch, query)
        pred_values, aligned_targets = align_prediction_targets(
            prediction, target_values, alignment, target_sample_ids=sample_ids,
        )
        aligned_targets_mask = aligned_mask
        losses = F.mse_loss(pred_values, aligned_targets, reduction="none")
        return reduce_prediction_loss(
            losses, unit="cell", valid_mask=aligned_targets_mask,
            sample_ids=sample_ids, cell_ids=sample_ids,
        )

    initial_loss, initial_count = loss_value()
    if not int(initial_count):
        raise RuntimeError("bound expression example unexpectedly has no targets")
    initial = float(initial_loss.detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, count = loss_value()
        if int(count):
            loss.backward()
            optimizer.step()
    final_loss, final_count = loss_value()
    if not int(final_count):
        raise RuntimeError("bound expression supervision disappeared during training")
    return initial, float(final_loss.detach())


def _train_bound_events(steps: int = 80) -> tuple[float, float]:
    """Train event logits after mapping biological labels to output columns."""
    binding = FeatureBinding.from_ids(
        ("gene-B", "gene-A"), (5, 12), domain="gene", model_capacity=14,
    )
    base = EventHistoryReferenceModel(
        2, 12, heads=3, max_history=16, binding=binding,
    )
    model = attach_event_view(base)
    rows = [
        ("gene-A", "gene-B", "gene-A", "gene-B"),
        (),
        ("gene-B", "gene-A", "gene-B", "gene-A"),
        ("gene-A", "gene-B", "gene-A", "gene-B"),
    ]
    labels = tuple(identity for row in rows for identity in row)
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    sample_ids = tuple(f"bound-history-{index}" for index in range(len(rows)))
    batch = {"event_ids": labels, "event_offsets": tuple(offsets), "sample_ids": sample_ids}
    query = PredictionQuery(
        "events", feature_ids=binding.identities, sample_ids=sample_ids,
        row_offsets=offsets,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    def loss_value() -> tuple[torch.Tensor, torch.Tensor]:
        prediction = model.query(batch, query)
        columns = categorical_target_columns(prediction, labels)
        unreduced = F.cross_entropy(prediction.values, columns, reduction="none")
        return reduce_prediction_loss(
            unreduced, unit="observation", sample_ids=prediction.sample_ids,
            row_offsets=prediction.row_offsets,
        )

    initial_loss, initial_count = loss_value()
    if not int(initial_count):
        raise RuntimeError("bound event example unexpectedly has no targets")
    initial = float(initial_loss.detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, count = loss_value()
        if int(count):
            loss.backward()
            optimizer.step()
    final_loss, final_count = loss_value()
    if not int(final_count):
        raise RuntimeError("bound event supervision disappeared during training")
    return initial, float(final_loss.detach())


def main() -> None:
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    expression_initial, expression_final, expression_model, expression_batch, values = (
        _train_expression()
    )
    event_initial, event_final, event_model, event_batch_data = _train_events()

    atac_targets = torch.stack((
        values[:, 0] > 0.45,
        values[:, 1] + values[:, 2] > 0.85,
        values[:, 3] > 0.45,
    ), dim=-1).to(dtype=torch.float32)
    atac_frozen_initial, atac_frozen_final = _train_atac(
        expression_model, expression_batch, atac_targets,
        train_base=False, seed=2027
    )
    atac_joint_initial, atac_joint_final = _train_atac(
        expression_model, expression_batch, atac_targets,
        train_base=True, seed=2027
    )
    history_atac_targets = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)
    history_atac_frozen_initial, history_atac_frozen_final = _train_history_atac(
        event_model, event_batch_data, history_atac_targets,
        train_base=False, seed=2028
    )
    history_atac_joint_initial, history_atac_joint_final = _train_history_atac(
        event_model, event_batch_data, history_atac_targets,
        train_base=True, seed=2028
    )
    peak_binding = FeatureBinding.from_ids(
        ("peak-A", "peak-B", "peak-C"), (1, 4, 9),
        domain="peak", model_capacity=11,
    )
    history_atac_bound_frozen_initial, history_atac_bound_frozen_final = _train_history_atac(
        event_model, event_batch_data, history_atac_targets,
        train_base=False, seed=2029, output_binding=peak_binding,
    )
    history_atac_bound_joint_initial, history_atac_bound_joint_final = _train_history_atac(
        event_model, event_batch_data, history_atac_targets,
        train_base=True, seed=2029, output_binding=peak_binding,
    )
    bound_expression_initial, bound_expression_final = _train_bound_expression()
    bound_event_initial, bound_event_final = _train_bound_events()
    results = {
        "expression_masked_mse": (expression_initial, expression_final),
        "event_next_event_ce": (event_initial, event_final),
        "atac_frozen_base_bce": (atac_frozen_initial, atac_frozen_final),
        "atac_joint_base_bce": (atac_joint_initial, atac_joint_final),
        "history_atac_frozen_base_bce": (
            history_atac_frozen_initial, history_atac_frozen_final
        ),
        "history_atac_joint_base_bce": (
            history_atac_joint_initial, history_atac_joint_final
        ),
        "history_atac_bound_frozen_base_bce": (
            history_atac_bound_frozen_initial, history_atac_bound_frozen_final
        ),
        "history_atac_bound_joint_base_bce": (
            history_atac_bound_joint_initial, history_atac_bound_joint_final
        ),
        "bound_expression_cell_mse": (bound_expression_initial, bound_expression_final),
        "bound_event_observation_ce": (bound_event_initial, bound_event_final),
    }
    for name, (initial, final) in results.items():
        if not (torch.isfinite(torch.tensor(initial)) and torch.isfinite(torch.tensor(final))):
            raise RuntimeError(f"{name} produced a non-finite loss")
        print(f"{name}: initial={initial:.6f} final={final:.6f}")
    for name in ("expression_masked_mse", "event_next_event_ce"):
        initial, final = results[name]
        if final >= initial:
            raise RuntimeError(f"{name} did not decrease in the deterministic example")


if __name__ == "__main__":
    main()
