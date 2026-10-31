"""Run small deterministic CPU training examples for Cellerator's NN modules.

This script is a composition demonstration over synthetic values. Its losses
are not biological evidence and do not validate a scientific objective.
"""
from __future__ import annotations

import copy

import torch

if __package__:
    from .reference_models import (
        EventHistoryReferenceModel,
        ExpressionReferenceModel,
        atac_binary_loss,
        attach_atac_view,
        attach_history_view,
        event_next_token_loss,
        masked_expression_mse,
    )
else:
    from reference_models import (
        EventHistoryReferenceModel,
        ExpressionReferenceModel,
        atac_binary_loss,
        attach_atac_view,
        attach_history_view,
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
    steps: int = 80,
) -> tuple[float, float]:
    """Train a paired ATAC observation head on an explicit history readout.

    The head is shared across histories. ``history_features`` is an explicitly
    exposed computational representation, not a required or canonical cell
    state. Each loss call obtains the base result once through ``predict``.
    """
    model_base = copy.deepcopy(base).cpu()
    torch.manual_seed(seed)
    model = attach_history_view(model_base, "atac", output_size=targets.shape[-1]).cpu()
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
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    history_atac_frozen_initial, history_atac_frozen_final = _train_history_atac(
        event_model, event_batch_data, history_atac_targets,
        train_base=False, seed=2028
    )
    history_atac_joint_initial, history_atac_joint_final = _train_history_atac(
        event_model, event_batch_data, history_atac_targets,
        train_base=True, seed=2028
    )
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
