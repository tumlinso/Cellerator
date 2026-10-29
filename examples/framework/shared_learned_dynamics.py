"""A shared learned transition advances two independent tensor-valued cells.

This example is optional because it requires the standalone Torch adapter.
"""
try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on optional Torch
    raise SystemExit("Install the optional Torch integration to run this example.") from exc

from cellerator.adapters.torch import ResidualStateUpdate, TorchModel
from cellerator.modeling import Cell


def main() -> None:
    torch.manual_seed(13)
    learned_rule = TorchModel(ResidualStateUpdate(state_size=3, hidden_size=5))
    shared_model = learned_rule.as_model()
    first = Cell(shared_model, torch.tensor([1.0, 0.0, 0.0]))
    second = Cell(shared_model, torch.tensor([0.0, 1.0, 0.0]))

    first_loss = first.advance(dt=0.2).state.square().sum()
    second_loss = second.advance(dt=0.2).state.square().sum()
    (first_loss + second_loss).backward()

    gradient_norm = sum(
        float(parameter.grad.norm())
        for parameter in learned_rule.parameters()
        if parameter.grad is not None
    )
    assert first.model is second.model
    assert not torch.equal(first.state, second.state)
    assert gradient_norm > 0
    print("independent outputs:", first.state.tolist(), second.state.tolist())
    print("shared parameter gradient norm:", round(gradient_norm, 6))


if __name__ == "__main__":
    main()
