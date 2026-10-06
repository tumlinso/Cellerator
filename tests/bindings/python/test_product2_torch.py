"""Optional CPU Torch integration tests for native Product2 topology updates."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from cellerator.torch import Product2Module


def test_product2_module_reprepares_when_state_dict_changes_indices():
    module = Product2Module(
        torch.tensor([4.0]), torch.tensor([0]), torch.tensor([1]),
    )
    x = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    first = module(x)
    old_prepared = module.prepared
    torch.testing.assert_close(first, torch.tensor([24.0]))

    state = module.state_dict()
    state["a"] = torch.tensor([1])
    state["b"] = torch.tensor([2])
    module.load_state_dict(state)
    updated = module(x)
    assert module.prepared is not old_prepared
    torch.testing.assert_close(updated, torch.tensor([60.0]))
    updated.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 20.0, 12.0]))
    torch.testing.assert_close(module.coefficients.grad, torch.tensor([15.0]))

    _, tangent = module.jvp(
        x.detach(), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5]),
    )
    torch.testing.assert_close(tangent, torch.tensor([83.5]))
