import pytest
import torch

from cellerator.experimental import ExecutionField
from cellerator.torch.product2 import Product2Module


def _field():
    first = Product2Module(
        torch.tensor([0.5, -1.25], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([1, 2], dtype=torch.int64),
    )
    second = Product2Module(
        torch.tensor([2.0], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    return ExecutionField((("first", first), ("second", second)), input_extent=3), first, second


def test_composed_prepared_field_matches_independent_torch_oracle_and_autograd():
    field, first, second = _field()
    x = torch.tensor([1.5, -2.0, 4.0], dtype=torch.float32, requires_grad=True)
    handles = field.prepared_handles
    assert all(handle is not None for handle in handles)

    y = field(field.bind(x))
    oracle = (x[0] * x[1] * first.coefficients[0]) * (x[1] * x[2] * first.coefficients[1]) * second.coefficients[0]
    torch.testing.assert_close(y, oracle.reshape(1))
    grad_x, grad_first, grad_second = torch.autograd.grad(
        y.sum(), (x, first.coefficients, second.coefficients)
    )
    ref_grad = torch.autograd.grad(
        oracle, (x, first.coefficients, second.coefficients)
    )
    for actual, expected in zip((grad_x, grad_first, grad_second), ref_grad):
        torch.testing.assert_close(actual, expected)
    assert field.prepared_handles == handles


def test_binding_reuse_and_rebind_preserve_prepared_handles_and_advance_generation():
    field, first, _ = _field()
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    binding = field.bind(x)
    handles = field.prepared_handles
    field(binding)
    field(binding)
    assert field.binding_generation == binding.generation
    assert field.prepared_handles == handles

    rebound = field.bind(torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32))
    assert rebound.generation == binding.generation + 1
    assert field.prepared_handles == handles

    optimizer = torch.optim.SGD(field.parameters(), lr=0.01)
    optimizer.zero_grad()
    field(rebound).sum().backward()
    optimizer.step()
    assert field.value_generation == 1
    assert field.prepared_handles == handles


def test_structure_mutation_requires_explicit_epoch_rebind_and_invalidates_old_binding():
    field, first, _ = _field()
    binding = field.bind(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    original_handle = field.prepared_handles[0]
    with torch.no_grad():
        first.a[0] = 2
    with pytest.raises(RuntimeError, match="rebind_structure"):
        field(binding)

    new_structure = field.rebind_structure()
    assert new_structure.epoch == 1
    assert field.structure_epoch == 1
    assert field.prepared_handles[0] is not original_handle
    with pytest.raises(RuntimeError, match="obsolete structure epoch"):
        field(binding)
    assert field(field.bind(binding.value)).shape == (1,)


def test_layout_and_extent_are_checked_at_binding():
    field, _, _ = _field()
    with pytest.raises(ValueError, match="rank-one strided CPU FP32"):
        field.bind(torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="extent"):
        field.bind(torch.ones(4, dtype=torch.float32))


def test_composition_checks_ordered_stage_extents():
    first = Product2Module(torch.ones(2), torch.tensor([0, 1]), torch.tensor([1, 2]))
    second = Product2Module(torch.ones(2), torch.tensor([0, 1]), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="stage 'second' indices require at least 3 input values.*provides 2"):
        ExecutionField((("first", first), ("second", second)), input_extent=3)
