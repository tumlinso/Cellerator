"""Installed-package integration tests for the Cellerator mechanism adapter."""

from __future__ import annotations

import gc
import os
import weakref

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from torch import nn

from cellerator import mechanisms_available, prepare_mechanism
from cellerator.torch import (
    Axis, BiologicalTensor, Identity, Incidence, Mechanism, MechanismModule,
    MechanismSpec, OutputContribution, guarded_step, load_checkpoint,
    save_checkpoint,
)


def ident(n: int) -> Identity:
    return Identity(n, n + 1000)


def axis(base: int, extent: int) -> Axis:
    return Axis(ident(base), ident(base + 1), ident(base + 2), ident(base + 3), extent)


def specification(*, coefficient_order=(20, 21)) -> MechanismSpec:
    input_axis = axis(1, 3)
    output_axis = axis(10, 2)
    coefficient_axis = Axis(ident(20), ident(22 if coefficient_order == (20, 21) else 222),
                            ident(23), ident(24), 2)
    coefficient_ids = tuple(ident(i) for i in coefficient_order)
    # Two contributions assemble into y[0]. The second mechanism repeats x[1]
    # twice, exercising multiplicity and zero-safe product derivatives.
    mechanisms = (
        Mechanism(ident(30), ident(20), (
            Incidence(0, ident(40), 0), Incidence(1, ident(41), 1),
        ), (OutputContribution(0, ident(50), 0, ident(60), 1.0),)),
        Mechanism(ident(31), ident(21), (
            Incidence(0, ident(40), 1), Incidence(1, ident(41), 1),
        ), (OutputContribution(0, ident(50), 0, ident(60), -0.5),
            OutputContribution(1, ident(51), 1, ident(61), 1.0))),
    )
    return MechanismSpec(input_axis, output_axis, coefficient_axis,
                        coefficient_ids, mechanisms)


def _native_available() -> bool:
    return mechanisms_available() and torch.cuda.is_available()


if os.environ.get("CELLERATOR_REQUIRE_NATIVE") == "1":
    if not _native_available():
        raise RuntimeError(
            "CELLERATOR_REQUIRE_NATIVE=1 requires CUDA indexed mechanisms"
        )


requires_native = pytest.mark.skipif(
    not _native_available(), reason="requires Cellerator CUDA indexed mechanisms"
)


def make_module(spec=None, *, precision="f32", initial=(0.7, -0.3), max_batch=11):
    spec = spec or specification()
    values = torch.tensor(initial, device="cuda", dtype=torch.float32)
    return MechanismModule(spec, values, max_batch=max_batch,
                           max_live_forwards=8, precision=precision)


def reference(x, k, spec, *, mixed=False):
    if mixed:
        # Forward rounds to storage values; backward uses the declared identity
        # straight-through gradient rather than autograd's half-rounding gradient.
        xs = x + (x.to(torch.float16).float() - x).detach()
        ks = k + (k.to(torch.float16).float() - k).detach()
    else:
        xs, ks = x, k
    y = torch.zeros((x.shape[0], spec.output_axis.extent), device=x.device, dtype=xs.dtype)
    for m in spec.mechanisms:
        product = ks[spec.coefficient_ids.index(m.coefficient)]
        for arg in m.arguments:
            product = product * xs[:, arg.index]
        for out in m.outputs:
            y[:, out.index] += float(out.scale) * product
    return y


@requires_native
def test_f32_full_vjp_repeated_slots_and_additive_outputs():
    spec = specification()
    module = make_module(spec)
    x = torch.tensor([[0.0, 0.4, -0.2], [0.3, -0.5, 0.7]],
                     device="cuda", requires_grad=True)
    y = module(x)
    ref_x = x.detach().double().requires_grad_()
    ref_k = module.coefficients.detach().double().requires_grad_()
    expected = reference(ref_x, ref_k, spec)
    torch.testing.assert_close(y.double(), expected, rtol=2e-6, atol=2e-7)
    cotangent = torch.tensor([[0.3, -0.7], [1.1, 0.2]], device="cuda")
    (y * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(x.grad.double(), ref_x.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(module.coefficients.grad.double(), ref_k.grad, rtol=1e-5, atol=1e-6)


@requires_native
def test_input_and_output_permutation_covariance_including_full_vjp():
    original = specification()
    input_order = [2, 0, 1]  # new coordinate j represents old coordinate input_order[j]
    output_order = [1, 0]
    inverse_input = [input_order.index(i) for i in range(len(input_order))]
    inverse_output = [output_order.index(i) for i in range(len(output_order))]
    permuted_input_axis = Axis(
        original.input_axis.domain, ident(180), original.input_axis.geometry,
        original.input_axis.partition, original.input_axis.extent,
    )
    permuted_output_axis = Axis(
        original.output_axis.domain, ident(181), original.output_axis.geometry,
        original.output_axis.partition, original.output_axis.extent,
    )
    permuted_mechanisms = tuple(
        Mechanism(
            mechanism.identity, mechanism.coefficient,
            tuple(Incidence(arg.slot, arg.role, inverse_input[arg.index], arg.axis)
                  for arg in mechanism.arguments),
            tuple(OutputContribution(out.slot, out.role, inverse_output[out.index],
                                     out.assembly, out.scale, out.axis)
                  for out in mechanism.outputs),
        )
        for mechanism in original.mechanisms
    )
    permuted = MechanismSpec(
        permuted_input_axis, permuted_output_axis, original.coefficient_axis,
        original.coefficient_ids, permuted_mechanisms,
    )
    native = make_module(original)
    reordered = make_module(permuted, initial=(0.7, -0.3))

    x = torch.tensor([[0.2, -0.7, 0.4], [0.8, 0.1, -0.3]],
                     device="cuda", requires_grad=True)
    x_permuted = x.detach()[:, input_order].clone().requires_grad_()
    cotangent = torch.tensor([[0.3, -0.5], [0.8, 0.2]], device="cuda")
    y = native(x)
    y_permuted = reordered(BiologicalTensor(x_permuted, permuted_input_axis))
    torch.testing.assert_close(y_permuted, y[:, output_order])
    with pytest.raises(ValueError, match="axis identity"):
        reordered(BiologicalTensor(x_permuted, original.input_axis))

    (y * cotangent).sum().backward()
    (y_permuted * cotangent[:, output_order]).sum().backward()
    torch.testing.assert_close(x_permuted.grad[:, inverse_input], x.grad)
    torch.testing.assert_close(reordered.coefficients.grad, native.coefficients.grad)


@requires_native
def test_mechanism_rejects_higher_order_gradient_request():
    module = make_module()
    x = torch.rand((2, 3), device="cuda", requires_grad=True)
    y = module(x)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(y.sum(), (x, module.coefficients), create_graph=True)


@requires_native
def test_biological_axis_identity_is_checked_independently_of_shape():
    module = make_module()
    same_extent_wrong_identity = axis(70, module.spec.input_axis.extent)
    x = torch.zeros((2, module.spec.input_axis.extent), device="cuda")
    with pytest.raises(ValueError, match="axis identity"):
        module(BiologicalTensor(x, same_extent_wrong_identity))
    assert module(BiologicalTensor(x, module.spec.input_axis)).shape == (2, 2)


@requires_native
def test_mixed_forward_and_ste_gradients_match_stored_value_reference():
    spec = specification()
    module = make_module(spec, precision="mixed_f16", initial=(0.12345, -0.7123))
    x = torch.tensor([[0.3333, 0.0, -0.2], [0.7, -0.4, 0.1]],
                     device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_k = module.coefficients.detach().clone().requires_grad_()
    y = module(x)
    expected = reference(ref_x, ref_k, spec, mixed=True)
    torch.testing.assert_close(y, expected, rtol=0, atol=2e-6)
    cotangent = torch.tensor([[0.2, 0.8], [-0.3, 0.5]], device="cuda")
    (y * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(module.coefficients.grad, ref_k.grad, rtol=2e-3, atol=2e-3)
    module.zero_grad(set_to_none=True)
    x_half = torch.tensor([[0.15, -0.35, 0.8]], device="cuda",
                          dtype=torch.float16, requires_grad=True)
    y_half = module(x_half)
    y_half.sum().backward()
    assert x_half.grad.dtype == torch.float16
    assert module.coefficients.grad.dtype == torch.float32


@requires_native
def test_mixed_initial_coefficients_must_round_to_finite_half():
    with pytest.raises(ValueError, match="finite FP16"):
        make_module(precision="mixed_f16", initial=(65520.0, 0.1))


@requires_native
def test_shared_module_use_accumulates_before_one_guarded_adam_step():
    module = make_module()
    shared = MechanismModule.shared_view(module)
    assert shared.coefficients is module.coefficients
    assert shared._native is module._native
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-2)
    before = module._native.generation
    x1 = torch.rand((3, 3), device="cuda", requires_grad=True)
    x2 = torch.rand((5, 3), device="cuda", requires_grad=True)
    loss = module(x1).square().sum() + module(x2).square().sum()
    loss.backward()
    assert module.coefficients.grad is not None
    assert guarded_step(module, optimizer)
    assert module._native.generation == before + 1
    with torch.no_grad(), pytest.raises(RuntimeError, match="outside guarded_step"):
        module.coefficients.add_(1.0)
        guarded_step(module, optimizer)


@requires_native
def test_from_handle_reuses_exact_parameter_and_optimizer_gradient():
    spec = specification()
    handle = prepare_mechanism(
        spec, np.asarray([0.7, -0.3], dtype=np.float32), device="cuda:0",
        max_batch=4, max_live_forwards=4,
    )
    first = MechanismModule.from_handle(handle)
    second = MechanismModule.from_handle(handle)
    assert first.coefficients is second.coefficients
    assert first._native is second._native is handle
    optimizer = torch.optim.Adam([first.coefficients], lr=1e-2)
    before = handle.generation
    x1 = torch.tensor([[0.2, 0.4, 0.8]], device="cuda", requires_grad=True)
    x2 = torch.tensor([[0.5, -0.1, 0.3]], device="cuda", requires_grad=True)
    (first(x1).sum() + second(x2).sum()).backward()
    assert first.coefficients.grad is not None
    assert torch.isfinite(first.coefficients.grad).all()
    assert guarded_step([first, second], optimizer)
    assert handle.generation == before + 1


@requires_native
@pytest.mark.parametrize("precision", ["f32", "mixed_f16"])
def test_out_of_band_coefficient_value_write_blocks_forward_and_checkpoint(tmp_path, precision):
    module = make_module(precision=precision)
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-2)
    x = torch.rand((2, 3), device="cuda")
    module(x).sum().backward()
    with torch.no_grad():
        module.coefficients.add_(0.125)
    # Keep the Python module check out of the way to exercise the native
    # TensorImpl alias baseline as well.
    module.coefficients._cellerator_expected_version = module.coefficients._version
    with pytest.raises(RuntimeError, match="version changed outside guarded update"):
        module(x)
    with pytest.raises(RuntimeError, match="version changed outside guarded update"):
        save_checkpoint(tmp_path / f"{precision}.pt", module, optimizer)


@requires_native
def test_coefficient_mutation_before_first_forward_is_rejected():
    module = make_module()
    x = torch.rand((2, 3), device="cuda")
    with torch.no_grad():
        module.coefficients.add_(0.125)
    with pytest.raises(RuntimeError, match="changed outside guarded_step"):
        module(x)


@requires_native
def test_guard_rejects_native_owner_missing_from_module_set():
    module = make_module()
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-2)
    module.coefficients.grad = torch.ones_like(module.coefficients)
    with pytest.raises(ValueError, match="owner is absent"):
        guarded_step([], optimizer)


@requires_native
def test_nondefault_stream_and_tail_batch():
    module = make_module(max_batch=11)
    stream = torch.cuda.Stream()
    x = torch.rand((7, 3), device="cuda", requires_grad=True)
    with torch.cuda.stream(stream):
        y = module(x)
        y.sum().backward()
    stream.synchronize()
    assert y.shape == (7, 2)
    assert x.grad is not None and module.coefficients.grad is not None


@requires_native
def test_checkpoint_restores_values_and_adam_for_next_step(tmp_path):
    module = make_module()
    optimizer = torch.optim.Adam([module.coefficients], lr=2e-3)
    x = torch.rand((4, 3), device="cuda")
    module(x).sum().backward()
    guarded_step(module, optimizer)
    path = tmp_path / "mechanism.pt"
    save_checkpoint(path, module, optimizer)
    expected = module(x).detach()

    restored = make_module()
    restored_optimizer = torch.optim.Adam([restored.coefficients], lr=2e-3)
    load_checkpoint(path, restored, restored_optimizer)
    torch.testing.assert_close(restored(x), expected)
    module.zero_grad(set_to_none=True)
    restored.zero_grad(set_to_none=True)
    module(x).square().sum().backward()
    restored(x).square().sum().backward()
    guarded_step(module, optimizer)
    guarded_step(restored, restored_optimizer)
    torch.testing.assert_close(restored.coefficients, module.coefficients)
    torch.testing.assert_close(restored(x), module(x))


@requires_native
def test_checkpoint_aligns_adam_moments_by_logical_coefficient_id(tmp_path):
    source = make_module()
    source_optimizer = torch.optim.Adam([source.coefficients], lr=1e-3)
    x = torch.rand((5, 3), device="cuda")
    source(x).square().sum().backward()
    guarded_step(source, source_optimizer)
    path = tmp_path / "permuted.pt"
    save_checkpoint(path, source, source_optimizer)

    reordered = make_module(specification(coefficient_order=(21, 20)))
    reordered_optimizer = torch.optim.Adam([reordered.coefficients], lr=1e-3)
    load_checkpoint(path, reordered, reordered_optimizer)
    torch.testing.assert_close(reordered(x), source(x))
    source.zero_grad(set_to_none=True)
    reordered.zero_grad(set_to_none=True)
    (source(x) * torch.tensor([0.2, -0.4], device="cuda")).sum().backward()
    (reordered(x) * torch.tensor([0.2, -0.4], device="cuda")).sum().backward()
    guarded_step(source, source_optimizer)
    guarded_step(reordered, reordered_optimizer)
    torch.testing.assert_close(reordered(x), source(x), rtol=1e-6, atol=1e-7)


@requires_native
def test_mixed_checkpoint_overflow_is_rejected_before_target_mutation(tmp_path):
    source = make_module(precision="mixed_f16")
    source_optimizer = torch.optim.Adam([source.coefficients], lr=1e-3)
    path = tmp_path / "mixed.pt"
    save_checkpoint(path, source, source_optimizer)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["native_owners"][0]["coefficients"][0] = 65520.0
    torch.save(payload, path)

    target = make_module(precision="mixed_f16", initial=(0.25, 0.5))
    target_optimizer = torch.optim.Adam([target.coefficients], lr=1e-3)
    before = target.coefficients.detach().clone()
    with pytest.raises(ValueError, match="finite FP16"):
        load_checkpoint(path, target, target_optimizer)
    assert not target._native.poisoned
    torch.testing.assert_close(target.coefficients, before)


@requires_native
def test_composed_torch_model_fits_coefficients_and_roundtrips_next_step(tmp_path):
    spec = specification()
    batch_capacity = 64
    mechanism = make_module(spec, initial=(0.1, 0.1), max_batch=batch_capacity)
    before = nn.Linear(3, 3, device="cuda")
    after = nn.Linear(2, 2, device="cuda")
    with torch.no_grad():
        before.weight.copy_(torch.eye(3, device="cuda"))
        before.bias.zero_()
        after.weight.copy_(torch.eye(2, device="cuda"))
        after.bias.zero_()
    for parameter in (*before.parameters(), *after.parameters()):
        parameter.requires_grad_(False)
    model = nn.Sequential(before, mechanism, after)
    optimizer = torch.optim.Adam([mechanism.coefficients], lr=3e-2)
    x = torch.rand((batch_capacity, 3), device="cuda")
    truth = torch.tensor([0.9, -0.4], device="cuda")
    target = reference(x, truth, spec)
    initial_loss = (model(x) - target).square().mean().item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        assert guarded_step(model, optimizer)
    assert (model(x) - target).square().mean().item() < initial_loss * 0.1

    checkpoint = tmp_path / "composed.pt"
    save_checkpoint(checkpoint, model, optimizer)
    expected = model(x).detach()
    restored_mechanism = make_module(spec, initial=(0.0, 0.0), max_batch=batch_capacity)
    restored = nn.Sequential(nn.Linear(3, 3, device="cuda"), restored_mechanism,
                             nn.Linear(2, 2, device="cuda"))
    restored_optimizer = torch.optim.Adam([restored_mechanism.coefficients], lr=1e-3)
    load_checkpoint(checkpoint, restored, restored_optimizer)
    torch.testing.assert_close(restored(x), expected)
    for candidate in (model, restored):
        candidate.zero_grad(set_to_none=True)
    (model(x) * torch.tensor([0.3, -0.6], device="cuda")).sum().backward()
    (restored(x) * torch.tensor([0.3, -0.6], device="cuda")).sum().backward()
    guarded_step(model, optimizer)
    guarded_step(restored, restored_optimizer)
    torch.testing.assert_close(restored(x), model(x), rtol=1e-6, atol=1e-7)
    for left, right in zip(model[0].parameters(), restored[0].parameters()):
        torch.testing.assert_close(left, right)
    for left, right in zip(model[2].parameters(), restored[2].parameters()):
        torch.testing.assert_close(left, right)


@requires_native
def test_live_tape_blocks_update_and_each_tape_has_one_backward():
    module = make_module()
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-2)
    x = torch.rand((2, 3), device="cuda", requires_grad=True)
    y = module(x)
    with pytest.raises(RuntimeError):
        guarded_step(module, optimizer)
    y.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError):
        y.sum().backward()


@requires_native
def test_stale_saved_input_is_rejected():
    module = make_module()
    x = torch.rand((2, 3), device="cuda", requires_grad=True)
    y = module(x)
    with torch.no_grad():
        x.add_(1.0)
    with pytest.raises(RuntimeError):
        y.sum().backward()


@requires_native
def test_abandoned_forward_releases_native_reader():
    module = make_module()
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-3)
    before = module._native.generation
    x = torch.rand((2, 3), device="cuda", requires_grad=True)
    abandoned = module(x)
    del abandoned, x
    gc.collect()
    torch.cuda.synchronize()
    module.coefficients.grad = torch.zeros_like(module.coefficients)
    assert guarded_step(module, optimizer)
    assert module._native.generation == before + 1


@requires_native
def test_python_owner_binding_releases_with_dropped_module():
    module = make_module()
    binding_ref = weakref.ref(module._binding)
    handle_ref = weakref.ref(module._native)
    del module
    gc.collect()
    assert binding_ref() is None
    assert handle_ref() is None


@requires_native
def test_grad_scaler_skip_does_not_publish_generation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = make_module()
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    before = module._native.generation
    x = torch.rand((2, 3), device="cuda")
    scaler.scale(module(x).sum()).backward()
    module.coefficients.grad.fill_(float("inf"))
    assert not guarded_step(module, optimizer, scaler=scaler)
    assert module._native.generation == before


@requires_native
def test_partial_optimizer_failure_poisons_owner_and_whole_checkpoint_restores(tmp_path):
    module = make_module()
    optimizer = torch.optim.Adam([module.coefficients], lr=1e-2)
    original = module.coefficients.detach().clone()
    checkpoint = tmp_path / "before-partial-step.pt"
    save_checkpoint(checkpoint, module, optimizer)
    module.coefficients.grad = torch.ones_like(module.coefficients)
    x = torch.rand((3, 3), device="cuda")
    expected = module(x).detach()
    failure_stream = torch.cuda.Stream()

    def partial_failure():
        with torch.cuda.stream(failure_stream), torch.no_grad():
            module.coefficients[0].add_(0.25)
        raise RuntimeError("simulated partial update")

    optimizer.step = partial_failure
    with pytest.raises(RuntimeError, match="simulated partial update"):
        guarded_step(module, optimizer)
    assert module._native.poisoned
    load_checkpoint(checkpoint, module, optimizer)
    assert not module._native.poisoned
    torch.testing.assert_close(module.coefficients, original)
    torch.testing.assert_close(module(x), expected)
    del optimizer.step
    optimizer.zero_grad(set_to_none=True)
    module(x).square().sum().backward()
    assert guarded_step(module, optimizer)


def test_invalid_declarations_fail_before_native_construction():
    input_axis = axis(1, 3)
    output_axis = axis(10, 1)
    coefficient_axis = Axis(ident(20), ident(22), ident(23), ident(24), 1)
    coefficient_ids = (ident(20),)
    first = Mechanism(ident(30), ident(20), (Incidence(0, ident(40), 0),),
                      (OutputContribution(0, ident(50), 0, ident(60)),))
    second = Mechanism(ident(31), ident(20), (Incidence(0, ident(41), 1),),
                       (OutputContribution(0, ident(51), 0, ident(61)),))
    with pytest.raises(ValueError, match="assembly identity"):
        MechanismSpec(input_axis, output_axis, coefficient_axis, coefficient_ids,
                      (first, second))


@requires_native
def test_shared_coefficient_binding_and_unused_parameter_gradient():
    base = specification()
    shared_second = Mechanism(
        base.mechanisms[1].identity, base.coefficient_ids[0],
        base.mechanisms[1].arguments, base.mechanisms[1].outputs,
    )
    spec = MechanismSpec(base.input_axis, base.output_axis, base.coefficient_axis,
                         base.coefficient_ids, (base.mechanisms[0], shared_second))
    module = make_module(spec)
    x = torch.rand((4, 3), device="cuda", requires_grad=True)
    module(x).square().sum().backward()
    assert torch.count_nonzero(module.coefficients.grad[1]).item() == 0
    assert torch.count_nonzero(module.coefficients.grad[0]).item() > 0


def test_identity_words_preserve_full_uint64_range():
    from cellerator._declarations import _signed_word

    high = Identity((1 << 64) - 1, 1 << 63)
    assert [_signed_word(word) for word in high.words] == [-1, -(1 << 63)]
