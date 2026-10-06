"""Shared-support biology composition against independent Torch mathematics.

Native tests require the installed CUDA adapter; the CPU reference test checks
the declared expression, and is never used as a native execution substitute.
"""

from __future__ import annotations

from dataclasses import replace
import os

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from cellerator import SharedSupportSpec, mechanisms_available
from cellerator.torch import (
    Axis, BiologicalTensor, Identity, guarded_step, load_checkpoint, save_checkpoint,
)
from cellerator.torch.biology import SharedSupportRelation


def axis(base, size):
    return Axis(*(Identity(base + i, 1000 + base + i) for i in range(4)), size)


def support():
    # Edges 0 and 1 have the same endpoints but distinct persistent identities.
    return SharedSupportSpec(
        source_axis=axis(10, 3), target_axis=axis(20, 2), edge_axis=axis(30, 5),
        edge_ids=tuple(Identity(100 + i, 200 + i) for i in range(5)),
        src=(0, 0, 1, 2, 1), dst=(0, 0, 0, 1, 1),
    )


def reference(x, w, s, a, spec):
    """Independent additive edge law, including duplicate endpoint pairs."""
    columns = []
    z = x * s
    for target in range(spec.target_axis.extent):
        terms = [w[e] * z[:, source] for e, (source, dest) in
                 enumerate(zip(spec.src, spec.dst)) if dest == target]
        columns.append(sum(terms, torch.zeros_like(z[:, 0])))
    return torch.stack(columns, dim=1) * a


def native_available():
    return mechanisms_available() and torch.cuda.is_available()


if os.environ.get("CELLERATOR_REQUIRE_NATIVE") == "1":
    if not native_available():
        raise RuntimeError("biology qualification requires CUDA indexed mechanisms")

requires_native = pytest.mark.skipif(
    not native_available(), reason="requires Cellerator CUDA indexed mechanisms")


def relation(spec=None, initial=(0.0, 0.6, -0.4, 0.8, 0.3)):
    return SharedSupportRelation(
        spec or support(), torch.tensor(initial, device="cuda", dtype=torch.float32),
        max_batch=8, max_live_forwards=8, precision="f32")


def inputs(batch, *, device="cuda", dtype=torch.float32):
    x = torch.linspace(-0.9, 1.1, batch * 3, device=device, dtype=dtype).reshape(batch, 3)
    s = torch.linspace(0.2, 1.2, batch * 3, device=device, dtype=dtype).reshape(batch, 3)
    a = torch.linspace(-0.5, 0.9, batch * 2, device=device, dtype=dtype).reshape(batch, 2)
    s[0, 1] = 0
    a[0, 1] = 0
    return tuple(t.requires_grad_() for t in (x, s, a))


def test_declared_shared_support_expression_cpu_double_finite_differences():
    spec = support()
    x, s, a = inputs(3, device="cpu", dtype=torch.float64)
    w = torch.tensor([0., .6, -.4, .8, .3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda xx, ww, ss, aa: reference(xx, ww, ss, aa, spec),
                                    (x, w, s, a), eps=1e-6, atol=1e-6, rtol=1e-5)


@requires_native
@pytest.mark.parametrize("batch", [8, 3])
def test_native_forward_full_vjp_tail_zero_trainable_values_and_duplicates(batch):
    model = relation()
    x, s, a = inputs(batch)
    before = model.coefficients.detach().clone()
    generation = model.mechanism._native.generation
    xx, ww, ss, aa = (t.detach().double().requires_grad_() for t in
                       (x, model.coefficients, s, a))
    expected = reference(xx, ww, ss, aa, model.spec)
    actual = model(x, s, a)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
    cotangent = torch.linspace(.1, .9, batch * 2, device="cuda").reshape(batch, 2)
    (actual * cotangent).sum().backward()
    (expected * cotangent.double()).sum().backward()
    for got, want in zip((x, model.coefficients, s, a), (xx, ww, ss, aa)):
        torch.testing.assert_close(got.grad.double(), want.grad, rtol=2e-5, atol=2e-6)
    # A zero coefficient remains differentiable, as do zero activity values.
    assert model.coefficients.grad[0].abs() > 0
    assert s.grad[0, 1].abs() > 0
    assert a.grad[0, 1].abs() > 0
    torch.testing.assert_close(model.coefficients, before, rtol=0, atol=0)
    assert model.mechanism._native.generation == generation
    torch.testing.assert_close(model.coefficients.grad[0], model.coefficients.grad[1])


@requires_native
def test_native_broadcast_activity_and_missingness_mask():
    model = relation()
    x = inputs(3)[0]
    s = torch.tensor([.6, 0., .8], device="cuda", requires_grad=True)
    a = torch.tensor([1., .4], device="cuda", requires_grad=True)
    mask = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]], device="cuda")
    y = model(x * mask, s, a)
    expected = reference(x * mask, model.coefficients, s, a, model.spec)
    torch.testing.assert_close(y, expected)
    grad = torch.autograd.grad(y.sum(), (x, s, a, model.coefficients))
    ref_grad = torch.autograd.grad(expected.sum(), (x, s, a, model.coefficients))
    for actual, ref in zip(grad, ref_grad):
        torch.testing.assert_close(actual, ref)
    assert torch.equal(grad[0][mask == 0], torch.zeros_like(grad[0][mask == 0]))


@requires_native
def test_native_source_target_and_edge_permutation_covariance():
    spec = support()
    source_order, target_order, edge_order = [2, 0, 1], [1, 0], [4, 2, 0, 3, 1]
    inverse_source = [source_order.index(i) for i in range(3)]
    inverse_target = [target_order.index(i) for i in range(2)]
    inverse_edge = [edge_order.index(i) for i in range(5)]
    permuted = replace(
        spec, source_axis=replace(spec.source_axis, order=Identity(900, 901)),
        target_axis=replace(spec.target_axis, order=Identity(902, 903)),
        edge_axis=replace(spec.edge_axis, order=Identity(904, 905)),
        edge_ids=tuple(spec.edge_ids[e] for e in edge_order),
        src=tuple(inverse_source[spec.src[e]] for e in edge_order),
        dst=tuple(inverse_target[spec.dst[e]] for e in edge_order))
    original = relation(spec)
    reordered = relation(permuted, original.coefficients.detach()[edge_order])
    x, s, a = inputs(3)
    xp, sp, ap = (v.detach()[:, order].clone().requires_grad_() for v, order in
                  ((x, source_order), (s, source_order), (a, target_order)))
    y, yp = original(x, s, a), reordered(xp, sp, ap)
    torch.testing.assert_close(yp[:, inverse_target], y)
    cot = torch.tensor([.3, -.7], device="cuda")
    (y * cot).sum().backward()
    (yp * cot[target_order]).sum().backward()
    for left, right, inv in ((x, xp, inverse_source), (s, sp, inverse_source),
                             (a, ap, inverse_target)):
        torch.testing.assert_close(right.grad[:, inv], left.grad)
    torch.testing.assert_close(reordered.coefficients.grad[inverse_edge], original.coefficients.grad)


@requires_native
def test_shared_owner_multiple_uses_accumulate_and_publish_one_adam_update():
    first = relation()
    second = SharedSupportRelation.shared_view(first)
    assert second.coefficients is first.coefficients
    assert second.mechanism._native is first.mechanism._native
    model = nn.ModuleList([first, second])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    assert len(optimizer.param_groups[0]["params"]) == 1
    x, s, a = inputs(3)
    generation = first.mechanism._native.generation
    before = first.coefficients.detach().clone()
    w = before.clone().requires_grad_()
    loss = first(x, s, a).square().sum() + second(x * .7, s, a).sum()
    expected = reference(x, w, s, a, first.spec).square().sum() + reference(x * .7, w, s, a, first.spec).sum()
    loss.backward()
    expected.backward()
    torch.testing.assert_close(first.coefficients.grad, w.grad)
    torch.testing.assert_close(first.coefficients, before, rtol=0, atol=0)
    assert first.mechanism._native.generation == generation
    assert guarded_step(model, optimizer)
    assert first.mechanism._native.generation == generation + 1
    assert not torch.equal(first.coefficients, before)
    torch.testing.assert_close(first(x, s, a), second(x, s, a))


class ComposedBiology(nn.Module):
    def __init__(self):
        super().__init__()
        self.before = nn.Linear(3, 3, device="cuda")
        self.source_generator = nn.Linear(2, 3, device="cuda")
        self.target_generator = nn.Linear(2, 2, device="cuda")
        self.relation = relation()
        self.after = nn.Linear(2, 2, device="cuda")

    def forward(self, x, context):
        s = self.source_generator(context).sigmoid()
        a = self.target_generator(context).sigmoid()
        return self.after(self.relation(self.before(x), s, a))


@requires_native
def test_activity_generators_pre_post_layers_optimizer_and_checkpoint_next_step(tmp_path):
    torch.manual_seed(175)
    model = ComposedBiology()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    x, _, _ = inputs(8)
    context = torch.linspace(-.8, .9, 16, device="cuda").reshape(8, 2)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model(x, context).square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert guarded_step(model, optimizer)
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name]), name
    checkpoint = tmp_path / "shared-support.pt"
    save_checkpoint(checkpoint, model, optimizer)
    restored = ComposedBiology()  # separately prepared owner, same logical identities
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.1)
    assert restored.relation.mechanism._native is not model.relation.mechanism._native
    load_checkpoint(checkpoint, restored, restored_optimizer)
    torch.testing.assert_close(restored(x, context), model(x, context))
    for candidate, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        candidate(x[:3], context[:3]).square().sum().backward()
        assert guarded_step(candidate, opt)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(right, left, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(restored(x, context), model(x, context), rtol=2e-6, atol=2e-7)


@requires_native
def test_axis_identity_capacity_and_higher_order_fail_explicitly():
    model = relation()
    x, s, a = inputs(3)
    for values in ((BiologicalTensor(x, axis(500, 3)), s, a),
                   (x, BiologicalTensor(s, axis(500, 3)), a),
                   (x, s, BiologicalTensor(a, axis(500, 2)))):
        with pytest.raises(ValueError, match="identity"):
            model(*values)
    with pytest.raises(ValueError, match="batch|capacity"):
        model(*inputs(9))
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(model(x, s, a).sum(), (x, model.coefficients), create_graph=True)


def test_shared_support_context_has_restricted_effective_weight_family():
    # For an all-positive 2x2 support, diagonal source/target activity preserves
    # the cross ratio. An unconstrained context-dependent W need not preserve it.
    base = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    effective = torch.tensor([.7, 1.3])[:, None] * base * torch.tensor([1.1, .6])[None, :]
    unconstrained = base.clone()
    unconstrained[0, 0] *= 2
    cross_ratio = lambda matrix: matrix[0, 0] * matrix[1, 1] / (matrix[0, 1] * matrix[1, 0])
    torch.testing.assert_close(cross_ratio(effective), cross_ratio(base))
    assert not torch.isclose(cross_ratio(unconstrained), cross_ratio(base))
