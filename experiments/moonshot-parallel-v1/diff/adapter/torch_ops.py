"""First-order CPU adapter; reusable numerical definitions live in native code.

Torch owns tensor version validation. Native forward owns copied saved primals;
backward returns gradients and performs no parameter or state update.
"""
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from patch import ops as patch_ops
from product import ops as product_ops
from ports import ops as ports_ops


def _array(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a Torch tensor')
    if tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
        raise ValueError(f'{name} requires CPU float32')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} requires contiguous storage')
    return tensor.detach().numpy().copy()


def _result(array):
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))


def _cotangent(tensor):
    return _array(tensor.contiguous(), 'cotangent')


class _Patch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, r, policy):
        y, ctx.native_tape = patch_ops.forward(
            _array(x, 'x'), _array(l, 'l'), _array(r, 'r'), policy=policy)
        ctx.save_for_backward(x, l, r)
        return _result(y)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # Reading this property checks every saved Tensor version before native VJP.
        _ = ctx.saved_tensors
        values = patch_ops.vjp(ctx.native_tape, _cotangent(gradient))
        return *(_result(value) for value in values), None


class _Process(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, a, b):
        y, ctx.native_tape = product_ops.forward(
            _array(x, 'x'), _array(k, 'k'), a, b)
        ctx.save_for_backward(x, k)
        return _result(y)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        _ = ctx.saved_tensors
        dx, dk = product_ops.vjp(ctx.native_tape, _cotangent(gradient))
        return _result(dx), _result(dk), None, None


class _Ports(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, e, d, weights, widths, src, dst):
        y, ctx.native_tape = ports_ops.forward(
            _array(h, 'h'), _array(e, 'e'), _array(d, 'd'),
            _array(weights, 'weights'), widths, src, dst)
        ctx.save_for_backward(h, e, d, weights)
        return _result(y)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        _ = ctx.saved_tensors
        values = ports_ops.vjp(ctx.native_tape, _cotangent(gradient))
        return *(_result(value) for value in values), None, None, None


def _indices(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.dtype != torch.int64 or value.requires_grad:
            raise ValueError(f'{name} requires static CPU int64 indices')
        value = value.detach().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in 'iu' or result.ndim != 1:
        raise ValueError(f'{name} requires a one-dimensional integer sequence')
    return np.array(result, dtype=np.int64, copy=True)


def patch(x, l, r, policy='fp32'):
    """Square matrix patch tanh(L X) R; fp32 or stored_half_ste policy."""
    return _Patch.apply(x, l, r, policy)


def process(x, k, a, b):
    """Indexed two-input products with independent trainable coefficients."""
    return _Process.apply(x, k, _indices(a, 'a'), _indices(b, 'b'))


def ports(h, e, d, weights, widths, src, dst):
    """Flattened heterogeneous actor states and encoder/decoder port transport."""
    return _Ports.apply(h, e, d, weights, _indices(widths, 'widths'),
                        _indices(src, 'src'), _indices(dst, 'dst'))


def patch_jvp(x, l, r, dx, dl, dr, policy='fp32'):
    """Explicit native directional evaluation; this is not Torch forward AD."""
    _, tape = patch_ops.forward(_array(x, 'x'), _array(l, 'l'), _array(r, 'r'), policy=policy)
    return _result(patch_ops.jvp(tape, _array(dx, 'dx'), _array(dl, 'dl'), _array(dr, 'dr')))


def process_jvp(x, k, a, b, dx, dk):
    _, tape = product_ops.forward(_array(x, 'x'), _array(k, 'k'),
                                 _indices(a, 'a'), _indices(b, 'b'))
    return _result(product_ops.jvp(tape, _array(dx, 'dx'), _array(dk, 'dk')))


def ports_jvp(h, e, d, weights, widths, src, dst, dh, de, dd, dweights):
    _, tape = ports_ops.forward(_array(h, 'h'), _array(e, 'e'), _array(d, 'd'),
                               _array(weights, 'weights'), _indices(widths, 'widths'),
                               _indices(src, 'src'), _indices(dst, 'dst'))
    return _result(ports_ops.jvp(tape, _array(dh, 'dh'), _array(de, 'de'),
                               _array(dd, 'dd'), _array(dweights, 'dweights')))
