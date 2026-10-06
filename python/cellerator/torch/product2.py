"""Optional Torch autograd adapter for Cellerator's CPU FP32 product2."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from cellerator import prepare_product2 as _prepare_native_product2


def _vector(value: Tensor, dtype: torch.dtype, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.device.type != "cpu" or value.dtype != dtype or value.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 CPU {dtype} tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use strided tensor layout")
    return value


def _admit(x: Tensor, k: Tensor, a: Tensor, b: Tensor) -> None:
    _vector(x, torch.float32, "x")
    _vector(k, torch.float32, "k")
    _vector(a, torch.int64, "a")
    _vector(b, torch.int64, "b")
    if a.numel() != k.numel() or b.numel() != k.numel():
        raise ValueError("k, a and b must have equal packet extents")
    if any(ids.numel() and bool(((ids < 0) | (ids >= x.numel())).any()) for ids in (a, b)):
        raise ValueError("packet index outside x extent")


def _array(tensor: Tensor, dtype) -> np.ndarray:
    return tensor.detach().contiguous().numpy().astype(dtype, copy=False)


def prepare_product2(a: Tensor, b: Tensor, input_count: int, *, structure_generation: int = 0):
    _vector(a, torch.int64, "a")
    _vector(b, torch.int64, "b")
    if a.numel() != b.numel():
        raise ValueError("a and b extents must match")
    return _prepare_native_product2(_array(a, np.int64), _array(b, np.int64),
                                    input_count, structure_generation=structure_generation)


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, k: Tensor, a: Tensor, b: Tensor, prepared=None) -> Tensor:
        _admit(x, k, a, b)
        if prepared is None:
            prepared = prepare_product2(a, b, x.numel())
        ctx.save_for_backward(x, k, a, b)
        ctx.primals = tuple(t.detach().contiguous().clone() for t in (x, k))
        ctx.indices = tuple(_array(t, np.int64) for t in (a, b))
        ctx.prepared = prepared
        y = prepared.forward(_array(ctx.primals[0], np.float32),
                             _array(ctx.primals[1], np.float32))
        return torch.from_numpy(y.copy())

    @staticmethod
    @once_differentiable
    def backward(ctx, g: Tensor):
        originals = ctx.saved_tensors
        _vector(g, torch.float32, "cotangent")
        if g.numel() != originals[1].numel():
            raise ValueError("cotangent extent differs from packet extent")
        gx, gk = ctx.prepared.vjp(
            _array(ctx.primals[0], np.float32), _array(ctx.primals[1], np.float32),
            _array(g, np.float32))
        return torch.from_numpy(gx.copy()), torch.from_numpy(gk.copy()), None, None, None


def product2(x: Tensor, k: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Evaluate native packets with first-order input and coefficient gradients."""
    return _Product.apply(x, k, a, b, None)


def product2_jvp(x: Tensor, k: Tensor, a: Tensor, b: Tensor,
                 dx: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate primal and native JVP as detached CPU FP32 results."""
    _admit(x, k, a, b)
    _vector(dx, torch.float32, "dx")
    _vector(dk, torch.float32, "dk")
    if dx.numel() != x.numel() or dk.numel() != k.numel():
        raise ValueError("JVP directions must match x and k extents")
    prepared = prepare_product2(a, b, x.numel())
    values = tuple(_array(t, np.float32) for t in (x, k, dx, dk))
    primal = prepared.forward(values[0], values[1])
    tangent = prepared.jvp(*values)
    return torch.from_numpy(primal.copy()), torch.from_numpy(tangent.copy())


class Product2Module(nn.Module):
    """Trainable coefficients with persistent indices and prepared topology."""

    def __init__(self, initial_coefficients: Tensor, a: Tensor, b: Tensor):
        super().__init__()
        _vector(initial_coefficients, torch.float32, "initial_coefficients")
        _vector(a, torch.int64, "a")
        _vector(b, torch.int64, "b")
        if a.numel() != initial_coefficients.numel() or b.numel() != a.numel():
            raise ValueError("coefficient and index packet extents differ")
        self.coefficients = nn.Parameter(initial_coefficients.detach().contiguous().clone())
        self.register_buffer("a", a.detach().contiguous().clone())
        self.register_buffer("b", b.detach().contiguous().clone())
        self.prepared = None
        self._prepared_indices = None

    def _prepare_for_input(self, input_count: int):
        indices = (self.a.detach().contiguous().clone(),
                   self.b.detach().contiguous().clone())
        stale = (self._prepared_indices is None
                 or self.prepared is None
                 or self.prepared.input_count != int(input_count)
                 or not torch.equal(indices[0], self._prepared_indices[0])
                 or not torch.equal(indices[1], self._prepared_indices[1]))
        if stale:
            self.prepared = prepare_product2(self.a, self.b, int(input_count))
            self._prepared_indices = indices

    def forward(self, x: Tensor) -> Tensor:
        _vector(x, torch.float32, "x")
        self._prepare_for_input(x.numel())
        return _Product.apply(x, self.coefficients, self.a, self.b, self.prepared)

    def jvp(self, x: Tensor, dx: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
        return product2_jvp(x, self.coefficients, self.a, self.b, dx, dk)
