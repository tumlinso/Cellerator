"""Native CPU FP32 product2 packets, ``y[i] = k[i] * x[a[i]] * x[b[i]]``.

Load the standalone Cellerator product library through
``CELLERATOR_PRODUCT2_LIBRARY``. Gradients and explicit JVP use its C ABI.
Backward owns copied primals and never updates inputs or coefficients. CUDA,
double backward, torch.func/vmap and autocast are outside this adapter contract.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable


class _Binding(ctypes.Structure):
    _fields_ = [(field, ctype)
                for name in ("x", "k", "y", "g", "dx", "dk", "dy", "gx", "gk")
                for field, ctype in ((name, ctypes.c_void_p),
                                     (name + "_count", ctypes.c_uint64))] + [
        (phase + "_" + kind + "_generation", ctypes.c_uint64)
        for kind in ("structure", "value", "parameter")
        for phase in ("expected", "current")]


class _Native:
    def __init__(self, path: str):
        self.lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
        self.lib.ce_product2_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.ce_product2_create.restype = ctypes.c_int
        self.lib.ce_product2_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ce_product2_destroy.restype = None
        for operation in ("forward", "vjp", "jvp"):
            fn = getattr(self.lib, "ce_product2_" + operation)
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Binding)]
            fn.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status:
            raise RuntimeError(f"native product2 {operation} rejected binding (status {status})")

    def _call(self, operation: str, x: Tensor, k: Tensor, a: Tensor, b: Tensor,
              **directions: Tensor):
        context = ctypes.c_void_p()
        self._check(self.lib.ce_product2_create(x.numel(), k.numel(),
            a.data_ptr(), a.numel(), b.data_ptr(), b.numel(), 0,
            ctypes.byref(context)), "create")
        try:
            names = {"forward": ("y",), "vjp": ("gx", "gk"), "jvp": ("dy",)}[operation]
            outputs = {name: torch.empty_like(x if name == "gx" else k)
                       for name in names}
            binding = _Binding()
            for name, value in dict(x=x, k=k, **directions, **outputs).items():
                setattr(binding, name, value.data_ptr())
                setattr(binding, name + "_count", value.numel())
            self._check(getattr(self.lib, "ce_product2_" + operation)(
                context, ctypes.byref(binding)), operation)
            return tuple(outputs[name] for name in names)
        finally:
            self.lib.ce_product2_destroy(context)

    def forward(self, x, k, a, b):
        return self._call("forward", x, k, a, b)[0]

    def vjp(self, x, k, a, b, g):
        return self._call("vjp", x, k, a, b, g=g)

    def jvp(self, x, k, a, b, dx, dk):
        return self._call("jvp", x, k, a, b, dx=dx, dk=dk)[0]


_LIBRARIES: dict[str, _Native] = {}


def _native() -> _Native:
    path = os.environ.get("CELLERATOR_PRODUCT2_LIBRARY")
    if not path:
        raise RuntimeError("set CELLERATOR_PRODUCT2_LIBRARY to the standalone native product library")
    path = str(Path(path).resolve(strict=True))
    if path not in _LIBRARIES:
        _LIBRARIES[path] = _Native(path)
    return _LIBRARIES[path]


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


def _copy(value: Tensor) -> Tensor:
    return value.detach().clone(memory_format=torch.contiguous_format)


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, k: Tensor, a: Tensor, b: Tensor) -> Tensor:
        _admit(x, k, a, b)
        ctx.save_for_backward(x, k, a, b)
        ctx.primals = tuple(_copy(v) for v in (x, k, a, b))
        ctx.native = _native()
        return ctx.native.forward(*ctx.primals)

    @staticmethod
    @once_differentiable
    def backward(ctx, g: Tensor):
        # Access triggers Torch's version check for every original primal/index.
        originals = ctx.saved_tensors
        _vector(g, torch.float32, "cotangent")
        if g.numel() != originals[1].numel():
            raise ValueError("cotangent extent differs from packet extent")
        dx, dk = ctx.native.vjp(*ctx.primals, _copy(g))
        return dx, dk, None, None


def product2(x: Tensor, k: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Evaluate native packets with first-order input and coefficient gradients."""
    return _Product.apply(x, k, a, b)


def product2_jvp(x: Tensor, k: Tensor, a: Tensor, b: Tensor,
                 dx: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate primal and native JVP as detached CPU FP32 results."""
    _admit(x, k, a, b)
    _vector(dx, torch.float32, "dx")
    _vector(dk, torch.float32, "dk")
    if dx.numel() != x.numel() or dk.numel() != k.numel():
        raise ValueError("JVP directions must match x and k extents")
    values = tuple(_copy(v) for v in (x, k, a, b, dx, dk))
    native = _native()
    return native.forward(*values[:4]), native.jvp(*values)


class Product2Module(nn.Module):
    """Trainable packet coefficients with persistent index buffers.

    ``state_dict`` records coefficients and indices. Restore between completed
    backward passes; changing a saved tensor before backward raises a version
    error. Optimizers remain external and own all parameter updates.
    """
    def __init__(self, initial_coefficients: Tensor, a: Tensor, b: Tensor):
        super().__init__()
        _vector(initial_coefficients, torch.float32, "initial_coefficients")
        _vector(a, torch.int64, "a")
        _vector(b, torch.int64, "b")
        if a.numel() != initial_coefficients.numel() or b.numel() != a.numel():
            raise ValueError("coefficient and index packet extents differ")
        if any(ids.numel() and bool((ids < 0).any()) for ids in (a, b)):
            raise ValueError("packet indices must be nonnegative")
        self.coefficients = nn.Parameter(_copy(initial_coefficients))
        self.register_buffer("a", _copy(a))
        self.register_buffer("b", _copy(b))

    def forward(self, x: Tensor) -> Tensor:
        return product2(x, self.coefficients, self.a, self.b)

    def jvp(self, x: Tensor, dx: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
        return product2_jvp(x, self.coefficients, self.a, self.b, dx, dk)
