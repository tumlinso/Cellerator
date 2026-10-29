"""Explicit, optional CPU quantization through a pinned ggml bridge.

This adapter stores numeric arrays with block quantization. It does not select
gene thresholds, infer an on/off encoding, or change a Cell's state implicitly.
Use :mod:`cellerator.state.quantization` for identity-aware per-feature state
codes and biological quantization policies.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


def _native():
    try:
        return import_module("._ggml_native", "cellerator")
    except ModuleNotFoundError as exc:
        if exc.name != "cellerator._ggml_native":
            raise
        raise ImportError(
            "The optional Cellerator ggml bridge is unavailable. Configure "
            "with CELLERATOR_ENABLE_GGML=ON and build the Python extension."
        ) from exc


def available() -> bool:
    """Whether the optional ggml bridge was included in this installation."""
    try:
        _native()
    except ImportError:
        return False
    return True


def supported_formats() -> tuple[str, ...]:
    """Return the installed bridge's block formats."""
    return tuple(_native().supported_formats())


def quantize(values: Any, format: str):
    """Quantize a CPU-convertible array explicitly and retain its shape.

    Gradient-tracked tensors are rejected: ggml storage quantization is a
    discrete, non-differentiable representation change.
    """
    if hasattr(values, "requires_grad") and values.requires_grad:
        raise ValueError("ggml quantization is not differentiable")
    if hasattr(values, "device") and str(values.device) != "cpu":
        raise ValueError("ggml quantization accepts CPU values only")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.isfinite(array).all():
        raise ValueError("ggml quantization requires finite values")
    return _native().quantize(np.ascontiguousarray(array), format)


def dequantize(values):
    """Reconstruct a NumPy float32 array from ``QuantizedValues``."""
    if not isinstance(values, _native().QuantizedValues):
        raise TypeError("values must be produced by cellerator.adapters.ggml.quantize")
    return values.dequantize()


__all__ = ["available", "dequantize", "quantize", "supported_formats"]
