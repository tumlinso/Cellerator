"""Cellerator's thin Python bindings for native numerical operations."""
from __future__ import annotations

import weakref
import operator
from typing import Any

import numpy as np

from . import _native
from ._declarations import (
    Axis, Identity, Incidence, Mechanism, MechanismSpec, OutputContribution,
    SharedSupportSpec, constructor_data,
)

_handles: weakref.WeakSet[Any] = weakref.WeakSet()


def _index(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0 or result > maximum:
        raise ValueError(f"{name} must fit the supported unsigned integer range")
    return result


def prepare_product2(a: np.ndarray, b: np.ndarray, input_count: int, *,
                     structure_generation: int = 0):
    """Prepare a reusable native CPU FP32 product packet topology."""
    return _native.prepare_product2(
        a, b, _index(input_count, "input_count", (1 << 64) - 1),
        _index(structure_generation, "structure_generation", (1 << 64) - 1),
    )


def product2_forward(x: np.ndarray, k: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _native.product2_forward(x, k, a, b)


def product2_vjp(x: np.ndarray, k: np.ndarray, a: np.ndarray, b: np.ndarray,
                 cotangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _native.product2_vjp(x, k, a, b, cotangent)


def product2_jvp(x: np.ndarray, k: np.ndarray, a: np.ndarray, b: np.ndarray,
                 dx: np.ndarray, dk: np.ndarray) -> np.ndarray:
    return _native.product2_jvp(x, k, a, b, dx, dk)


def mechanisms_available() -> bool:
    """Return whether this installation contains CUDA indexed mechanisms."""
    return _native.mechanisms_available()


def prepare_mechanism(spec: MechanismSpec, initial_coefficients: np.ndarray, *,
                      device: int | str, max_batch: int,
                      max_live_forwards: int = 8, precision: str = "f32"):
    """Construct the sole native parameter owner and prepared mechanism program.

    Initial values cross the Python/native boundary as contiguous host FP32;
    the native owner copies them to the selected CUDA device.
    """
    if not isinstance(spec, MechanismSpec):
        raise TypeError("spec must be a MechanismSpec")
    max_batch = _index(max_batch, "max_batch", (1 << 63) - 1)
    max_live_forwards = _index(max_live_forwards, "max_live_forwards", (1 << 32) - 1)
    if max_batch == 0 or max_live_forwards == 0:
        raise ValueError("prepared capacities must be positive")
    if isinstance(device, str):
        if device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("native mechanism device must select a CUDA device")
        device = _index(int(device.partition(":")[2] or 0), "device", (1 << 31) - 1)
    else:
        device = _index(device, "device", (1 << 31) - 1)
    values = np.asarray(initial_coefficients)
    if values.dtype != np.dtype(np.float32) or values.ndim != 1 or not values.flags.c_contiguous:
        raise ValueError("initial_coefficients must be a contiguous rank-1 NumPy float32 array")
    if values.size != len(spec.coefficient_ids):
        raise ValueError("initial coefficient extent must match the declared coefficient axis")
    if not np.isfinite(values).all():
        raise ValueError("initial coefficients must be finite")
    if precision not in ("f32", "mixed_f16"):
        raise ValueError("precision must be 'f32' or 'mixed_f16'")
    if precision == "mixed_f16" and not (np.abs(values) < 65520.0).all():
        raise ValueError("mixed coefficients must remain finite after FP16 conversion")
    if not mechanisms_available():
        raise RuntimeError("CUDA indexed mechanisms are unavailable in this Cellerator build")
    handle = _native.MechanismHandle(
        values, *constructor_data(spec), max_batch, max_live_forwards,
        0 if precision == "f32" else 1, device,
    )
    handle.spec = spec
    _handles.add(handle)
    return handle


def _known_mechanism_handles():
    return tuple(_handles)


__all__ = [
    "Axis", "Identity", "Incidence", "Mechanism", "MechanismSpec",
    "OutputContribution", "SharedSupportSpec", "prepare_mechanism",
    "prepare_product2", "product2_forward", "product2_vjp", "product2_jvp",
    "mechanisms_available",
]
