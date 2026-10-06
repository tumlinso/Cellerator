"""Optional PyTorch adapters for Cellerator native numerical owners.

CPU Product2 uses the core extension and does not require the optional CUDA
mechanism adapter. Mechanism symbols are imported lazily so a CPU-only
installation can use Product2 with PyTorch alone.
"""
from __future__ import annotations

import importlib

from cellerator import SharedSupportSpec
from .product2 import Product2Module, prepare_product2, product2, product2_jvp

_MECHANISM_EXPORTS = {
    "Axis", "BiologicalTensor", "Identity", "Incidence", "Mechanism",
    "MechanismModule", "MechanismSpec", "OutputContribution",
    "SharedSupportRelation", "guarded_step", "load_checkpoint",
    "prepare_mechanism", "save_checkpoint",
}


def __getattr__(name: str):
    if name not in _MECHANISM_EXPORTS:
        raise AttributeError(name)
    try:
        importlib.import_module("._torch", __name__)
        from .biology import SharedSupportRelation
        from .mechanism import (
            Axis, BiologicalTensor, Identity, Incidence, Mechanism,
            MechanismModule, MechanismSpec, OutputContribution, guarded_step,
            load_checkpoint, save_checkpoint,
        )
        from cellerator import prepare_mechanism
    except ImportError as exc:
        raise ImportError(
            "Cellerator CUDA mechanism adapters require a build with optional libtorch support"
        ) from exc
    values = locals()
    return values[name]


__all__ = sorted(_MECHANISM_EXPORTS | {
    "Product2Module", "SharedSupportSpec", "prepare_product2", "product2", "product2_jvp",
})
