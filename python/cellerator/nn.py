"""Optional Torch model modules with stock ``torch.nn`` ownership semantics."""
from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name == "Product2Module":
        return getattr(import_module(".torch.nn", "cellerator"), name)
    if name in {"MechanismModule", "SharedSupportRelation", "BiologicalTensor",
                "RankedGeneSlice", "KaryoplastEventSlice"}:
        return getattr(import_module(".torch.nn", "cellerator"), name)
    raise AttributeError(name)


__all__ = [
    "Product2Module", "MechanismModule", "SharedSupportRelation",
    "BiologicalTensor", "RankedGeneSlice", "KaryoplastEventSlice",
]
