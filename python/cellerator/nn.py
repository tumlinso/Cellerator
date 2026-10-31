"""Optional Torch model modules with stock ``torch.nn`` ownership semantics."""
from __future__ import annotations

from importlib import import_module

_FOUNDATION = {
    "FeatureEncoder": ".torch.features",
    "RelationAttention": ".torch.relation_attention",
    "ModelWithViews": ".torch.views",
    "grouped_linear": ".torch.grouped_linear",
    "GroupedLinear": ".torch.grouped_linear",
}


def __getattr__(name: str):
    if name in _FOUNDATION:
        return getattr(import_module(_FOUNDATION[name], "cellerator"), name)
    if name == "Product2Module":
        return getattr(import_module(".torch.nn", "cellerator"), name)
    if name in {"MechanismModule", "SharedSupportRelation", "BiologicalTensor",
                "RankedGeneSlice", "KaryoplastEventSlice"}:
        return getattr(import_module(".torch.nn", "cellerator"), name)
    raise AttributeError(name)


__all__ = [
    *_FOUNDATION,
    "Product2Module", "MechanismModule", "SharedSupportRelation",
    "BiologicalTensor", "RankedGeneSlice", "KaryoplastEventSlice",
]
