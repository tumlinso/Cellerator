"""Cellerator model modules built on Torch's Module and optimizer contracts."""
from .product2 import Product2Module
from . import mechanism as _mechanism
from . import routing as _routing
from . import biology as _biology

_NAMES = (
    "BiologicalTensor", "MechanismModule", "SharedSupportRelation",
    "RankedGeneSlice", "KaryoplastEventSlice",
)


def __getattr__(name: str):
    if name in ("RankedGeneSlice", "KaryoplastEventSlice"):
        return getattr(_routing, name)
    if name == "SharedSupportRelation":
        return getattr(_biology, name)
    if name in ("BiologicalTensor", "MechanismModule"):
        return getattr(_mechanism, name)
    raise AttributeError(name)


__all__ = ["Product2Module", *_NAMES]
