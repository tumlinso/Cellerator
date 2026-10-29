"""Caller-built worksets and access to the prepared mechanism route index."""
from __future__ import annotations

from torch import Tensor

from .execution import SparseSupportModule
from .structure import PerCellWorkset


def affected_mechanism_workset(module: SparseSupportModule,
                               entity_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Look up prepared mechanism postings for the supplied entity IDs."""
    if not isinstance(module, SparseSupportModule):
        raise TypeError("module must be a Cellerator SparseSupportModule")
    return module.affected_mechanism_workset(entity_ids)


__all__ = ["PerCellWorkset", "affected_mechanism_workset"]
