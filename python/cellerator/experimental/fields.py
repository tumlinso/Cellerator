"""Experimental composition of already-prepared Torch/native operations.

This module deliberately does not compile a graph.  An ``ExecutionField`` is
an ordered composition of existing ``Product2Module`` instances, each of which
retains its own native prepared operation and Torch parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import uuid

import torch
from torch import Tensor, nn

from ..torch.product2 import Product2Module


@dataclass(frozen=True)
class StageStructure:
    """Immutable logical description of one ordered Product2 stage."""

    name: str
    input_extent: int
    output_extent: int
    a: tuple[int, ...]
    b: tuple[int, ...]


@dataclass(frozen=True)
class FieldStructure:
    """Logical structure identity, independent of values and physical binding."""

    epoch: int
    input_extent: int
    stages: tuple[StageStructure, ...]


@dataclass(frozen=True)
class FieldLayout:
    """Physical contract currently admitted by the native Product2 adapter."""

    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    rank: int = 1
    tensor_layout: torch.layout = torch.strided
    index_mapping: str = "identity"
    value_mapping: str = "identity"


@dataclass(frozen=True)
class FieldBinding:
    """A concrete input binding for one field structure epoch."""

    _owner_token: str
    structure_epoch: int
    generation: int
    value: Tensor


class ExecutionField(nn.Module):
    """A small experimental field over prepared Product2 Torch modules.

    Stage order is execution order.  Each stage's output extent is its packet
    count and must equal the next stage's prepared input extent.  ``bind``
    creates an explicit input/layout binding; the binding can be reused for
    changing values in the same tensor.  Rebinding the same extent preserves
    native prepared handles.  Changed stage indices require the explicit
    ``rebind_structure`` call and advance the immutable structure epoch.
    """

    def __init__(self, stages: Iterable[tuple[str, Product2Module]], *,
                 input_extent: int):
        super().__init__()
        named = tuple(stages)
        if input_extent < 0:
            raise ValueError("input_extent must be nonnegative")
        if not named:
            raise ValueError("an execution field needs at least one stage")
        names = tuple(name for name, _ in named)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("stage names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        if any(not isinstance(module, Product2Module) for _, module in named):
            raise TypeError("experimental fields currently accept Product2Module stages")

        self.stage_names = names
        self.stages = nn.ModuleList(module for _, module in named)
        self.layout = FieldLayout()
        self._owner_token = uuid.uuid4().hex
        self._structure_epoch = 0
        self._binding_generation = 0
        self._value_generation = 0
        self._last_parameter_versions: tuple[int, ...] | None = None
        self._input_extent = int(input_extent)
        self._structure = self._capture_structure(epoch=0)
        self._prepare_stages(self._structure)

    @property
    def structure(self) -> FieldStructure:
        return self._structure

    @property
    def structure_epoch(self) -> int:
        return self._structure_epoch

    @property
    def binding_generation(self) -> int:
        return self._binding_generation

    @property
    def value_generation(self) -> int:
        self._observe_parameter_versions()
        return self._value_generation

    @property
    def prepared_handles(self) -> tuple[object, ...]:
        """Return the existing native prepared owners, in execution order."""
        return tuple(stage.prepared for stage in self.stages)

    def _capture_structure(self, *, epoch: int) -> FieldStructure:
        cursor = self._input_extent
        records: list[StageStructure] = []
        for name, module in zip(self.stage_names, self.stages):
            if (module.coefficients.device.type != "cpu"
                    or module.coefficients.dtype != torch.float32
                    or module.coefficients.ndim != 1
                    or module.a.device.type != "cpu" or module.a.dtype != torch.int64
                    or module.a.ndim != 1 or module.b.device.type != "cpu"
                    or module.b.dtype != torch.int64 or module.b.ndim != 1):
                raise ValueError(
                    f"stage {name!r} must retain CPU FP32 coefficients and CPU int64 indices"
                )
            a = tuple(int(i) for i in module.a.detach().cpu().tolist())
            b = tuple(int(i) for i in module.b.detach().cpu().tolist())
            if any(index < 0 or index >= cursor for index in (*a, *b)):
                required = max((*a, *b), default=-1) + 1
                raise ValueError(
                    f"stage {name!r} indices require at least {required} input values, "
                    f"but the preceding field stage provides {cursor}"
                )
            records.append(StageStructure(name, cursor, len(a), a, b))
            cursor = len(a)
        for left, right in zip(records, records[1:]):
            if left.output_extent != right.input_extent:
                raise ValueError(
                    f"stage {right.name!r} expects {right.input_extent} values, "
                    f"but {left.name!r} produces {left.output_extent}"
                )
        return FieldStructure(epoch, self._input_extent, tuple(records))

    def _prepare_stages(self, structure: FieldStructure) -> None:
        for module, record in zip(self.stages, structure.stages):
            # This is the existing Product2Module preparation/reuse boundary;
            # native arithmetic and handle ownership remain with that module.
            module._prepare_for_input(record.input_extent)

    def bind(self, value: Tensor) -> FieldBinding:
        """Bind an input tensor after checking the field's physical contract."""
        if not isinstance(value, Tensor):
            raise TypeError("field input must be a torch.Tensor")
        if (value.device.type != self.layout.device or value.dtype != self.layout.dtype
                or value.ndim != self.layout.rank or value.layout != self.layout.tensor_layout):
            raise ValueError("field input must be a rank-one strided CPU FP32 tensor")
        if value.numel() != self._input_extent:
            raise ValueError("field input extent differs from the immutable structure")
        self._assert_structure_current()
        self._binding_generation += 1
        return FieldBinding(self._owner_token, self._structure_epoch,
                            self._binding_generation, value)

    def _assert_structure_current(self) -> None:
        current = self._capture_structure(epoch=self._structure_epoch)
        if current.stages != self._structure.stages:
            raise RuntimeError(
                "stage indices changed; call rebind_structure() to prepare the new structure"
            )

    def rebind_structure(self) -> FieldStructure:
        """Explicitly accept changed indices and prepare the new structure epoch."""
        candidate = self._capture_structure(epoch=self._structure_epoch + 1)
        self._prepare_stages(candidate)
        self._structure_epoch += 1
        self._structure = candidate
        return candidate

    def _observe_parameter_versions(self) -> None:
        versions = tuple(int(module.coefficients._version) for module in self.stages)
        if self._last_parameter_versions is None:
            self._last_parameter_versions = versions
        elif versions != self._last_parameter_versions:
            self._value_generation += 1
            self._last_parameter_versions = versions

    def forward(self, binding: FieldBinding) -> Tensor:
        if not isinstance(binding, FieldBinding) or binding._owner_token != self._owner_token:
            raise TypeError("binding must come from this ExecutionField")
        if binding.structure_epoch != self._structure_epoch:
            raise RuntimeError("field binding belongs to an obsolete structure epoch")
        self._assert_structure_current()
        value = binding.value
        if value.numel() != self._input_extent:
            raise ValueError("bound input extent differs from the field structure")
        for module in self.stages:
            value = module(value)
        self._observe_parameter_versions()
        return value


__all__ = [
    "ExecutionField", "FieldBinding", "FieldLayout", "FieldStructure", "StageStructure",
]
