"""Shared biological support with differentiable context activities.

The support is prepared once as native one-input product mechanisms. Torch
owns the source and target activity products; Cellerator owns the shared edge
coefficients and their input/parameter VJP. This is the exact composition
``y = target_activity * (W @ (source_activity * x))``.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator

import torch
from torch import Tensor, nn

from .mechanism import (
    Axis, BiologicalTensor, Identity, Incidence, Mechanism, MechanismModule,
    MechanismSpec, OutputContribution,
)


@dataclass(frozen=True)
class SharedSupportSpec:
    """Persistent directed support over typed source, target and edge axes.

    Edge order is coefficient order. Equal ``(src, dst)`` pairs remain distinct
    relation members with separate logical identities and trainable values.
    """

    source_axis: Axis
    target_axis: Axis
    edge_axis: Axis
    edge_ids: tuple[Identity, ...]
    src: tuple[int, ...]
    dst: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("source_axis", "target_axis", "edge_axis"):
            if not isinstance(getattr(self, name), Axis):
                raise TypeError(f"{name} must be an Axis")
        ids = tuple(i if isinstance(i, Identity) else Identity(*i)
                    for i in self.edge_ids)
        object.__setattr__(self, "edge_ids", ids)
        for name in ("src", "dst"):
            try:
                indices = tuple(operator.index(i) for i in getattr(self, name))
            except TypeError as exc:
                raise TypeError(f"{name} indices must be integers") from exc
            object.__setattr__(self, name, indices)
        if not (len(ids) == len(self.src) == len(self.dst) == self.edge_axis.extent):
            raise ValueError("edge identities and src/dst counts must match edge-axis extent")
        if len(set(ids)) != len(ids):
            raise ValueError("edge logical identities must be unique, including duplicate pairs")
        if any(i < 0 or i >= self.source_axis.extent for i in self.src):
            raise ValueError("src index is outside the source axis")
        if any(i < 0 or i >= self.target_axis.extent for i in self.dst):
            raise ValueError("dst index is outside the target axis")

    def mechanism_spec(self) -> MechanismSpec:
        """Lower each persistent edge to one additive product contribution."""
        mechanisms = tuple(
            Mechanism(edge, edge,
                      (Incidence(0, self.source_axis.domain, source),),
                      (OutputContribution(0, self.target_axis.domain, target,
                                          self.target_axis.order),))
            for edge, source, target in zip(self.edge_ids, self.src, self.dst)
        )
        return MechanismSpec(self.source_axis, self.target_axis, self.edge_axis,
                             self.edge_ids, mechanisms)


class SharedSupportRelation(nn.Module):
    """Prepared FP32 relation composed with differentiable activity products.

    Inputs have shape ``[batch, source_extent]``. Each activity has that same
    batch shape on its own axis, or shape ``[axis_extent]`` for a shared activity
    broadcast across the batch. No forward/backward operation changes shared
    coefficients. Use ``guarded_step`` and the existing checkpoint helpers for
    updates and restoration, just as with ``MechanismModule``.
    """

    def __init__(self, spec: SharedSupportSpec, initial_coefficients: Tensor,
                 *, max_batch: int, max_live_forwards: int = 8,
                 precision: str = "f32"):
        super().__init__()
        if not isinstance(spec, SharedSupportSpec):
            raise TypeError("spec must be a SharedSupportSpec")
        if precision != "f32":
            raise ValueError("SharedSupportRelation supports explicit FP32 precision only")
        self.spec = spec
        self.mechanism = MechanismModule(
            spec.mechanism_spec(), initial_coefficients, max_batch=max_batch,
            max_live_forwards=max_live_forwards, precision=precision,
        )

    @property
    def coefficients(self) -> nn.Parameter:
        """The sole trainable edge leaf, borrowing native canonical storage."""
        return self.mechanism.coefficients

    @property
    def max_batch(self) -> int:
        return self.mechanism.max_batch

    @property
    def max_live_forwards(self) -> int:
        return self.mechanism.max_live_forwards

    @property
    def precision(self) -> str:
        return self.mechanism.precision

    def _tensor(self, value: Tensor | BiologicalTensor, axis: Axis,
                name: str) -> Tensor:
        if isinstance(value, BiologicalTensor):
            if value.axis != axis:
                raise ValueError(f"{name} biological axis identity does not match support")
            value = value.tensor
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor or BiologicalTensor")
        if value.dtype != torch.float32 or not value.is_cuda:
            raise ValueError(f"{name} must be CUDA float32")
        if value.device != self.coefficients.device:
            raise ValueError(f"{name} device must match native coefficients")
        return value

    def forward(self, x: Tensor | BiologicalTensor,
                source_activity: Tensor | BiologicalTensor,
                target_activity: Tensor | BiologicalTensor) -> Tensor:
        x = self._tensor(x, self.spec.source_axis, "input")
        source_activity = self._tensor(source_activity, self.spec.source_axis,
                                       "source_activity")
        target_activity = self._tensor(target_activity, self.spec.target_axis,
                                       "target_activity")
        if x.ndim != 2 or x.shape[1] != self.spec.source_axis.extent:
            raise ValueError("input must have shape [batch, source-axis extent]")
        batch = x.shape[0]
        if batch <= 0 or batch > self.max_batch:
            raise ValueError("input batch must be positive and within prepared capacity")
        for value, axis, name in (
            (source_activity, self.spec.source_axis, "source_activity"),
            (target_activity, self.spec.target_axis, "target_activity"),
        ):
            if tuple(value.shape) not in ((axis.extent,), (batch, axis.extent)):
                raise ValueError(f"{name} must have shape [extent] or [batch, extent]")
        # The contiguous product is a fresh activity intermediate, leaving both
        # caller tensors and the canonical coefficient owner unchanged.
        q = (x * source_activity).contiguous()
        return self.mechanism(q) * target_activity

    @classmethod
    def shared_view(cls, owner: "SharedSupportRelation") -> "SharedSupportRelation":
        """Reuse exactly the same native program, storage and Torch leaf."""
        if not isinstance(owner, SharedSupportRelation):
            raise TypeError("owner must be a SharedSupportRelation")
        view = cls.__new__(cls)
        nn.Module.__init__(view)
        view.spec = owner.spec
        view.mechanism = MechanismModule.shared_view(owner.mechanism)
        return view
