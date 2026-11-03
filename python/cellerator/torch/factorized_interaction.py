"""Prepared native execution for gated joint factorized interactions.

The plan owns immutable structure only. Callers continue to own the five
numerical tensors and their optimizer; autograd returns their gradients.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from ._native_nn import native_nn


class PreparedFactorizedInteraction:
    """A reusable structural plan for ``x, z, alpha, bias, gamma`` tensors.

    ``records`` contains mappings with ``logical_id``, ordered ``inputs``,
    ``target``, ``alpha``, ``bias``, and optional ``terms`` pairs of
    ``(context_occurrence, gamma_parameter)``. Extents are occurrence and
    parameter counts, not biological IDs or tensor storage positions.

    Host structure is prepared once here. The native adapter uploads and
    caches one immutable device layout per CUDA device on first use, then
    reuses it across calls. Each call creates its own gate tape and scratch.
    """

    __slots__ = ("_native", "_plan")

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        inputs: int,
        contexts: int,
        outputs: int,
        alpha_rows: int,
        biases: int,
        gammas: int,
        pattern_id: int = 1,
        structure_id: int = 1,
        structure_epoch: int = 1,
        max_mechanisms: int = 32,
        max_inputs: int = 64,
        max_contexts: int = 64,
    ) -> None:
        if isinstance(records, (str, bytes)):
            raise TypeError("records must be an iterable of mappings")
        frozen_records = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("each interaction record must be a mapping")
            frozen_records.append(dict(record))
        native = native_nn("native")
        if not hasattr(native, "PreparedFactorizedInteraction"):
            raise ImportError(
                "factorized CUDA execution requires building the optional NN CUDA "
                "and Python adapters"
            )
        self._native = native
        self._plan = native.PreparedFactorizedInteraction(
            frozen_records,
            inputs,
            contexts,
            outputs,
            alpha_rows,
            biases,
            gammas,
            pattern_id,
            structure_id,
            structure_epoch,
            max_mechanisms,
            max_inputs,
            max_contexts,
        )

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        gamma: torch.Tensor,
        *,
        output_dtype: torch.dtype = torch.float16,
        route: str = "packet",
    ) -> torch.Tensor:
        """Execute the prepared law with first-order autograd support."""
        output_types = {
            torch.float16: "float16",
            torch.float32: "float32",
        }
        try:
            native_dtype = output_types[output_dtype]
        except (KeyError, TypeError) as exc:
            raise TypeError("output_dtype must be torch.float16 or torch.float32") from exc
        if route not in {"packet", "direct"}:
            raise ValueError("route must be 'packet' or 'direct'")
        return self._native.factorized_interaction_cuda(
            self._plan, x, z, alpha, bias, gamma, native_dtype, route
        )


__all__ = ["PreparedFactorizedInteraction"]
