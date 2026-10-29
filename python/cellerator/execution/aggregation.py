"""Explicit owner reduction for completed scheduled expert outputs."""
from __future__ import annotations

import operator
from typing import Any, Callable

from .engine import ExecutionResult


def reduce_execution_outputs(
    result: ExecutionResult,
    *,
    owner: Callable | None = None,
    scale: Callable = operator.mul,
    combine: Callable = operator.add,
) -> dict[Any, Any]:
    """Weight and reduce completed occurrences in original selection order.

    The default owner is ``(selection.origin, selection.row_id)``. Supply an
    explicit owner function when identities span devices or map to another state
    owner. ``scale(value, weight)`` and ``combine(previous, contribution)`` retain
    arbitrary backend values and, with ordinary tensor operators, their autograd
    graphs. Signed weights and repeated/overlapping selections remain explicit;
    this performs a weighted sum, without implicit normalization.

    No model state is changed. The caller applies each returned value through its
    own updater, for example ``cell.advance(inputs=value)``. Incomplete failed-run
    reports are rejected before invoking reduction callbacks.
    """
    if not isinstance(result, ExecutionResult):
        raise TypeError("result must be an ExecutionResult")
    if owner is not None and not callable(owner):
        raise TypeError("owner must be callable")
    if not callable(scale) or not callable(combine):
        raise TypeError("scale and combine must be callable")
    if len(result.outputs) != len(result.selections):
        raise ValueError("output count does not match selected occurrences")
    groups = {(selection.key, selection.origin) for selection in result.selections}
    if tuple(result.completed_decisions) != tuple(range(len(groups))):
        raise ValueError("cannot reduce an incomplete execution result")
    reduced = {}
    for selected, value in zip(result.selections, result.outputs):
        identity = (selected.origin, selected.row_id) if owner is None else owner(selected)
        contribution = scale(value, selected.weight)
        reduced[identity] = combine(reduced[identity], contribution) if identity in reduced else contribution
    return reduced
