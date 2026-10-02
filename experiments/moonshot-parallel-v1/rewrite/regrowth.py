"""CPU M08 residual capacity: y + V tanh(U x), with external optimization.

U starts nonzero and V starts zero. The first backward gives V gradients; U
becomes live after the caller's optimizer step. Recycling is an explicit
publication operation and belongs after the controller drains old tapes.

Adam policy: clear incompatible coordinate moments, including AMSGrad maxima,
and preserve the parameter's scalar step for partial *and* whole resets. Adam
has no per-coordinate age, so recycled coordinates retain the existing bias
correction age. This is a declared limitation, not a fresh-coordinate Adam.
"""

from collections.abc import Iterable

import torch
from torch import nn


def _indices(rows: Iterable[int], size: int, device: torch.device) -> torch.Tensor:
    values = list(rows)
    if any(isinstance(i, bool) or not isinstance(i, int) for i in values):
        raise ValueError("coordinates must be integers")
    if len(set(values)) != len(values) or any(i < 0 or i >= size for i in values):
        raise ValueError("coordinates must be distinct and in bounds")
    return torch.tensor(values, dtype=torch.long, device=device)


def _moment_plan(optimizer, parameter, affected_rows=None, *, axis=0):
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("moment reset supports Adam and AdamW only")
    if not any(parameter is p for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("parameter is not owned by optimizer")
    if axis < 0 or axis >= parameter.ndim:
        raise ValueError("axis is out of bounds")
    indices = None if affected_rows is None else _indices(
        affected_rows, parameter.shape[axis], parameter.device
    )
    moments = []
    for name, value in optimizer.state.get(parameter, {}).items():
        if name == "step":
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                raise ValueError("Adam step must be scalar")
            continue
        if name not in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            raise ValueError(f"unsupported Adam state: {name}")
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f"incompatible Adam moment shape: {name}")
        if value.device != parameter.device or value.dtype != parameter.dtype:
            raise ValueError(f"incompatible Adam moment storage: {name}")
        moments.append(value)
    return moments, indices, axis


@torch.no_grad()
def _clear_plan(plan):
    moments, indices, axis = plan
    for moment in moments:
        if indices is None:
            moment.zero_()
        else:
            moment.index_fill_(axis, indices, 0)


def reset_adam_moments(optimizer, parameter, affected_rows=None, *, axis=0):
    """Clear validated moments for one parameter or selected coordinates.

    None clears every coordinate; an empty iterable changes nothing. Scalar
    step and every other parameter's state remain untouched. Call explicitly
    after a basis change; partial clears are safe only for a coordinate-local
    change. Validation completes before any moment is mutated.
    """
    _clear_plan(_moment_plan(optimizer, parameter, affected_rows, axis=axis))


def _seeded_rows(count, input_dim, seed, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    magnitude = 0.05 + 0.15 * torch.rand(
        count, input_dim, generator=generator, dtype=dtype
    )
    sign = torch.randint(0, 2, (count, input_dim), generator=generator) * 2 - 1
    return magnitude * sign


class ResidualRegrowth(nn.Module):
    """A fixed-capacity CPU branch; parameters and seed are state_dict data.

    U has shape [capacity, input_dim], V [output_dim, capacity]. Dimension
    properties are read-only and inferred from these parameters. Recycling
    zero-output slots preserves the current function exactly. Set discard_live
    explicitly to discard a live contribution, for example after the controller
    has absorbed it into an extracted mechanism.
    """

    def __init__(self, input_dim, output_dim, capacity, seed=0, dtype=torch.float64):
        super().__init__()
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
               for n in (input_dim, output_dim, capacity)):
            raise ValueError("dimensions and capacity must be positive integers")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CPU reference supports float32 and float64")
        self.U = nn.Parameter(_seeded_rows(capacity, input_dim, seed, dtype))
        self.V = nn.Parameter(torch.zeros(output_dim, capacity, dtype=dtype))
        self.register_buffer("last_seed", torch.tensor(seed, dtype=torch.int64))

    @property
    def input_dim(self):
        return self.U.shape[1]

    @property
    def output_dim(self):
        return self.V.shape[0]

    @property
    def capacity(self):
        return self.U.shape[0]

    def forward(self, x, existing_output=None):
        if x.device.type != "cpu" or self.U.device.type != "cpu" or self.V.device.type != "cpu":
            raise ValueError("this reference branch requires CPU tensors")
        if x.ndim < 1 or x.shape[-1] != self.input_dim or x.dtype != self.U.dtype:
            raise ValueError("input shape or dtype does not match branch")
        contribution = torch.tanh(x @ self.U.T) @ self.V.T
        if existing_output is None:
            return contribution
        if (existing_output.shape != contribution.shape
                or existing_output.dtype != contribution.dtype
                or existing_output.device != contribution.device):
            raise ValueError("existing output must match contribution exactly")
        return existing_output + contribution

    @torch.no_grad()
    def recycle(self, slots, *, seed, optimizer=None, discard_live=False):
        """Reseed U rows and zero V columns without replacing parameters.

        Validate both moment sets before changing any state. Stale gradients
        on the recycled coordinates are cleared; unaffected gradients survive.
        This performs no optimizer step and cannot be called with live tapes.
        The caller is responsible for enforcing that publication boundary.
        """
        indices = _indices(slots, self.capacity, self.U.device)
        if self.U.device.type != "cpu" or self.V.device.type != "cpu":
            raise ValueError("this reference branch requires CPU tensors")
        if not discard_live and torch.count_nonzero(self.V[:, indices]).item():
            raise ValueError("recycling a live contribution requires discard_live=True")
        incoming = _seeded_rows(len(indices), self.input_dim, seed, self.U.dtype)
        seed_tensor = torch.tensor(seed, dtype=self.last_seed.dtype)
        plans = []
        if optimizer is not None:
            coordinates = indices.tolist()
            plans = [_moment_plan(optimizer, self.U, coordinates),
                     _moment_plan(optimizer, self.V, coordinates, axis=1)]
        for plan in plans:
            _clear_plan(plan)
        self.U.index_copy_(0, indices, incoming)
        self.V.index_fill_(1, indices, 0)
        for parameter, axis in ((self.U, 0), (self.V, 1)):
            if parameter.grad is not None:
                parameter.grad.index_fill_(axis, indices, 0)
        if indices.numel():
            self.last_seed.copy_(seed_tensor)
