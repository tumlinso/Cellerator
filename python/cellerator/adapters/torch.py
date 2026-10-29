"""Optional PyTorch modules for learnable cellular dynamics.

Import this module explicitly to opt in to PyTorch. The backend-neutral model
and cell packages do not import it, and cell state stays with the caller.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cellerator.modeling import Model, StepContext, StepResult


class TorchDynamics(nn.Module):
    """A trainable state transition whose parameters are shared by cells.

    Subclasses implement ``forward(state, context)`` and may return any state
    object or a :class:`StepResult`. Cell state never lives in this module.
    """

    def step(self, state: Any, context: StepContext) -> StepResult:
        value = self(state, context)
        return value if isinstance(value, StepResult) else StepResult(value)

    def as_model(self, observation: Callable[[Any, StepContext], Any] | nn.Module | None = None) -> Model:
        """Bind this parameter-sharing rule to the generic model interface."""
        return Model(self.step, observation, parameters=self)


class TorchModel(nn.Module):
    """Torch-owned shared dynamics and observation modules.

    Use ``step``/``observe`` with ``Cell`` or call ``forward`` directly.
    Optimizers see shared parameters once, regardless of cell count.
    """

    def __init__(self, dynamics: nn.Module,
                 observation: nn.Module | Callable[[Any, StepContext], Any] | None = None):
        super().__init__()
        if not isinstance(dynamics, nn.Module):
            raise TypeError("dynamics must be a torch.nn.Module")
        if observation is not None and not callable(observation):
            raise TypeError("observation must be callable or None")
        self.dynamics = dynamics
        if isinstance(observation, nn.Module):
            self.observation = observation
        else:
            self.observation = None
        self._observation_fn = observation if observation is not None and self.observation is None else None

    def forward(self, state: Any, context: StepContext | None = None) -> Any:
        return self.dynamics(state, StepContext() if context is None else context)

    def step(self, state: Any, context: StepContext) -> StepResult:
        value = self.forward(state, context)
        return value if isinstance(value, StepResult) else StepResult(value)

    def observe(self, state: Any, context: StepContext | None = None) -> Any:
        if context is None:
            context = StepContext()
        if self.observation is not None:
            return self.observation(state, context)
        if self._observation_fn is not None:
            return self._observation_fn(state, context)
        return state

    def as_model(self) -> Model:
        """Expose registered Torch parameters through the generic model API."""
        return Model(self.step, self.observe, parameters=self)


class ResidualStateUpdate(TorchDynamics):
    """Learn a feature-wise change to a tensor state, optionally conditioned
    on a same-shaped external tensor. The caller chooses units and time scale.
    """

    def __init__(self, state_size: int, hidden_size: int | None = None,
                 input_size: int = 0, *, activation: type[nn.Module] = nn.SiLU):
        super().__init__()
        if state_size <= 0 or input_size < 0:
            raise ValueError("state_size must be positive and input_size nonnegative")
        hidden_size = hidden_size or state_size
        self.state_size = int(state_size)
        self.input_size = int(input_size)
        self.delta = nn.Sequential(
            nn.Linear(self.state_size + self.input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, self.state_size),
        )

    def forward(self, state: Tensor, context: StepContext) -> Tensor:
        if state.shape[-1] != self.state_size:
            raise ValueError("last state dimension does not match state_size")
        pieces = [state]
        if self.input_size:
            external = context.inputs
            if not isinstance(external, Tensor) or external.shape[-1] != self.input_size:
                raise ValueError("context.inputs must be a tensor with the declared input_size")
            pieces.append(external)
        elif context.inputs is not None:
            raise ValueError("this update was constructed without external inputs")
        delta = self.delta(torch.cat(pieces, dim=-1))
        return state + context.dt * delta


class GatedInteractor(nn.Module):
    """Jointly transform an explicitly supplied sparse participant group.

    Each participant receives an independent sigmoid gate, then gated features
    are summed (all may contribute at once) and passed through a nonlinear joint
    computation. Membership and routing are supplied by the caller; this module
    does not infer gene identity or a biological complex.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 *, normalize: bool = False):
        super().__init__()
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("feature sizes must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.normalize = bool(normalize)
        self.encode = nn.Linear(self.input_size, hidden_size)
        self.gate = nn.Linear(self.input_size, 1)
        self.joint = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                                   nn.SiLU(), nn.Linear(hidden_size, output_size))

    def forward(self, participants: Sequence[Tensor] | Tensor,
                context: Any = None) -> Tensor:
        del context  # relation-specific context may be captured by a caller wrapper
        if isinstance(participants, Tensor):
            if participants.ndim < 2:
                raise ValueError("participant tensor needs [..., participants, features]")
            values = participants
        else:
            if not participants:
                raise ValueError("an interactor requires at least one participant")
            if len(participants) == 1:
                values = participants[0].unsqueeze(-2)
            else:
                try:
                    values = torch.stack(tuple(participants), dim=-2)
                except RuntimeError as exc:
                    raise ValueError("participants must have matching shapes and devices") from exc
        if values.shape[-1] != self.input_size:
            raise ValueError("participant feature size does not match input_size")
        hidden = torch.tanh(self.encode(values))
        weights = torch.sigmoid(self.gate(values))
        pooled = (weights * hidden).sum(dim=-2)
        if self.normalize:
            pooled = pooled / weights.sum(dim=-2).clamp_min(1.0)
        return self.joint(pooled)


class EventPredictionHead(nn.Module):
    """Predict a categorical next-event distribution over caller identities."""

    def __init__(self, state_size: int, event_ids: Sequence[Hashable]):
        super().__init__()
        identities = tuple(event_ids)
        if state_size <= 0 or not identities:
            raise ValueError("state_size and event_ids must be nonempty")
        if len(set(identities)) != len(identities):
            raise ValueError("event identities must be unique")
        self.event_ids = identities
        self.logits = nn.Linear(int(state_size), len(identities))

    def forward(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.logits.in_features:
            raise ValueError("last state dimension does not match state_size")
        return self.logits(state)

    def probabilities(self, state: Tensor) -> Tensor:
        return torch.softmax(self(state), dim=-1)

    def sample(self, state: Tensor, *, generator: torch.Generator | None = None) -> Hashable | tuple[Hashable, ...]:
        """Sample event identity/identities; event meaning stays caller-owned."""
        probabilities = self.probabilities(state)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        picks = torch.multinomial(flat, 1, generator=generator).squeeze(-1).tolist()
        result = tuple(self.event_ids[index] for index in picks)
        return result[0] if not state.shape[:-1] else result


@dataclass(frozen=True)
class SparseRouting:
    """Ragged, caller-visible assignment of batch rows to shared experts.

    ``expert_ids`` contains one expert identity per assignment and
    ``row_offsets`` divides that flat list into rows. Repeated identities are
    allowed when a model intentionally makes multiple contributions.
    ``weights`` is an optional differentiable per-assignment gate.
    """

    expert_ids: Tensor
    row_offsets: Tensor
    weights: Tensor | None = None

    @classmethod
    def from_dense(cls, expert_ids: Tensor, weights: Tensor | None = None,
                   *, mask: Tensor | None = None) -> "SparseRouting":
        """Build ragged assignments from ``[batch, choices]`` expert IDs.

        ``mask`` marks eligible assignments. Padded IDs are never routed.
        Without a mask, every listed ID is an active assignment.
        """
        if expert_ids.ndim != 2 or expert_ids.dtype != torch.long:
            raise ValueError("expert_ids must be a rank-2 torch.long tensor")
        if weights is not None and weights.shape != expert_ids.shape:
            raise ValueError("weights must match expert_ids shape")
        if weights is not None and not weights.is_floating_point():
            raise ValueError("weights must use a floating tensor dtype")
        if mask is None:
            mask = torch.ones_like(expert_ids, dtype=torch.bool)
        if mask.shape != expert_ids.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean and match expert_ids shape")
        flat_ids = expert_ids[mask]
        flat_weights = None if weights is None else weights[mask]
        counts = mask.sum(dim=1, dtype=torch.long)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        return cls(flat_ids, offsets, flat_weights)

    @property
    def batch_size(self) -> int:
        return self.row_offsets.numel() - 1

    def row_indices(self, *, device: torch.device | None = None) -> Tensor:
        counts = self.row_offsets[1:] - self.row_offsets[:-1]
        rows = torch.arange(self.batch_size, device=self.row_offsets.device)
        result = torch.repeat_interleave(rows, counts)
        return result if device is None else result.to(device=device)


class SparseExpertBank(nn.Module):
    """Dispatch only selected batch rows through shared, registered experts.

    Inputs are rows ``[batch, ..., feature]``. Each routed row can visit any
    number of experts, including none or several. Expert outputs must share
    shape and be on the input device. Weighted contributions combine with an
    explicit ``sum`` or weighted ``mean`` rule. This class owns model routing;
    it makes no device placement or scheduling decisions.

    Supply ``routing`` on each call for caller-controlled routing, or provide a
    learnable ``router`` and ``top_k`` here. Learned routes use a hard top-k
    identity choice and sigmoid gates by default, so selected experts can
    contribute independently. No cell-type or trajectory labels are consumed.
    """

    def __init__(self, experts: Sequence[nn.Module], *,
                 router: nn.Module | None = None,
                 top_k: int | None = None,
                 gate_mode: str = "sigmoid",
                 combine: str = "sum",
                 output_size: int | None = None):
        super().__init__()
        if not experts or not all(isinstance(expert, nn.Module) for expert in experts):
            raise ValueError("experts must contain one or more torch.nn.Module values")
        if router is not None and not isinstance(router, nn.Module):
            raise TypeError("router must be a torch.nn.Module or None")
        if (router is None) != (top_k is None):
            raise ValueError("learned routing requires both router and top_k")
        if top_k is not None and not 1 <= top_k <= len(experts):
            raise ValueError("top_k must be within the expert count")
        if gate_mode not in {"sigmoid", "softmax"}:
            raise ValueError("gate_mode must be 'sigmoid' or 'softmax'")
        if combine not in {"sum", "mean"}:
            raise ValueError("combine must be 'sum' or 'mean'")
        if output_size is not None and output_size <= 0:
            raise ValueError("output_size must be positive")
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.top_k = top_k
        self.gate_mode = gate_mode
        self.combine = combine
        self.output_size = output_size

    @property
    def expert_count(self) -> int:
        return len(self.experts)

    def route(self, x: Tensor) -> SparseRouting:
        """Select sparse routes from learned logits, preserving gate gradients."""
        if self.router is None or self.top_k is None:
            raise RuntimeError("this expert bank uses caller-provided routing")
        logits = self.router(x)
        if not isinstance(logits, Tensor) or logits.shape != (*x.shape[:-1], self.expert_count):
            raise ValueError("router must return [..., expert_count] logits aligned to input rows")
        if x.ndim != 2:
            raise ValueError("learned routing currently expects [batch, features] inputs")
        values, ids = logits.topk(self.top_k, dim=-1)
        gates = torch.sigmoid(values) if self.gate_mode == "sigmoid" else torch.softmax(values, dim=-1)
        return SparseRouting.from_dense(ids, gates)

    def forward(self, x: Tensor, routing: SparseRouting | None = None,
                *, return_routing: bool = False) -> Tensor | tuple[Tensor, SparseRouting]:
        if not isinstance(x, Tensor) or x.ndim < 2:
            raise ValueError("x must be a tensor shaped [batch, ..., features]")
        route = self.route(x) if routing is None and self.router is not None else routing
        if route is None:
            raise ValueError("caller-provided SparseRouting is required when no router is configured")
        if route.expert_ids.ndim != 1 or route.expert_ids.dtype != torch.long:
            raise ValueError("routing expert_ids must be a rank-1 torch.long tensor")
        if route.row_offsets.ndim != 1 or route.row_offsets.dtype != torch.long:
            raise ValueError("routing row_offsets must be a rank-1 torch.long tensor")
        if route.batch_size != x.shape[0] or route.row_offsets.device != route.expert_ids.device:
            raise ValueError("routing batch extent and route devices must match x")
        if route.row_offsets.numel() == 0 or int(route.row_offsets[0]) != 0:
            raise ValueError("row_offsets must start at zero")
        if int(route.row_offsets[-1]) != route.expert_ids.numel():
            raise ValueError("row_offsets must end at the assignment count")
        if bool(torch.any(route.row_offsets[1:] < route.row_offsets[:-1])):
            raise ValueError("row_offsets must be monotone")
        if route.weights is not None:
            if route.weights.ndim != 1 or route.weights.numel() != route.expert_ids.numel():
                raise ValueError("routing weights must be rank-1 and align with assignments")
            if not route.weights.is_floating_point():
                raise ValueError("routing weights must use a floating tensor dtype")
            if route.weights.device != x.device:
                raise ValueError("routing weights must be on the input device")
        if route.expert_ids.numel() and (
            int(route.expert_ids.min()) < 0 or int(route.expert_ids.max()) >= self.expert_count
        ):
            raise ValueError("routing contains an expert identity outside this bank")
        if route.expert_ids.device != x.device:
            raise ValueError("routing expert identities must be on the input device")

        flat_x = x.reshape(x.shape[0], -1, x.shape[-1])
        rows = route.row_indices(device=x.device)
        output: Tensor | None = None
        denominator: Tensor | None = None
        for expert_id, expert in enumerate(self.experts):
            positions = torch.nonzero(route.expert_ids == expert_id, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            selected_rows = rows.index_select(0, positions)
            selected = flat_x.index_select(0, selected_rows)
            value = expert(selected)
            if (not isinstance(value, Tensor) or value.ndim < 2
                    or value.shape[:-1] != selected.shape[:-1]):
                raise ValueError("every expert must return [assignments, ..., features] and preserve leading extents")
            if value.device != x.device:
                raise ValueError("expert outputs must remain on the input device")
            if output is None:
                output = value.new_zeros((x.shape[0], *value.shape[1:]))
                if self.combine == "mean":
                    denominator = value.new_zeros((x.shape[0], *([1] * (value.ndim - 1))))
            elif value.shape[1:] != output.shape[1:] or value.device != output.device:
                raise ValueError("all expert outputs must share trailing shape and device")
            elif value.dtype != output.dtype:
                raise ValueError("all expert outputs must share a dtype")
            if route.weights is None:
                weight = value.new_ones((positions.numel(), *([1] * (value.ndim - 1))))
            else:
                weight = route.weights.index_select(0, positions).to(dtype=value.dtype)
                weight = weight.reshape(-1, *([1] * (value.ndim - 1)))
            output = output.index_add(0, selected_rows, value * weight)
            if denominator is not None:
                denominator = denominator.index_add(0, selected_rows, weight)
        if output is None:
            width = self.output_size
            if width is None:
                width = next((getattr(expert, "out_features", None)
                              for expert in self.experts
                              if getattr(expert, "out_features", None) is not None), None)
            if width is None:
                raise ValueError("fully empty routing requires output_size or experts exposing out_features")
            zero = flat_x.sum(dim=-1, keepdim=True)[..., :1] * 0
            output = zero.expand(*flat_x.shape[:-1], width)
        if denominator is not None:
            output = output / denominator.clamp_min(torch.finfo(output.dtype).eps)
        output = output.reshape(x.shape[0], *x.shape[1:-1], output.shape[-1])
        return (output, route) if return_routing else output


__all__ = [
    "EventPredictionHead", "GatedInteractor", "ResidualStateUpdate",
    "SparseExpertBank", "SparseRouting", "TorchDynamics", "TorchModel",
]
