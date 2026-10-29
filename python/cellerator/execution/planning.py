"""Sparse selected work, advisory demand forecasts, and topology-aware choices.

Logical expert demand is immutable input. Placement planning changes where/how
that work could execute, never which experts the model selected. Plans are
estimates and proposals; they do not assert residency or launch computation.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any

from .residency import MemoryCategory, ParameterKey
from .topology import HardwareTopology, TransferRoute, UnreachableDevice, _number, _size


@dataclass(frozen=True, slots=True)
class SelectedWork:
    key: ParameterKey
    origin: Any
    row_id: Any
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.key, ParameterKey):
            raise TypeError("key must be a ParameterKey")
        hash(self.origin)
        if not math.isfinite(float(self.weight)):
            raise ValueError("selection weight must be finite")


@dataclass(frozen=True, slots=True)
class ExpertWorklist:
    """Compact expert-major permutation of caller-selected occurrences.

    Duplicate/overlapping selections and weights are retained. No padded
    cell-by-expert matrix or replacement expert is constructed.
    """

    selections: tuple[SelectedWork, ...]
    expert_keys: tuple[ParameterKey, ...]
    expert_offsets: tuple[int, ...]
    dispatch_to_selection: tuple[int, ...]

    def __post_init__(self):
        for name in ("selections", "expert_keys", "expert_offsets", "dispatch_to_selection"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not all(isinstance(value, SelectedWork) for value in self.selections):
            raise TypeError("worklist selections must be SelectedWork values")
        if not all(isinstance(key, ParameterKey) for key in self.expert_keys):
            raise TypeError("worklist expert keys must be ParameterKey values")
        if len(set(self.expert_keys)) != len(self.expert_keys):
            raise ValueError("worklist expert keys must be unique")
        offsets = tuple(_size(value, "expert offset") for value in self.expert_offsets)
        permutation = tuple(_size(value, "selection index") for value in self.dispatch_to_selection)
        if (len(offsets) != len(self.expert_keys) + 1 or not offsets or offsets[0] != 0
                or offsets[-1] != len(self.selections)
                or any(start >= end for start, end in zip(offsets, offsets[1:]))):
            raise ValueError("expert offsets must delimit nonempty expert groups")
        if len(permutation) != len(self.selections) or set(permutation) != set(range(len(self.selections))):
            raise ValueError("dispatch permutation must preserve every selected occurrence exactly once")
        for group, key in enumerate(self.expert_keys):
            if any(self.selections[permutation[index]].key != key
                   for index in range(offsets[group], offsets[group + 1])):
                raise ValueError("worklist group key does not match its selected expert revisions")
        object.__setattr__(self, "expert_offsets", offsets)
        object.__setattr__(self, "dispatch_to_selection", permutation)

    @classmethod
    def from_selections(cls, selections: Iterable[SelectedWork]) -> "ExpertWorklist":
        values = tuple(selections)
        groups: dict[ParameterKey, list[int]] = {}
        for index, selection in enumerate(values):
            if not isinstance(selection, SelectedWork):
                raise TypeError("selections must contain SelectedWork values")
            groups.setdefault(selection.key, []).append(index)
        offsets = [0]
        permutation = []
        for members in groups.values():
            permutation.extend(members)
            offsets.append(len(permutation))
        return cls(values, tuple(groups), tuple(offsets), tuple(permutation))

    @property
    def selection_to_dispatch(self) -> tuple[int, ...]:
        inverse = [0] * len(self.selections)
        for dispatch, selection in enumerate(self.dispatch_to_selection):
            inverse[selection] = dispatch
        return tuple(inverse)

    def demands(self, *, activation_bytes_per_row: int, output_bytes_per_row: int) -> tuple["ExpertDemand", ...]:
        activation_size = _size(activation_bytes_per_row)
        output_size = _size(output_bytes_per_row)
        groups: dict[tuple[ParameterKey, Any], list[Any]] = {}
        for selection in self.selections:
            groups.setdefault((selection.key, selection.origin), []).append(selection.row_id)
        return tuple(ExpertDemand(key, origin, len(rows) * activation_size,
                                 len(rows) * output_size, len(rows), tuple(rows))
                     for (key, origin), rows in groups.items())


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    key: ParameterKey
    nbytes: int
    compute_seconds: float = 0.0
    allowed_devices: tuple[Any, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.key, ParameterKey):
            raise TypeError("key must be a ParameterKey")
        object.__setattr__(self, "nbytes", _size(self.nbytes))
        object.__setattr__(self, "compute_seconds", _number(self.compute_seconds, "compute_seconds"))
        if self.allowed_devices is not None:
            object.__setattr__(self, "allowed_devices", tuple(self.allowed_devices))


@dataclass(frozen=True, slots=True)
class ExpertPlacement:
    key: ParameterKey
    device: Any
    ready: bool = True
    executable: bool = True

    def __post_init__(self):
        if not isinstance(self.key, ParameterKey):
            raise TypeError("key must be a ParameterKey")
        hash(self.device)


@dataclass(frozen=True, slots=True)
class ExpertDemand:
    """Actual selected work; byte counts cover the entire group, not one row."""

    key: ParameterKey
    origin: Any
    activation_bytes: int
    output_bytes: int
    uses: int = 1
    work_ids: tuple[Any, ...] = ()

    def __post_init__(self):
        if not isinstance(self.key, ParameterKey):
            raise TypeError("key must be a ParameterKey")
        hash(self.origin)
        object.__setattr__(self, "activation_bytes", _size(self.activation_bytes))
        object.__setattr__(self, "output_bytes", _size(self.output_bytes))
        object.__setattr__(self, "uses", _size(self.uses, "uses"))
        object.__setattr__(self, "work_ids", tuple(self.work_ids))
        if self.work_ids and len(self.work_ids) != self.uses:
            raise ValueError("work_ids must identify every selected occurrence")


class ExecutionStrategy(str, Enum):
    LOCAL = "local"
    REMOTE_ACTIVATIONS = "remote_activations"
    MOVE_WEIGHTS = "move_weights"
    REPLICATE = "replicate"


@dataclass(frozen=True, slots=True)
class PlannedTransfer:
    category: MemoryCategory
    route: TransferRoute


@dataclass(frozen=True, slots=True)
class CostCandidate:
    strategy: ExecutionStrategy
    execution_device: Any
    parameter_source: Any
    estimated_seconds: float
    transfers: tuple[PlannedTransfer, ...] = ()
    new_parameter_bytes: int = 0
    feasible: bool = True
    reason: str | None = None
    dependencies: tuple[int, ...] = ()
    activation_reservations: tuple[tuple[Any, int], ...] = ()


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    demand: ExpertDemand
    chosen: CostCandidate
    alternatives: tuple[CostCandidate, ...]

    @property
    def key(self):
        return self.demand.key

    @property
    def origin(self):
        return self.demand.origin

    @property
    def execution_device(self):
        return self.chosen.execution_device

    @property
    def strategy(self):
        return self.chosen.strategy

    @property
    def estimated_seconds(self):
        return self.chosen.estimated_seconds


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    decisions: tuple[DispatchDecision, ...]
    projected_memory: Mapping[Any, Mapping[MemoryCategory, int]]
    peak_memory: Mapping[Any, Mapping[MemoryCategory, int]]
    proposed_replicas: tuple[ExpertPlacement, ...] = ()

    @property
    def estimated_seconds(self) -> float:
        """Sequential horizon estimate; not a measured distributed makespan."""
        return sum(decision.estimated_seconds for decision in self.decisions)


class PlanningError(RuntimeError):
    def __init__(self, demand: ExpertDemand, candidates: tuple[CostCandidate, ...], reason: str):
        self.demand = demand
        self.candidates = candidates
        super().__init__(reason)


class PlacementPlanner:
    """Compare feasible execution mechanisms against explicit snapshots.

    Ready placements must name exact revisions. ``memory_usage``, if provided,
    includes ALL baseline bytes, including parameters. If omitted, baseline
    parameter bytes are derived from ready placements. Activation workspace is
    reserved per sequential decision; persistent replicas affect later choices.
    Device capacity zero means zero bytes, not unknown/unlimited memory.

    Reuse hints only amortize resource cost. A replica is proposed once and can
    serve subsequent decisions through explicit dependency indices. The caller
    must materialize it before dependent work. This planner performs no mutation
    of the actual residency manager or model routing.
    """

    def __init__(self, topology: HardwareTopology, experts: Iterable[ExpertSpec],
                 placements: Iterable[ExpertPlacement], *, memory_usage=None,
                 retention_penalty_seconds_per_byte: float = 0.0):
        if not isinstance(topology, HardwareTopology):
            raise TypeError("topology must be a HardwareTopology")
        self.topology = topology
        values = tuple(experts)
        if not all(isinstance(expert, ExpertSpec) for expert in values):
            raise TypeError("experts must contain ExpertSpec values")
        self.experts = {expert.key: expert for expert in values}
        if len(self.experts) != len(values):
            raise ValueError("expert revision keys must be unique")
        self.placements = tuple(placements)
        seen = set()
        for placement in self.placements:
            if not isinstance(placement, ExpertPlacement):
                raise TypeError("placements must contain ExpertPlacement values")
            if placement.key not in self.experts or placement.device not in topology.devices:
                raise ValueError("placement expert/device is not registered")
            identity = placement.key, placement.device
            if identity in seen:
                raise ValueError("duplicate expert/device placement")
            seen.add(identity)
        self.penalty = _number(retention_penalty_seconds_per_byte, "retention penalty")
        self._baseline = {device: {category: 0 for category in MemoryCategory}
                          for device in topology.devices}
        if memory_usage is None:
            for placement in self.placements:
                if placement.ready:
                    self._baseline[placement.device][MemoryCategory.PARAMETER] += self.experts[placement.key].nbytes
        else:
            for device, categories in memory_usage.items():
                if device not in self._baseline:
                    raise ValueError("memory usage refers to an unregistered device")
                for category, nbytes in categories.items():
                    self._baseline[device][MemoryCategory(category)] = _size(nbytes)
        for device, categories in self._baseline.items():
            if sum(categories.values()) > topology.devices[device].memory_bytes:
                raise ValueError("baseline memory exceeds declared device capacity")

    def _supports(self, expert, device):
        return (self.topology.devices[device].kind != "host"
                and (expert.allowed_devices is None or device in expert.allowed_devices))

    def plan(self, demands: Iterable[ExpertDemand], *, repeats: Mapping | None = None,
             allow_replication: bool = True) -> ExecutionPlan:
        values = tuple(demands)
        if not all(isinstance(demand, ExpertDemand) for demand in values):
            raise TypeError("demands must contain ExpertDemand values")
        usage = {device: dict(categories) for device, categories in self._baseline.items()}
        peak = {device: dict(categories) for device, categories in usage.items()}
        # Value records producer decision index and executable flag.
        copies = {(placement.key, placement.device): (None, placement.executable)
                  for placement in self.placements if placement.ready}
        decisions = []
        proposed = []
        for demand in values:
            if demand.key not in self.experts:
                raise PlanningError(demand, (), "selected expert revision is not registered")
            if demand.origin not in self.topology.devices:
                raise PlanningError(demand, (), "demand origin is not registered")
            expert = self.experts[demand.key]
            repeat = 1.0 if repeats is None else repeats.get((demand.key, demand.origin), 1.0)
            repeat = _number(repeat, "repeat hint", positive=True)
            if repeat < 1:
                raise ValueError("repeat hints must be at least one")
            workspace = demand.activation_bytes + demand.output_bytes
            candidates = []

            def candidate(strategy, device, source, transfer_specs=(), new_bytes=0, dependency=None):
                dependencies = () if dependency is None else (dependency,)
                if not self._supports(expert, device):
                    return
                reservations = ((device, workspace),)
                if strategy is ExecutionStrategy.REMOTE_ACTIVATIONS:
                    reservations += ((demand.origin, demand.output_bytes),)
                fits = all(sum(usage[where].values()) + size + (new_bytes if where == device else 0)
                           <= self.topology.devices[where].memory_bytes
                           for where, size in reservations)
                if not fits:
                    candidates.append(CostCandidate(strategy, device, source, math.inf,
                        new_parameter_bytes=new_bytes, feasible=False, reason="insufficient memory",
                        dependencies=dependencies, activation_reservations=reservations))
                    return
                try:
                    transfers = tuple(PlannedTransfer(category, self.topology.route(start, end, size))
                                      for category, start, end, size in transfer_specs)
                except UnreachableDevice:
                    candidates.append(CostCandidate(strategy, device, source, math.inf,
                        new_parameter_bytes=new_bytes, feasible=False, reason="unreachable transfer path",
                        dependencies=dependencies, activation_reservations=reservations))
                    return
                communication = sum(transfer.route.seconds for transfer in transfers)
                compute = expert.compute_seconds * demand.uses / self.topology.devices[device].compute_scale
                if strategy is ExecutionStrategy.REPLICATE:
                    cost = communication + compute * repeat + self.penalty * new_bytes
                else:
                    cost = (communication + compute) * repeat
                candidates.append(CostCandidate(strategy, device, source, cost, transfers,
                                                new_bytes, dependencies=dependencies,
                                                activation_reservations=reservations))

            sources = [(device, producer, executable) for (key, device), (producer, executable) in copies.items()
                       if key == demand.key]
            for device, producer, executable in sources:
                if device == demand.origin and executable:
                    candidate(ExecutionStrategy.LOCAL, device, device, dependency=producer)
                elif device != demand.origin and executable:
                    candidate(ExecutionStrategy.REMOTE_ACTIVATIONS, device, device, (
                        (MemoryCategory.ACTIVATION, demand.origin, device, demand.activation_bytes),
                        (MemoryCategory.ACTIVATION, device, demand.origin, demand.output_bytes)),
                        dependency=producer)
            if (demand.key, demand.origin) not in copies:
                for source, producer, _ in sources:
                    transfer = ((MemoryCategory.PARAMETER, source, demand.origin, expert.nbytes),)
                    candidate(ExecutionStrategy.MOVE_WEIGHTS, demand.origin, source, transfer,
                              expert.nbytes, producer)
                    if allow_replication:
                        candidate(ExecutionStrategy.REPLICATE, demand.origin, source, transfer,
                                  expert.nbytes, producer)
            feasible = [item for item in candidates if item.feasible]
            if not feasible:
                raise PlanningError(demand, tuple(candidates), "no feasible execution for the exact selected expert revision")
            # Enumeration order is stable and resolves exact-cost ties.
            chosen = min(feasible, key=lambda item: item.estimated_seconds)
            device = chosen.execution_device
            for where, nbytes in chosen.activation_reservations:
                current_peak = dict(usage[where])
                if where == device:
                    current_peak[MemoryCategory.PARAMETER] += chosen.new_parameter_bytes
                current_peak[MemoryCategory.ACTIVATION] += nbytes
                for category, size in current_peak.items():
                    peak[where][category] = max(peak[where][category], size)
            if chosen.strategy is ExecutionStrategy.REPLICATE:
                usage[device][MemoryCategory.PARAMETER] += expert.nbytes
                copies[demand.key, device] = (len(decisions), True)
                proposed.append(ExpertPlacement(demand.key, device, ready=False))
            decisions.append(DispatchDecision(demand, chosen, tuple(candidates)))

        def frozen(mapping):
            return MappingProxyType({device: MappingProxyType(dict(categories))
                                     for device, categories in mapping.items()})
        return ExecutionPlan(tuple(decisions), frozen(usage), frozen(peak), tuple(proposed))


@dataclass(frozen=True, slots=True)
class DemandForecast:
    key: ParameterKey
    origin: Any
    expected_uses: float


class RollingDemandPredictor:
    """Bounded EWMA demand hints for prefetch/placement, never model gating."""

    def __init__(self, *, decay: float = 0.8, max_entries: int = 10000):
        self.decay = _number(decay, "decay")
        if self.decay >= 1:
            raise ValueError("decay must be less than one")
        self.max_entries = _size(max_entries, "max_entries")
        if self.max_entries == 0:
            raise ValueError("max_entries must be positive")
        self._scores: OrderedDict[tuple[ParameterKey, Any], float] = OrderedDict()

    def observe(self, demands: Iterable[ExpertDemand]) -> None:
        counts: dict[tuple[ParameterKey, Any], int] = {}
        for demand in demands:
            if not isinstance(demand, ExpertDemand):
                raise TypeError("observe expects ExpertDemand values")
            identity = demand.key, demand.origin
            counts[identity] = counts.get(identity, 0) + demand.uses
        for identity in tuple(self._scores):
            self._scores[identity] *= self.decay
        for identity, count in counts.items():
            self._scores[identity] = self._scores.get(identity, 0.0) + (1 - self.decay) * count
            self._scores.move_to_end(identity)
        while len(self._scores) > self.max_entries:
            self._scores.popitem(last=False)

    def forecast(self, *, windows: float = 1.0, limit: int | None = None) -> tuple[DemandForecast, ...]:
        horizon = _number(windows, "windows", positive=True)
        if limit is not None:
            limit = _size(limit, "limit")
        ranked = sorted(self._scores.items(), key=lambda item: -item[1])
        if limit is not None:
            ranked = ranked[:limit]
        return tuple(DemandForecast(key, origin, score * horizon)
                     for (key, origin), score in ranked if score > 0)
