"""Sparse, discrete-count reaction networks with explicit hazard dependencies.

Propensities are inverse-time hazards, constant between state changes. Learned
hazards are welcome, but their complete state read sets must be declared (or
left unspecified to invalidate all channels). This is a CPU reference backend.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
import random
from types import MappingProxyType
from typing import Any

from ..modeling import StepContext, StepResult
from .core import JumpChannel, JumpOccurrence


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("molecule counts must be integers")
    if value < 0:
        raise ValueError("molecule counts must be nonnegative")
    return int(value)


class SpeciesLayout:
    """Immutable identity-to-slot mapping; identifiers never mean storage slots."""
    __slots__ = ("_species", "_slots")

    def __init__(self, species: Iterable[Hashable]):
        self._species = tuple(species)
        slots = {species_id: slot for slot, species_id in enumerate(self._species)}
        if len(slots) != len(self._species):
            raise ValueError("species identities must be unique")
        self._slots = MappingProxyType(slots)

    @property
    def species(self) -> tuple[Hashable, ...]:
        return self._species

    def slot(self, species: Hashable) -> int:
        return self._slots[species]

    def __len__(self) -> int:
        return len(self._species)


class ReactionState(Mapping):
    """Immutable discrete counts with identity-addressed access and explicit layout."""
    __slots__ = ("_layout", "_counts")

    def __init__(self, layout: SpeciesLayout, counts: Sequence[int]):
        values = tuple(_count(value) for value in counts)
        if len(values) != len(layout):
            raise ValueError("count length does not match species layout")
        self._layout, self._counts = layout, values

    @property
    def layout(self) -> SpeciesLayout:
        return self._layout

    @property
    def counts(self) -> tuple[int, ...]:
        return self._counts

    def __getitem__(self, species: Hashable) -> int:
        return self._counts[self._layout.slot(species)]

    def __iter__(self):
        return iter(self._layout.species)

    def __len__(self) -> int:
        return len(self._layout)


@dataclass(frozen=True, slots=True)
class Reaction:
    """One named channel and sparse net count changes.

    ``dependencies=None`` is a safe all-species dependency. An explicit empty
    tuple declares a state-independent hazard. Consumption feasibility is
    always included in the dependency index. Callbacks must be pure and must
    not depend on time/dt or mutable external parameters during an SSA span.
    """
    name: str
    changes: Mapping[Hashable, int]
    propensity: Callable[[Mapping, StepContext], float]
    dependencies: tuple[Hashable, ...] | None = None
    time_dependent: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("reaction name must be a nonempty string")
        if not callable(self.propensity):
            raise TypeError("propensity must be callable")
        changes = {}
        for species, delta in self.changes.items():
            if isinstance(delta, bool) or not isinstance(delta, Integral):
                raise TypeError("stoichiometric changes must be integers")
            if delta:
                changes[species] = int(delta)
        object.__setattr__(self, "changes", MappingProxyType(changes))
        if self.dependencies is not None:
            object.__setattr__(self, "dependencies", tuple(self.dependencies))

    @classmethod
    def mass_action(cls, name: str, reactants: Mapping, products: Mapping,
                    rate: float) -> Reaction:
        """Count hazard ``rate * product(comb(count, stoichiometry))``.

        ``rate`` is a stochastic per-combination rate, not an automatically
        volume-adjusted concentration rate. Catalysts may appear on both sides.
        """
        k = float(rate)
        if not math.isfinite(k) or k < 0:
            raise ValueError("mass-action rate must be finite and nonnegative")
        reactants = {s: _count(n) for s, n in reactants.items()}
        products = {s: _count(n) for s, n in products.items()}
        reactants = {s: n for s, n in reactants.items() if n}
        products = {s: n for s, n in products.items() if n}
        changes = {s: products.get(s, 0) - reactants.get(s, 0)
                   for s in dict.fromkeys((*reactants, *products))}

        def hazard(state, context):
            value = k
            for species, order in reactants.items():
                value *= math.comb(state[species], order)
            return value

        return cls(name, changes, hazard, tuple(reactants))


class _CountView(Mapping):
    """Borrowed read-only callback view; do not retain it outside a callback."""
    __slots__ = ("layout", "counts")

    def __init__(self, layout, counts):
        self.layout, self.counts = layout, counts

    def __getitem__(self, key):
        return self.counts[self.layout.slot(key)]

    def __iter__(self):
        return iter(self.layout.species)

    def __len__(self):
        return len(self.layout)


class PropensityWorkspace:
    """Private count copy and cached hazards, updated only by declared dependencies.

    ``fire`` validates every sparse change before committing any count. Hazard
    recomputation is a separate operation, exposed through ``refresh``. Each
    workspace belongs to one simulation/state and must not be shared across
    concurrent trajectories. Rebuild it when external parameters change.
    """
    def __init__(self, network: ReactionNetwork, state: ReactionState):
        network._check_state(state)
        self.network = network
        self._counts = list(state.counts)
        self._view = _CountView(network.layout, self._counts)
        self._rates = [0.0] * len(network.reactions)
        self._dirty = set(range(len(network.reactions)))
        self.evaluations = 0

    @property
    def pending(self) -> tuple[int, ...]:
        return tuple(sorted(self._dirty))

    @property
    def rates(self) -> tuple[float, ...]:
        if self._dirty:
            raise RuntimeError("propensities are stale; refresh the pending worklist")
        return tuple(self._rates)

    def snapshot(self) -> ReactionState:
        return ReactionState(self.network.layout, self._counts)

    def invalidate_all(self) -> None:
        self._dirty.update(range(len(self.network.reactions)))

    def refresh(self, context: StepContext) -> tuple[int, ...]:
        pending = self.pending
        updates = []
        for index in pending:
            reaction = self.network.reactions[index]
            feasible = all(self._counts[slot] + delta >= 0
                           for slot, delta in self.network._changes[index])
            raw = reaction.propensity(self._view, context) if feasible else 0.0
            if isinstance(raw, bool):
                raise TypeError("propensities must be numeric, not bool")
            rate = float(raw)
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(f"invalid propensity for {reaction.name!r}")
            updates.append((index, rate))
        # Failed callbacks cannot leave a partially refreshed cache usable.
        for index, rate in updates:
            self._rates[index] = rate
        self.evaluations += len(pending)
        self._dirty.clear()
        return pending

    def fire(self, channel: int | str) -> tuple[Hashable, ...]:
        index = self.network.channel_index(channel)
        changes = self.network._changes[index]
        proposed = [(slot, self._counts[slot] + delta) for slot, delta in changes]
        if any(value < 0 for _, value in proposed):
            raise ValueError("reaction would produce a negative molecule count")
        for slot, value in proposed:
            self._counts[slot] = value
            self._dirty.update(self.network._dependents[slot])
        return tuple(self.network.layout.species[slot] for slot, _ in changes)


@dataclass(frozen=True, slots=True)
class ReactionHistoryScore:
    """Path log density under supplied hazards, relative to event-time measure.

    This scores a fully observed channel/time history. It does not marginalize
    hidden events, fit parameters, or supply an observation likelihood.
    """
    log_likelihood: float
    log_hazard: float
    integrated_hazard: float
    terminal_state: ReactionState


class ReactionNetwork:
    """Immutable sparse reaction structure shared by independent trajectories."""
    __slots__ = ("_layout", "_reactions", "_names", "_changes", "_dependents")

    def __init__(self, species: Iterable[Hashable], reactions: Iterable[Reaction]):
        self._layout = SpeciesLayout(species)
        self._reactions = tuple(reactions)
        if any(not isinstance(r, Reaction) for r in self.reactions):
            raise TypeError("reactions must contain Reaction instances")
        self._names = {reaction.name: i for i, reaction in enumerate(self.reactions)}
        if len(self._names) != len(self.reactions):
            raise ValueError("reaction names must be unique")
        self._changes = tuple(tuple((self.layout.slot(s), delta)
                                    for s, delta in reaction.changes.items())
                              for reaction in self.reactions)
        dependents = [set() for _ in self.layout.species]
        for i, reaction in enumerate(self.reactions):
            reads = self.layout.species if reaction.dependencies is None else reaction.dependencies
            for species_id in reads:
                dependents[self.layout.slot(species_id)].add(i)
            for slot, delta in self._changes[i]:
                if delta < 0:
                    dependents[slot].add(i)
        self._dependents = tuple(tuple(sorted(indices)) for indices in dependents)

    @property
    def layout(self) -> SpeciesLayout:
        return self._layout

    @property
    def reactions(self) -> tuple[Reaction, ...]:
        return self._reactions

    def state(self, counts: Mapping[Hashable, int]) -> ReactionState:
        """Construct state by identity; omitted species start at zero."""
        for species in counts:
            self.layout.slot(species)
        return ReactionState(self.layout, tuple(counts.get(s, 0) for s in self.layout.species))

    def _check_state(self, state: ReactionState) -> None:
        if not isinstance(state, ReactionState) or state.layout is not self.layout:
            raise ValueError("state must use this network's species layout; remap by identity")

    def channel_index(self, channel: int | str) -> int:
        if isinstance(channel, str):
            return self._names[channel]
        if isinstance(channel, bool) or not isinstance(channel, Integral):
            raise TypeError("channel must be a name or integer index")
        if not 0 <= channel < len(self.reactions):
            raise IndexError("reaction channel out of bounds")
        return int(channel)

    def affected(self, species: Iterable[Hashable]) -> tuple[int, ...]:
        """Deterministic deduplicated propensity worklist for changed identities."""
        return tuple(sorted({i for s in species for i in self._dependents[self.layout.slot(s)]}))

    def workspace(self, state: ReactionState) -> PropensityWorkspace:
        return PropensityWorkspace(self, state)

    def apply(self, state: ReactionState, channel: int | str) -> ReactionState:
        work = self.workspace(state)
        work.fire(channel)
        return work.snapshot()

    def channels(self) -> tuple[JumpChannel, ...]:
        """Adapt to the existing generic Gillespie interface (full refresh)."""
        if any(r.time_dependent for r in self.reactions):
            raise ValueError("time-dependent hazards require a different SSA algorithm")
        channels = []
        for index, reaction in enumerate(self.reactions):
            def rate(state, context, i=index):
                self._check_state(state)
                if any(state.counts[slot] + delta < 0 for slot, delta in self._changes[i]):
                    return 0.0
                return self.reactions[i].propensity(state, context)
            def update(state, context, i=index):
                return self.apply(state, i)
            channels.append(JumpChannel(reaction.name, rate, update))
        return tuple(channels)

    def conservation_residuals(self, weights: Mapping[Hashable, int]) -> dict[str, int]:
        """Exact weighted net change per event; zero means a structural invariant."""
        normalized = {}
        for species, weight in weights.items():
            self.layout.slot(species)
            if isinstance(weight, bool) or not isinstance(weight, Integral):
                raise TypeError("conservation weights must be integers")
            normalized[species] = int(weight)
        return {r.name: sum(normalized.get(s, 0) * delta for s, delta in r.changes.items())
                for r in self.reactions}

    def score_history(self, state: ReactionState, events: Iterable[JumpOccurrence],
                      context: StepContext) -> ReactionHistoryScore:
        """Replay and score an ordered complete event history on ``[time,time+dt)``.

        The density is ``sum(log(fired_hazard)) - integral(total_hazard)``.
        Zero-hazard events have negative-infinite log density; infeasible count
        transitions and malformed histories are rejected. Hazards obey the
        same state-only dependency contract as :meth:`simulate`.
        """
        if any(r.time_dependent for r in self.reactions):
            raise ValueError("time-dependent hazards require integrated-hazard scoring")
        now, duration = float(context.time), float(context.dt)
        end = now + duration
        if not all(math.isfinite(x) for x in (now, duration, end)) or duration < 0:
            raise ValueError("time and nonnegative duration must be finite")
        if duration > 0 and end <= now:
            raise ArithmeticError("horizon cannot advance at binary64 precision")
        work = self.workspace(state)
        log_hazard, exposure = 0.0, 0.0
        for event in events:
            event_time = float(event.time)
            if not math.isfinite(event_time) or not now < event_time < end:
                raise ValueError("event times must increase strictly inside the horizon")
            index = self.channel_index(event.channel)
            work.refresh(context)
            rates = work.rates
            total = math.fsum(rates)
            if not math.isfinite(total):
                raise OverflowError("total propensity is not finite")
            exposure += total * (event_time - now)
            log_hazard += math.log(rates[index]) if rates[index] > 0 else -math.inf
            work.fire(index)
            now = event_time
        work.refresh(context)
        total = math.fsum(work.rates)
        if not math.isfinite(total):
            raise OverflowError("total propensity is not finite")
        exposure += total * (end - now)
        return ReactionHistoryScore(log_hazard - exposure, log_hazard, exposure, work.snapshot())

    def simulate(self, state: ReactionState, context: StepContext, *,
                 max_events: int = 100_000) -> StepResult:
        """Direct SSA with sparse hazard refresh and a half-open event horizon.

        State-only hazards are exact under the declared dependency contract.
        Time-dependent hazards require integrated hazards/thinning and are
        rejected. Binary64 physical time and hazards; exact integer counts.
        """
        if any(r.time_dependent for r in self.reactions):
            raise ValueError("time-dependent hazards require a different SSA algorithm")
        if isinstance(max_events, bool) or not isinstance(max_events, Integral) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        now, duration = float(context.time), float(context.dt)
        end = now + duration
        if not all(math.isfinite(x) for x in (now, duration, end)) or duration < 0:
            raise ValueError("time and nonnegative duration must be finite")
        if duration > 0 and end <= now:
            raise ArithmeticError("horizon cannot advance at binary64 precision")
        rng = random if context.rng is None else context.rng
        work = self.workspace(state)
        events = []
        while now < end:
            # Time/dt are deliberately fixed for the hazard callback contract.
            work.refresh(context)
            rates = work.rates
            total = math.fsum(rates)
            if not math.isfinite(total):
                raise OverflowError("total propensity is not finite")
            if total == 0:
                break
            u = float(rng.random())
            if not 0 <= u < 1:
                raise ValueError("rng.random() must return a value in [0, 1)")
            wait = -math.log1p(-u) / total
            event_time = now + wait
            if event_time >= end:
                break
            if event_time <= now:
                raise ArithmeticError("event time cannot advance at binary64 precision")
            if len(events) == max_events:
                raise RuntimeError("max_events exceeded before the horizon")
            pick = float(rng.random())
            if not 0 <= pick < 1:
                raise ValueError("rng.random() must return a value in [0, 1)")
            threshold, cumulative = pick * total, 0.0
            selected = None
            for index, rate in enumerate(rates):
                cumulative += rate
                if rate > 0 and threshold < cumulative:
                    selected = index
                    break
            if selected is None:  # fsum/ordinary-sum last-bit discrepancy
                selected = next(i for i in reversed(range(len(rates))) if rates[i] > 0)
            work.fire(selected)
            now = event_time
            events.append(JumpOccurrence(self.reactions[selected].name, now))
        return StepResult(state=work.snapshot(), events=tuple(events), time=end,
                          metadata={"jump_count": len(events),
                                    "propensity_evaluations": work.evaluations})

    def __call__(self, state: ReactionState, context: StepContext) -> StepResult:
        return self.simulate(state, context)
