"""Sparse, identity-preserving interactions between model state entries.

The graph records relationships that could participate. State predicates decide
which relationships are evaluated for a particular update. No dense adjacency
matrix or biological ontology is required.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


State = Mapping[Hashable, Any]
Gate = Callable[[State, Any], bool]
Operation = Callable[[tuple[Any, ...], Any], Any]
Reducer = Literal["replace", "sum", "append"]


def _unique(values: Iterable[Hashable], label: str) -> tuple[Hashable, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate identities")
    return result


@dataclass(frozen=True)
class Interaction:
    """One ordered, possibly multi-input state transformation.

    ``inputs`` and ``outputs`` retain their declared order. ``kind`` labels the
    relation shape (directed, bipartite or hyperedge) without assigning
    biological meaning to identities. ``active_if`` gates execution while the
    structural relation remains present in the graph.

    The operation receives input values as a tuple and an opaque context. It may
    return a mapping from declared output identities to values, a value tuple
    aligned with ``outputs``, or a scalar for a single output.
    """

    id: Hashable
    inputs: tuple[Hashable, ...]
    outputs: tuple[Hashable, ...]
    operation: Operation = field(compare=False, repr=False)
    kind: Literal["directed", "bipartite", "hyperedge", "custom"] = "custom"
    active_if: Gate | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, Hashable):
            raise TypeError("interaction id must be hashable")
        _unique(self.inputs, "inputs")
        _unique(self.outputs, "outputs")
        if not self.outputs:
            raise ValueError("an interaction must declare at least one output")
        if not callable(self.operation):
            raise TypeError("operation must be callable")
        if self.kind == "hyperedge" and len(self.inputs) < 2:
            raise ValueError("a hyperedge must have at least two input participants")

    @classmethod
    def directed(cls, identity: Hashable, source: Hashable, target: Hashable,
                 operation: Operation, *, active_if: Gate | None = None) -> "Interaction":
        return cls(identity, (source,), (target,), operation, "directed", active_if)

    @classmethod
    def bipartite(cls, identity: Hashable, sources: Iterable[Hashable],
                  targets: Iterable[Hashable], operation: Operation, *,
                  active_if: Gate | None = None) -> "Interaction":
        return cls(identity, tuple(sources), tuple(targets), operation,
                   "bipartite", active_if)

    @classmethod
    def hyperedge(cls, identity: Hashable, participants: Iterable[Hashable],
                  targets: Iterable[Hashable], operation: Operation, *,
                  active_if: Gate | None = None) -> "Interaction":
        return cls(identity, tuple(participants), tuple(targets), operation,
                   "hyperedge", active_if)


@dataclass(frozen=True)
class InteractionResult:
    """Output of a synchronous evaluation over the selected worklist."""

    state: Mapping[Hashable, Any]
    writes: Mapping[Hashable, Any]
    evaluated: tuple[Hashable, ...]
    skipped: tuple[Hashable, ...]


class ActiveWorklist:
    """Stable deduplicating queue of candidate interaction identities."""

    def __init__(self, identities: Iterable[Hashable] = ()) -> None:
        self._queue: deque[Hashable] = deque()
        self._pending: set[Hashable] = set()
        self.extend(identities)

    def add(self, identity: Hashable) -> None:
        if identity not in self._pending:
            self._pending.add(identity)
            self._queue.append(identity)

    def extend(self, identities: Iterable[Hashable]) -> None:
        for identity in identities:
            self.add(identity)

    def pop(self) -> Hashable:
        identity = self._queue.popleft()
        self._pending.remove(identity)
        return identity

    def drain(self) -> tuple[Hashable, ...]:
        result = tuple(self._queue)
        self._queue.clear()
        self._pending.clear()
        return result

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(tuple(self._queue))


class InteractionGraph:
    """Sparse incidence index and state-dependent interaction evaluator.

    Relations are immutable after construction; re-create the graph to change
    structural identity/order. A candidate lookup visits only relations
    incident to changed input identities. Evaluation reads a common state
    snapshot and returns writes separately, giving explicit synchronous update
    semantics. Use ``reducer='sum'`` to combine multiple active writers to one
    output; the default rejects such conflicts.
    """

    def __init__(self, interactions: Iterable[Interaction]) -> None:
        ordered = tuple(interactions)
        by_id: dict[Hashable, Interaction] = {}
        incident: dict[Hashable, list[Hashable]] = defaultdict(list)
        produced: dict[Hashable, list[Hashable]] = defaultdict(list)
        for relation in ordered:
            if not isinstance(relation, Interaction):
                raise TypeError("interactions must contain Interaction values")
            if relation.id in by_id:
                raise ValueError(f"duplicate interaction identity: {relation.id!r}")
            by_id[relation.id] = relation
            for identity in relation.inputs:
                incident[identity].append(relation.id)
            for identity in relation.outputs:
                produced[identity].append(relation.id)
        self._ordered = ordered
        self._by_id = by_id
        self._incident = {key: tuple(value) for key, value in incident.items()}
        self._produced = {key: tuple(value) for key, value in produced.items()}

    @property
    def interactions(self) -> tuple[Interaction, ...]:
        return self._ordered

    def __len__(self) -> int:
        return len(self._ordered)

    def __iter__(self):
        return iter(self._ordered)

    def __getitem__(self, identity: Hashable) -> Interaction:
        return self._by_id[identity]

    def incident(self, identity: Hashable) -> tuple[Interaction, ...]:
        """Return input-side interactions in declaration order."""
        return tuple(self._by_id[i] for i in self._incident.get(identity, ()))

    def producing(self, identity: Hashable) -> tuple[Interaction, ...]:
        return tuple(self._by_id[i] for i in self._produced.get(identity, ()))

    def candidates(self, changed: Iterable[Hashable]) -> tuple[Interaction, ...]:
        """Find the stable sparse union of interactions reading changed state."""
        ids: set[Hashable] = set()
        for identity in changed:
            ids.update(self._incident.get(identity, ()))
        return tuple(relation for relation in self._ordered if relation.id in ids)

    def worklist(self, *, changed: Iterable[Hashable] | None = None,
                 state: State | None = None, context: Any = None,
                 include_inactive: bool = False) -> ActiveWorklist:
        relations = (self._ordered if changed is None else
                     self.candidates(changed))
        if include_inactive:
            return ActiveWorklist(r.id for r in relations)
        if state is None:
            raise ValueError("state is required to evaluate state-dependent gates")
        return ActiveWorklist(r.id for r in relations
                              if r.active_if is None or r.active_if(state, context))

    @staticmethod
    def _writes(relation: Interaction, value: Any) -> dict[Hashable, Any]:
        if isinstance(value, Mapping):
            unexpected = set(value) - set(relation.outputs)
            if unexpected:
                raise ValueError(f"interaction {relation.id!r} returned undeclared outputs {unexpected!r}")
            missing = set(relation.outputs) - set(value)
            if missing:
                raise ValueError(f"interaction {relation.id!r} omitted outputs {missing!r}")
            return {identity: value[identity] for identity in relation.outputs}
        if len(relation.outputs) == 1:
            return {relation.outputs[0]: value}
        if not isinstance(value, Sequence) or len(value) != len(relation.outputs):
            raise ValueError("multi-output operation must return a mapping or aligned sequence")
        return dict(zip(relation.outputs, value))

    def evaluate(self, state: State, *, context: Any = None,
                 worklist: Iterable[Hashable] | ActiveWorklist | None = None,
                 changed: Iterable[Hashable] | None = None,
                 reducer: Reducer = "replace") -> InteractionResult:
        """Evaluate active sparse relations against one state snapshot.

        ``worklist`` contains interaction identities. When omitted, all active
        relations are selected, or only relations incident to ``changed``.
        ``sum`` keeps upstream tensor/array arithmetic; ``append`` accumulates
        opaque values as tuples. ``replace`` requires at most one active writer
        per output.
        """
        if reducer not in ("replace", "sum", "append"):
            raise ValueError("reducer must be 'replace', 'sum', or 'append'")
        if worklist is None:
            selected = self.worklist(changed=changed, state=state, context=context)
            ids = selected.drain()
        elif isinstance(worklist, ActiveWorklist):
            ids = worklist.drain()
        else:
            ids = tuple(dict.fromkeys(worklist))
        writes: dict[Hashable, Any] = {}
        skipped: list[Hashable] = []
        evaluated: list[Hashable] = []
        for identity in ids:
            try:
                relation = self._by_id[identity]
            except KeyError as exc:
                raise KeyError(f"worklist references unknown interaction {identity!r}") from exc
            if relation.active_if is not None and not relation.active_if(state, context):
                skipped.append(identity)
                continue
            missing = tuple(source for source in relation.inputs if source not in state)
            if missing:
                raise KeyError(f"interaction {identity!r} requires missing inputs {missing!r}")
            values = tuple(state[source] for source in relation.inputs)
            produced = self._writes(relation, relation.operation(values, context))
            for target, value in produced.items():
                if target not in writes:
                    writes[target] = value
                elif reducer == "replace":
                    raise ValueError(f"multiple active interactions write {target!r}; choose an explicit reducer")
                elif reducer == "sum":
                    writes[target] = writes[target] + value
                else:
                    prior = writes[target]
                    writes[target] = (prior if isinstance(prior, tuple) else (prior,)) + (value,)
            evaluated.append(identity)
        updated: MutableMapping[Hashable, Any] = dict(state)
        for target, value in writes.items():
            if reducer == "sum" and target in state:
                value = state[target] + value
            elif reducer == "append" and target in state:
                prior = state[target]
                value = (prior if isinstance(prior, tuple) else (prior,)) + (
                    value if isinstance(value, tuple) else (value,))
            updated[target] = value
        return InteractionResult(updated, writes, tuple(evaluated), tuple(skipped))

    @classmethod
    def from_bipartite(cls, edges: Iterable[tuple[Hashable, Hashable]],
                       operation: Operation, *,
                       active_if: Gate | None = None) -> "InteractionGraph":
        """Group source-target edges into one sparse multi-input update per target."""
        sources_by_target: dict[Hashable, list[Hashable]] = {}
        seen: set[tuple[Hashable, Hashable]] = set()
        for source, target in edges:
            edge = (source, target)
            if edge in seen:
                raise ValueError(f"duplicate bipartite edge: {edge!r}")
            seen.add(edge)
            sources_by_target.setdefault(target, []).append(source)
        interactions = [Interaction.bipartite(
            ("bipartite", target), sources, (target,), operation,
            active_if=active_if,
        ) for target, sources in sources_by_target.items()]
        return cls(interactions)

    @classmethod
    def from_hyperedges(cls,
                        hyperedges: Iterable[tuple[Hashable, Iterable[Hashable], Iterable[Hashable]]],
                        operation: Operation, *,
                        active_if: Gate | None = None) -> "InteractionGraph":
        """Create ordered multi-participant interactions with overlapping members."""
        return cls(Interaction.hyperedge(identity, participants, outputs, operation,
                                         active_if=active_if)
                   for identity, participants, outputs in hyperedges)


__all__ = ["ActiveWorklist", "Interaction", "InteractionGraph", "InteractionResult"]
