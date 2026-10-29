"""Optional adapter to Cellerator's host-native sparse relation primitives.

The Python interaction graph accepts arbitrary hashable logical identities and
Python callbacks. This adapter maps a graph's local identities to compact
native IDs, builds a reusable incidence plan, and exposes native CPU gather,
joint reduction, and scatter. Importing the interactions package does not load
the extension; call :func:`native_available` or construct an index explicitly.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
import importlib
from typing import Any

from .core import ActiveWorklist, InteractionGraph


def native_available() -> bool:
    """Return whether the optional ``_modeling_native`` module can be loaded."""
    try:
        importlib.import_module("cellerator._modeling_native")
    except ImportError:
        try:
            importlib.import_module("_modeling_native")
        except ImportError:
            return False
    return True


def _native_module():
    try:
        return importlib.import_module("cellerator._modeling_native")
    except ImportError:
        try:
            return importlib.import_module("_modeling_native")
        except ImportError as exc:
            raise RuntimeError(
                "native sparse interactions are unavailable; build the optional Cellerator modeling component"
            ) from exc


@dataclass(frozen=True)
class NativeIncidencePlan:
    """Native physical incidence plan plus its Python logical identity maps."""

    plan: Any
    entity_ids: tuple[Hashable, ...]
    interaction_ids: tuple[Hashable, ...]
    entity_id_to_slot: Mapping[Hashable, int]
    interaction_id_to_edge_id: Mapping[Hashable, int]

    @property
    def edge_ids(self) -> tuple[Hashable, ...]:
        """Logical interaction IDs in the plan's stable native edge order."""
        native_ids = self.plan.edge_ids
        edge_id_to_interaction = {v: k for k, v in self.interaction_id_to_edge_id.items()}
        return tuple(edge_id_to_interaction[i] for i in native_ids)

    @property
    def entity_slots(self) -> tuple[int, ...]:
        return tuple(self.plan.slots)

    @property
    def roles(self) -> tuple[int, ...]:
        return tuple(self.plan.roles)

    def gather(self, values: Any) -> Any:
        """Gather entity rows into incidence order (native contiguous f32/f64)."""
        return _native_module().gather(self.plan, values)

    def joint_reduce(self, values: Any, *, reduction: str = "sum",
                     role: int | None = None) -> Any:
        """Reduce participant rows once per active interaction."""
        return _native_module().joint_reduce(self.plan, values, reduction, role)

    def scatter_add(self, edge_values: Any, *, role: int | None = None) -> Any:
        """Scatter edge values to entity rows, retaining repeated incidences."""
        return _native_module().scatter_add(self.plan, edge_values, role)

    def joint_reduce_scatter(self, values: Any, *, reduction: str = "sum",
                             input_role: int | None = None,
                             output_role: int | None = None) -> Any:
        """Run sparse reduce then scatter without a dense pairwise intermediate."""
        return _native_module().joint_reduce_scatter(
            self.plan, values, reduction, input_role, output_role)


class NativeInteractionIndex:
    """Native incidence index for one immutable Python interaction graph.

    Compact IDs are local to this adapter. Python biological/entity identities
    remain authoritative and are retained in the reverse maps. Input and output
    incidences use roles 0 and 1, respectively.
    """

    INPUT_ROLE = 0
    OUTPUT_ROLE = 1

    def __init__(self, graph: InteractionGraph, *, relation_id: int = 1,
                 structure_id: int = 1, structure_epoch: int = 1) -> None:
        if not isinstance(graph, InteractionGraph):
            raise TypeError("graph must be an InteractionGraph")
        self._native = _native_module()
        self.graph = graph
        self._interaction_ids = tuple(r.id for r in graph.interactions)
        self._edge_ids = {identity: ordinal + 1
                          for ordinal, identity in enumerate(self._interaction_ids)}
        entities: dict[Hashable, int] = {}
        edges: list[tuple[int, list[tuple[int, int]]]] = []
        for relation in graph.interactions:
            participants: list[tuple[int, int]] = []
            for role, identities in ((self.INPUT_ROLE, relation.inputs),
                                     (self.OUTPUT_ROLE, relation.outputs)):
                for identity in identities:
                    if identity not in entities:
                        # Keep native logical entity ID zero available as a
                        # sentinel in the underlying support stack.
                        entities[identity] = len(entities) + 1
                    participants.append((role, entities[identity]))
            edges.append((self._edge_ids[relation.id], participants))
        self._entity_to_id = entities
        self._id_to_entity = tuple(entities)
        self.relation = self._native.SparseRelation(
            edges, relation_id, structure_id, structure_epoch)
        self.activity = self._native.ActivityState(self.relation, True)

    @property
    def entity_ids(self) -> tuple[Hashable, ...]:
        return self._id_to_entity

    @property
    def interaction_ids(self) -> tuple[Hashable, ...]:
        return self._interaction_ids

    @property
    def structure_epoch(self) -> int:
        return int(self.relation.structure_epoch)

    @property
    def activity_generation(self) -> int:
        return int(self.activity.generation)

    def set_active(self, interaction_ids: Iterable[Hashable], active: bool) -> None:
        """Set explicit structural activity by logical interaction identity."""
        native_ids = []
        for identity in interaction_ids:
            try:
                native_ids.append(self._edge_ids[identity])
            except KeyError as exc:
                raise KeyError(f"unknown interaction identity {identity!r}") from exc
        self.activity.set_active(native_ids, bool(active))

    def query(self, *, changed: Iterable[Hashable] | None = None,
              state: Mapping[Hashable, Any] | None = None,
              context: Any = None) -> ActiveWorklist:
        """Select active native candidates, then apply Python state gates.

        ``changed`` narrows lookup to interactions that read changed entities.
        Native activity and arbitrary Python predicates are both respected.
        """
        if changed is None:
            native_work = self.activity.worklist()
        else:
            terms = []
            for identity in changed:
                try:
                    terms.append((self.INPUT_ROLE, self._entity_to_id[identity]))
                except KeyError:
                    continue
            if not terms:
                return ActiveWorklist()
            native_work = self.activity.query(terms, False)
        active_ids = set(native_work.edge_ids)
        selected: list[Hashable] = []
        candidate_ids = tuple(identity for identity in self._interaction_ids
                              if self._edge_ids[identity] in active_ids)
        if any(self.graph[identity].active_if is not None for identity in candidate_ids) and state is None:
            raise ValueError("state is required to evaluate Python interaction gates")
        for identity in candidate_ids:
            gate = self.graph[identity].active_if
            if gate is None or gate(state, context):
                selected.append(identity)
        return ActiveWorklist(selected)

    def plan(self, worklist: Iterable[Hashable] | ActiveWorklist,
             entity_order: Iterable[Hashable], *, domain_id: int = 1,
             order_id: int = 1) -> NativeIncidencePlan:
        """Compile selected interactions for one explicit logical entity order."""
        identities = (worklist.drain() if isinstance(worklist, ActiveWorklist)
                      else tuple(dict.fromkeys(worklist)))
        entity_ids = tuple(entity_order)
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("entity_order contains duplicate identities")
        unknown_interactions = set(identities) - self._edge_ids.keys()
        if unknown_interactions:
            raise KeyError(f"worklist contains identities outside the graph: {unknown_interactions!r}")
        native_ids = [self._edge_ids[identity] for identity in identities]
        if len(native_ids) != len(set(native_ids)):
            raise ValueError("worklist contains duplicate interaction identities")
        entity_id_to_native = dict(self._entity_to_id)
        next_entity_id = max(entity_id_to_native.values(), default=0) + 1
        for identity in entity_ids:
            if identity not in entity_id_to_native:
                entity_id_to_native[identity] = next_entity_id
                next_entity_id += 1
        required_entities = {entity for identity in identities
                             for entity in self.graph[identity].inputs + self.graph[identity].outputs}
        missing_entities = required_entities - set(entity_ids)
        if missing_entities:
            raise ValueError(f"entity_order omits active interaction participants: {missing_entities!r}")
        # Native ActiveWorklist takes edge ordinals, while public IDs are
        # deliberately assigned as ordinal + 1 to avoid its zero sentinel.
        selected = self._native.ActiveWorklist(self.relation,
                                               [edge_id - 1 for edge_id in native_ids],
                                               self.activity_generation)
        layout = self._native.EntityLayout(
            [entity_id_to_native[identity] for identity in entity_ids],
            domain_id, order_id)
        plan = self._native.IncidencePlan(selected, layout)
        return NativeIncidencePlan(
            plan, entity_ids, identities,
            {identity: slot for slot, identity in enumerate(entity_ids)},
            {identity: self._edge_ids[identity] for identity in identities},
        )


__all__ = ["NativeIncidencePlan", "NativeInteractionIndex", "native_available"]
