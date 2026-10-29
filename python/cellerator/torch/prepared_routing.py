"""Thin owner-keyed adapter for the native prepared route-union index."""
from __future__ import annotations

from dataclasses import dataclass
from operator import index
from threading import RLock
from typing import Iterable


@dataclass(frozen=True)
class PreparedRoutingIdentity:
    """Dependencies that make one route projection valid for a model owner."""

    owner_identity: int
    relation_id: int
    structure_id: int
    structure_epoch: int
    route_generation: int
    binding_generation: int

    def __post_init__(self):
        raw_values = (self.owner_identity, self.relation_id, self.structure_id,
                      self.structure_epoch, self.route_generation,
                      self.binding_generation)
        if any(isinstance(value, bool) for value in raw_values):
            raise TypeError("prepared routing identity fields must be integers")
        try:
            values = tuple(index(value) for value in raw_values)
        except TypeError as error:
            raise TypeError("prepared routing identity fields must be integers") from error
        if any(value <= 0 for value in (values[0], values[1], values[2], values[3], values[5])):
            raise ValueError("owner, relation, structure, epoch, and binding identities must be positive")
        if values[4] < 0:
            raise ValueError("route generation must be nonnegative")
        for name, value in zip(self._fields(), values):
            object.__setattr__(self, name, value)

    @staticmethod
    def _fields():
        return ("owner_identity", "relation_id", "structure_id",
                "structure_epoch", "route_generation", "binding_generation")

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return tuple(getattr(self, name) for name in self._fields())


class PreparedEntityMechanismRouting:
    """Prepare an exact global mechanism-union index from route-table rows.

    ``entity_mechanisms`` is the existing entity-indexed route table. Its
    zero-based mechanism IDs and current discrete route policy are consumed as
    supplied; this adapter neither scores nor interprets route memberships.
    The native object owns the immutable prepared projection and its per-stream
    query workspaces. Per-query outputs are fresh tensors suitable for saved
    autograd worksets.
    """

    def __init__(self, native_module, entity_mechanisms: Iterable[Iterable[int]],
                 mechanism_count: int, identity: PreparedRoutingIdentity,
                 device: int):
        if not isinstance(identity, PreparedRoutingIdentity):
            raise TypeError("identity must be a PreparedRoutingIdentity")
        rows = tuple(tuple(map(int, row)) for row in entity_mechanisms)
        self.identity = identity
        self.entity_count = len(rows)
        self.mechanism_count = int(mechanism_count)
        if self.mechanism_count <= 0:
            raise ValueError("mechanism_count must be positive")
        self.device = int(device)
        self._native = native_module.PreparedEntityMechanismRouting(
            rows, self.entity_count, self.mechanism_count, *identity.as_tuple(),
            self.device)

    def query(self, entity_ids, *, identity: PreparedRoutingIdentity):
        """Return device ``(offsets:int64, mechanisms:int32)`` for each cell."""
        if identity != self.identity:
            raise ValueError("prepared route dependencies changed; rebuild the index")
        return self._native.query(entity_ids, *identity.as_tuple())


class _PreparedRouteIndexCache:
    """One module-owned prepared index, replaced when an explicit key changes."""

    def __init__(self):
        self._lock = RLock()
        self._binding_key = None
        self._binding_generation = 0
        self._identity = None
        self._router = None

    @property
    def identity(self):
        with self._lock:
            return self._identity

    def observe_binding(self, device_index: int, parameter_pointer: int,
                        native_binding_identity: int) -> int:
        key = (int(device_index), int(parameter_pointer),
               int(native_binding_identity))
        if key[0] < 0 or key[1] <= 0 or key[2] <= 0:
            raise ValueError("prepared route binding identity is invalid")
        with self._lock:
            if key != self._binding_key:
                self._binding_key = key
                self._binding_generation += 1
                self._invalidate_locked()
            return self._binding_generation

    def get(self, native_module, entity_mechanisms, mechanism_count: int,
            identity: PreparedRoutingIdentity, device_index: int):
        with self._lock:
            if self._router is not None and identity == self._identity:
                return self._router, False
            self._invalidate_locked()
            router = PreparedEntityMechanismRouting(
                native_module, entity_mechanisms, mechanism_count,
                identity, device_index)
            self._identity = identity
            self._router = router
            return router, True

    def invalidate(self) -> None:
        with self._lock:
            self._invalidate_locked()

    def _invalidate_locked(self) -> None:
        self._identity = None
        self._router = None
