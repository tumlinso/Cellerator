"""Versioned host-backed parameter residency with asynchronous demand pins.

Payloads and transfers remain caller-owned/backend-specific. This module tracks
bytes and lifetime; it never chooses model work or changes expert selection.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
import operator
import threading
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping


class MemoryCategory(str, Enum):
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    CELL_STATE = "cell_state"
    OPTIMIZER = "optimizer"


_CATEGORIES = tuple(MemoryCategory)
_KEEP_LATEST = object()
_DEFAULT_SOURCE = object()


class _ResidencyFuture(Future):
    """Read-only completion handle; caller cancellation cannot orphan a reservation."""

    def cancel(self) -> bool:
        return False

    def set_result(self, result: Any) -> None:
        raise RuntimeError("residency futures are completed by the manager")

    def set_exception(self, exception: BaseException) -> None:
        raise RuntimeError("residency futures are completed by the manager")

    def _resolve(self, result: Any) -> None:
        Future.set_result(self, result)

    def _reject(self, exception: BaseException) -> None:
        Future.set_exception(self, exception)


@dataclass(frozen=True, slots=True)
class ParameterKey:
    """Caller-defined expert identity plus immutable parameter revision."""

    expert_id: Hashable
    revision: Hashable

    def __post_init__(self) -> None:
        hash(self.expert_id)
        hash(self.revision)


@dataclass(slots=True)
class _Backing:
    key: ParameterKey
    payload: Any
    source: Hashable
    nbytes: int
    category: MemoryCategory


@dataclass(slots=True)
class _Entry:
    key: ParameterKey
    device: Hashable
    nbytes: int
    category: MemoryCategory
    future: Future
    state: str  # pending, completing, ready, failed
    payload: Any = None
    pins: set[int] = field(default_factory=set)
    source: bool = False
    touched: int = 0
    transfer_source: Any = None
    source_pin: tuple["_Entry", int] | None = None


class ResidencyError(RuntimeError):
    """A residency request failed without changing logical model selection."""


class CapacityError(ResidencyError):
    """Capacity is insufficient after evicting every safe cache candidate."""


class ResidencyLease:
    """One request pin. Keep it until queued or active device work has finished."""

    __slots__ = ("_manager", "_entry", "_token", "_released")

    def __init__(self, manager: "ResidencyManager", entry: _Entry, token: int):
        self._manager = manager
        self._entry = entry
        self._token = token
        self._released = False

    @property
    def future(self) -> Future:
        """Future resolving to this exact revision's resident payload."""
        return self._entry.future

    @property
    def key(self) -> ParameterKey:
        return self._entry.key

    @property
    def device(self) -> Hashable:
        return self._entry.device

    @property
    def source(self) -> Hashable:
        """Source actually used; an already pending target may have another source."""
        return self._entry.transfer_source

    def result(self, timeout: float | None = None) -> Any:
        """Wait for transfer completion and return the pinned payload."""
        return self.future.result(timeout)

    def release(self) -> None:
        """Release once; repeated calls are harmless."""
        if not self._released:
            self._released = True
            self._manager._release(self._entry, self._token)

    def __enter__(self) -> Any:
        return self.result()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class ResidencyManager:
    """Host-backed cache for caller-selected versioned work.

    ``capacities`` maps device identities to byte budgets. Device transfers are
    supplied as ``transfer(payload, source, target)`` and may return a
    ``concurrent.futures.Future`` or a completed payload. The manager never
    waits on transfers, creates worker threads or imports a device framework.
    The callback itself must return promptly if it is to run asynchronously.
    Host backing is retained when a replica is evicted.

    Capacity includes both in-flight reservations and completed resident copies.
    Demand ``acquire`` pins before a transfer is launched. Every use must release
    its lease only after dependent work has completed. A failed shared transfer
    fails all waiting leases and drops all their pins/reservations. ``prefetch``
    is advisory and unpinned, so it may be evicted before its future is consumed.

    A source entry registered on a managed device is caller-owned and cannot be
    evicted. Unpinned completed destination entries are evicted in deterministic
    least-recently-used order. Payload size accounting is supplied by the caller;
    bytes held by unmanaged host backing are not included in a device budget.
    """

    def __init__(
        self,
        capacities: Mapping[Hashable, int],
        transfer: Callable[[Any, Hashable, Hashable], Any],
        *,
        dispose: Callable[[Any, Hashable], None] | None = None,
    ) -> None:
        if not isinstance(capacities, Mapping) or not capacities:
            raise ValueError("capacities must be a nonempty device-to-byte mapping")
        parsed = {}
        for device, capacity in capacities.items():
            hash(device)
            if isinstance(capacity, bool):
                raise TypeError("device capacities must be nonnegative integers")
            capacity = operator.index(capacity)
            if capacity < 0:
                raise ValueError("device capacities must be nonnegative")
            parsed[device] = capacity
        if not callable(transfer):
            raise TypeError("transfer must be callable")
        if dispose is not None and not callable(dispose):
            raise TypeError("dispose must be callable or None")
        self.capacities = MappingProxyType(parsed)
        self._transfer = transfer
        self._dispose = dispose
        self._backings: dict[ParameterKey, _Backing] = {}
        self._latest: dict[Hashable, Hashable] = {}
        self._entries: dict[tuple[ParameterKey, Hashable], _Entry] = {}
        self._usage: dict[Hashable, dict[MemoryCategory, int]] = {
            device: {category: 0 for category in _CATEGORIES} for device in parsed
        }
        self._lock = threading.RLock()
        self._clock = 0
        self._token = 0
        self._disposal_errors: list[BaseException] = []

    @staticmethod
    def _size(nbytes: Any) -> int:
        if isinstance(nbytes, bool):
            raise TypeError("nbytes must be a nonnegative integer")
        result = operator.index(nbytes)
        if result < 0:
            raise ValueError("nbytes must be nonnegative")
        return result

    @staticmethod
    def _category(category: MemoryCategory) -> MemoryCategory:
        if not isinstance(category, MemoryCategory):
            raise TypeError("category must be a MemoryCategory")
        return category

    def _touch(self, entry: _Entry) -> None:
        self._clock += 1
        entry.touched = self._clock

    def _remove_entry_locked(self, entry: _Entry) -> Any:
        identity = (entry.key, entry.device)
        if self._entries.get(identity) is not entry:
            return None
        del self._entries[identity]
        if entry.device in self._usage:
            self._usage[entry.device][entry.category] -= entry.nbytes
        if entry.source:
            return None  # Registered payload is owned by the caller.
        return entry.payload

    def _make_room_locked(self, device: Hashable, nbytes: int) -> list[Any]:
        if device not in self.capacities:
            return []  # Unmanaged placement has no budget in this manager.
        capacity = self.capacities[device]
        used = sum(self._usage[device].values())
        if nbytes > capacity:
            raise CapacityError(f"{nbytes} bytes exceed {device!r} capacity {capacity}")
        need = max(0, used + nbytes - capacity)
        candidates = sorted((entry for entry in self._entries.values()
                             if entry.device == device and entry.state == "ready"
                             and not entry.source and not entry.pins),
                            key=lambda entry: entry.touched)
        selected = []
        freed = 0
        for candidate in candidates:
            if freed >= need:
                break
            selected.append(candidate)
            freed += candidate.nbytes
        if freed < need:
            raise CapacityError(f"no evictable capacity remains on {device!r}")
        dispose = []
        for victim in selected:
            payload = self._remove_entry_locked(victim)
            if payload is not None:
                dispose.append((payload, device))
        return dispose

    def _dispose_items(self, items) -> None:
        if self._dispose is None:
            return
        errors = []
        for payload, device in items:
            try:
                self._dispose(payload, device)
            except Exception as exc:  # complete cleanup even when one backend release fails
                errors.append(exc)
        if errors:
            with self._lock:
                self._disposal_errors.extend(errors)

    def register(
        self,
        key: ParameterKey,
        payload: Any,
        *,
        nbytes: int,
        source: Hashable = "host",
        category: MemoryCategory = MemoryCategory.PARAMETER,
    ) -> None:
        """Register one immutable host-backed revision; revisions are explicit."""
        if not isinstance(key, ParameterKey):
            raise TypeError("key must be a ParameterKey")
        size = self._size(nbytes)
        category = self._category(category)
        hash(source)
        disposals = []
        with self._lock:
            if key in self._backings:
                raise ValueError("revision is already registered; use a new revision key")
            identity = (key, source)
            if source in self.capacities:
                disposals = self._make_room_locked(source, size)
            future = _ResidencyFuture()
            future._resolve(payload)
            entry = _Entry(key, source, size, category, future, "ready",
                           payload=payload, source=True, transfer_source=source)
            self._touch(entry)
            self._entries[identity] = entry
            if source in self._usage:
                self._usage[source][category] += size
            self._backings[key] = _Backing(key, payload, source, size, category)
            self._latest[key.expert_id] = key.revision
        self._dispose_items(disposals)

    def _begin_transfer(self, entry: _Entry, backing: _Backing) -> None:
        """Start callback outside the lock; pending entry and demand pins exist."""
        try:
            result = self._transfer(backing.payload, backing.source, entry.device)
        except BaseException as exc:
            self._finish(entry, error=exc)
            return
        if isinstance(result, Future):
            result.add_done_callback(lambda completed: self._copy_future(entry, completed))
        else:
            self._finish(entry, payload=result)

    def _copy_future(self, entry: _Entry, source_future: Future) -> None:
        try:
            self._finish(entry, payload=source_future.result())
        except BaseException as exc:
            self._finish(entry, error=exc)

    def _finish(self, entry: _Entry, *, payload: Any = None,
                error: BaseException | None = None) -> None:
        disposals = []
        with self._lock:
            identity = (entry.key, entry.device)
            if self._entries.get(identity) is not entry or entry.state != "pending":
                # This can only follow explicit invalidation after a bad backend
                # callback race. Keep the late resource from leaking silently.
                if error is None and payload is not None:
                    disposals.append((payload, entry.device))
                failed_entry = True
            else:
                failed_entry = False
                if error is None:
                    # Keep the entry non-evictable while resolving outside the
                    # lock: user Future callbacks run synchronously and may
                    # wait on other threads that need this manager's lock.
                    entry.payload = payload
                    entry.state = "completing"
                    self._touch(entry)
                else:
                    entry.state = "failed"
                    if entry.device in self._usage:
                        self._usage[entry.device][entry.category] -= entry.nbytes
                    del self._entries[identity]
                    entry.pins.clear()
                source_pin = entry.source_pin
                entry.source_pin = None
                if source_pin is not None:
                    self._release_locked(*source_pin)
        self._dispose_items(disposals)
        if not failed_entry:
            if error is not None:
                entry.future._reject(error)
            else:
                entry.future._resolve(payload)
                with self._lock:
                    if self._entries.get((entry.key, entry.device)) is entry \
                            and entry.state == "completing":
                        entry.state = "ready"
                        self._touch(entry)

    def _pin_locked(self, entry: _Entry) -> int:
        token = self._token
        self._token += 1
        entry.pins.add(token)
        self._touch(entry)
        return token

    def _release_locked(self, entry: _Entry, token: int) -> None:
        entry.pins.discard(token)
        if self._entries.get((entry.key, entry.device)) is entry:
            self._touch(entry)

    def _entry_locked(self, key: ParameterKey, device: Hashable, *,
                      source: Any = _DEFAULT_SOURCE) -> tuple[_Entry, bool, list[Any], _Backing | None]:
        identity = (key, device)
        existing = self._entries.get(identity)
        if existing is not None:
            self._touch(existing)
            return existing, False, [], None
        backing = self._backings.get(key)
        if backing is None:
            raise KeyError(f"no host backing is registered for {key!r}")
        if source is _DEFAULT_SOURCE:
            selected = backing
            source_entry_pin = None
        elif source == backing.source:
            selected = backing
            source_entry_pin = None
        else:
            source_identity = (key, source)
            source_entry = self._entries.get(source_identity)
            if source_entry is None:
                raise KeyError(f"no exact-version resident source {source!r} for {key!r}")
            if source_entry.state != "ready" or not source_entry.future.done():
                raise ResidencyError("explicit transfer source is not fully resident")
            if source_entry.nbytes != backing.nbytes or source_entry.category != backing.category:
                raise ResidencyError("explicit transfer source size/category disagrees with registered revision")
            source_token = self._pin_locked(source_entry)
            source_entry_pin = (source_entry, source_token)
            selected = _Backing(key, source_entry.payload, source, source_entry.nbytes,
                                source_entry.category)
        if device not in self.capacities:
            # Unmanaged targets can transfer but have no capacity accounting;
            # lease lifetime still pins the logical placement against invalidation.
            disposals = []
        else:
            try:
                disposals = self._make_room_locked(device, backing.nbytes)
            except Exception:
                if source_entry_pin is not None:
                    self._release_locked(*source_entry_pin)
                raise
        future = _ResidencyFuture()
        at_source = device == selected.source
        entry = _Entry(key, device, selected.nbytes, selected.category, future,
                       "ready" if at_source else "pending",
                       payload=selected.payload if at_source else None, source=at_source,
                       transfer_source=selected.source, source_pin=source_entry_pin)
        self._touch(entry)
        self._entries[identity] = entry
        if device in self._usage:
            self._usage[device][entry.category] += entry.nbytes
        if at_source:
            future._resolve(selected.payload)
            if source_entry_pin is not None:
                self._release_locked(*source_entry_pin)
                entry.source_pin = None
        return entry, not at_source, disposals, selected

    def acquire(self, key: ParameterKey, device: Hashable, *,
                source: Hashable = _DEFAULT_SOURCE) -> ResidencyLease:
        """Pin a demand copy before transfer starts; caller releases after use."""
        hash(device)
        with self._lock:
            entry, start, disposals, backing = self._entry_locked(key, device, source=source)
            token = self._token
            self._token += 1
            entry.pins.add(token)
            self._touch(entry)
            lease = ResidencyLease(self, entry, token)
        self._dispose_items(disposals)
        # Transfers are kicked only after the request pin is visible.
        if start:
            self._begin_transfer(entry, backing)
        return lease

    def prefetch(self, key: ParameterKey, device: Hashable, *,
                 source: Hashable = _DEFAULT_SOURCE) -> Future:
        """Start or join a transfer without pinning; completion remains advisory."""
        hash(device)
        with self._lock:
            entry, start, disposals, backing = self._entry_locked(key, device, source=source)
            future = entry.future
        self._dispose_items(disposals)
        if start:
            self._begin_transfer(entry, backing)
        return future

    def _release(self, entry: _Entry, token: int) -> None:
        with self._lock:
            self._release_locked(entry, token)

    def resident(self, key: ParameterKey, device: Hashable) -> bool:
        """True only for a completed, successful entry of this exact revision."""
        with self._lock:
            entry = self._entries.get((key, device))
            return entry is not None and entry.state == "ready" and entry.future.done() \
                and entry.future.exception() is None

    def evict(self, key: ParameterKey, device: Hashable) -> bool:
        """Evict one safe completed copy; caller-owned backing is never evicted."""
        with self._lock:
            entry = self._entries.get((key, device))
            if entry is None or entry.state != "ready" or entry.source or entry.pins:
                return False
            payload = self._remove_entry_locked(entry)
        self._dispose_items(() if payload is None else ((payload, device),))
        return True

    @property
    def disposal_errors(self) -> tuple[BaseException, ...]:
        """Backend disposal failures; cache accounting still reflects eviction."""
        with self._lock:
            return tuple(self._disposal_errors)

    def usage(self, device: Hashable) -> Mapping[MemoryCategory, int]:
        """Resident plus in-flight reserved bytes, separated by storage role."""
        with self._lock:
            if device not in self._usage:
                raise KeyError(f"device {device!r} has no managed capacity")
            return MappingProxyType(dict(self._usage[device]))

    def latest_revision(self, expert_id: Hashable) -> Hashable:
        """Return the last registered revision (caller defines its meaning)."""
        with self._lock:
            return self._latest[expert_id]

    def invalidate(self, expert_id: Hashable, *,
                   keep_revision: Hashable = _KEEP_LATEST) -> tuple[ParameterKey, ...]:
        """Drop older revisions by default, preserving the latest host backing.

        Supplying ``keep_revision`` explicitly selects the revision to retain.
        All matching versions are checked before any mutation; an active or
        in-flight demand pin rejects the entire invalidation. Device copies are
        disposed after metadata/usage is atomically removed. Caller-owned source
        payloads are unregistered but never passed to ``dispose``.
        """
        hash(expert_id)
        disposals = []
        with self._lock:
            revisions = [key.revision for key in self._backings if key.expert_id == expert_id]
            if not revisions:
                return ()
            retained = self._latest[expert_id] if keep_revision is _KEEP_LATEST else keep_revision
            if retained not in revisions:
                raise KeyError(f"revision {retained!r} is not registered for {expert_id!r}")
            dropping = {key for key in self._backings
                        if key.expert_id == expert_id and key.revision != retained}
            blocked = [entry for entry in self._entries.values()
                       if entry.key in dropping and (entry.pins or entry.state != "ready")]
            if blocked:
                raise ResidencyError("cannot invalidate pinned or in-flight revisions")
            for key in dropping:
                for identity, entry in tuple(self._entries.items()):
                    if entry.key != key:
                        continue
                    payload = self._remove_entry_locked(entry)
                    if payload is not None:
                        disposals.append((payload, entry.device))
                del self._backings[key]
            self._latest[expert_id] = retained
        self._dispose_items(disposals)
        return tuple(sorted(dropping, key=lambda key: repr(key.revision)))

    def purge(self, expert_id: Hashable) -> tuple[ParameterKey, ...]:
        """Explicitly remove all versions after callers release their leases."""
        hash(expert_id)
        with self._lock:
            keys = [key for key in self._backings if key.expert_id == expert_id]
            if not keys:
                return ()
            # A sentinel is not a legal revision. Perform the same atomic pin
            # check then remove all matching records.
            blocked = [entry for entry in self._entries.values()
                       if entry.key.expert_id == expert_id and (entry.pins or entry.state != "ready")]
            if blocked:
                raise ResidencyError("cannot purge pinned or in-flight revisions")
            disposals = []
            for key in keys:
                for entry in tuple(self._entries.values()):
                    if entry.key == key:
                        payload = self._remove_entry_locked(entry)
                        if payload is not None:
                            disposals.append((payload, entry.device))
                del self._backings[key]
            self._latest.pop(expert_id, None)
        self._dispose_items(disposals)
        return tuple(sorted(keys, key=lambda key: repr(key.revision)))
