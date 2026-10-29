"""Experimental inherited accessibility gates for state-dependent computation.

Rules describe caller-selected required/forbidden inputs. A compact bit mask
combines those conditions with per-program open/closed state. Forked states share
immutable structure and bit values; changes replace bits locally. A gate may be
passed into ``InteractionGraph`` without changing the model's own route choices.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AccessibilityRule:
    """An opaque program identity and caller-defined state requirements."""

    identity: Hashable
    required: tuple[Hashable, ...] = ()
    forbidden: tuple[Hashable, ...] = ()
    revision: int = 0
    initially_open: bool = True

    def __post_init__(self) -> None:
        try:
            hash(self.identity)
            for identity in (*self.required, *self.forbidden):
                hash(identity)
        except TypeError as exc:
            raise TypeError("accessibility identities must be hashable") from exc
        if len(set(self.required)) != len(self.required):
            raise ValueError("required accessibility inputs must be unique")
        if len(set(self.forbidden)) != len(self.forbidden):
            raise ValueError("forbidden accessibility inputs must be unique")
        if set(self.required).intersection(self.forbidden):
            raise ValueError("an input cannot be both required and forbidden")
        if isinstance(self.revision, bool) or self.revision < 0:
            raise ValueError("rule revision must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class AccessibilityChange:
    kind: str
    identity: Hashable
    enabled: bool
    generation: int
    provenance: Any = None


@dataclass(frozen=True, slots=True)
class Eligibility:
    allowed: bool
    structure_revision: int
    program_revision: int
    state_generation: int


@dataclass(frozen=True, slots=True)
class _CompiledRule:
    revision: int
    open_bit: int
    required_bits: int
    forbidden_bits: int


class AccessibilityState:
    """A model-independent open/closed state and compact active-input bitset.

    Rules and identity indexes are shared unchanged by :meth:`fork`; mutable
    states are Python integers, so a child starts sharing their immutable values
    and each update creates a branch-local integer. State provenance records
    discrete changes without deciding how biology produced the change.
    """

    def __init__(self, rules: Iterable[AccessibilityRule], *,
                 structure_revision: int = 0) -> None:
        ordered = tuple(rules)
        if isinstance(structure_revision, bool) or structure_revision < 0:
            raise ValueError("structure revision must be a nonnegative integer")
        rule_index: dict[Hashable, int] = {}
        input_index: dict[Hashable, int] = {}
        for i, rule in enumerate(ordered):
            if not isinstance(rule, AccessibilityRule):
                raise TypeError("rules must contain AccessibilityRule values")
            if rule.identity in rule_index:
                raise ValueError(f"duplicate accessibility rule {rule.identity!r}")
            rule_index[rule.identity] = i
            for identity in (*rule.required, *rule.forbidden):
                input_index.setdefault(identity, len(input_index))
        compiled: dict[Hashable, _CompiledRule] = {}
        open_bits = 0
        for rule in ordered:
            open_bit = 1 << rule_index[rule.identity]
            if rule.initially_open:
                open_bits |= open_bit
            required = sum(1 << input_index[i] for i in rule.required)
            forbidden = sum(1 << input_index[i] for i in rule.forbidden)
            compiled[rule.identity] = _CompiledRule(
                rule.revision, open_bit, required, forbidden
            )
        self._rules = ordered
        self._rule_index = rule_index
        self._input_index = input_index
        self._compiled = compiled
        self._structure_revision = int(structure_revision)
        self._open_bits = open_bits
        self._active_bits = 0
        self._generation = 0
        self._changes: tuple[AccessibilityChange, ...] = ()

    @property
    def structure_revision(self) -> int:
        return self._structure_revision

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def changes(self) -> tuple[AccessibilityChange, ...]:
        return self._changes

    @property
    def storage_bytes(self) -> int:
        return max(1, (len(self._rules) + 7) // 8) + max(1, (len(self._input_index) + 7) // 8)

    def shares_structure_with(self, other: "AccessibilityState") -> bool:
        return isinstance(other, AccessibilityState) and self._compiled is other._compiled

    def fork(self) -> "AccessibilityState":
        """Create an independent branch with the same current inherited state."""
        child = object.__new__(AccessibilityState)
        child._rules = self._rules
        child._rule_index = self._rule_index
        child._input_index = self._input_index
        child._compiled = self._compiled
        child._structure_revision = self._structure_revision
        child._open_bits = self._open_bits
        child._active_bits = self._active_bits
        child._generation = self._generation
        child._changes = self._changes
        return child

    def _rule(self, identity: Hashable) -> _CompiledRule:
        try:
            return self._compiled[identity]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown accessibility program: {identity!r}") from exc

    def _input_bit(self, identity: Hashable) -> int:
        try:
            return 1 << self._input_index[identity]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown accessibility input: {identity!r}") from exc

    def is_open(self, identity: Hashable) -> bool:
        return bool(self._open_bits & self._rule(identity).open_bit)

    def input_active(self, identity: Hashable) -> bool:
        return bool(self._active_bits & self._input_bit(identity))

    def _record(self, kind: str, identity: Hashable, enabled: bool, provenance: Any) -> None:
        self._generation += 1
        self._changes += (AccessibilityChange(kind, identity, enabled,
                                               self._generation, provenance),)

    def set_open(self, identity: Hashable, opened: bool, *, provenance: Any = None) -> bool:
        rule = self._rule(identity)
        was_open = bool(self._open_bits & rule.open_bit)
        if was_open == bool(opened):
            return False
        self._open_bits = (self._open_bits | rule.open_bit) if opened else (self._open_bits & ~rule.open_bit)
        self._record("program", identity, bool(opened), provenance)
        return True

    def set_input(self, identity: Hashable, active: bool, *, provenance: Any = None) -> bool:
        bit = self._input_bit(identity)
        was_active = bool(self._active_bits & bit)
        if was_active == bool(active):
            return False
        self._active_bits = (self._active_bits | bit) if active else (self._active_bits & ~bit)
        self._record("input", identity, bool(active), provenance)
        return True

    def set_inputs(self, updates: Iterable[tuple[Hashable, bool]], *,
                   provenance: Any = None) -> int:
        """Validate all input identities before changing the branch state."""
        staged: list[tuple[Hashable, int, bool]] = []
        seen: set[Hashable] = set()
        for identity, active in updates:
            if identity in seen:
                raise ValueError(f"duplicate input {identity!r} in accessibility update")
            seen.add(identity)
            bit = self._input_bit(identity)
            if bool(self._active_bits & bit) != bool(active):
                staged.append((identity, bit, bool(active)))
        if not staged:
            return 0
        for _, bit, active in staged:
            self._active_bits = (self._active_bits | bit) if active else (self._active_bits & ~bit)
        self._generation += 1
        self._changes += tuple(AccessibilityChange("input", identity, active,
                                                    self._generation, provenance)
                               for identity, _, active in staged)
        return len(staged)

    def eligibility(self, identity: Hashable, *,
                    context_inputs: Iterable[Hashable] = ()) -> Eligibility:
        """Combine the persistent active mask and temporary context input mask."""
        rule = self._rule(identity)
        context_bits = 0
        for input_id in context_inputs:
            context_bits |= self._input_bit(input_id)
        active = self._active_bits | context_bits
        allowed = bool(self._open_bits & rule.open_bit)
        allowed = allowed and (active & rule.required_bits) == rule.required_bits
        allowed = allowed and (active & rule.forbidden_bits) == 0
        return Eligibility(allowed, self._structure_revision, rule.revision,
                          self._generation)

    def eligible(self, identity: Hashable, *,
                 context_inputs: Iterable[Hashable] = ()) -> bool:
        return self.eligibility(identity, context_inputs=context_inputs).allowed

    def eligible_programs(self, *, context_inputs: Iterable[Hashable] = ()) -> tuple[Hashable, ...]:
        context_bits = 0
        for identity in context_inputs:
            context_bits |= self._input_bit(identity)
        active = self._active_bits | context_bits
        result = []
        for identity in self._rule_index:
            rule = self._compiled[identity]
            if (self._open_bits & rule.open_bit and
                    (active & rule.required_bits) == rule.required_bits and
                    (active & rule.forbidden_bits) == 0):
                result.append(identity)
        return tuple(result)

    def gate(self, identity: Hashable) -> Callable[[Mapping[Any, Any], Any], bool]:
        """Make an InteractionGraph-compatible active predicate.

        If context is a mapping, its optional ``accessibility_inputs`` member
        supplies additional transient input identities. The gate merely answers
        eligibility; the model remains responsible for all routing decisions.
        """
        self._rule(identity)

        def active_if(_state: Mapping[Any, Any], context: Any) -> bool:
            incoming = context.get("accessibility_inputs", ()) if isinstance(context, Mapping) else ()
            return self.eligible(identity, context_inputs=incoming)
        return active_if


class HysteresisThreshold:
    """Caller-parameterized Schmitt gate for noisy state-dependent access."""

    def __init__(self, open_at: float, close_at: float, *, minimum_dwell: float = 0.0) -> None:
        self.open_at = float(open_at)
        self.close_at = float(close_at)
        self.minimum_dwell = float(minimum_dwell)
        if not all(map(math.isfinite, (self.open_at, self.close_at, self.minimum_dwell))):
            raise ValueError("thresholds and dwell time must be finite")
        if self.close_at >= self.open_at:
            raise ValueError("close_at must be below open_at")
        if self.minimum_dwell < 0:
            raise ValueError("minimum_dwell must be nonnegative")
        self._last_time: float | None = None
        self._last_change: float | None = None

    def update(self, opened: bool, signal: float, time: float) -> bool:
        signal, time = float(signal), float(time)
        if not math.isfinite(signal) or not math.isfinite(time):
            raise ValueError("signal and time must be finite")
        if self._last_time is not None and time < self._last_time:
            raise ValueError("hysteresis time cannot move backwards")
        self._last_time = time
        requested = signal > self.close_at if opened else signal >= self.open_at
        if requested != bool(opened):
            if self._last_change is not None and time - self._last_change < self.minimum_dwell:
                return bool(opened)
            self._last_change = time
        return requested


__all__ = [
    "AccessibilityChange", "AccessibilityRule", "AccessibilityState",
    "Eligibility", "HysteresisThreshold",
]
