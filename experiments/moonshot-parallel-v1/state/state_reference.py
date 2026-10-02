"""CPU experimental state/placement contracts. No native owner replacement."""
from dataclasses import dataclass, replace
from types import MappingProxyType
import numpy as np


@dataclass(frozen=True, order=True)
class Coordinate:
    actor: int
    local_slot: int
    incarnation: int

    def __post_init__(self):
        if min(self.actor, self.local_slot, self.incarnation) < 0:
            raise ValueError("negative coordinate identity")


@dataclass(frozen=True)
class Epoch:
    structure: int = 0
    values: int = 0
    activity: int = 0
    parameters: int = 0

    def __post_init__(self):
        if min(self.structure, self.values, self.activity, self.parameters) < 0:
            raise ValueError("negative generation")


@dataclass(frozen=True)
class Slot:
    coordinate: Coordinate
    canonical_offset: int
    support: bool = True
    capacity: bool = True
    activity: bool | None = None
    residual: bool = False

    def __post_init__(self):
        if not isinstance(self.canonical_offset, int) or self.canonical_offset < 0:
            raise ValueError("invalid canonical offset")


class StateOwner:
    """Borrow caller storage; metadata names canonical slots, including capacity.

    Native integration must translate CE IDs and use its native generation API.
    Caller mutations of storage require touch('values'); raw ndarray mutation
    cannot be intercepted by this illustrative adapter.
    """
    def __init__(self, values, coordinates, epoch=Epoch()):
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError("canonical storage must be a rank-1 ndarray")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("canonical storage must have floating dtype")
        coordinates = tuple(coordinates)
        if len(coordinates) != len(values) or len(set(coordinates)) != len(coordinates):
            raise ValueError("one distinct identity per canonical offset required")
        if len({(c.actor, c.local_slot) for c in coordinates}) != len(coordinates):
            raise ValueError("multiple incarnations of one slot cannot be current")
        self.values = values
        self.coordinates = coordinates
        self.offsets = MappingProxyType({c: i for i, c in enumerate(coordinates)})
        self.epoch = epoch

    def touch(self, generation):
        if generation not in Epoch.__dataclass_fields__:
            raise ValueError("unknown generation")
        self.epoch = replace(self.epoch, **{generation: getattr(self.epoch, generation) + 1})


class StateView:
    """Physical order with replicas; all reads bind one saved owner epoch.

    Repeated bindings of the same coordinate are physical aliases. Distinct
    coordinates cannot share canonical offsets. Residual placement is metadata,
    and structural support, trainable capacity and optional activity stay apart.
    """
    def __init__(self, owner, slots):
        self.owner = owner
        self.slots = tuple(slots)
        self.epoch = owner.epoch
        self.validate()

    def validate(self):
        if self.owner.epoch != self.epoch:
            raise ValueError("stale saved generation")
        flags = {}
        for s in self.slots:
            if self.owner.offsets.get(s.coordinate) != s.canonical_offset:
                raise ValueError("stale incarnation or invalid canonical alias")
            if s.support and not s.capacity:
                raise ValueError("structural slot requires capacity")
            if s.activity and not s.support:
                raise ValueError("activity requires structural support")
            identity_flags = (s.support, s.capacity, s.activity)
            if flags.setdefault(s.coordinate, identity_flags) != identity_flags:
                raise ValueError("aliases disagree on logical support/activity/capacity")

    def gather(self):
        self.validate()
        return self.owner.values[[s.canonical_offset for s in self.slots]]

    def pullback(self, physical_cotangents):
        """Transpose of physical replication: SUM, never average replicas."""
        self.validate()
        g = np.asarray(physical_cotangents)
        if g.shape != (len(self.slots),):
            raise ValueError("physical cotangent extent mismatch")
        out = np.zeros_like(self.owner.values, dtype=np.result_type(g, self.owner.values))
        np.add.at(out, [s.canonical_offset for s in self.slots], g)
        return out

    def assemble(self, contributions, output_coordinates):
        """Each contribution ID appears once and has one declared logical owner.

        Contributions are (ID, coordinate, scalar). Multiple different processes
        may reduce into one output. Output is fresh; it cannot overwrite inputs.
        """
        self.validate()
        outputs = tuple(output_coordinates)
        if len(set(outputs)) != len(outputs):
            raise ValueError("duplicate output ownership")
        supported = {s.coordinate for s in self.slots if s.support}
        if not set(outputs) <= supported:
            raise ValueError("output owner has no structural support")
        positions = {c: i for i, c in enumerate(outputs)}
        result = np.zeros(len(outputs), dtype=self.owner.values.dtype)
        seen = set()
        for process, coordinate, value in contributions:
            if process in seen or coordinate not in positions:
                raise ValueError("duplicate contribution or undeclared output owner")
            scalar = np.asarray(value)
            if scalar.ndim != 0:
                raise ValueError("contribution must be a scalar")
            seen.add(process)
            result[positions[coordinate]] += scalar
        return result


@dataclass(frozen=True)
class Footprint:
    coordinate: Coordinate
    reads: frozenset[int]
    writes: frozenset[int]
    contexts: frozenset[int] = frozenset()
    program: str = "generic"

    def __post_init__(self):
        if any(i < 0 for support in (self.reads, self.writes, self.contexts) for i in support):
            raise ValueError("negative support site")


def placement_cost(group, width):
    """Illustrative traffic/padding/reduction proxy; no execution-time claim."""
    if width <= 0:
        raise ValueError("positive packet width required")
    if not group:
        return 0.0
    if len(group) > width:
        return float("inf")
    reads = set().union(*(f.reads for f in group))
    writes = set().union(*(f.writes for f in group))
    contexts = set().union(*(f.contexts for f in group))
    reductions = sum(len(f.writes) for f in group) - len(writes)
    return (16 + 4 * len(reads) + 4 * len(writes) + 2 * reductions
            + .25 * (width - len(group)) + 3 * len({f.program for f in group})
            + .1 * len(contexts))


def pack(footprints, width=16):
    """Exact small-support shortlist, deterministic under input permutation.

    Tagged read/write/context overlap nominates co-placement. Every identity is
    returned exactly once; neither similar support nor equal values merge IDs.
    """
    if width <= 0:
        raise ValueError("positive packet width required")
    fs = sorted(footprints, key=lambda f: f.coordinate)
    if len({f.coordinate for f in fs}) != len(fs):
        raise ValueError("duplicate logical footprint")
    groups = [[f] for f in fs]
    while True:
        candidates = []
        for i, left in enumerate(groups):
            for j in range(i + 1, len(groups)):
                right = groups[j]
                if len(left) + len(right) > width:
                    continue
                overlap = any(a.reads & b.reads or a.writes & b.writes or a.contexts & b.contexts
                              for a in left for b in right)
                if not overlap:
                    continue
                merged = sorted(left + right, key=lambda f: f.coordinate)
                gain = placement_cost(left, width) + placement_cost(right, width) - placement_cost(merged, width)
                if gain > 0:
                    candidates.append((-gain, tuple(f.coordinate for f in merged), i, j, merged))
        if not candidates:
            break
        _, _, i, j, merged = min(candidates)
        groups[i] = merged
        del groups[j]
    return sorted(tuple(f.coordinate for f in g) for g in groups)


def local_port_transport(states, encoders, decoders, adjacency, local_fields=None):
    """Heterogeneous private widths; independent E_i and D_i, supplied A[dst,src].

    Returns phi_i(h_i) + D_i sum_j A_ij E_j h_j. Fields are already evaluated
    values; caller owns the local laws. The common message width is an interface,
    never a global hidden-coordinate dictionary or parameter-sharing policy.
    """
    hs = [np.asarray(x) for x in states]
    es = [np.asarray(x) for x in encoders]
    ds = [np.asarray(x) for x in decoders]
    a = np.asarray(adjacency)
    n = len(hs)
    if not n or len(es) != n or len(ds) != n or a.shape != (n, n):
        raise ValueError("invalid actor counts/adjacency")
    if es[0].ndim != 2 or es[0].shape[0] == 0:
        raise ValueError("positive port width required")
    p = es[0].shape[0]
    phi = [np.zeros_like(h) for h in hs] if local_fields is None else [np.asarray(x) for x in local_fields]
    if len(phi) != n:
        raise ValueError("invalid local-field count")
    for h, e, d, f in zip(hs, es, ds, phi):
        if h.ndim != 1 or e.shape != (p, len(h)) or d.shape != (len(h), p) or f.shape != h.shape:
            raise ValueError("invalid local port dimensions")
        if not all(np.isfinite(v).all() for v in (h, e, d, f)):
            raise ValueError("nonfinite local port operands")
    if not np.isfinite(a).all():
        raise ValueError("nonfinite adjacency")
    messages = np.stack([e @ h for e, h in zip(es, hs)])
    incoming = a @ messages
    return [f + d @ incoming[i] for i, (d, f) in enumerate(zip(ds, phi))]
