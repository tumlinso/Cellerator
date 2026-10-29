"""Explicit hardware/link metadata and message-size-aware transfer estimates.

These estimates describe caller-supplied topology. They neither discover actual
hardware nor measure bandwidth, contention, overlap, or kernel performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import operator
from types import MappingProxyType
from typing import Any, Iterable


def _size(value, name="nbytes") -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = operator.index(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _number(value, name, *, positive=False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    id: Any
    kind: str = "cpu"
    memory_bytes: int = 0
    compute_scale: float = 1.0
    host: Any = None

    def __post_init__(self):
        hash(self.id)
        object.__setattr__(self, "memory_bytes", _size(self.memory_bytes, "memory_bytes"))
        object.__setattr__(self, "compute_scale", _number(self.compute_scale, "compute_scale", positive=True))


@dataclass(frozen=True, slots=True)
class LinkSpec:
    source: Any
    target: Any
    bandwidth_bytes_per_second: float
    latency_seconds: float = 0.0
    bidirectional: bool = False

    def __post_init__(self):
        object.__setattr__(self, "bandwidth_bytes_per_second",
                           _number(self.bandwidth_bytes_per_second, "bandwidth", positive=True))
        object.__setattr__(self, "latency_seconds", _number(self.latency_seconds, "latency"))

    def seconds(self, nbytes: int) -> float:
        size = _size(nbytes)
        return 0.0 if size == 0 else self.latency_seconds + size / self.bandwidth_bytes_per_second


@dataclass(frozen=True, slots=True)
class TransferRoute:
    path: tuple[Any, ...]
    nbytes: int
    seconds: float


class UnreachableDevice(ValueError):
    pass


class HardwareTopology:
    """Directed links with deterministic shortest transfer paths.

    Each edge charges latency plus payload/bandwidth. Multi-hop routes charge
    each store-and-forward hop; pipelining/collectives require another model.
    An absent edge means unavailable, not an assumed PCIe/NVLink/network path.
    """

    def __init__(self, devices: Iterable[DeviceSpec], links: Iterable[LinkSpec] = ()):
        values = tuple(devices)
        if not all(isinstance(device, DeviceSpec) for device in values):
            raise TypeError("devices must contain DeviceSpec values")
        mapped = {device.id: device for device in values}
        if len(mapped) != len(values):
            raise ValueError("device identities must be unique")
        self.devices = MappingProxyType(mapped)
        edges = {device.id: [] for device in values}
        supplied = tuple(links)
        for link in supplied:
            if not isinstance(link, LinkSpec):
                raise TypeError("links must contain LinkSpec values")
            if link.source not in edges or link.target not in edges:
                raise ValueError("link endpoints must be registered devices")
            edges[link.source].append(link)
            if link.bidirectional and link.source != link.target:
                edges[link.target].append(LinkSpec(link.target, link.source,
                    link.bandwidth_bytes_per_second, link.latency_seconds))
        self.links = supplied
        self._edges = {key: tuple(value) for key, value in edges.items()}

    def route(self, source: Any, target: Any, nbytes: int) -> TransferRoute:
        size = _size(nbytes)
        if source not in self.devices or target not in self.devices:
            raise KeyError("route endpoint is not registered")
        if source == target or size == 0:
            return TransferRoute((source,) if source == target else (source, target), size, 0.0)
        # Sequence tie-breaks avoid comparing arbitrary caller device IDs.
        sequence = 0
        queue = [(0.0, sequence, source, (source,))]
        best = {source: 0.0}
        while queue:
            cost, _, current, path = heapq.heappop(queue)
            if cost != best[current]:
                continue
            if current == target:
                return TransferRoute(path, size, cost)
            for link in self._edges[current]:
                candidate = cost + link.seconds(size)
                if candidate < best.get(link.target, math.inf):
                    best[link.target] = candidate
                    sequence += 1
                    heapq.heappush(queue, (candidate, sequence, link.target, path + (link.target,)))
        raise UnreachableDevice(f"no transfer path from {source!r} to {target!r}")

    def transfer_time(self, source: Any, target: Any, nbytes: int) -> float:
        return self.route(source, target, nbytes).seconds
