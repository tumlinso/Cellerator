"""Conservative, simultaneous exchange between cells and finite compartments.

The state stores amounts, not concentrations. Each link transfers the same
species with opposite signs at its two endpoints. Biological rate laws remain
caller supplied; no species, cell type, or tissue ontology is assumed.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ExchangeLink:
    """A persistent link identity and its dimensionless incidence weight."""

    identity: Hashable
    cell: Hashable
    compartment: Hashable
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ExchangeState:
    """Identity-addressed amounts in the plan's explicit species order."""

    cells: Mapping[Hashable, Sequence[float]]
    compartments: Mapping[Hashable, Sequence[float]]


@dataclass(frozen=True, slots=True)
class ExchangeResult:
    state: ExchangeState
    transfers: Mapping[Hashable, np.ndarray]
    passive_dt_limit: float
    limited: bool


def _vector(value: Any, size: int, name: str, *, nonnegative: bool = True) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite species values")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _volumes(ids: tuple, values: Mapping, name: str) -> dict:
    if set(values) != set(ids):
        raise ValueError(f"{name} must match the plan identities exactly")
    result = {key: float(values[key]) for key in ids}
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError(f"{name} must be positive and finite")
    return result


def concentrations(amounts: Mapping, volumes: Mapping) -> dict[Hashable, np.ndarray]:
    """Return fresh amount/volume vectors without changing either input."""
    volume = _volumes(tuple(amounts), volumes, "volumes")
    output = {}
    for identity, value in amounts.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("amounts must be finite nonnegative species vectors")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            output[identity] = array / volume[identity]
    return output


class ExchangePlan:
    """Sparse cell/compartment incidence, separate from mutable amounts/volumes.

    For link e=(cell, compartment), the requested amount from cell to compartment
    is ``dt * weight[e] * (permeability * (c_cell-c_compartment) + active_rate[e])``.
    Permeability has volume/time units; active rates have amount/time units.
    Positive active rates secrete, negative rates take up. Link weights are
    explicit and are never normalized. All requests use the same input snapshot.

    With ``donor_policy='limit'``, all simultaneous outgoing requests from each
    donor/species share its *initial* amount proportionally. Incoming material
    cannot be spent during the same step. Limiting changes the requested model;
    it ensures a budget constraint, not integration accuracy. The default rejects
    overspending and enforces the sufficient passive explicit-Euler positivity
    bound ``dt <= min(volume / sum incident weighted permeability)``.
    """

    def __init__(self, cells: Iterable[Hashable], compartments: Iterable[Hashable],
                 species: Iterable[Hashable], links: Iterable[ExchangeLink]) -> None:
        self._cells, self._compartments = tuple(cells), tuple(compartments)
        self._species, self._links = tuple(species), tuple(links)
        for label, identities in (("cells", self.cells), ("compartments", self.compartments),
                                  ("species", self.species)):
            if len(set(identities)) != len(identities):
                raise ValueError(f"duplicate {label} identity")
        if not self.species:
            raise ValueError("at least one species is required")
        cell_ids, compartment_ids = set(self.cells), set(self.compartments)
        seen = set()
        for link in self.links:
            if not isinstance(link, ExchangeLink):
                raise TypeError("links must contain ExchangeLink values")
            if link.identity in seen:
                raise ValueError("duplicate link identity")
            seen.add(link.identity)
            if link.cell not in cell_ids or link.compartment not in compartment_ids:
                raise ValueError("link endpoint is absent from the plan")
            if not math.isfinite(link.weight) or link.weight < 0:
                raise ValueError("link weight must be finite and nonnegative")

    cells = property(lambda self: self._cells)
    compartments = property(lambda self: self._compartments)
    species = property(lambda self: self._species)
    links = property(lambda self: self._links)

    def _parameters(self, cell_volumes, compartment_volumes, permeability):
        cv = _volumes(self.cells, cell_volumes, "cell_volumes")
        ev = _volumes(self.compartments, compartment_volumes, "compartment_volumes")
        raw = np.asarray(permeability)
        p = _vector(np.full(len(self.species), raw.item()) if raw.ndim == 0 else raw,
                    len(self.species), "permeability")
        cd = {key: np.zeros_like(p) for key in self.cells}
        ed = {key: np.zeros_like(p) for key in self.compartments}
        with np.errstate(over="raise", invalid="raise"):
            for link in self.links:
                conductance = link.weight * p
                cd[link.cell] += conductance
                ed[link.compartment] += conductance
        bound = math.inf
        for volume, degree in ((cv, cd), (ev, ed)):
            for identity, rates in degree.items():
                positive = rates > 0
                if positive.any():
                    bound = min(bound, float(np.min(volume[identity] / rates[positive])))
        return cv, ev, p, bound

    def passive_dt_limit(self, *, cell_volumes: Mapping, compartment_volumes: Mapping,
                         permeability: Any = 1.0) -> float:
        """Sufficient passive nonnegative-Euler bound; infinity for no exchange."""
        return self._parameters(cell_volumes, compartment_volumes, permeability)[3]

    def step(self, state: ExchangeState, *, cell_volumes: Mapping,
             compartment_volumes: Mapping, dt: float, permeability: Any = 1.0,
             active_rates: Mapping | None = None, donor_policy: str = "reject",
             enforce_stability: bool = True) -> ExchangeResult:
        """Compute a new state and auditable signed transfers; never mutate input.

        ``active_rates`` maps link identities to species vectors. Omitted links
        have zero active rate. Rates and permeability must already represent the
        input snapshot. Finite reservoirs can be ordinary compartments; external
        creation, destruction and fixed-concentration reservoirs are not implicit.
        """
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if donor_policy not in ("reject", "limit"):
            raise ValueError("donor_policy must be 'reject' or 'limit'")
        cv, ev, p, bound = self._parameters(cell_volumes, compartment_volumes, permeability)
        if enforce_stability and dt > bound:
            raise ValueError(f"dt exceeds the passive positivity bound {bound}")
        n = len(self.species)
        snapshots = []
        for ids, values in ((self.cells, state.cells), (self.compartments, state.compartments)):
            if set(ids) != set(values):
                raise ValueError("state identities must match the plan exactly")
            snapshots.append({key: _vector(values[key], n, "amount") for key in ids})
        cells, env = snapshots
        active = {} if active_rates is None else dict(active_rates)
        if set(active) - {link.identity for link in self.links}:
            raise ValueError("active rate names an unknown link")
        active = {key: _vector(value, n, "active rate", nonnegative=False)
                  for key, value in active.items()}
        cc, ec = concentrations(cells, cv), concentrations(env, ev)
        outgoing = [{key: [[] for _ in range(n)] for key in values} for values in snapshots]
        requested = {}
        with np.errstate(over="raise", invalid="raise"):
            for link in self.links:
                q = dt * link.weight * (p * (cc[link.cell] - ec[link.compartment])
                                        + active.get(link.identity, 0.0))
                requested[link.identity] = q
                for s, amount in enumerate(q):
                    side, donor = (0, link.cell) if amount >= 0 else (1, link.compartment)
                    outgoing[side][donor][s].append(abs(float(amount)))
        scales = [{key: np.ones(n) for key in values} for values in snapshots]
        limited = False
        for side, values in enumerate(snapshots):
            for identity, amount in values.items():
                for s in range(n):
                    demand = math.fsum(outgoing[side][identity][s])
                    if demand > amount[s]:
                        if donor_policy == "reject":
                            raise ValueError("simultaneous outgoing transfers exceed donor amount")
                        scales[side][identity][s] = (0.0 if amount[s] == 0 else
                                                    math.nextafter(float(amount[s]) / demand, 0.0))
                        limited = True
        incoming = [{key: [[] for _ in range(n)] for key in values} for values in snapshots]
        spent = [{key: [[] for _ in range(n)] for key in values} for values in snapshots]
        transfers = {}
        for link in self.links:
            q = requested[link.identity]
            transfer = np.where(q >= 0, scales[0][link.cell], scales[1][link.compartment]) * q
            transfers[link.identity] = transfer
            for s, amount in enumerate(transfer):
                ds, donor, rs, receiver = ((0, link.cell, 1, link.compartment) if amount >= 0
                                           else (1, link.compartment, 0, link.cell))
                spent[ds][donor][s].append(abs(float(amount)))
                incoming[rs][receiver][s].append(abs(float(amount)))
        output = []
        for side, values in enumerate(snapshots):
            result = {}
            for identity, old in values.items():
                updated = np.array([math.fsum([float(old[s]),
                                               -math.fsum(spent[side][identity][s]),
                                               math.fsum(incoming[side][identity][s])])
                                    for s in range(n)])
                if not np.isfinite(updated).all() or np.any(updated < 0):
                    raise ArithmeticError("exchange exceeds numerical range; reduce the step")
                result[identity] = updated
            output.append(result)
        return ExchangeResult(ExchangeState(*output), transfers, bound, limited)


__all__ = ["ExchangeLink", "ExchangeState", "ExchangeResult", "ExchangePlan", "concentrations"]
