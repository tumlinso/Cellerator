"""Sparse geometric neighborhoods and caller-defined spatial coupling."""
from .index import Neighbor, SpatialEdge, SpatialIndex, diffuse
from .coupling import ExchangeLink, ExchangePlan, ExchangeResult, ExchangeState, concentrations

__all__ = [
    "Neighbor", "SpatialEdge", "SpatialIndex", "diffuse",
    "ExchangeLink", "ExchangePlan", "ExchangeResult", "ExchangeState", "concentrations",
]
