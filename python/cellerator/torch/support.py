"""Logical sparse support contracts and native preparation operations."""
from .structure import SupportSpec
from .routing import prepare_sparse_slice
from .geometry import compile_sparse_support_ce_geo

__all__ = ["SupportSpec", "prepare_sparse_slice", "compile_sparse_support_ce_geo"]
