"""Compatibility imports for the pre-consolidation sparse support module.

New code should import contracts from :mod:`structure`, route consumers from
:mod:`routing`, and synthetic data from :mod:`testing`.
"""
from .structure import (
    SupportSpec, ParameterBlock, sparse_parameter_layout, _stable_scalar_ids,
    PerCellWorkset, GateValues, _snapshot_edges,
)
from .gating import learned_gates
from .layout import _logical_values_for_ordinals
from .execution import (
    _ParameterReadLease, _SparseParameterRead, SparseSupportModule,
)
from .routing import (
    prepare_sparse_slice, compile_sparse_support_ce_geo, RankedGeneSlice,
    KaryoplastEventSlice, _sum_history_sources,
    _validate_relation_input_policy, _relation_source_inputs,
)
from .testing import make_next_event_fixture, make_masked_gene_fixture

__all__ = [
    "SupportSpec", "ParameterBlock", "sparse_parameter_layout",
    "PerCellWorkset", "GateValues", "learned_gates", "SparseSupportModule",
    "RankedGeneSlice", "KaryoplastEventSlice", "prepare_sparse_slice",
    "compile_sparse_support_ce_geo", "make_next_event_fixture",
    "make_masked_gene_fixture",
]
