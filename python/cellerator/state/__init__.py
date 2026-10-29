"""Structured state, identity mappings, and sparse/ragged layouts."""

from .core import LayoutMap, RaggedState, SparseState, StateReference, StateSnapshot, StructuredState
from .quantization import QuantizationDiagnostics, QuantizedState, quantize_state
from .quantization_profiles import QuantizationProfile
from .storage import SoAStateStore, StateHandle, StorageSpec

__all__ = [
    "LayoutMap", "RaggedState", "SparseState", "StateReference",
    "StateSnapshot", "StructuredState", "QuantizationDiagnostics",
    "QuantizationProfile", "QuantizedState", "quantize_state", "SoAStateStore", "StateHandle",
    "StorageSpec",
]
