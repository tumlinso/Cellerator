"""State-aware sparse relationships and conditional update worklists."""

from .core import ActiveWorklist, Interaction, InteractionGraph, InteractionResult
from .native import NativeIncidencePlan, NativeInteractionIndex, native_available

__all__ = ["ActiveWorklist", "Interaction", "InteractionGraph", "InteractionResult",
           "NativeIncidencePlan", "NativeInteractionIndex", "native_available"]
