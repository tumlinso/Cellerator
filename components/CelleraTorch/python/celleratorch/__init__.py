"""Biology-shaped PyTorch adapters for prepared Cellerator mechanisms.

The native extension is deliberately loaded from an explicit shared-library
path.  This lets a Python Torch distribution use a matching CelleraTorch build
without loading the host's separate libtorch into the same process.
"""

from .mechanism import (
    Axis,
    BiologicalTensor,
    Identity,
    Incidence,
    Mechanism,
    MechanismModule,
    MechanismSpec,
    OutputContribution,
    guarded_step,
    load_checkpoint,
    save_checkpoint,
)
from .biology import SharedSupportRelation, SharedSupportSpec

__all__ = [
    "Axis", "BiologicalTensor", "Identity", "Incidence", "Mechanism",
    "MechanismModule", "MechanismSpec", "OutputContribution",
    "guarded_step", "load_checkpoint", "save_checkpoint",
    "SharedSupportRelation", "SharedSupportSpec",
]
