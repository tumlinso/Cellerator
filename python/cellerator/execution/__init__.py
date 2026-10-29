"""Population execution, topology planning and versioned parameter residency.

Use Model/Cell for numerical behavior, Population for identity and communication,
and an explicit work selection/plan for resource scheduling. Execution effects
are caller-directed mechanisms and carry no biological interpretation.
"""
from .context import ActiveContext, ContextCapacityError, ContextInputs, Ejection, RoutedInput
from .effects import Broadcast, Create, Eject, Fork, Remove, Resume, Send, Suspend
from .population import (
    CausalityError, Emission, ExecutionFailure, ExecutionRecord, LocalExecutor,
    Population, PopulationStep, WorkBatch,
)
from .residency import (
    CapacityError, MemoryCategory, ParameterKey, ResidencyError, ResidencyLease,
    ResidencyManager,
)
from .topology import DeviceSpec, HardwareTopology, LinkSpec, TransferRoute, UnreachableDevice
from .planning import (
    CostCandidate, DemandForecast, DispatchDecision, ExecutionPlan, ExecutionStrategy,
    ExpertDemand, ExpertPlacement, ExpertSpec, ExpertWorklist, PlacementPlanner,
    PlannedTransfer, PlanningError, RollingDemandPredictor, SelectedWork,
)

__all__ = [
    "ActiveContext", "Broadcast", "CausalityError", "ContextCapacityError",
    "ContextInputs", "Create", "Eject", "Ejection", "Emission", "ExecutionFailure",
    "ExecutionRecord", "Fork", "LocalExecutor", "Population", "PopulationStep",
    "Remove", "Resume", "RoutedInput", "Send", "Suspend", "WorkBatch",
    "CapacityError", "CostCandidate", "DemandForecast", "DeviceSpec",
    "DispatchDecision", "ExecutionPlan", "ExecutionStrategy", "ExpertDemand",
    "ExpertPlacement", "ExpertSpec", "ExpertWorklist", "HardwareTopology",
    "LinkSpec", "MemoryCategory", "ParameterKey", "PlacementPlanner",
    "PlannedTransfer", "PlanningError", "ResidencyError", "ResidencyLease",
    "ResidencyManager", "RollingDemandPredictor", "SelectedWork", "TransferRoute",
    "UnreachableDevice",
]

from .engine import (
    BoundExecution, CollectiveExecution, ExecutionEngine, ExecutionRequest,
    ExecutionResult, ParameterRevision, ParameterStore, PlanExecutionError,
    StaleRevisionError, TorchCollectiveBinding,
)
from .aggregation import reduce_execution_outputs
__all__ += [
    "BoundExecution", "CollectiveExecution", "ExecutionEngine", "ExecutionRequest",
    "ExecutionResult", "ParameterRevision", "ParameterStore", "PlanExecutionError",
    "StaleRevisionError", "TorchCollectiveBinding",
    "reduce_execution_outputs",
]
