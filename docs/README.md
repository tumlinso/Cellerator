# Cellerator documentation

This documentation describes an early PyTorch extension for cellular dynamical model building. PyTorch remains the general tensor, standard-layer, autograd, optimizer, and composition framework. Cellerator adds reusable operations and execution semantics shaped by cellular structure.

## Reader route

1. Start with the [book entry](index.qmd) and [getting started](getting_started.qmd).
2. Read [scope](scope.qmd), [architecture](architecture.qmd), [model building](model_building.qmd), and [operations](operations.qmd).
3. Use [layout and execution](layout_execution.qmd) to understand identity, eligibility, physical realization, and runtime boundaries.
4. For implementation, see [Python bindings](development/python-bindings.md), [developer reference](developer_reference.qmd), and the agent-facing [source routing guide](development/routing.md).
5. Use the [current implementation record](current_implementation.qmd) for the capability boundary and the [reference and history index](experiments_and_history.qmd) for subsystem contracts, experiments, and preserved material.

The book navigation is intentionally focused on current product guidance and maintained references. Historical subsystem chapters and source-bound records remain in the repository and are routed from the history index. This is not a task tracker: live task state belongs to Project Control.

## Evidence policy

Keep implementation placement, target architecture, experimental status, and historical evidence distinct. Preserve source-bound reports and snapshots. State correctness, scientific validity, package usability, and performance separately. Update contracts and callers together when implementation moves; a target in an architecture diagram is not proof of a supported API.
