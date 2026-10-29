# Framework push integration contract

User authorized a broad one-hour implementation push beginning 2026-10-29 17:39 UTC, with substantial parallel Luna work and focused validation. This file coordinates implementation, not Todo authority. Preserve existing changes; no commits/pushes, GPU workloads, installs, or legacy rewrites. Current user mandate supersedes older Torch-extension-only product framing.

## Shared Python contracts

New packages: cellerator.modeling, state, events, interactions, dynamics, execution, inference, adapters. Existing tokenization, ops, nn, torch and experimental stay compatible. No mandatory Torch in new core. State may be custom objects, sparse/ragged structures, arrays, tensors or externally maintained handles.

`cellerator.modeling.StepContext`: frozen dataclass with `time: float=0.0`, `dt: float=1.0`, `inputs: Any=None`, `parameters: Any=None`, `rng: Any=None`, `metadata: Mapping[str, Any]` default empty.
`cellerator.modeling.StepResult`: dataclass with `state: Any`, `events: tuple=()`, `effects: tuple=()`, `time: float|None=None`, `metadata: Mapping[str, Any]` default empty.
A dynamics callable has `(state, context: StepContext) -> StepResult`; wrappers may normalize raw state to StepResult. Observation callable `(state, context) -> Any`.
`Model(dynamics, observation=None, parameters=None)` exposes `step(state, context)` -> StepResult and `observe(state, context)` -> Any. Model parameters remain shared, individual states remain separate.
`Cell(model, state, *, id=None, time=0.0)` exposes `advance(dt=1.0, *, inputs=None, rng=None)` -> StepResult, `observe(...)`, persistent id/state/time; no universal dense conversion.

Events package owns `Event(kind, *, time=None, target=None, payload=None, source=None, metadata=None)` and history/trajectory; core treats events as opaque to avoid cycles. State and interactions are independent of modeling. Dynamics, execution and inference depend on modeling. Adapters depend on optional external providers; import provider only when needed.

## Boundaries

- core modeling: protocols/model/Cell/composition, no generic biological ontology
- state: structured fields, identity-addressed sparse/ragged storage, layout mapping, snapshots/generations; no mandatory contiguous tensor
- events: event sequences, histories, trajectories, lineage, tokenizer interop via explicit optional methods
- interactions: sparse relations, overlapping group membership, active worklists, joint interaction execution and explicit aggregation
- dynamics: discrete/continuous/stochastic/update composition; reference CPU algorithms, no false GPU claims
- execution: populations, context routing/ejection, suspension/resumption, shared-model batching, scheduling separate from model choices
- inference: generic observation losses, uncertain candidates, event/trajectory reconstruction tools with explicit supplied hypotheses
- adapters: Torch learned modules/autograd-preserving interfaces and AnnData input/output mapping; preserve upstream semantics
- native: reusable typed sparse worklist/interaction primitives with focused native validation and native bindings where practical

No placeholder-only packages. Each owner delivers working code plus concise focused smoke checks and API notes. Communicate inter-package changes promptly. Root owns python/cellerator/__init__.py, pyproject.toml and root CMake integration; do not edit these. Distinct package owners own their package __init__.py. Target implementation handoff by 18:20 UTC; integration finishes by 18:39 UTC.
