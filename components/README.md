# Cellerator Components

`components/` contains repositories or integrations whose dependency boundary
must remain explicit. A directory here belongs to one of two categories:

1. a **framework adapter**, which translates an external framework boundary
   into Cellerator-owned contracts; or
2. a **privileged compiler component**, which may compose Cellerator with
   higher-level compilation, storage, placement, or runtime services without
   becoming part of libCellerator.

A component is not automatically conceptually outside Cellerator.

CellPack is not a component. The CellPack/CP-BP biological geometry compiler is
owned by `include/Cellerator/geometry/` and `src/geometry/`. Historical
`CellPack::` targets and `include/CellPack/` paths are compatibility surfaces.

## CelleraTorch

`components/CelleraTorch/` is the explicit Torch and libtorch adapter.

It may provide:

- tensor views over Cellerator-owned memory;
- custom-op registration;
- framework modules that invoke native Cellerator operations;
- deliberate conversion at framework boundaries.

It must not redefine Cellerator's native math, structure ownership, planner, execution order, or parameter allocation.

## Privileged compiler components

`components/CellShard/` is the registered privileged compiler component.
CellShard may own planning and runtime *above* Cellerator: global atom and
artifact composition, durable storage, placement, sharding, transport,
residency, distributed scheduling, and cross-worker recovery. It may embed or
invoke libCellerator as an operation provider and local execution engine.

That privilege does not reverse the ownership boundary. Cellerator continues
to own biological relation mathematics, semantic geometry, physical projection
meaning, local candidate capabilities, prepared native execution, and local
complete-cost selection. CellShard must consume those contracts rather than
redefine them. Cellerator must consume CellShard artifacts through explicit,
versioned, non-owning or opaque-delivery boundaries rather than importing
CellShard's global planner into its native core.

## Dependency rule

libCellerator is independently buildable and has no required CellShard
dependency. The root build keeps `CELLERATOR_ENABLE_CELLSHARD=OFF` by default;
enabling the registered source tree is an explicit combined-build choice.
Neither the presence of the submodule nor a combined target may make CellShard
headers, callbacks, schedulers, storage, or runtime mandatory for standalone
Cellerator consumers.

Framework adapters depend downward on Cellerator contracts. Privileged compiler
components may orchestrate above Cellerator, but they do not create a competing
local numerical ecosystem or absorb Cellerator's math, geometry, projections,
or execution-session ownership. The frozen JBC charter and its exact boundary
tests are recorded in
[`docs/JBC/CE_JBC_B03_PRIVILEGED_COMPONENT_CHARTER.md`](../docs/JBC/CE_JBC_B03_PRIVILEGED_COMPONENT_CHARTER.md).
