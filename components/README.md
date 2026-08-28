# Cellerator Components

`components/` contains optional framework integrations whose dependency
boundary is useful to keep distinct. CelleraTorch is currently its only major
inhabitant.

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

## Component rule

Components depend on Cellerator contracts. They do not create competing
numerical ecosystems or own native planning, runtime, geometry, or math.
