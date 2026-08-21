# Cellerator Components

`components/` contains tightly coupled Cellerator subsystems whose build or dependency boundary is useful to keep distinct.

A component is not automatically conceptually outside Cellerator.

## CellPack

`components/CellPack/` is the current home of Cellerator's biological geometry compiler and the CP-BP v1 implementation.

CellPack is part of the Cellerator architecture. It observes biological sparse structure, produces semantic execution geometry, constructs and validates physical execution images, and supplies statistics used by the core planner.

Its current location is transitional. Stable domain, geometry, structure, and projection contracts may move into the canonical `include/Cellerator/` and `src/` trees as the core ABI settles.

Read:

- `components/CellPack/AGENTS.md`
- `docs/cellpack_cp_bp.qmd`
- `docs/core_execution_cp_math.qmd`

## CelleraTorch

`components/CelleraTorch/` is the explicit Torch and libtorch adapter.

It may provide:

- tensor views over Cellerator-owned memory;
- custom-op registration;
- framework modules that invoke native Cellerator operations;
- deliberate conversion at framework boundaries.

It must not redefine Cellerator's native math, structure ownership, planner, execution order, or parameter allocation.

## Component rule

Components depend on Cellerator contracts. They do not create competing numerical ecosystems.
