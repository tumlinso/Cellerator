# `src/compute`

This directory is the compiled home for Cellerator operations, physical
projections, execution candidates, reusable operators, and native training.
Runtime resources, preprocessing, geometry compilation, planning, and
inter-project adapters have separate canonical owners.

## Target ownership

`src/compute/` should converge on:

- operation implementations;
- physical projection construction;
- native and vendor kernel backends;
- candidate measurement support consumed by the planner;
- epilogues;
- graph and order transforms;
- common sparse, masked, block-sparse, dense-fragment, reduction, and relation operations;
- training-oriented forward and backward primitives;
- execution instrumentation.

The public contracts belong under `include/Cellerator/compute/` or another canonical public Cellerator ABI location.

## Rules

- Name code by biological or mathematical contract, not by one storage backend.
- Treat CSR, SELL, BSR, Blocked-ELL, dense fragments, and CP-BP tiles as physical projections.
- Do not make a workflow or model own reusable math.
- Do not create a second runtime context when the core execution session can own the resource.
- Do not reconstruct Cellerator-native structures into conventional formats inside a hot path.
- Keep launch bindings separate from reusable prepared state.
- Keep order and identity explicit.
- Include conversion, preparation, epilogue, and synchronization in planner cost.
- Preserve independent reference implementations and strong fallbacks.
- Use vendor libraries when they win the measured total cost, not by default doctrine.
- Keep architecture-specific kernels behind portable semantic contracts.

## Current important areas

- `operation/`: authoritative operation preparation and dispatch.
- `projection/`: physical representations and activation/conversion.
- `candidate/`: native, vendor, sparse, and Tensor Core candidates.
- `operators/`: reusable mathematical and sparse operators.
- `training/`: native forward/backward training primitives.
- `matrix/convert/`: matrix conversion and bucket machinery.
- `neighbors/`: reusable scoring and search math.

Historical CP-Math v1 implementation lives under `compat/cp_math_v1/`.
Forwarding headers under `include/Cellerator/compute/math/` do not confer
current ownership. Runtime resources belong under `src/runtime/`, planning
under `src/planner/`, and preprocessing under `src/preprocess/`.

Read `docs/core_execution_cp_math.qmd` before adding new runtime or planner abstractions.
