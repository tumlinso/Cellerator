# `src/compute`

This directory is the current compiled home for reusable Cellerator execution machinery.

It is transitional: older sparse operators, preprocessing kernels, model-adjacent math, and the emerging CP-Math runtime coexist here.

## Target ownership

`src/compute/` should converge on:

- operation implementations;
- physical projection construction;
- native and vendor kernel backends;
- planner and autotuner support;
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

- `math/`: experimental CP-Math contracts, planner, runtime, projections, backends, referee, and epilogue.
- `sparse/`: existing reusable sparse operators and projections.
- `matrix/convert/`: conversion and bucket machinery.
- `preprocess/`: reusable preprocessing math.
- `neighbors/`: reusable scoring and search math.
- `ml/`: model-adjacent native operations.
- `runtime/`: older execution-context surfaces that must be reconciled with core CP-Math ownership.

Read `docs/core_execution_cp_math.qmd` before adding new runtime or planner abstractions.
