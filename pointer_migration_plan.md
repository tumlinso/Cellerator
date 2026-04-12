# Pointer Migration Plan

## Purpose

This repository is migrating away from `std::vector`-first host surfaces in performance-sensitive code. The target state is pointer-first, layout-explicit, preallocated code in hot paths and at host-device boundaries. This plan is the repository-level reference for that migration.

## Core Rules

1. Do not introduce new `std::vector` usage in hot paths, GPU-facing code, batch assembly, cross-device merge code, or frequently repeated inference or training loops.
2. Prefer raw pointers plus explicit size or stride fields, `std::unique_ptr<T[]>`, aligned allocations, or workspace-owned slabs when data is contiguous.
3. Keep ownership stable across repeated calls. Allocate once, reuse many times.
4. Keep layouts explicit. Prefer structure-of-arrays and packed contiguous storage over nested containers.
5. Do not hide hot-path costs behind helper abstractions that obscure allocation, copying, synchronization, or layout transforms.
6. Public APIs for hot subsystems must stop requiring `std::vector` once their migration phase begins.

## Acceptable Exceptions

- Small control-plane code that is not on a repeated performance path.
- Tests and micro-bench fixtures.
- One-shot host-side export helpers where the code is explicitly outside steady-state execution.

Even in those cases, pointer-stable storage is still preferred when the code is likely to be reused in a hot path later.

## Current Heavy Violations

The current largest `std::vector` concentrations are:

- `src/compute/neighbors/forward_neighbors/forward_neighbors.cu`
- `src/compute/neighbors/cuvs_sharded_knn.cu`
- `src/apps/series_workbench.cc`
- `src/apps/series_workbench_cuda.cu`
- `src/apps/series_workbench.hh`
- `src/ingest/series/series_ingest.cuh`
- `src/models/developmental_time/dT_dataloader.hh`
- `src/models/dense_reduce/dR_dataloader.hh`
- `src/compute/graph/supernode_reduce.cuh`
- `src/compute/graph/record_table.cuh`
- `src/compute/graph/forward_prune.cuh`

The first migration budget should go to the compute-neighbor surfaces, then model dataloaders and infer helpers, then ingest and graph assembly, then workbench control surfaces that still leak into hot paths.

## Target End State

Each hot subsystem should converge on these patterns:

- A plain data struct with pointer fields, sizes, capacities, and explicit ownership notes.
- A workspace or arena that reserves all repeated temporary storage up front.
- No nested dynamic containers in steady-state code.
- No per-call heap growth during repeated execution once dimensions are known.
- No host merge logic that depends on downloading full intermediate tables into fresh vectors.
- No public hot-path API that requires `std::vector` inputs or outputs.

## Preferred Building Blocks

- Raw pointer plus count interfaces.
- `std::unique_ptr<T[]>` for owned contiguous host buffers.
- Explicit `reserve`, `setup`, `reset`, and `clear` routines on workspaces.
- Fixed-capacity or caller-provided scratch regions for temporary results.
- Packed arrays with offset tables instead of nested containers.
- Reusable device buffers mirrored by reusable host staging slabs.

## Disallowed Patterns In Hot Code

- `std::vector<std::vector<T>>`
- returning freshly allocated vectors from repeated helper calls
- per-row or per-batch vector construction inside training, inference, or search loops
- vector-backed public batch structs for GPU-facing code
- vector-backed merge state for multi-device top-k or candidate aggregation
- vector growth used as an implicit allocator strategy

## Migration Order

### Phase 1: Forward Neighbors And KNN

Scope:

- `src/compute/neighbors/forward_neighbors/*`
- `src/compute/neighbors/cuvs_sharded_knn.*`

Required changes:

- Replace vector-backed record batches, query batches, result tables, shard storage, and merge scratch with pointer-first structs.
- Replace nested per-device and per-row vectors with flat buffers plus offset tables.
- Move host merge scratch into reusable workspaces.
- Stop downloading candidate tables into fresh host vectors when a workspace or direct device-side merge can be used.
- Remove `std::vector` from public forward-neighbor APIs.

Exit criteria:

- No `std::vector` in repeated search or build loops.
- No nested vectors in merge code.
- Public neighbor APIs accept pointer-plus-size surfaces or workspace-backed tables.

### Phase 2: Model Samplers And Infer Paths

Scope:

- `src/models/rngFetch.hh`
- `src/models/developmental_time/*`
- `src/models/dense_reduce/*`

Required changes:

- Replace vector-returning samplers with caller-filled buffers or workspace-backed spans.
- Replace bucket storage with flat row tables plus bucket offset arrays.
- Replace per-batch CSR metadata assembly vectors with preallocated host scratch.
- Remove vector-backed batch aggregation in inference helpers.

Exit criteria:

- No per-batch dynamic vector growth in sampler or inference steady state.
- Batch assembly uses reusable buffers with explicit capacities.

### Phase 3: Ingest Layout And Series Planning

Scope:

- `src/ingest/series/*`
- hot ingest-adjacent paths in `src/apps/series_workbench_cuda.cu`

Required changes:

- Replace vector-backed layout accumulation with explicit arrays and offset buffers.
- Replace dataset and part planning vectors in throughput-sensitive code with preallocated planning slabs.
- Keep one-shot manifest parsing separate from throughput-critical conversion surfaces.

Exit criteria:

- Conversion and browse-cache build paths do not rely on vector growth for large metadata or per-part accumulation.
- H2D and D2H staging code is backed by stable buffers.

### Phase 4: Graph And Trajectory Assembly

Scope:

- `src/compute/graph/*`
- `src/trajectory/*`

Required changes:

- Replace vector-backed record tables, CSR graphs, tree overlays, and supernode tables with explicit buffer structs.
- Replace temporary nested containers with flat edge tables and scratch offsets.
- Separate offline graph construction helpers from reusable runtime surfaces.

Exit criteria:

- Core graph data structures are pointer-first and capacity-aware.
- No nested vectors remain in graph construction code that is expected to scale.

### Phase 5: Workbench API Cleanup

Scope:

- `src/apps/series_workbench.hh`
- `src/apps/series_workbench.cc`

Required changes:

- Narrow vector use to clearly one-shot UI or filesystem discovery paths.
- Remove vector-heavy data structures from any API that feeds ingest, browse-cache construction, or preprocess execution.
- Split control-plane summaries from execution-plane layouts.

Exit criteria:

- Workbench does not force vector-based representations onto hot subsystems.
- Any remaining vectors are isolated to one-shot inspection or UI summary code.

## Review Checklist

Before landing a migration change, verify:

1. The new surface makes allocation points obvious.
2. Repeated calls do not allocate after setup for stable dimensions.
3. Data ownership is pointer-stable across device upload, kernel launch, and merge phases.
4. Nested containers were replaced with flat storage plus offsets.
5. Public APIs did not reintroduce `std::vector` on a hot path.
6. The rewritten code still matches the real Volta bottleneck rather than a style preference.

## Immediate Policy For New Work

Until migration is complete:

- treat the current `std::vector` footprint as technical debt
- do not copy existing vector-heavy patterns into new code
- when touching one of the files listed above, reduce vector use instead of preserving it
- if a change cannot remove an existing vector-heavy surface yet, document the remaining blocker in the patch notes
