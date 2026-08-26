# Cellerator

Cellerator is a performance-first biological execution system for modern accelerators.

It is built around one premise: cellular data and regulatory systems are not random sparse matrices. They contain repeated supports, modules, hierarchies, conditional programs, shared regulatory structure, local coherence, and reusable sequence logic. Cellerator compiles those regularities into execution order, physical layout, kernel schedules, and communication plans.

Cellerator is not intended to become a biological wrapper around sparse matrix multiplication. CSR, SELL, BSR, Blocked-ELL, dense fragments, vendor libraries, and custom kernels are tools available to the planner. None is the conceptual center.

## The Computational Model

The target Cellerator model consists of:

- **biological domains**, such as sequence coordinates, motif occurrences, regulatory elements, genes, transcripts, proteins, cells, modules, pathways, developmental states, and latent dimensions;
- **typed relations** between domains;
- **immutable structures** that define topology and execution geometry;
- **mutable value planes** that carry expression, accessibility, activity, learned weights, or state;
- **first-class order and partition identity**;
- **semantic geometry** compiled from biological organization;
- **multiple physical projections** selected for particular operations, devices, precisions, and reuse regimes;
- **prepared execution plans** whose per-run bindings remain cheap to change.

Canonical biological order is primarily an external API concern. Internally, compatible operations should remain in packed execution order until a consumer explicitly requires a different order.

## Core Subsystems

### CellPack and CP-BP

CellPack is Cellerator's biological geometry compiler.

The implemented CP-BP v1 pipeline can:

1. sample sparse biological support;
2. construct feature-support bitsets;
3. discover candidate feature groupings;
4. score merge cost exactly;
5. optimize and freeze a reusable packing plan;
6. apply it to full partitions;
7. build compact cell-block records;
8. infer bounded local row order;
9. emit warp-oriented tiles;
10. execute a feature-weighted row reduction directly from those tiles;
11. validate generalization and stability;
12. fit a replaceable V100 execution-cost model;
13. persist a pointer-free plan, order, and tile image.

That work is a strong foundation. Its current row-oriented physical projection and `N=1` consumer are not the final universal format.

### Core execution and CP-Math

CP-Math is being absorbed into Cellerator core.

Its target role is:

- operation contracts;
- physical projection management;
- kernel and backend registration;
- analytical planning and bounded autotuning;
- prepared-operation lifecycle;
- workspace and stream integration;
- fused epilogues;
- graph-wide order optimization;
- native and vendor execution;
- forward, backward, and multi-GPU planning.

The current CP-Math implementation is experimental. It already includes SpMM contracts, feature-order identity, backend capability queries, a structural planner, physical CSR and BELL views, a CPU referee, a runtime context, and tests. Its v1 ownership model is not the final ABI.

### Baseplane

Baseplane is Cellerator's subordinate sequence-computation library.

It remains a separate repository because packed sequence and bit-level predicates require a distinct implementation discipline. It is not a separate conceptual or numerical ecosystem.

The long-term boundary is deliberately thin:

```text
packed sequence
    → masks, events, segments, and static relations
    → regulatory structure and dynamic value planes
    → expression and cellular state
    → downstream biological behavior
```

Sequence and level should become different domains, axes, structures, or views in one biological execution model.

### CellShard

CellShard owns storage and distribution:

- canonical and sharded persistence;
- execution-envelope publication;
- fetch, caching, transport, and upload;
- generation and compatibility checks;
- delivery to active workers.

Cellerator owns the meaning and execution of packed biological structures. CellShard may persist and transport opaque Cellerator images, but it does not infer their geometry or choose their kernels.

### CelleraTorch

CelleraTorch is the explicit Torch and libtorch adapter. Native structures, parameters, planning, and kernels remain Cellerator-owned.

## Current State

The repository currently contains both the emerging architecture and older sparse-matrix-centered systems.

Implemented:

- CP-BP-01 through CP-BP-13;
- CPK1 pointer-free persistence;
- direct native weighted-row reduction;
- hardware-cost calibration for the v1 packed path and CSR fallback;
- sparse preprocessing and model primitives;
- CSR, SELL, Blocked-ELL, quantized, cuSPARSE, CUB, NCCL, and custom CUDA paths;
- experimental CP-Math contracts and backends;
- Baseplane integration at the build level.

Still in migration:

- universal domain, order, geometry, partition, structure, and value-generation identity;
- structure/value separation across all paths;
- reusable prepared operations with independent launch bindings;
- graph-wide preservation of execution order;
- multiple native CP-BP physical projections;
- empirical end-to-end CP-Math planning;
- direct Baseplane-to-Cellerator operand and fusion contracts;
- training-oriented transpose and backward projections.

See [Current implementation](docs/current_implementation.qmd) for the audited status and [Migration roadmap](docs/migration_roadmap.qmd) for the transition.

## Repository Map

```text
include/Cellerator/
    public ABI, matrix, runtime, quantized, interop, and compute contracts

src/
    compiled runtime, compute, preprocessing, models, trajectory, and support

components/CellPack/
    CP-BP compiler, native tiles, validation, persistence, tests, and benchmarks

components/CelleraTorch/
    Torch and libtorch adapter

tests/
    repository-level compile and runtime checks

bench/
    repository-level benchmarks and benchmark mutex

docs/
    authoritative architecture, performance, and migration documentation
```

The current location of CellPack under `components/` is organizational, not conceptual. CellPack is part of Cellerator's core architecture.

## Build

Cellerator requires CUDA.

```bash
cmake -S . -B build
cmake --build build -j 4
```

A core-oriented build without CelleraTorch is:

```bash
cmake -S . -B build \
  -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
cmake --build build -j 4
```

The source build resolves dependencies in this order:

1. an explicitly configured source directory;
2. a sibling checkout;
3. an installed CMake package.

The expected sibling layout can live under any parent directory; the former
CellStack wrapper is not required. For example:

```text
~/src/
├── Baseplane/
├── CellShard/
└── Cellerator/
```

Related repositories: [Baseplane](https://github.com/tumlinso/Baseplane) owns
sequence computation, and [CellShard](https://github.com/tumlinso/CellShard)
owns storage, persistence, and delivery. Cellerator consumes them as sibling
source checkouts or installed CMake packages; neither dependency is vendored.

The current local default targets Tesla V100 and `sm_70`, while stable semantic contracts must remain portable across NVIDIA generations.

## Tests

The repository currently runs focused binaries directly rather than relying on `ctest`.

Examples include:

```bash
./build/sparseOpsRuntimeTest
./build/exactSearchRuntimeTest
./build/abiRuntimeTest
```

CellPack provides focused tests and benchmarks through its component CMake targets. Use the repository benchmark mutex for every benchmark or profiler run on shared hardware.

## Documentation

Start here:

- [Scope](scope.md)
- [Architecture](docs/architecture.qmd)
- [Current implementation](docs/current_implementation.qmd)
- [Biological execution model](docs/biological_execution_model.qmd)
- [CellPack and CP-BP](docs/cellpack_cp_bp.qmd)
- [Core execution and CP-Math](docs/core_execution_cp_math.qmd)
- [Baseplane integration](docs/baseplane_integration.qmd)
- [Storage, distribution, and interop](docs/storage_distribution_and_interop.qmd)
- [Performance and validation](docs/performance_validation.qmd)
- [Migration roadmap](docs/migration_roadmap.qmd)
- [Developer reference](docs/developer_reference.qmd)

`AGENTS.md` contains the non-negotiable implementation rules for coding agents.
