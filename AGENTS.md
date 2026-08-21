# Cellerator Repository Guidance

## Authority

This file is the operational contract for coding agents working in Cellerator.

Read these documents in this order before substantial architectural work:

1. `AGENTS.md`
2. `scope.md`
3. `docs/architecture.qmd`
4. `docs/current_implementation.qmd`
5. The nearest component-level `AGENTS.md`
6. `docs/migration_roadmap.qmd`
7. Relevant tests, benchmarks, and source

The implementation is authoritative for what exists today. The architecture documents are authoritative for what new work must converge toward. Closed TODO ledgers and historical performance reports are evidence, not architecture.

When current code and the target architecture differ, do not silently normalize the current code into the future contract. Preserve compatibility where necessary, identify the transitional boundary, and move the implementation toward the documented end state.

## Project Thesis

Cellerator is a performance-first biological execution system.

Its purpose is to exploit the regularized, modular, hierarchical, repeated, correlated, and highly non-random organization of cellular systems. Biological semantics matter when they reveal computational structure that reduces execution time, memory traffic, synchronization, conversion, or communication. Semantics do not justify a slower representation by themselves.

Cellerator is not:

- a biological wrapper around sparse matrix multiplication;
- a generic sparse matrix library with biological names;
- a framework that hides layouts, copies, streams, or launch structure;
- a storage owner;
- a Torch-owned runtime;
- a requirement that every workload use one sparse format;
- a reason to preserve an abstraction after measurements show that it is costly.

The core abstraction is moving toward a biological sparse program:

- biological domains define what axes contain;
- typed relations connect domains;
- immutable structures define topology and execution geometry;
- mutable value planes carry quantitative state;
- order and partition are first-class;
- multiple physical projections may represent one logical structure;
- the planner chooses the fastest correct strategy for the whole operation or graph.

## Current Implementation Status

The repository is transitional.

Implemented and valuable:

- CP-BP sampling, support extraction, candidate discovery, exact merge scoring, constrained optimization, frozen packing plans, full-data application, compact cell-block records, bounded local row ordering, warp tiles, statistical validation, a replaceable V100 cost model, and pointer-free persistence;
- direct CP-BP execution for feature-weighted row reduction without reconstruction to CSR or BELL;
- CPK1 pointer-free plan, order, and tile images, wrapped for persistence and staging through CellShard;
- CSR, SELL, Blocked-ELL, quantized sparse, preprocessing, model, trajectory, distributed, and framework-adapter code that remains useful as implementation evidence and fallback machinery;
- an experimental CP-Math surface under `include/Cellerator/compute/math/` and `src/compute/math/`;
- Baseplane packed sequence, bit-plane, motif-scan, count, mask, and compact-event primitives.

Not yet the end-state core:

- CP-Math is not yet a unified Cellerator execution library;
- its current planner performs structural pruning rather than measured end-to-end selection;
- its prepared execution lifetime still captures per-run bindings and stream-owned runtime state;
- the current CP-BP native tile orientation is proven primarily for the `N=1` weighted-row-reduction regime;
- execution order, domain identity, partition identity, geometry identity, and value generation are not yet universal operand properties;
- sparse structure and mutable values are not yet separated everywhere;
- CPK1 is a successful v1 execution image, not the universal future runtime IR;
- Baseplane does not yet share the complete Cellerator domain and relation ABI.

Do not write documentation or code that presents these unfinished transitions as complete.

## Non-Negotiable Architectural Invariants

### 1. Performance governs

A new abstraction, hierarchy, index, projection, cache, conversion, or preprocessing pass must pay rent through at least one of:

- lower end-to-end latency;
- higher throughput;
- fewer bytes moved;
- fewer launches;
- less synchronization;
- better cache or dense-operand reuse;
- lower communication;
- lower preparation cost after amortization;
- materially cheaper downstream computation.

Conceptual elegance is not sufficient.

### 2. Biological axes have identity

Dimensions alone never establish biological equivalence.

Every core operand axis must eventually identify:

- the biological domain;
- the exact order;
- the geometry interpretation when relevant;
- the partition or ownership view when relevant.

Two arrays with the same shape but different genes, cells, regulatory elements, coordinate chunks, latent modules, or permutations are not interchangeable.

Do not add a new persistent core ABI that identifies a biological axis only by length.

### 3. Execution order is first-class

Canonical gene, cell, feature, or coordinate order is primarily an external boundary convention.

Internal operations should remain in a compatible execution order across whole computation graphs. Do not gather into packed order and scatter back to canonical order after every operation. Canonicalization must be explicit, costed, and justified by a consumer contract.

Kernels must declare the orders they consume and produce. Order transforms must be hoisted, fused, cached by value generation, or eliminated where possible.

### 4. Structure and values are separate

Immutable topology and mutable numerical state must have independent identities and lifetimes.

Examples include:

- one regulatory graph shared by many cells and time points;
- one sparse neural topology with changing learned weights;
- one CP-BP geometry with multiple precision value planes;
- one Baseplane sequence program reused across many cellular states;
- forward and transpose projections sharing logical edge identities.

Changing values must not force structure reconstruction. A structural change creates a new structure epoch. A value change creates a new value generation.

### 5. Semantic geometry is not a physical format

Semantic geometry defines stable execution organization:

- domain ordering;
- feature and row grouping;
- module boundaries;
- nested partitions;
- canonical identity recovery;
- logical relation topology.

A physical projection defines bytes and schedules for one operation, orientation, precision, accelerator, or reuse regime.

One semantic geometry may generate several physical projections. A row-masked tile, feature-major tile, CTA macro-tile, dense MMA fragment, CSR, SELL, BSR, Blocked-ELL, and transpose projection may all be valid projections of one structure.

No single layout is the repository-wide default by doctrine.

### 6. Native formats must stay native through execution

A Cellerator-native structure is valuable only when kernels consume it directly.

Do not introduce hidden reconstruction to CSR, COO, BELL, or dense tensors in a hot path. A conventional format may be selected explicitly by the planner when it wins, but conversion cost must be visible and included in the decision.

### 7. Prepared plans and launch bindings have different lifetimes

A prepared operation may own or reference:

- immutable structure;
- a selected projection;
- a selected backend or kernel;
- persistent vendor descriptors;
- persistent preprocessing;
- graph-stable buffers;
- workspace requirements.

It must not require re-preparation merely because the following changed:

- dense input pointer;
- output pointer;
- bias pointer;
- scalar coefficients;
- stream;
- value generation;
- caller-provided transient workspace.

Streams belong to execution sessions or launch bindings, not immutable plans.

### 8. Planning minimizes total measured cost

The planner selects the fastest correct strategy, not the cleanest abstraction and not the smallest representation.

Candidate cost includes:

- projection construction;
- persistent preprocessing;
- input packing;
- kernel execution;
- epilogue;
- output order transformation;
- transient workspace;
- synchronization;
- communication;
- expected reuse.

Use cheap analytical models to reject and shortlist. Use bounded empirical autotuning for serious candidates. Persist measurements with device-performance and build identities.

### 9. Runtime plurality is expected

Cellerator may contain several formats, kernel families, and vendor paths.

Plurality is acceptable when:

- candidates have precise capability contracts;
- correctness is independently checked;
- planner selection is cheap;
- preparation and conversion costs are represented;
- fallbacks remain available;
- the regime in which each path wins is measured.

Do not force the preferred native layout into workloads where CSR, SELL, BSR, Blocked-ELL, or dense execution is faster.

### 10. Baseplane is subordinate to Cellerator

Baseplane is a separate repository because sequence computation is a distinct engineering problem. It is not an independent numerical ecosystem.

Baseplane owns sequence-specialized primitives and representations. Cellerator owns the shared biological execution model, cross-domain relations, numerical propagation, planning, and sequence-to-state integration.

The intended boundary is extremely narrow:

- shared domain and order identities;
- validity-aware packed sequence views;
- bit planes, masks, event streams, segments, and relation-builder outputs;
- direct producer-consumer fusion where materialization would be wasteful;
- one planner that can compare materialized and fused sequence paths.

Never describe Baseplane as merely an external provider of bit utilities.

### 11. CellPack and CP-BP compile geometry

CellPack and CP-BP own:

- structural observation;
- support extraction;
- candidate generation;
- semantic grouping;
- row and feature ordering;
- nested partition design;
- statistics required by downstream scheduling;
- construction and validation of execution images.

They do not own canonical storage, transport, or general runtime resource management.

The current CP-BP v1 pipeline is evidence to preserve, not a universal fixed representation.

### 12. CP-Math becomes core execution

CP-Math is moving into Cellerator core as:

- operation contracts;
- structure and value binding;
- physical projection management;
- kernel and backend registry;
- planner and autotuner;
- prepared operation lifecycle;
- epilogue composition;
- graph and order optimization;
- execution and profiling hooks.

Do not move the current files mechanically and freeze their v1 ABI. Migrate their useful pieces into the target ownership model.

### 13. CellShard owns persistence and distribution

CellShard owns:

- canonical and sharded storage;
- durable containers;
- execution-envelope publication;
- fetch, cache, transport, and staging;
- generation and compatibility validation;
- storage-oriented partition artifacts.

Cellerator owns:

- biological structure semantics;
- geometry compilation;
- projection meaning;
- kernel selection;
- execution order;
- numerical execution.

CellShard may store and transport opaque Cellerator execution images. It must not rediscover or redefine their biological execution semantics.

### 14. Frameworks are adapters

CelleraTorch and future framework integrations expose Cellerator-owned data and operations.

Framework adapters must not become the canonical allocator, planner, structure owner, or implementation of Cellerator-native operators. Hidden conversion through Torch in a repeated hot path is forbidden.

### 15. Steady-state execution is allocation-free and synchronization-explicit

Hot paths must not contain:

- repeated `cudaMalloc` or `cudaFree`;
- descriptor reconstruction;
- host-visible per-tile decisions;
- implicit device-wide synchronization;
- per-cell kernel launches;
- pointer forests;
- repeated structure hashing;
- hidden device-host round trips.

Use caller-owned or session-owned memory, explicit streams, persistent descriptors, sectioned pointer-free images, and batched work queues.

## Existing Work To Preserve

Preserve unless measurements or correctness require a versioned replacement:

- CP-BP sampled support and exact scoring pipeline;
- frozen packing-plan identities and canonical recovery maps;
- bounded local row ordering;
- compact zero-free mask grammar;
- direct native tile consumption;
- exact host references and reconstruction tests;
- statistical held-out, null, bootstrap, and stability validation;
- pointer-free aligned persistence and relocation;
- the CP-BP V100 measurement corpus as historical calibration evidence;
- Baseplane allocation-free packed sequence primitives;
- existing CSR, SELL, Blocked-ELL, cuSPARSE, CUB, NCCL, and dense paths as candidate backends and baselines;
- explicit low-level visibility of layout, residency, launch, and transfer costs.

Preservation does not mean retaining current ownership or public names forever. Prefer adapters and versioned migration over wholesale rewrites.

## Superseded Historical Assumptions

The following statements are obsolete as repository-wide rules:

- Blocked-ELL is the universal native Cellerator layout.
- CellShard owns Cellerator layout derivation.
- Cellerator is primarily sparse ML over CellShard matrices.
- CP-BP is merely a better permutation for BELL or ELLPACK.
- CPK1 is the final runtime representation.
- Baseplane is external to Cellerator conceptually.
- canonical row output is the normal internal postcondition;
- vendor libraries should be preferred before the planner evaluates the full cost;
- one sparse batch ABI can represent every biological operand;
- one matrix-wide format should be selected for a whole structure.

These may remain true for a particular current subsystem. They are not architectural defaults.

## Forbidden Without Measured Justification

Do not introduce any of the following without a written cost model and relevant benchmark:

- a mandatory canonical-order output in an internal operator;
- a hidden sparse-format conversion;
- a dense materialization between a sparse biological producer and its only consumer;
- a second copy of immutable structure owned by a workflow layer;
- a physical format selected globally when tile- or region-level plurality is possible;
- a biological hierarchy duplicated into separate value arrays;
- per-run plan preparation for changing pointers or streams;
- a host-side decision inside repeated tile execution;
- a Baseplane event buffer when a downstream kernel can consume the predicate directly;
- a new universal container in place of several measured projections;
- a compatibility abstraction that blocks the fastest reasonable hardware path;
- an architecture-specific property in the stable semantic ABI when runtime dispatch would suffice.

## Performance Claims

Every performance claim must identify:

- exact hardware and topology;
- compiler, CUDA, driver, and relevant library versions;
- build mode and architecture;
- data shape and structural distribution;
- dtype and accumulation policy;
- warmup and repeat counts;
- whether setup, transfer, conversion, synchronization, and output transformation are included;
- expected reuse;
- relevant baselines;
- numerical tolerance and correctness result;
- benchmark-mutex use.

A microkernel win is not an end-to-end win.

At minimum report the applicable subset of:

- latency and throughput;
- achieved DRAM bandwidth;
- bytes per useful biological interaction;
- nanoseconds per useful biological interaction;
- useful interactions per DRAM byte;
- launch count;
- host time;
- warp execution efficiency;
- branch efficiency;
- register and shared-memory use;
- L1 and L2 behavior;
- preparation and projection cost;
- packing amortization;
- persistent metadata size;
- memory expansion;
- communication bytes;
- forward and backward time.

Benchmark against the strongest relevant path. An intentionally weak baseline is not evidence.

Benchmark and profiler jobs must use the repository benchmark mutex and any active GPU resource reservation mechanism.

## Hardware Policy

The current local performance baseline is Volta `sm_70` on Tesla V100.

Stable semantic contracts must remain portable across NVIDIA generations. Physical projections and kernels may specialize for:

- Volta;
- Ampere;
- Hopper;
- Blackwell;
- later accelerators.

Runtime dispatch owns architecture selection.

Do not use Ampere- or Hopper-only instructions in the portable ABI. Do not weaken the V100 path merely to make one kernel source look uniform.

For current CUDA work, state the dominant expected limiter:

- HBM traffic;
- L2 or shared-memory reuse;
- register pressure;
- occupancy;
- launch overhead;
- PCIe or NVLink;
- atomics;
- synchronization;
- host preparation.

## Code Ownership

### Shared ABI

Stable, trivially copyable, pointer-light or pointer-free contracts belong under `include/Cellerator/`.

The shared ABI should contain domain, order, geometry, partition, structure, value-generation, operand-view, and execution contracts. It must remain small enough for Baseplane and adapters to depend on without importing the complete runtime.

### Core execution

Operation contracts, planners, projections, prepared operations, runtime sessions, kernel dispatch, and common epilogues belong under `include/Cellerator/compute/` and `src/compute/` until the final core layout is settled.

Do not create another independent runtime context when the existing Cellerator session can own the resource.

### CellPack

CellPack currently lives under `components/CellPack/`. New work must follow `components/CellPack/AGENTS.md`.

Long term, its compiler-facing semantic contracts may move into canonical Cellerator include and source trees. Do not use the current component location as evidence that CellPack is conceptually outside Cellerator.

### Baseplane

Baseplane remains a separate repository and build target. Integration code belongs on the Cellerator side unless it is a general sequence primitive.

### CellShard

Storage, persistence, delivery, and distributed data movement belong in CellShard. Cellerator may define the opaque image and compatibility requirements that CellShard transports.

### CelleraTorch

Torch-facing views, custom-op registration, and framework wrappers belong in `components/CelleraTorch/`. Reusable math remains in native Cellerator.

### Workflows, models, preprocessing, and trajectory code

These layers may orchestrate native operations. They must not duplicate reusable math, memory management, planning, or structure semantics.

When a workflow needs a new numerical primitive, implement the primitive in the owning core layer first.

## C++ and CUDA Style

- Use C++17 and CUDA 17 unless the build policy changes.
- Use four-space indentation.
- Use `snake_case` for files, functions, variables, and POD structures.
- Keep ownership, residency, stream, capacity, identity, and lifetime explicit.
- Prefer trivially copyable structs at ABI and persistence boundaries.
- Prefer structure-of-arrays for columnar or warp-coalesced access.
- Prefer explicit contiguous buffers to allocator-heavy container graphs in hot paths.
- Keep private device helpers near their kernels.
- Split files by behavior rather than by an entire workflow.
- Keep hot implementation details visible enough to profile.
- Do not compress code at the cost of obscuring ownership or bounds.
- Do not use `std::vector` as the default public surface for repeated GPU-facing work.
- Do not use raw ownership where RAII can preserve the same performance and pointer stability.

See `style_hint.md` for local file-shape guidance.

## Build and Test

Configure:

```bash
cmake -S . -B build
cmake --build build -j 4
```

For a core build without CelleraTorch:

```bash
cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
cmake --build build -j 4
```

Cellerator currently requires CUDA and resolves sibling CellShard and Baseplane source trees before installed packages.

Run focused binaries directly. The repository currently does not use `ctest` as its primary test runner.

Before changing a core contract:

1. add or update an independent reference test;
2. add adversarial identity, order, capacity, and stale-generation tests;
3. run affected host and CUDA tests;
4. run Compute Sanitizer for new device formats or pointer rebinding;
5. benchmark only after correctness passes;
6. record exact commands and environment.

## Documentation Rules

The authoritative documentation spine is:

- `README.md`
- `scope.md`
- `docs/architecture.qmd`
- `docs/current_implementation.qmd`
- `docs/biological_execution_model.qmd`
- `docs/cellpack_cp_bp.qmd`
- `docs/core_execution_cp_math.qmd`
- `docs/baseplane_integration.qmd`
- `docs/storage_distribution_and_interop.qmd`
- `docs/performance_validation.qmd`
- `docs/migration_roadmap.qmd`

When behavior or architecture changes, update the relevant document in the same change.

Closed TODO files are historical execution records. They must not be cited as the current architecture without an explicit historical qualifier.

Do not create a second architecture summary in a local README. Local READMEs should explain only the purpose and constraints of that directory and link back to the authoritative spine.

## Review Checklist

Before accepting a core change, answer:

- What biological regularity is exploited?
- What concrete hardware cost is reduced?
- Which domains and orders are consumed and produced?
- Does the change preserve execution order across downstream consumers?
- Is immutable structure separate from mutable values?
- Is this semantic geometry or a physical projection?
- Which alternative projections or backends were considered?
- Are conversion, preparation, epilogue, and synchronization included in the cost?
- Does the hot path allocate, synchronize, hash, or rebuild descriptors?
- Can the result be captured in a CUDA Graph?
- Does training require a transpose or backward projection?
- Does Baseplane integration materialize an avoidable intermediate?
- Does CellShard remain storage and transport only?
- Is the performance claim compared with the strongest relevant baseline?
- Does the change make future multi-GPU partitioning harder?
- Is a versioned compatibility path required?

If those questions cannot be answered, the change is not ready to define a new core contract.

## Commit and Pull Request Notes

Keep commits narrow and descriptive.

Pull requests that touch core execution should state:

- architectural invariant affected;
- source and destination domains and orders;
- structure and value lifetimes;
- selected or added physical projections;
- exact tests run;
- sanitizer results;
- benchmark commands and deltas;
- hardware and toolchain assumptions;
- compatibility or migration strategy.
