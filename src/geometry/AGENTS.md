# CellPack and CP-BP Guidance

## Mission

CellPack is Cellerator's biological geometry compiler.

It converts measured biological structure into reusable execution geometry and validated physical projections. It exists because biological supports, modules, co-access patterns, row neighborhoods, regulatory programs, and repeated subgraphs contain regularity that generic sparse formats do not know how to exploit.

CellPack is not conceptually outside Cellerator. Its current component boundary is an implementation and build boundary.

The current CP-BP v1 pipeline is a completed experimental foundation. New work must preserve its proven contracts while moving toward a broader operation-aware compiler.

## Authority

Read before changing CellPack:

1. repository `AGENTS.md`;
2. `scope.md`;
3. `docs/architecture.qmd`;
4. `docs/cellpack_cp_bp.qmd`;
5. `docs/core_execution_cp_math.qmd`;
6. this file;
7. the relevant source, test, benchmark, and historical ledger.

Closed CP-BP TODO files describe how v1 was implemented. They are not the target architecture.

## Current v1 implementation

CP-BP v1 currently provides:

- deterministic sampled support extraction;
- sampled sparse materialization;
- feature-major support bitsets and counts;
- approximate candidate generation;
- exact pair and merge-cost scoring;
- a deterministic constrained host optimizer;
- a frozen packing plan with canonical feature recovery;
- application of that plan to full partitions;
- compact cell-block records;
- bounded local row ordering;
- host and CUDA warp-tile construction;
- a direct feature-weighted row-reduction reference and CUDA kernel;
- held-out, null, bootstrap, and relearned-stability validation;
- a replaceable V100 cost model;
- a pointer-free aligned CPK1 image containing plan maps, row order, and tiles;
- rebinding and direct execution after CellShard-owned upload.

The first direct operation is:

```text
y[row] = Σ value[row, feature] × weight[canonical_feature]
```

It is effectively an `N=1` sparse-dense operation.

The historical V100 benchmark showed the direct tile path winning strongly at high sharing, winning at medium sharing, and losing to the existing CSR path at low sharing. That result is a regime map, not proof of a universal layout.

## Non-Negotiable Invariants

### Preserve canonical biological identity

Every feature and row must remain recoverable.

Packing may reorder, group, partition, and encode. It may not make canonical identity ambiguous.

Required identities include:

- source feature-axis identity;
- source row-domain identity;
- structure or plan identity;
- geometry identity;
- structure epoch;
- value generation where values are mutable;
- partition identity when a view is partial.

Never substitute a live pointer for a durable identity.

### Semantic geometry and physical projections are different

CellPack must produce a semantic geometry that can outlive one kernel.

Semantic geometry may contain:

- feature order;
- row order;
- feature groups;
- row groups;
- module boundaries;
- nested warp, CTA, GPU, and node partitions;
- cross-partition edges;
- canonical recovery maps;
- statistics and priors used by downstream planning.

Physical projections may include:

- the current row-masked warp tiles;
- feature-major masked tiles;
- CTA macro-tiles;
- dense MMA fragments;
- CSR;
- SELL-C-sigma;
- BSR;
- Blocked-ELL;
- transpose or backward views;
- quantized value layouts.

Do not add a physical-layout field to the semantic contract when a projection catalog is sufficient.

### The current row-masked tile is one projection

The v1 tile grammar is valuable and must remain supported.

It is not the universal representation for:

- medium or large dense RHS width;
- transpose execution;
- learned sparse-value gradients;
- feature-major reductions;
- sequence-conditioned relations;
- high-density fragments;
- low-sharing heavy-tail regions.

New work should add projections or versioned tile classes rather than contort one representation into every operation.

### Structure and values are separate

The v1 CPK1 image couples structure and value order closely enough for immutable data execution. The future core must allow:

- one geometry with several value planes;
- changing values without rebuilding geometry;
- fp16, bf16, fp8, integer, or fp32 planes over one structure;
- forward and transpose value mappings;
- static topology with learned weights;
- many cells or time points sharing one regulatory topology.

A serializer may colocate structure and values, but runtime ownership must remain separable.

### Static compilation, cheap execution

Expensive discovery, clustering, graph partitioning, projection construction, and validation belong outside steady-state execution.

Runtime may:

- select among precompiled projections;
- skip inactive precompiled modules;
- bind new values;
- choose a measured kernel;
- consume a device work queue.

Runtime must not:

- rediscover modules;
- rebuild feature groups;
- sort complete structures on the host;
- repack every minibatch;
- walk biological object graphs;
- allocate per tile;
- canonicalize after every operation.

### No hidden conventional lowering

A native CellPack structure must not secretly reconstruct CSR, COO, BELL, or dense data in the hot path.

A conventional projection is allowed when it is an explicit planner candidate and its construction and execution costs are represented.

### One objective is not enough

The current row-active-block-reference objective is a useful v1 surrogate.

The future optimizer must account for:

- operation family;
- dense RHS width `N`;
- dtype and accumulation;
- row-mask and feature-mask distributions;
- dense-operand reuse;
- lane imbalance;
- descriptor-lane visits;
- metadata bytes;
- register and shared-memory pressure;
- epilogue;
- input and output ordering;
- forward and transpose needs;
- expected reuse;
- projection build cost;
- graph capture;
- communication cuts.

Do not replace the current objective with another elegant surrogate and call it execution cost. Calibrate against measured kernels.

## Compiler Architecture

The target compiler has two stages.

### Stage A: semantic geometry

Inputs may include:

- sparse support;
- value distributions;
- operation profile;
- known biological modules;
- regulatory relations;
- temporal or state-conditioned activity;
- hardware deployment profile;
- expected reuse;
- distributed partition requirements.

The compiler produces:

- one or more candidate semantic geometries;
- feature and row order;
- module and group membership;
- nested partitions;
- canonical identity maps;
- cross-partition relation summaries;
- geometry statistics.

### Stage B: physical projection construction

For each selected operation and deployment profile, construct one or more projections:

```text
semantic geometry
    ├── row-masked N=1 projection
    ├── feature-major small-N projection
    ├── CTA macro-tile projection
    ├── dense-fragment projection
    ├── CSR or SELL fallback
    └── transpose/backward projection
```

The core planner decides which projection is used. CellPack may precompute likely candidates and persist common projections.

## Optimization Strategy

### Use biological hierarchy as a prior

Known modules, pathways, enhancer-promoter groups, chromatin domains, lineages, or gene families may initialize or softly constrain the optimizer.

They are not mandatory truth.

A biologically named module that increases total execution cost remains interpretation metadata unless another operation benefits enough to justify it.

### Optimize rows and features together

Feature grouping is only useful relative to rows that execute together. Row grouping is only useful relative to the feature blocks those rows activate.

Use an alternating or multilevel approach:

1. initialize feature and row communities;
2. coarsen the bipartite or hypergraph structure;
3. partition under hardware capacities;
4. refine features given row groups;
5. refine rows given feature groups;
6. classify physical tiles;
7. measure or estimate total operation cost;
8. retain exact rollback and validation.

### Candidate algorithms

Appropriate tools include:

- MinHash and LSH for sparse candidate discovery;
- weighted bipartite partitioning;
- hypergraph partitioning;
- biclustering;
- constrained community detection;
- multilevel coarsening;
- separator-based partitioning;
- local move, swap, split, and merge refinement;
- domain-specific priors;
- exact local cost oracles;
- measured tile-class lookup models.

No generic clustering algorithm is accepted merely because it groups similar features.

### Nested partitions

Geometry should be able to express:

```text
node
  → GPU
      → biological supermodule
          → CTA macro-tile
              → warp tile
                  → local feature block
```

Each level must correspond to reuse, scheduling, memory, or communication value.

## Statistics To Produce

Construction should calculate enough information that runtime scheduling is almost free.

Per tile or macro-tile, record or make available:

- row count;
- feature count;
- nnz;
- density;
- distinct feature blocks;
- descriptor count;
- row-mask popcount distribution;
- feature-mask popcount distribution;
- lane-work mean, variance, and maximum;
- unique dense-RHS rows required;
- estimated dense-RHS reuse;
- metadata bytes;
- value bytes;
- partial-block fraction;
- dense-fragment candidates;
- heavy-row indicators;
- module activity frequency;
- forward and transpose locality;
- partition-cut edges;
- precision and quantization range;
- likely kernel class.

Only compact scheduling fields belong in the hot tile header. Detailed distributions belong in a planner sidecar or calibration artifact.

## Physical Format Rules

### Pointer-free persistence

Persistent images use:

- versioned headers;
- explicit section kinds;
- relative offsets;
- explicit lengths and capacities;
- stable identities;
- alignment;
- checksums at the transport envelope;
- validation before rebinding.

A self-describing image is for loading, validation, and planning. Kernels consume compact rebound views rather than parsing a general directory repeatedly.

### Projection directory

The future image should support optional sections:

```cpp
enum class projection_kind : std::uint32_t {
    native_row_masked,
    native_feature_masked,
    native_cta_macrotile,
    dense_fragment,
    csr,
    sell,
    bsr,
    blocked_ell,
    transpose_native
};
```

Do not persist every possible projection. Persist those with sufficient expected reuse. Build others lazily and cache them.

### Metadata must pay rent

Persist metadata when it removes repeated runtime work or enables a better kernel.

Reject metadata that:

- duplicates values;
- is read by every lane but rarely useful;
- encodes one kernel's schedule into the semantic ABI;
- can be derived cheaply once during plan preparation;
- increases transfer enough to erase its scheduling benefit.

## Interaction With Core Execution

CellPack does not select a final kernel in isolation.

It provides:

- structure identities;
- candidate geometries;
- projection capabilities;
- construction cost;
- tile statistics;
- persistent sections;
- validation.

Cellerator core provides:

- operation semantics;
- actual dense RHS width and dtype;
- expected reuse;
- device profile;
- planner and autotuner;
- graph-wide order decisions;
- launch bindings;
- execution and profiling.

A hardware cost model inside CellPack is an input to the core planner, not a second independent planner.

## Interaction With Baseplane

Baseplane may provide:

- sequence bit planes;
- motif masks;
- compact events;
- segments;
- static sequence-derived relations;
- sequence-domain priors for grouping;
- producer capabilities that can be fused into Cellerator kernels.

CellPack may compile sequence-derived relations into the same semantic geometry machinery used for expression and regulatory graphs.

Do not make Baseplane emit CPK1 bytes directly. It may write into a neutral relation builder owned by the shared Cellerator ABI.

## Interaction With CellShard

CellShard may:

- store an opaque Cellerator image;
- wrap it in a versioned execution envelope;
- validate dataset, partition, generation, and checksum;
- fetch and upload the image contiguously;
- route it to workers.

CellShard must not:

- infer feature groups;
- decide biological geometry;
- reinterpret projection sections;
- rebuild tiles;
- select kernels.

## Performance Requirements

A CellPack claim must report the relevant subset of:

- total build time;
- projection build time;
- persisted bytes;
- metadata bytes;
- bytes per useful edge;
- descriptor-lane efficiency;
- row and feature reuse;
- kernel time;
- epilogue and order-transform time;
- end-to-end operation time;
- expected reuse and break-even count;
- forward and backward time;
- graph-capture compatibility;
- communication cut and bytes;
- memory expansion.

Required baselines may include:

- canonical CSR and cuSPARSE;
- the existing Cellerator CSR path;
- SELL;
- current Blocked-ELL;
- current CP-BP row-masked tiles;
- feature-only ordering;
- row-only ordering;
- combined semantic geometry;
- hybrid projection;
- dense execution when density warrants it.

Use real and adversarial structure traces. Full-width synthetic blocks are not sufficient.

## Correctness Requirements

Every new geometry or projection must test:

- domain identity;
- order identity;
- geometry identity;
- structure epoch;
- value generation;
- canonical feature recovery;
- canonical row recovery;
- forward reconstruction;
- empty rows and tiles;
- partial feature blocks;
- partial warp tiles;
- bit 31 and full masks;
- duplicate or overlapping relations where legal;
- invalid capacities and offsets;
- stale identities;
- deterministic construction with fixed inputs;
- reference numerical equivalence;
- projection-to-projection logical equivalence;
- forward and transpose consistency when training is supported;
- archive, rebind, and device execution;
- Compute Sanitizer for new device views.

## Forbidden Changes

Do not:

- declare one physical format universal;
- remove CSR or another fallback because a native path won one benchmark;
- make canonical output mandatory for internal consumers;
- merge structure and mutable values into an inseparable core ABI;
- place stream or per-run pointers in a durable plan identity;
- use manual cell labels as required layout input;
- move geometry inference into CellShard;
- treat correlation or biological coherence as the final optimizer objective;
- add per-cell dynamic routing;
- build modules during forward;
- claim Tensor Core benefit from density alone;
- tune only on regular full-block synthetic data;
- change v1 persistence without versioning;
- erase the exact host referee.

## Near-Term Priority

Before accelerating the remaining CUDA packing evaluator, complete the core contracts that determine what the evaluator should optimize:

1. first-class domain and order identity;
2. structure and value separation;
3. reusable plan and launch-binding split;
4. projection catalog;
5. low-sharing and feature-major native kernels;
6. real-data benchmark corpus;
7. end-to-end planner cost.

A faster optimizer for the wrong objective is not the next bottleneck.

## Review Checklist

Before merging CellPack work, verify:

- Is this semantic geometry or a physical projection?
- Which operation regimes does it target?
- Does it preserve canonical identity without forcing canonical execution?
- Can values change without rebuilding structure?
- Does the planner see construction and conversion cost?
- Is runtime discovery eliminated?
- Is metadata compact and useful in the hot path?
- Are multiple projection candidates retained where appropriate?
- Does the change support future transpose and multi-GPU needs?
- Are real and adversarial structures tested?
- Is the strongest relevant baseline included?
- Is the v1 path preserved through a versioned adapter?
