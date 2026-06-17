# AGENTS.md — CellPack

## Mission

CellPack is Cellerator's static semantic sparse-packing subsystem. Its job is to organize biological sparse matrices into compute-efficient, module-packed layouts that reduce memory-access entropy, index traffic, padding, launch fragmentation, and runtime indirection.

CellPack should transform canonical sparse biological data, typically CSR/COO or CellShard-backed sparse partitions, into static packed computational regions. Those regions may use Blocked-ELL, Sliced-ELL, BCSR, bitmap/tile formats, dense tiles, quantized variants, or residual CSR depending on the local structure.

The goal is not to make runtime kernels reason about biology. The goal is to use data-derived biological structure offline to compile the matrix into a low-entropy memory layout. Runtime kernels should see flat descriptors, contiguous buffers, predictable access patterns, and minimal metadata.

The central rule is:

> Dynamic selection is allowed. Dynamic assembly is forbidden.

CellPack may eventually support dynamic computation by skipping precompiled blocks when confidence is high enough. It must not dynamically discover, gather, bucket, or assemble modules in the hot path.

## Current Cellerator Relationship

Cellerator already contains much of the spirit of this idea, but the current sparse-layout implementation should be treated as scaffolding rather than the final design. Existing Blocked-ELL, quantized Blocked-ELL, conversion, sparse compute, and benchmarking surfaces are useful references, but CellPack should not assume that the current NNZ-bucketing or CSR-to-Blocked-ELL paths are semantically correct.

CellPack belongs inside Cellerator because it is a compute-layout compiler and runtime execution surface, not canonical storage or ingest. CellShard remains the owner of durable source storage, ingest, `.csh5` / `.cspack` publication, shard ownership, and canonical retrieval. CellPack consumes explicit matrix views and emits compute-oriented packed layouts, metadata, and execution descriptors.

CellPack should preserve the Cellerator philosophy:

- keep layout, memory residency, transfer, and launch costs visible;
- avoid hidden high-level workflow abstractions in hot paths;
- keep Torch, AnnData, Scanpy, and CellShard interop at explicit boundaries;
- favor measured low-level operators over convenience layers.

## Non-Goals

CellPack is not:

- a manual annotation system;
- a cell-type labeling system;
- a pathway database;
- a Scanpy, AnnData, or scvi-tools replacement;
- a generic sparse matrix library unrelated to biological compute;
- a dynamic sparse router that assembles modules at runtime;
- a per-cell attention mechanism that creates scattered gather patterns;
- a storage owner for canonical biological datasets;
- a reason to hide format conversion costs from callers.

If an implementation requires runtime feature lookup, runtime module construction, per-cell scattered module gathering, or repeated dynamic bucketing, it is probably outside CellPack's intended design.

## Core Terms

### Feature module

A feature module is a data-derived storage/co-access unit. It may correspond to a gene program, regulatory module, chromatin-accessibility module, peak neighborhood, pathway-like group, co-detection group, or task-specific predictive feature block, but it is not accepted merely because it has a biological name.

A feature module earns first-class storage status only if packing it improves or plausibly improves compute behavior: locality, block density, dense RHS reuse, padding, index traffic, or skippability.

### Row signature

A row signature is a compact description of how a row uses feature modules: module occupancy, density, activation, or related structural statistics. Row signatures are used offline to order rows and form row groups. They must not become a hot-path manual cell-type lookup.

### Packed region

A packed region is a flat executable storage region with one concrete sparse or dense layout. Runtime kernels consume packed regions, not biological trees.

### Conditional region

A conditional region is a packed region that is meaningful enough to store separately and substantial enough to possibly skip for some microbatches. It is precompiled and contiguous. Optional gating may select or skip it, but the region itself is never assembled dynamically.

### Residual region

A residual region holds rare, tiny, irregular, or low-confidence structure that should not pollute the primary hot layout. Residual data may use CSR, Sliced-ELL, or another compact fallback.

### Layout epoch

A layout epoch is a period during which the physical row/feature permutations and packed regions are fixed. Training may collect statistics that inform a future repack, but the layout must not mutate every batch.

### Route mask and route tape

A route mask is an optional bitmask or compact list of precompiled regions selected for a microbatch. A route tape records the exact regions executed during forward so backward can replay only those regions.

## Design Principles

1. **Static packing first.** Do expensive module discovery, row ordering, format selection, and packing outside the hot path.

2. **Runtime gets flat descriptors.** Kernels should consume arrays of packed-region descriptors and contiguous value/index buffers. They should not traverse biological hierarchies.

3. **Modules are co-access units.** A module is a storage and computation decision, not a manual biological label.

4. **No manual cell annotations.** CellPack may infer row structure from the data, but it must not require human cell-type labels. Labels may be used for external evaluation only, not as required layout input.

5. **The row axis is evidence, not identity.** Row structure helps discover feature modules, row signatures, and row ordering. Runtime should not ask what cell type a row is.

6. **Hybrid layout is expected.** Blocked-ELL is important, but no single sparse format should be forced onto every region.

7. **Conditionally meaningful structure should be isolated.** If a module is real but only useful in a subset of rows or tasks, store it separately so it does not inflate the common hot layout.

8. **Dynamic selection may skip static blocks.** Optional gating may choose not to execute precompiled conditional regions. Gating must be coarse, cheap, microbatch-friendly, and recorded for backward.

9. **Dynamic assembly is forbidden.** Do not dynamically gather genes into modules, build sparse blocks, rebucket rows, or repack feature groups during forward/backward.

10. **Memory traffic is the primary enemy.** Favor fewer bytes, fewer index reads, fewer metadata reads, fewer irregular dense RHS loads, and fewer synchronization points over theoretical elegance.

11. **A biological module that slows computation is metadata, not a hot-path format.** Keep it for interpretation if useful, but do not force it into the primary compute layout.

12. **Every optimization needs a baseline.** Compare against raw CSR/cuSPARSE, current Blocked-ELL, NNZ-sorted layouts, feature-permuted layouts, and hybrid packed layouts before claiming a win.

## Intended Architecture

CellPack should be organized as a layout compiler plus a runtime substrate.

```text
canonical sparse input
    ↓
feature-module discovery / import
    ↓
row-signature construction
    ↓
row and feature permutation
    ↓
region planning and cost modeling
    ↓
hybrid format selection
    ↓
static packed layout emission
    ↓
flat runtime execution
    ↓
optional route-mask execution and backward replay
```

### 1. Input boundary

CellPack should accept explicit sparse matrix views and metadata through Cellerator/CellShard boundaries. It should not own source ingest or canonical dataset mutation.

Supported initial inputs may include:

- CSR / compressed-by-row views;
- COO / triplet views;
- sharded compressed views;
- future CellShard runtime matrix views;
- precomputed feature-module assignments for early testing.

### 2. Discovery boundary

Feature-module discovery may eventually use multiple signals:

- co-detection / binary occupancy;
- coexpression or covariance on rows, metacells, or local aggregates;
- matrix factorization usage patterns;
- graph-community structure;
- feature-neighborhood structure for peaks or genomic intervals;
- model usage, gradients, or block-error statistics from a previous layout epoch;
- optional biological priors as weighted evidence, never as mandatory truth.

Early implementations may accept precomputed modules or simple unsupervised module assignments. The interface should leave room for better discovery without entangling discovery with runtime execution.

### 3. Row-signature boundary

Rows should be ordered by static signatures derived from module occupancy and structural profile. Within row-signature groups, hardware-oriented refinements such as NNZ sorting are allowed.

Initial row ordering may use:

- module occupancy bitsets or counts;
- module density vectors;
- row NNZ as a secondary key;
- local similarity over sparse signatures;
- task-specific row usage statistics when available.

Do not require manual row annotations.

### 4. Region planner

The planner should form candidate row-group × feature-module regions and score them by compute value. It should classify regions as primary, shared, conditional, residual, or discarded.

A candidate region should be scored by at least:

- expected nonzero density;
- expected block/tile density;
- padding ratio;
- index bytes per useful value;
- row-block width variance;
- dense RHS locality;
- expected active frequency;
- residual fallback cost;
- launch and grouping cost;
- quantization/compression opportunity;
- amortized layout-build cost.

Biological coherence may be included as a discovery and stability signal, but it must not override severe compute cost.

### 5. Format selector

CellPack should support a hybrid format decision per region.

Recommended initial categories:

| Region property | Preferred layout |
| --- | --- |
| regular row-block width, tolerable padding | Blocked-ELL |
| high tile density | dense tile or BCSR/bitmap tile |
| variable width but locally grouped rows | Sliced-ELL |
| rare, tiny, scattered, or low-confidence data | CSR/residual |
| recurrent low-bit-value-compatible data | quantized Blocked-ELL or quantized tile |

The format selector should be conservative. A region that cannot beat a fallback should stay fallback.

### 6. Packed storage ABI

Persisted and runtime metadata should use offsets rather than pointer forests. The format must be versioned, self-describing, and explicit about row/feature permutations.

A future packed descriptor may resemble:

```cpp
struct cellpack_region_desc {
    std::uint32_t region_id;
    std::uint32_t parent_id;
    std::uint32_t flags;
    std::uint32_t layout_kind;

    std::uint32_t row_begin;
    std::uint32_t row_count;
    std::uint32_t feature_begin;
    std::uint32_t feature_count;

    std::uint32_t block_size;
    std::uint32_t width_class;
    std::uint32_t index_offset;
    std::uint32_t value_offset;
    std::uint32_t aux_offset;

    std::uint32_t weight_offset;
    std::uint32_t output_offset;
};
```

Use concrete Cellerator types when implementing. The shape above is conceptual, not a mandatory ABI.

### 7. Runtime execution

Runtime should execute flat packed regions in a deterministic order or in cost-model grouped order. It should group compatible regions by layout kind, block size, width class, and dense RHS access pattern.

Runtime must avoid:

- per-region host synchronization in hot loops;
- per-call allocation in steady state;
- pointer chasing through biological trees;
- runtime module construction;
- scattered feature gathers inside inner loops;
- per-cell kernel launches;
- hidden CSR/COO conversions.

### 8. Optional block gating

Block gating is a future extension, not a requirement for the first format. When implemented, it must follow these rules:

- gates select or skip precompiled regions;
- gates operate at row-group or microbatch granularity when possible;
- gates are cheap relative to the skipped compute;
- gates produce compact masks or active-region lists;
- selected regions are grouped before launch;
- forward records a route tape;
- backward replays the route tape;
- optimizer updates for inactive module parameters are lazy or active-set-aware;
- an always-compute fallback path remains available for correctness and benchmarking.

Do not implement per-cell arbitrary dynamic routing unless benchmarks show that grouping and launch overhead are under control.

## Directory Guidance

Prefer a clear in-repo layout such as:

```text
include/Cellerator/cellpack/
    format.hh              # public descriptors, enums, flags, versioning
    plan.hh                # planner-facing host-side contracts
    pack.hh                # packer entrypoints
    runtime.cuh            # CUDA-facing region views and runtime contracts
    gating.hh              # optional route-mask/tape contracts, later

src/cellpack/
    plan/
    pack/
    runtime/
    discovery/
    tests or test helpers as appropriate

tests/
    cellpack_*_test.cc/cu

bench/cellpack/
    cellpack_layout_bench.cu
    cellpack_spmm_bench.cu
```

Use the repository's existing include style: public in-repo callers should include `Cellerator/...` paths rather than reaching into `src/` directly.

## Implementation Rules

- Keep APIs explicit about ownership, device/host residency, stream usage, and scratch buffers.
- Do not allocate scratch memory repeatedly inside steady-state kernels or hot loops.
- Do not hide expensive conversions behind convenient constructors.
- Prefer deterministic layout decisions when inputs and seeds are fixed.
- Keep persisted formats pointer-free; use offsets and lengths.
- Keep inverse row/feature maps explicit so correctness tests can reconstruct original order.
- Treat external priors as optional signals, not required inputs.
- Avoid dependencies that make core Cellerator heavier unless they are isolated behind optional build flags.
- Preserve a no-gating execution path.
- Preserve residual fallback paths.
- Add measurements before replacing an existing sparse path.

## Correctness Requirements

Every CellPack layout must be able to prove equivalence to the source matrix within the selected precision and transform policy.

Required tests should include:

- row permutation and inverse permutation round trip;
- feature permutation and inverse permutation round trip;
- region coverage and non-overlap where applicable;
- source-to-packed-to-source reconstruction on small fixtures;
- equivalence of SpMM/projection against CSR baseline;
- residual-region correctness;
- deterministic packing for fixed inputs;
- invalid input handling;
- version/descriptor validation;
- route-tape backward equivalence when gating is added;
- no-gating and gated outputs match within expected tolerance when all regions are active.

## Performance Requirements

A CellPack change should report the relevant metrics for any path it claims to improve.

Important metrics:

- padded bytes;
- index bytes per useful nonzero;
- region count;
- launch count;
- distribution of `ell_width` / width class;
- block/tile fill ratio;
- residual NNZ fraction;
- H2D/D2H bytes if conversion is included;
- achieved HBM bandwidth;
- forward time;
- backward time when applicable;
- optimizer time when applicable;
- layout build time and expected amortization;
- end-to-end epoch or batch time for model-facing paths.

Benchmark against at least the closest relevant baselines:

```text
raw CSR/cuSPARSE
current Blocked-ELL
NNZ-sorted Blocked-ELL
feature-permuted Blocked-ELL
feature + row-signature-permuted Blocked-ELL
hybrid CellPack regions
hybrid CellPack + oracle gating, when gating exists
hybrid CellPack + learned gating, when gating exists
```

Do not claim a performance improvement from reduced theoretical FLOPs alone. Memory traffic, launch count, and end-to-end wall time decide.

## Review Checklist

Before merging CellPack work, verify:

- Does the hot path consume static descriptors instead of building modules?
- Are row and feature permutations explicit and invertible?
- Are conditional modules stored separately without polluting the primary layout?
- Is manual cell annotation unnecessary?
- Is there a residual fallback for rare or irregular data?
- Does the format selector avoid forcing Blocked-ELL on bad regions?
- Are dynamic decisions limited to selecting/skipping precompiled blocks?
- Is there a no-gating correctness path?
- Are scratch allocation, synchronization, and transfer costs visible?
- Are tests comparing against CSR/source truth?
- Are benchmarks comparing against the right baselines?

## Roadmap

### M0 — CellPack skeleton and ABI

- Add public format descriptors, enums, and version constants.
- Add region descriptor validation.
- Add row/feature permutation structures and inverse-map utilities.
- Add small fixture tests.

### M1 — Static module-packed baseline

- Accept precomputed feature-module assignments.
- Build feature permutation from module order.
- Build row signatures from module occupancy.
- Emit a flat packed-region plan.
- Preserve residual CSR fallback.

### M2 — Baseline packing and reconstruction

- Pack CSR/COO into module-contiguous regions.
- Support at least CSR residual and Blocked-ELL candidate regions.
- Add round-trip reconstruction tests.
- Add CSR-baseline SpMM/projection equivalence tests.

### M3 — Cost model and hybrid layout selection

- Add padding/index/width/fill metrics.
- Choose among Blocked-ELL, Sliced-ELL, dense/tile, and residual where supported.
- Add layout benchmark binaries.

### M4 — Optional gating experiment

- Add static oracle masks over precompiled conditional regions.
- Add route tape for backward replay.
- Benchmark no-gating vs oracle-gating.
- Only then consider learned gates.

### M5 — Feedback-guided repacking

- Collect region usage, gradients, skip errors, and residual pressure.
- Support offline layout-epoch repacking.
- Keep runtime layout immutable within an epoch.

## North Star

CellPack succeeds when biological sparse data becomes easier for hardware to consume:

- fewer scattered loads;
- fewer index bytes;
- less padding;
- better dense RHS locality;
- fewer tiny launches;
- explicit residual handling;
- optional skipped compute only over static blocks;
- correctness preserved against canonical sparse input.

The final product should feel like a compiler: it studies the data, chooses a physical layout, emits static executable regions, and then gets out of the kernel's way.
