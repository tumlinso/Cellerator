# Cellerator performance-first replacement map for `std::*` data structures

**Source basis:** Cellerator working tree at commit `9216a42c21c5250c835eb16f9cff6e25799e9172`, inspected on 2026-08-29. The worktree was dirty only in `AGENTS.md`. This report is source-grounded. Project todo/workflow authority was unavailable during inspection, so no claims are made here about task ownership or active implementation lanes.

## 1. Executive verdict

Cellerator should not replace `std::vector<T>` with a generic `buffer<T>`, `T* + size`, or a custom vector clone. That would preserve the same conceptual failure while discarding useful safety.

The current core uses `std::vector` as an accidental universal representation for at least six different structures:

1. immutable compiled plans,
2. synchronized structure-of-arrays tables,
3. ragged relations,
4. fixed or tightly bounded sets,
5. temporary construction workspaces,
6. CPU/GPU transfer and residency boundaries.

Each class needs a different replacement. The correct migration is therefore a data-model and algorithm migration, not a container migration.

The strongest existing model is already inside Cellerator. CPK1, FMP1, CTP1, fixed-capacity runtime session tables, and raw execution bindings use explicit requirements, caller-provided storage, pointer-free durable images, typed hot views, and rebinding. Those patterns should become the default Cellerator idiom. The older geometry, sampling, optimizer, graph, and neighbor code should converge toward them.

The worst current areas are:

- the packing optimizer, which owns per-block vectors, copies member lists during scoring, rebuilds global lookup arrays after local mutations, validates globally after each mutation, uses node-based maps and sets for dense integer domains, and deep-copies the entire optimizer state for rollback;
- the gene-support to candidate-discovery to merge-score CUDA chain, which repeatedly stages device-capable data through host ownership and performs avoidable allocation, transfer, and synchronization;
- graph and forward-neighbor code, which encodes fixed-width or relational structures as growable arrays, duplicates representations, and frequently materializes host results that could remain on-device;
- the public geometry plan types, which expose vectors where the data are immutable compiled images with strong cross-array invariants.

The report recommends a small amount of generic low-level plumbing, but no generic owning sequence. Ownership remains domain-specific.

## 2. Governing rules

### 2.1 Core rule

Every owning collection in production Cellerator must answer, in its type and header:

- what the elements mean;
- which axis they index;
- whether ordering is canonical;
- whether the data are mutable;
- who owns the storage;
- which memory domain contains it;
- how capacity is established;
- which arrays share cardinality;
- whether the representation can move or be persisted;
- which access pattern the representation is optimized for.

A type that answers only “N values of T” is not sufficiently designed for the core library.

### 2.2 Two-form representation

Use two forms for substantial structures:

1. **Durable/build form:** a pointer-free image with a fixed header, relative offsets, explicit counts, alignment, schema identity, and validation.
2. **Hot form:** a small typed view containing raw pointers, extents, strides, and validated alias/alignment guarantees.

The image is relocatable and persistable. The view is bare-metal and compiler-friendly. Raw pointers appear exactly where they are useful, after the representation has been designed.

### 2.3 Allocation policy

Allocation is explicit and domain-bearing. The core vocabulary should distinguish at least:

- ordinary host memory;
- NUMA-bound host memory;
- page-locked host memory;
- write-combined page-locked upload memory;
- device memory;
- managed memory, explicit and non-default;
- caller/session workspace memory.

Allocation must not be hidden behind shared ownership, implicit growth, or an operation that can synchronize the device unexpectedly.

### 2.4 Preparation versus execution

Preparation may inspect shapes and choose an explicitly requested implementation variant. Execution on a prepared/sealed session must not allocate, grow, rebuild descriptors, copy metadata to inspect a scalar, or synchronize the entire device.

This is deterministic dispatch and preparation, not a runtime optimizer. No online benchmark tuner is proposed.

### 2.5 Fixed-capacity fast paths

When a domain is bounded, make the bound structural. Examples include candidate `K <= 8`, exact-search `K <= 32`, feature-block width, small route sets, and per-mutation touched blocks.

The fast representation should have an explicit extension path, not silently become a generic vector. The extension can use a separately named segmented representation selected during preparation.

### 2.6 Agent compatibility

The performance architecture must remain easy for agentic coding and CPP Context Compiler to recover:

- every public structure is defined in one obvious domain header;
- headers contain a compact memory diagram and invariants;
- hot views are plain PODs;
- type-erased allocators, virtual container hierarchies, and template metaprogramming frameworks are forbidden;
- templates are confined to small internal compile-time kernel specializations;
- macros are limited to compiler annotations and feature detection;
- assembly, if ever justified, lives beside a readable reference implementation and a short semantic contract;
- every image has `static_assert` checks for layout and alignment;
- source files use named records instead of anonymous `pair`/`tuple` protocols.

## 3. Thin foundational vocabulary

The following headers are sufficient. They are plumbing, not replacement containers.

### 3.1 `include/Cellerator/memory/domain.hh`

```cpp
#pragma once

#include <Cellerator/types.hh>

namespace cellerator::memory {

enum class domain : u8 {
    host = 0,
    host_numa = 1,
    host_pinned = 2,
    host_pinned_write_combined = 3,
    device = 4,
    managed = 5,
    external = 6
};

struct placement {
    domain kind = domain::external;
    i16 device_ordinal = -1;
    i16 numa_node = -1;
    u32 flags = 0;
};

} // namespace cellerator::memory
```

### 3.2 `include/Cellerator/memory/allocation.hh`

```cpp
#pragma once

#include <Cellerator/memory/domain.hh>

namespace cellerator::memory {

struct allocation {
    void *base = nullptr;
    u64 bytes = 0;
    u32 alignment = 0;
    placement where{};
    u32 generation = 0;
};

struct allocation_request {
    u64 bytes = 0;
    u32 alignment = 64;
    placement where{};
};

status allocate(const allocation_request &, allocation *) noexcept;
status release(allocation *) noexcept;

} // namespace cellerator::memory
```

There is no reference count. Long-lived ownership belongs to a session, compiled object, or caller. Hot code receives views or generation-checked handles.

### 3.3 `include/Cellerator/memory/view.hh`

```cpp
#pragma once

#include <Cellerator/types.hh>

namespace cellerator::memory {

template<class T>
struct array_view {
    T *data = nullptr;
    u32 count = 0;
};

template<class T>
struct const_array_view {
    const T *data = nullptr;
    u32 count = 0;
};

template<class T>
struct matrix_view {
    T *data = nullptr;
    u32 rows = 0;
    u32 columns = 0;
    u32 row_stride = 0;
};

} // namespace cellerator::memory
```

These are intentionally weak primitives. Domain APIs should expose `feature_permutation_view`, `candidate_pair_view`, `trajectory_record_view`, and similar names rather than leaking generic views everywhere.

### 3.4 `include/Cellerator/memory/workspace.hh`

```cpp
#pragma once

#include <Cellerator/memory/domain.hh>

namespace cellerator::memory {

struct workspace {
    unsigned char *base = nullptr;
    u64 bytes = 0;
    u64 cursor = 0;
    placement where{};
};

status reset(workspace *) noexcept;
status take_bytes(workspace *, u64 bytes, u32 alignment, void **out) noexcept;

template<class T>
status take(workspace *ws, u64 count, u32 alignment, T **out) noexcept;

} // namespace cellerator::memory
```

A workspace does not grow. Requirements are queried before execution. A session may own one workspace per device and stream, and CPU algorithms may use one per worker or NUMA node.

### 3.5 `include/Cellerator/memory/image.hh`

```cpp
#pragma once

#include <Cellerator/types.hh>

namespace cellerator::memory {

struct image_header {
    u32 magic = 0;
    u16 schema_version = 0;
    u16 flags = 0;
    u64 total_bytes = 0;
    u32 required_alignment = 0;
    u32 section_count = 0;
    u64 identity = 0;
};

struct rel32 { u32 byte_offset = 0; };
struct rel64 { u64 byte_offset = 0; };

struct image_buffer {
    void *base = nullptr;
    u64 bytes = 0;
    memory::placement where{};
};

} // namespace cellerator::memory
```

Use `rel32` when a format is explicitly limited to less than 4 GiB and define a separate wide schema when the requirement overflows. Do not silently widen every local index.

### 3.6 `include/Cellerator/memory/generation_marks.hh`

A generation-mark table replaces repeated clearing and many temporary `set`/`unordered_set` uses over dense integer domains:

```cpp
struct generation_marks {
    u32 *marks = nullptr;
    u32 count = 0;
    u32 generation = 1;
};
```

Advancing a generation logically clears the table. Wraparound performs one real clear.

### 3.7 `include/Cellerator/memory/flat_table.hh`

This is not a generic unordered-map replacement. It should contain only reusable probing mechanics for trivially copyable keys and values, while each user defines a named domain table with:

- exact capacity established at preparation;
- a fixed load factor;
- no rehashing;
- an explicit empty/sentinel encoding or generation array;
- a deterministic hash;
- contiguous storage;
- no per-entry allocation.

Sorting, counting, direct indexing, or CSR should be preferred whenever the domain allows them.

### 3.8 `include/Cellerator/compiler/hints.hh`

Provide compact Clang/GCC/NVCC-compatible annotations:

- `CELLERATOR_RESTRICT`;
- `CELLERATOR_ASSUME(condition)`;
- `CELLERATOR_ASSUME_ALIGNED(ptr, alignment)`;
- `CELLERATOR_FORCEINLINE`;
- `CELLERATOR_NOINLINE`;
- likely/unlikely hints.

Assumptions may only appear after a validating boundary has established the condition. They are not substitutes for validation.

## 4. Full production migration map

### Legend

- **Eliminate:** the current collection should cease to exist.
- **Image:** immutable offset-based representation plus typed view.
- **Table:** named exact SoA/AoS or fixed-width table.
- **Relation:** offsets plus members or sorted packed keys.
- **Workspace:** caller/session scratch with queried capacity.
- **Move:** leave the Cellerator core rather than receive a deep rewrite in place.
- **Cold keep:** STL may remain in a boundary adapter with no measurable execution effect.

| Current source | What the container is really representing | Replacement | Verdict | Priority |
|---|---|---|---|---|
| `include/Cellerator/geometry/planner.hh`, `src/geometry/planner.cc` | Immutable compiled permutations, signatures, modules, row groups, regions | `static_plan_image` and `static_plan_view`; flat sorted module-feature and row-signature relations | Image + relation | P0/P1 |
| `include/Cellerator/geometry/packing_plan.hh`, `src/geometry/packing_plan.cc` | Frozen immutable packing plan currently split across six arrays | One `packing_plan_image`, preferably folded into or shared with CPK1 | Image | P0 |
| `include/Cellerator/geometry/pack.hh`, `src/geometry/pack.cc` | Temporary coordinate object stream and reconstructed CSR | Direct count/scan/fill into physical projection; optional diagnostic coordinate image; exact CSR image | Eliminate + image | P1 |
| `include/Cellerator/geometry/gating.hh`, `src/geometry/gating.cc` | Ordered active-region selection and replay identity | Region bitset plus compact ordered IDs, or deterministic replay token | Table/image | P1 |
| `include/Cellerator/geometry/gating_cuda.cuh`, `src/geometry/gating_cuda.cu` | Region-to-coordinate ragged relation and aligned coordinate fields | `compiled_route_image` with region offsets and coordinate SoA; direct count/scan/fill | Image + relation | P1 |
| `include/Cellerator/geometry/layout_metrics.hh`, `src/geometry/layout_metrics.cc` | One metrics record per region and temporary region-row widths | Exact region metrics table; one flat row-width workspace keyed by region offsets | Table + workspace | P1 |
| `include/Cellerator/geometry/layout_selector.hh`, `src/geometry/layout_selector.cc` | One physical-layout decision per region | Region-indexed selection table; launch groups generated by fixed/sorted exact scratch | Table | P1 |
| `include/Cellerator/geometry/candidate_relation.hh`, `src/geometry/candidate_relation.cc` | Canonical unique feature-pair relation | Packed 64-bit pair keys plus aligned evidence arrays, in-place radix sort/compact | Relation/image | P1 |
| `src/geometry/optimizer_state.hh`, `src/geometry/optimizer.cc` | Mutable bounded feature-block partition, support union cache, proposal graph, rollback history | Fixed-stride block member slab, direct feature map, dense or pooled union cache, mutation journal, packed proposal tables, generation marks | Algorithm + table/workspace | P0 |
| `include/Cellerator/geometry/statistical_validation.hh`, `src/geometry/statistical_validation.cc` | Validation units, grouped rows, split order, null-edge membership | Sorted group-row relation with offsets; packed edge table; generation marks; exact workspace | Relation/workspace | P1 |
| `include/Cellerator/geometry/record_statistical_validation.hh`, `src/geometry/record_statistical_validation.cc` | Reusable decode and comparison scratch | Exact decode workspace; fixed-capacity row buffer with separately named overflow representation | Workspace | P2 |
| `include/Cellerator/compute/sampling.hh`, `src/compute/dataset/sampling.cc` | Canonical row selection plus provenance and bounded top-K selection state | `sample_selection_image`; compact stratum descriptors; explicit bounded heaps or radix selection | Image + bounded table | P1 |
| `src/compute/dataset/sampling_materialization.cc` | One sampled CSR and ordering scratch | One `sampled_csr_image`; consume already sorted sample rows; caller workspace | Image + workspace | P1 |
| `include/Cellerator/geometry/gene_support_bitset.hh`, `src/compute/dataset/gene_support_bitset.cu` | Dense gene-by-sampled-cell bit matrix, counts, row map, provenance | Domain-bearing `gene_support_image`; direct host or device build | Image | P0 |
| `include/Cellerator/geometry/gene_candidate_discovery.hh`, `src/geometry/candidate_discovery/*` | Device/host sketch pipeline and unique packed feature pairs | Prepared device workspace image; device-resident packed pair image; explicit host materialization adapter | Device image/workspace | P0 |
| `include/Cellerator/geometry/merge_cost.hh`, `src/geometry/merge_cost*` | Candidate-pair score relation | Consume device support and candidate image directly; device score image; pinned boundary transfer | Device relation | P0 |
| `src/compute/projection/physical_feature_major.cc`, `physical_transpose.cc` | Construction/validation scratch for already good pointer-free images | Caller workspace and count/scan/fill; retain FMP1/CTP1 image APIs | Workspace | P2 |
| `include/Cellerator/compute/operators/graph/record_table.cuh` | Immutable trajectory record SoA plus builder | `trajectory_record_image`; separate chunked builder or exact two-pass build | Image + builder | P1/P2 |
| `.../graph/slab_index.cuh` | Embryo/time run table and one future interval per row | Embryo span image; packed `uint2`/interval table; binary-search assignment | Table | P2 |
| `.../graph/forward_candidates.cuh`, `forward_prune.cuh` | Bounded top-K edge table and final bounded-degree graph | Fused K-specialized scoring/pruning into fixed-width edge table; optional CSR projection | Eliminate + fixed table | P1 |
| `.../graph/supernode_reduce.cuh` | Tree topology, member relation, centroids, aggregated DAG | Separate topology/member/Euler images; sort-reduce packed DAG edges | Images + relation | P1 |
| `.../graph/branch_detect.cuh` | Top two outgoing masses | Two scalar accumulators; no allocation and no sort | Eliminate | P0 quick win |
| `.../graph/incremental_insert.cuh` | Slab-to-input-row ragged relation | Slab assignment CSR/offset relation built by count/scan/fill | Relation | P2 |
| `include/Cellerator/examples/trajectory/trajectory_query.cuh` | Fresh materialized query results | Direct member/subtree spans; caller path buffer or iterator | Views; move example | P2 |
| `include/Cellerator/compute/neighbors/exact_search.hh`, `src/.../exact_search.cu` | Bounded device top-K | Keep public raw views; specialize K and redesign internal candidate storage/warp merge | Kernel rewrite, no owner | P1 |
| `include/Cellerator/compute/neighbors/forward_neighbors/fn_*`, `src/.../forward_neighbors.cu` | High-level index/storage/workflow plus many duplicated physical representations | Split storage/index building downstream; keep low-level scoring/refinement kernels and prepared views in Cellerator | Move + redesign | P0 architecture |
| `src/compute/neighbors/scoring/cuvs_sharded_knn.cu` | Legacy growable host/device shard staging | Move downstream or rewrite as requirements-driven prepared multi-GPU plan | Move/retire | P2 |
| `include/Cellerator/compute/core/host_buffer.hh` | A custom vector clone with geometric growth | Delete from production core; no universal replacement | Eliminate | P0 policy |
| `include/Cellerator/runtime/device_buffer.cuh`, graph workspace buffer | Shared-owned device allocation with hidden lifetime and blocking transfer helpers | Session allocation handles, raw device views, stream-aware explicit copies | Eliminate | P0 |
| `include/Cellerator/runtime/scratch.cuh`, `src/runtime/runtime.cu` | One grow-by-free/reallocate scratch buffer | Prepared suballocated slab per device/stream; stream-ordered pool fallback outside sealed execution | Workspace/runtime | P0 |
| `components/CelleraTorch/*` | Framework boundary and framework-owned lifetimes | Permit STL only where required by Torch ABI; convert immediately at the Cellerator boundary | Cold/boundary keep | P3 |
| `tests`, `bench`, `examples`, `compat/legacy_sparse` | Reference fixtures, diagnostics, legacy code | STL allowed unless it contaminates measured setup or production headers | Allowlisted | Outside core |

## 5. Detailed replacement designs

## 5.1 Static geometry plan

### Current problem

`static_plan` owns eleven vectors: row and feature permutations, inverse maps, signatures, modules, row groups, regions, and execution-axis offsets. `src/geometry/planner.cc` then builds nested module and row objects, linearly searches modules, sorts each row’s module list, and copies the result into the final plan.

The vectors hide a compiled relational schema:

- feature to module assignment;
- canonical feature order;
- row to unique signature assignment;
- signature to module membership;
- row-group to row range;
- row-group and module to packed region.

### Replacement

Define `static_plan_header` and `static_plan_view` in `include/Cellerator/geometry/static_plan.hh`. Sections:

- `row_permutation[row_count]`;
- `inverse_row_permutation[row_count]`;
- `feature_permutation[feature_count]`;
- `inverse_feature_permutation[feature_count]`;
- `signature_offsets[signature_count + 1]`;
- `signature_module_ids[signature_entry_count]`;
- `row_groups[row_group_count]`;
- `modules[module_count]`;
- `regions[region_count]`;
- `feature_block_offsets[module_count + 1]`;
- `row_group_offsets[row_group_count + 1]`;
- optional direct `row_group_module_to_region` table when repeated region lookup is hot.

Build it using flat records:

1. Emit `(module_id, feature_id)` for non-residual features.
2. Radix-sort by packed 64-bit key.
3. Run-length encode modules and write the feature permutation and module descriptors directly.
4. Validate row offsets.
5. Emit canonical `(row_id, module_id)` entries, filtering residual and invalid IDs.
6. Sort by `(row_id, module_id)` and compact duplicates.
7. Build per-row signature spans and hashes.
8. Sort row descriptors by `(hash, length, content, row_id)` using indirect indices.
9. Run-length encode equal signatures into row groups.
10. Count regions, prefix-sum, and fill the final image.

Use binary search or a compact direct module-index map instead of `find_module_index` linear scans. The final object is immutable.

### Expected benefit

- removes nested allocations and repeated growth;
- converts linear module searches to direct or logarithmic lookup;
- permits one exact requirements query and one build allocation;
- makes the plan directly persistable and transferable;
- enables raw `restrict` views for packing kernels;
- prevents cross-vector cardinality drift.

Confidence: high.

## 5.2 Frozen packing plan and CPK1 convergence

`frozen_packing_plan` is already vector-free, but it owns six independent `unique_ptr<u32[]>` arrays. That is a halfway state, not the target.

Replace it with one offset-based `packing_plan_image`, or make CPK1’s semantic plan section the sole frozen representation. The owner contains one `memory::allocation`; the hot `packing_plan_view` remains raw pointers and counts.

Do not force all large data into one allocation when independent placement or lifetime is useful. The requirements API may expose one metadata image plus separately placeable large value sections. What must disappear is accidental per-array heap ownership.

Confidence: very high.

## 5.3 Packing and coordinate compilation

### Current problem

`src/geometry/pack.cc` creates one `packed_coordinate` object per nonzero, performs a linear region scan for every coordinate, and later copies and sorts the entire coordinate array to reconstruct CSR. `gating_cuda.cu` copies and sorts the coordinate array again to group by region.

### Replacement

The production path should not materialize `packed_coordinate_plan`.

Compile directly:

1. Precompute direct region lookup from row-group and feature-block IDs.
2. First pass: count entries per output region or physical block.
3. Prefix-sum counts into offsets.
4. Second pass: fill row IDs, local feature IDs, and values directly into the final physical projection.
5. Build inverse maps only when a consumer actually needs them.

For CSR reconstruction, fill exact row counts, scan, and scatter directly. Preserve an optional `packed_coordinate_image` only for diagnostics, tests, and format-neutral debugging.

Confidence: very high. The only benchmark-dependent detail is whether a direct dense row-group by feature-block table or a smaller search structure wins for a given plan size.

## 5.4 Gating and replay

### Current problem

Route masks and tapes are vectors of region IDs. Validation allocates a new vector, sorts it, and scans for duplicates. Oracle-match validation builds another owning mask. The CUDA compiled coordinate plan owns four vectors and sorts coordinates again.

### Replacement

Define:

```cpp
struct route_selection_view {
    const u64 *active_words;
    const u32 *ordered_region_ids;
    u32 region_count;
    u32 active_count;
};
```

The bitset supports O(1) membership. The ordered compact IDs preserve deterministic launch order. If routes are always generated in canonical region order, duplicate validation reduces to monotonicity checks and no scratch.

A replay tape should record the minimum sufficient identity:

- exact ordered IDs when an externally supplied route is possible;
- otherwise `{plan_identity, route_policy_id, microbatch, route_hash}` rather than duplicating the same list.

The compiled route representation becomes:

- `region_offsets[region_count + 1]`;
- `row_ids[nnz]`;
- `feature_ids[nnz]`;
- `values[nnz]`;
- optional prepared launch descriptors grouped by geometry.

Count, scan, and fill it directly. The current one-block-per-region atomic kernels are then free to evolve independently of the ownership cleanup.

Confidence: high.

## 5.5 Layout metrics and selection

`row_widths_by_region` is a `vector<vector<u32>>` even though every row belongs to a known region range. Replace it with one flat row-width workspace sized from the plan’s row-group/region offsets. Region IDs are currently assigned densely, so direct indexing should replace `find_region` and `find_entry` scans.

`layout_metrics_plan` and `layout_selection_plan` become exact one-record-per-region tables. Launch-group uniqueness can use:

- a tiny fixed local table when the set of format/width classes is statically small;
- otherwise an exact scratch array sorted once.

`apply_layout_selection` must not deep-copy the entire semantic plan merely to change physical layout fields. Keep semantic geometry immutable and store physical selection as a separate region-indexed projection.

Confidence: high.

## 5.6 Candidate relations

Candidate endpoint IDs are 32-bit, so the canonical key is naturally:

```cpp
u64 key = (u64(min_feature) << 32) | u64(max_feature);
```

Use one packed key array, radix sort, and in-place duplicate compaction. Evidence fields should be parallel arrays only when kernels consume them independently; otherwise use a 16- or 32-byte aligned named record. This choice is access-pattern dependent and should be benchmarked.

Public ownership becomes an image/view contract. The current owning relation object and per-array `unique_ptr` fields disappear.

Confidence: high.

## 5.7 Packing optimizer

This is the highest-value host-side redesign.

### Current costs proven by source

`src/geometry/optimizer_state.hh`:

- each block owns `members` and `union_words` vectors;
- move and swap proxy scoring copies member vectors;
- every merge, move, and swap rebuilds `feature_to_slot` globally;
- every mutation calls full validation, which allocates and clears a `seen` vector;
- materialization rebuilds several vectors;
- union bitsets allocate lazily per block.

`src/geometry/optimizer.cc`:

- merge proposals use `std::map<pair<...>, ...>`;
- fanout uses `std::map<u32,u32>`;
- blacklists and batch conflict tracking use `std::set`;
- proposals are repeatedly allocated and sorted;
- rollback performs `optimizer_state snapshot = *state`, then replaces the whole state on rejection.

### Replacement state

Define in `include/Cellerator/geometry/optimizer_state.hh`:

```cpp
struct alignas(16) optimizer_block_desc {
    u32 stable_key;
    u32 generation;
    u16 member_count;
    u8 active;
    u8 union_state;
    u32 reserved;
};

struct optimizer_state_view {
    optimizer_block_desc *blocks;
    u32 *members;             // slot-major, fixed stride
    u32 *feature_to_slot;
    u32 slot_count;
    u32 feature_count;
    u32 member_stride;
};
```

Fast representation:

- one 64-byte-aligned descriptor table;
- one fixed-stride member slab `slot_count * maximum_block_width`;
- one direct `feature_to_slot[feature_count]` map;
- no per-block allocation;
- no global lookup rebuild after local mutations.

Extension representation:

- a separately named segmented member pool for block widths above the fast configured limit;
- selected explicitly during optimizer preparation;
- no automatic online tuning.

### Union cache

The fastest default should be benchmarked between two explicit modes:

1. **Dense union slab:** one aligned bitset row per block slot. This costs roughly one additional support matrix but makes cached union access direct and allows merges to OR rows. It is likely attractive because extra memory is explicitly acceptable and optimizer scoring reuses unions heavily.
2. **Prepared cache pool:** exact N cached union rows keyed by `(slot,generation)`, used when a full slab exceeds the caller’s memory budget.

Singleton blocks alias source support directly. Move and swap rebuild only the two touched union rows. Merge ORs the two existing rows when valid.

For host scoring, benchmark a 64-bit support-word projection because it halves loop iterations relative to current 32-bit words. Keep a 32-bit device projection for `__popc`-oriented CUDA kernels if that wins. Duplicate representations are allowed when their reuse amortizes conversion.

### Rollback

Replace deep snapshots with a preallocated mutation journal. One journal record stores:

- mutation kind;
- touched slots and generations;
- old block descriptors;
- old member slices for the touched blocks;
- changed `feature_to_slot` entries;
- union-cache validity and cache slot metadata.

A rejected batch is reversed in LIFO order. Journal storage is bounded by batch size and maximum block width, queried during preparation.

### Proposal machinery

- pack block-pair keys into `u64`;
- sort/compact or use an exact flat table, benchmarked against candidate count;
- replace fanout maps with dense `u16/u32[slot_count]` counters plus generation marks;
- replace blacklist sets with a small exact open-address table or sorted fixed list;
- replace batch conflict sets with generation-mark arrays over block slots and features;
- use partial selection for `proposal_shortlist` instead of sorting everything when candidate volume is large;
- allocate all proposal and journal storage once from the optimizer workspace.

### Validation

Keep full validation at initialization, explicit debug gates, accepted-phase boundaries, serialization, and tests. Production local mutations validate only the touched blocks and changed feature mappings. A compile-time or runtime debug flag can retain full validation without contaminating release performance.

### Materialization

Materialize execution geometry only when exact oracle evaluation needs it. Use exact output sections and direct writes. Keep semantic mutable state separate from the frozen plan image.

Confidence: very high for the structural rewrite. Dense-union versus pooled-union policy requires measurement.

## 5.8 Sampling

### Current hidden structure

`sample_plan` stores selected rows, hashes, row strata, per-row double weights, and a provenance object with several vectors and strings. Some of these are redundant:

- selected global rows are already canonical and can reproduce global-index hashes;
- per-row weights are constant within a stratum and can be represented by stratum totals and sample counts;
- strata often fit in 8 or 16 bits;
- string names are cold metadata, not execution data.

### Replacement

Define `sample_selection_image`:

```cpp
struct sample_stratum_desc {
    u64 upper_bound_inclusive;
    u64 population_rows;
    u64 sampled_rows;
};

struct sample_selection_header {
    memory::image_header common;
    u64 population_rows;
    u64 selected_rows;
    u64 seed;
    u64 split_identity;
    u32 algorithm_id;
    u32 algorithm_version;
    u32 identity_kind;
    u32 stratum_count;
    memory::rel64 selected_global_rows;
    memory::rel64 selected_strata;   // optional u8/u16 section
    memory::rel64 strata;
};
```

Drop stored hashes from the durable image unless a measured consumer needs them. Recompute them during validation. Replace per-row double weights with exact stratum ratios.

For exact lowest-hash selection:

- use an explicitly capacity-bounded max heap for small K;
- use deterministic radix selection plus final sort for larger K;
- parallel CPU selection uses per-worker exact buffers and a final deterministic merge;
- density-stratified selection uses one contiguous heap slab with per-stratum offsets, not a vector of priority queues.

`sampling_materialization.cc` should consume sorted selected rows directly and build one `sampled_csr_image` containing row offsets, feature IDs, values, and sampled-position-to-global-row mapping.

Confidence: high. The exact heap/radix crossover must be benchmarked, not runtime-autotuned.

## 5.9 Statistical validation

### Validation units

Replace `identities`, `row_to_unit`, and `vector<vector<row>> unit_rows` plus hash tables with:

```cpp
struct validation_units_view {
    const u64 *unit_ids;
    const u32 *unit_offsets;
    const u32 *unit_rows;
    const u32 *row_to_unit;
    u32 unit_count;
    u32 row_count;
};
```

Build it by sorting `(group_identity, row)` once and run-length encoding. In row-identity mode, avoid allocation entirely when the caller’s canonical row order already suffices.

### Uniqueness

Use radix-sorted identity scratch or an exact flat identity set. Sorting is preferable when the same order feeds later grouping.

### Null-edge models

Encode a feature pair as a packed 64-bit key. Use:

- sorted unique keys for batch generation and deterministic traversal;
- a named exact `edge_membership_table` for repeated random membership tests during rewiring;
- generation marks for temporary visited/degree state.

### Record validation

Expose exact workspace requirements. Use a fixed local row decode buffer for the common bounded case and a separately named overflow buffer when the record exceeds it. Do not return or grow a general sequence.

Confidence: high.

## 5.10 Gene support, candidate discovery, and merge scoring

This is the strongest end-to-end CUDA opportunity.

### Current pipeline fracture

The current code allocates multiple independent device arrays, copies host support to the GPU, launches work, synchronizes, copies support/candidates/scores back to host ownership, and later uploads those structures again for the next stage. Candidate discovery also repeatedly queries and reallocates CUB scratch and copies scalar counts to the host to direct subsequent stages.

### Replacement images

`gene_support_image`:

- header and provenance;
- support words;
- detected counts;
- sampled-position-to-global-row map;
- explicit memory domain and device ordinal in the owner, not embedded pointers in the image.

`candidate_pair_image`:

- packed 64-bit canonical feature-pair keys;
- exact count/capacity;
- optional evidence arrays;
- device-resident by default for the CUDA path.

`merge_score_image`:

- pair keys or stable candidate positions;
- score numerators/denominators or compact score records;
- device-resident until a caller explicitly requests host materialization.

### Prepared workspace

One query computes the maximum required storage for:

- sketches;
- band keys and gene IDs;
- sorted alternates;
- run-length output;
- bucket offsets;
- raw pair emission;
- unique pair output;
- CUB radix-sort, scan, select, and reduce temporary storage;
- scalar counters stored on device.

The session allocates one workspace slab per stream. CUB receives slices from that slab. No stage frees or reallocates it.

### Execution

1. Build gene support directly in its final host or device domain.
2. Compact nonempty genes on device.
3. Generate sketches and band keys.
4. Sort and run-length encode on device.
5. Emit packed candidate pairs.
6. Sort/unique pairs on device.
7. Feed the resulting device view directly into merge scoring.
8. Keep scores on device for optimizer/projection consumers when possible.
9. Use one asynchronous transfer to pinned host memory only at an actual host boundary.

Replace unconditional `cudaDeviceSynchronize` with stream ordering and events. Copy scalar counts to host only when a host decision is genuinely unavoidable; otherwise launch bounded kernels from device counters or retain maximum-capacity output with an explicit device count.

Confidence: very high. This removes proven transfer and synchronization work, though the final speedup magnitude requires measurement.

## 5.11 Physical projections

FMP1 and CTP1 already embody the target architecture: pointer-free payloads, exact requirements, caller buffers, typed views, validation, and rebinding. Keep them.

Remove their remaining construction vectors by:

- querying exact scratch;
- count/scan/fill;
- using generation marks for validation;
- allowing direct device construction only where measured.

Do not replace these APIs with a universal image framework that hides their domain semantics.

Confidence: very high.

## 5.12 Trajectory and graph structures

### Trajectory records

Replace `TrajectoryRecordTable` with an immutable SoA image:

- cell/global identity;
- embryo identity;
- developmental time;
- row-major latent matrix;
- optional precomputed embryo spans.

Appending belongs in a separate builder. The preferred build is exact two-pass. When streaming append is required, use fixed-size record chunks and a final freeze, not geometric growth.

### Slabs and windows

- `EmbryoRowSpan` becomes an exact run table.
- `FutureWindowBounds` becomes one packed interval per row, such as `{begin,count}`.
- delta-slab assignment searches embryo-local ordered slabs rather than scanning every slab for every row.

### Forward edges

The candidate and final graph are bounded by configuration. Fuse scoring and pruning into a fixed-width row table:

```cpp
struct alignas(16) forward_edge {
    u32 dst;
    float score;
    float delta_t;
    u32 flags;
};

struct forward_edge_table_view {
    forward_edge *edges;
    u8 *degree;
    u32 rows;
    u32 stride;
};
```

Public layout stays plain. Internal CUDA dispatch specializes K = 1, 2, 4, and 8. Emit CSR only for consumers that need CSR.

### Tree overlay

Split the current all-in-one object into optional images:

- parent topology: `parent`, `parent_score`, `depth`;
- child relation: offsets and child IDs;
- Euler index: `tin`, `tout`, `euler_to_node`, `node_to_euler`;
- binary-lifting table only when LCA/path operations require it.

Subtree queries then return a contiguous Euler span rather than scanning all nodes and allocating a result.

### Supernodes

Build in two passes:

1. assign supernode IDs and count members;
2. scan member counts, fill member CSR, and reduce time/mass/centroids into exact arrays.

For the supernode DAG, emit packed `(src,dst)` keys and associated mass/score records, radix-sort, reduce duplicates, and build CSR. Replace the forest of `unordered_map`s.

### Branch detection

Track the two largest outgoing masses in scalar variables. The current temporary vector and sort disappear completely.

### Incremental insert

Build a slab-to-input-row relation with counts, prefix sums, and member IDs. No vector per slab and no linear search through already-created buffers.

Confidence: high.

## 5.13 Exact search kernels

The public exact-search API is already close to ideal: raw device views and a fixed maximum K. Keep that interface.

Internally, the current large per-thread candidate arrays risk register pressure and local-memory spills, and lane-0 serial merges leave warp resources idle. Implement K-specialized kernels and benchmark:

- warp-cooperative top-K insertion/merge;
- compact candidate records using local 32-bit IDs and packed tie metadata;
- AoS versus SoA candidate scratch;
- shared-memory versus register exchange;
- direct final fixed-width output.

No assembly is justified before inspecting generated SASS and measuring a compiler failure. CUDA intrinsics and warp primitives should be the first implementation.

Confidence: medium-high because kernel details are measurement-sensitive.

## 5.14 Forward neighbors

This subsystem currently mixes low-level math, index construction, storage ownership, format duplication, routing policy, and query workflow. That conflicts with Cellerator’s low-level biomath boundary.

The correct migration is a split, not an in-place container beautification.

### Keep in Cellerator

- exact-search and refinement kernels;
- native sparse scoring kernels;
- prepared route/scoring descriptors;
- raw input/output views;
- explicit scratch requirements;
- low-level result comparison and merge primitives.

### Move downstream

- dataset/index ownership;
- cell-ID lookup ownership;
- shard construction;
- eager creation of several physical sparse formats;
- high-level same-embryo policy;
- ingestion and preprocessing;
- durable query-result objects.

### New low-level boundary

A `forward_neighbor_prepared_view` should identify exactly one selected physical projection and its metadata. The caller owns its storage. Cellerator operates on it without shared ownership.

Results use a fixed-width top-K table and store only core fields. Derived distance should not coexist with both similarity and squared distance unless a measured consumer requires all three. Optional metadata is a separate projection.

The current two-pass same-embryo strategy should become one search/comparator policy, not two complete searches merged on the host. Rank and merge should remain on-device, with one final pinned transfer when host output is required.

Confidence: high for the split, medium for the final physical query representation pending benchmark and downstream interface review.

## 5.15 Runtime memory

### Delete `host_buffer`

`host_buffer` is a vector clone with geometric growth. Adopting it as the anti-vector solution would preserve the problem and create a private STL. Remove it from production code.

### Delete shared-owned `device_buffer`

The runtime `device_buffer` combines `shared_ptr`, `cudaMalloc`, blocking upload/download helpers, and hidden lifetime. Replace it with:

- session-owned `allocation_handle {slot,generation}`;
- raw `device_view<T>` for launches;
- explicit `upload_async` and `download_async` functions taking a stream;
- explicit event/stream completion requirements;
- no implicit synchronization in destructors or copy helpers.

### Scratch

Replace grow-by-free/reallocate scratch with:

- one prepared slab per device and stream;
- deterministic suballocation by offsets;
- session sealing after preparation;
- an optional stream-ordered pool for non-sealed cold preparation paths.

### Host placement

For large, long-lived host images, provide explicit allocator variants for:

- aligned ordinary pages;
- `mmap` plus optional `MADV_HUGEPAGE`;
- NUMA-bound pages;
- page-locked transfer staging;
- write-combined one-way upload staging.

These are explicit choices, not a hidden universal heuristic. GPU-adjacent host staging should be benchmarked per topology.

## 6. Disposition of standard-library families

| Family | Core policy |
|---|---|
| `std::vector` | Prohibited as production-core ownership. Replace according to domain. Allowed in tests, references, and explicitly cold adapters. |
| `std::map`, `std::set`, `std::unordered_map`, `std::unordered_set` | Prohibited in hot/planning core unless a benchmark proves a named exact flat table or sort-based structure loses. Current uses have denser, more specific alternatives. |
| `std::priority_queue` | Replace sampling uses with explicit bounded heaps or radix selection over caller workspace. |
| `std::shared_ptr` | Prohibited in core runtime and prepared objects. Use session ownership, handles, leases, and raw hot views. Framework adapters may retain it. |
| `std::unique_ptr<T[]>` | Do not use to own parallel image sections. One cold pimpl or one external object may remain if it has no execution consequence. |
| `std::string`, `std::string_view` | Keep in cold CLI/diagnostic adapters. Remove from hot images and provenance contracts; use numeric IDs, hashes, and `{char*,length}` views where text input is required. |
| `std::pair`, `std::tuple` | Replace hot or persisted protocols with named POD records. They may remain as local cold return conveniences if codegen is identical. |
| `std::array` | Not a performance problem by itself, but plain fixed arrays are preferred in C-compatible PODs and CUDA locals. Permit only where it improves clarity without expanding context. |
| `std::optional`, `std::variant`, `std::function`, `std::any` | Keep absent from the core. Use explicit tagged PODs and function pointers. |
| STL algorithms | Not categorically banned. `std::sort` is acceptable for cold small data when it wins. Dense integer domains should prefer counting, radix, direct indexing, scan, or sorting networks as appropriate. |

## 7. Migration order and dependency graph

## Wave 0: freeze the direction and establish baselines

1. Add `docs/data_structure_policy.md` containing the rules above.
2. Add an allowlist-based CI/source check that rejects new owning STL containers under production `include/Cellerator` and `src` paths.
3. Record current correctness outputs, allocation counts, construction time, transfer bytes, synchronization count, kernel launches, peak host memory, and device memory for each target subsystem.
4. Keep tests/bench/reference code allowlisted.

This wave changes no representation.

## Wave 1: memory and image substrate

1. Add the thin memory/domain/workspace/image/compiler headers.
2. Add explicit allocation counters and stream-aware copy helpers.
3. Convert `frozen_packing_plan` to one image or merge it into CPK1.
4. Add status-only allocation failure paths.
5. Do not create a generic owning sequence.

## Wave 2A: immutable geometry images

Parallel-safe after Wave 1:

- static plan;
- direct pack compilation;
- gating/route image;
- layout metrics and selection tables;
- candidate relation image;
- physical projection scratch cleanup.

## Wave 2B: sample and validation relations

Parallel-safe after Wave 1:

- sample selection and provenance image;
- sampled CSR image;
- validation unit relation;
- edge membership and null-model workspaces;
- record-validation scratch.

## Wave 2C: device-resident sparse pipeline

Parallel-safe after Wave 1:

- gene support image;
- candidate pair device image;
- merge score device image;
- prepared CUB workspace;
- stream/event execution;
- host materialization adapter.

## Wave 3: optimizer rewrite

Depends on the packing-plan and support/candidate image contracts. Implement:

- block slab;
- direct feature map;
- union cache modes;
- mutation journal;
- proposal tables and marks;
- local validation;
- exact plan freeze.

## Wave 4: graph structures and exact search

- trajectory image;
- fixed-width edge table and fused pruning;
- tree/supernode/DAG images;
- query spans;
- K-specialized exact-search kernels.

## Wave 5: forward-neighbor split and runtime cleanup

- extract high-level storage and index building downstream;
- retain prepared low-level kernels/views;
- replace runtime device-buffer ownership;
- delete `host_buffer` and graph-local buffer clones;
- finalize sealed-session no-allocation enforcement.

## Wave 6: secondary surfaces

- convert public examples to the new contracts;
- keep Torch STL at the adapter edge only;
- move or retire legacy sparse and cuVS workflow code;
- remove obsolete compatibility types once all consumers migrate.

## 8. Required performance and correctness gates

No migration is accepted because it looks lower-level. It must be measured.

### 8.1 Universal gates

- identical semantic output or an explicitly versioned equivalent;
- deterministic output where the current contract is deterministic;
- no allocation during prepared/sealed execution;
- no hidden synchronization;
- no increase in total end-to-end transfer bytes;
- no increase in total pipeline latency unless a later stage produces a larger net win;
- source and header structure remains recoverable by CPP Context Compiler;
- Clang, GCC, and NVCC builds remain supported.

### 8.2 CPU measurements

Record:

- wall time and cycles;
- instructions;
- branches and branch misses;
- cache misses;
- allocation count and allocated bytes;
- page faults;
- peak RSS;
- NUMA locality for large workspaces;
- construction versus reuse cost separately.

### 8.3 GPU measurements

Record:

- total kernel time and end-to-end time;
- kernel launch count;
- `cudaMalloc`/free or pool operation count;
- H2D, D2H, and D2D bytes;
- device and stream synchronization count;
- achieved memory throughput;
- register count, spills/local-memory traffic, occupancy, and warp-stall reasons;
- workspace high-water mark;
- prepared versus first-use cost separately.

### 8.4 Required benchmark cases

1. **Static plan build:** repeated modules, singleton modules, many unique row signatures, high duplicate signature rate, residual-heavy layouts.
2. **Packing compile:** sparse real-like scRNA-seq rows, adversarial region counts, existing CPK1 real-regime fixtures.
3. **Optimizer:** 512, several thousand, and approximately transcriptome-scale features; block widths 8/16/32; sparse and broad support; accepted and rejected batches; dense versus pooled union cache.
4. **Sampling:** million- and 11-million-row populations; K from tiny to the exact maximum; density strata with ties; stable-ID mode.
5. **Gene pipeline:** support build through unique candidates through merge score, measured as one chain rather than isolated kernels only.
6. **Graph:** K = 4/8, multiple latent dimensions, narrow and broad future windows, sparse and branch-heavy trajectories.
7. **Forward neighbors:** prepared index cost, warm query throughput, same-embryo policy, final transfer volume.
8. **Runtime:** sealed repeated launches proving zero allocation and zero global synchronization.

## 9. Compiler and hardware implementation notes

### V100 first

- Align and lay out device arrays according to the actual warp access pattern, not a blanket SoA rule.
- Use K-specialized internal kernels while keeping public headers non-templated.
- Track register pressure and spills whenever records are kept per thread.
- Use `__restrict__` and alignment assumptions only at validated boundaries and re-check occupancy because stronger alias information can increase register caching.
- Prefer explicit preallocated workspaces to repeated allocator calls.
- Keep device intermediates resident and use pinned host memory only at genuine transfer boundaries.

### Later compute capabilities

Keep architecture-specific kernels behind a stable plain-C-style dispatch table keyed by compute capability. The public views should not encode Volta-only instruction details. Add later kernels beside, not through, a template hierarchy that expands every header.

### Assembly

No current container replacement justifies inline PTX or host assembly by itself. Use it only after:

1. a benchmark identifies a material gap;
2. generated LLVM IR/PTX/SASS shows the compiler cannot express the desired operation;
3. a compact reference implementation exists;
4. the assembly is isolated in one clearly named file with architecture guards and equivalence tests.

## 10. What should remain unchanged

- CPK1’s pointer-free image concept;
- FMP1 and CTP1 requirements/build/validate/rebind contracts;
- fixed-capacity session registries;
- raw launch bindings;
- explicit operation descriptors and plain function pointers;
- exact-search public raw views;
- canonical global identities and recoverable gene IDs;
- the separation between semantic geometry and physical projection;
- offline/preparation-time selection rather than runtime autotuning.

## 11. Firm conclusions versus benchmark questions

### Firm conclusions

- Do not replace vectors with `host_buffer`; delete `host_buffer` from production use.
- Public immutable plans should be images, not vector aggregates.
- Nested vectors representing ragged relations should become offsets plus members.
- Dense integer-domain maps/sets should become direct arrays, generation marks, sorted packed keys, or exact flat tables.
- Optimizer rollback must stop copying the entire state.
- Gene support, candidate pairs, and merge scores should remain device-resident across the CUDA pipeline.
- Prepared launches must not allocate or synchronize globally.
- Shared ownership does not belong in the low-level core runtime.
- Forward-neighbor storage/index workflow should be split from low-level Cellerator math.

### Benchmark-dependent choices

- dense full optimizer union slab versus bounded cache pool;
- 32-bit versus 64-bit host support-word projection;
- direct dense region lookup versus compact binary/hash lookup;
- AoS versus SoA for candidate/evidence records;
- heap versus radix-select crossover in sampling;
- exact flat hash versus sort/compact for proposal deduplication;
- shared-memory versus register warp top-K implementation;
- huge-page and NUMA placement policies for specific host workloads;
- direct device construction of FMP1/CTP1 versus host construction and transfer.

No numeric speedup is claimed before these measurements. The source proves avoidable allocation, copying, transfer, synchronization, and asymptotically poor lookup in several paths; it does not by itself establish a percentage improvement.

## 12. Final recommended architecture

Cellerator’s data layer should settle on four concrete forms:

1. **Immutable images:** compiled plans, persistent geometry, projections, sample selections, support bitsets, candidate relations, graph topology.
2. **Typed hot views:** raw restricted pointers, counts, strides, and domain identity derived from validated images or caller storage.
3. **Prepared mutable state:** optimizer block slabs, session registries, fixed-width graph tables, exact flat domain tables.
4. **Explicit workspaces:** one queried allocation per algorithm/session/stream, reset and reused without growth.

Everything else should be either a cold adapter, a test oracle, or moved out of the Cellerator core.

The central prohibition is not “no C++.” It is “no unspecified structure.” Each piece of memory should be shaped like the computation that consumes it.
