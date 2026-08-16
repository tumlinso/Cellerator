---
slug: "cellpack-bp04-packing-plan-optimizer"
status: "done"
execution: "closed"
owner: "codex-cp-bp-04"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T14:09:02Z"
last_reviewed_at: "2026-08-16T15:29:50Z"
stale_after_days: 3
objective: "CP-BP-04: Optimize exact candidate gains into constrained reusable gene blocks and a durable PackingPlan."
---

# Current Objective

## Summary

Turn canonical feature relations and sampled structural support into globally reusable constrained feature blocks, validate serious candidate geometries with the exact evaluator, and freeze a semantic two-sided PackingPlan. CP-BP-04 v1 is feature-first; rows remain identity-ordered with explicit fixed-width grouping.

## Quick Start

- Why this stream exists: Cellerator needs a reusable block grammar, not merely an evaluator or one-time feature permutation.
- In scope: deterministic candidate ingestion, variable-width constrained feature-block coarsening, feature moves/swaps, proxy batches with exact-oracle checkpoints, identity/fixed-width row geometry, owning mutable state, and an immutable semantic plan contract.
- Out of scope / dependencies: candidate sketching, general block splitting, row optimization, physical full-dataset packing, silent canonical-ID reassignment, GPU evaluator/search, and durable serialization.
- Required skills: `todo-orchestrator`; `cuda` for cost/evaluator integration and benchmarking.
- Required references: CP-BP-00, CP-BP-02, CP-BP-03, `todos/cellpack-packing-plan-evaluator.md`, and current CellPack planner/evaluator headers.

## Planning Notes

- Existing completed work provides exact evaluation of externally supplied two-sided row/feature geometry and a separate hypothetical cost policy. It does not infer gene blocks and must not be mistaken for this step.
- Width 32 is important because a within-block occupancy mask fits `uint32_t`, but 8/16/32 remain benchmarkable/configurable rather than universal assumptions.
- Full optimizer acceptance depends on CP-BP-02/03. Contract work and refinement mechanics against supplied scored edges can proceed before those streams complete.
- The post-evaluator audit below establishes that the CPU evaluator is sufficient as the initial exact oracle. A CUDA evaluator is deferred until measured oracle volume, rather than assumed production scale, makes it a bottleneck.
- CP-BP-01 is now complete and supplies gene-major sampled-cell support bitsets plus detected-cell counts and provenance. CP-BP-04 must consume that support through a non-owning adapter without taking ownership of sampling or discovery.

## Assumptions

- `PackingPlan` preserves original gene ID -> block/local position, block -> member genes, permutation/inverse permutation as required, and compatibility/version information.
- `packing_plan_view` is the stable semantic evaluator input. `static_plan` remains an adapter source and must not be used as mutable optimizer state without keeping its duplicated module/row-group descriptors synchronized.
- The canonical feature-block membership is the mutable optimizer authority. Execution permutations, inverses, boundaries, and feature-to-block/local maps are derived caches rebuilt and validated at explicit checkpoint/freeze boundaries.
- CP-BP-04 v1 optimizes feature geometry only. It still supplies valid row geometry through implicit identity permutation plus materialized fixed-width row-group boundaries.

## Suggested Skills

- `todo-orchestrator`
- `cuda`

## Useful Reference Files

- `todos/cellpack-packing-plan-evaluator.md`
- `components/CellPack/include/CellPack/planner.hh`
- `components/CellPack/include/CellPack/evaluator.hh`
- `components/CellPack/src/planner.cc`
- `components/CellPack/src/evaluator.cc`
- `include/Cellerator/compute/gene_support_bitset.hh`
- `todos/cellpack-bp01-support-extraction.md`
- `todos/cellpack-bp02-candidate-discovery.md`
- `todos/cellpack-bp03-exact-merge-cost.md`
- `todos/cellpack-bp05-apply-frozen-plan.md`

## Plan

1. Add the shared candidate-relation and immutable semantic plan contracts, then add private mutable-plan state that materializes `packing_plan_view` without using `static_plan`.
2. Implement deterministic feature-first constrained coarsening from canonical candidate relations and sampled gene support.
3. Add bounded feature moves and swaps, proxy batching, exact evaluator checkpoints, deterministic rollback, and final exact verification.
4. Freeze the accepted feature geometry together with identity/fixed-width row geometry and compatibility metadata required by CP-BP-05.
5. Validate mappings, width constraints, determinism, oracle monotonicity, and plan compatibility; benchmark optimizer and oracle time separately.

## Tasks

- [x] Audit the completed reference evaluator for optimizer readiness and record the handoff contract.
- [x] Complete the conceptual Step 5 v1 architecture, candidate contract, mutable/frozen plan contract, oracle cadence, row strategy, test plan, benchmark plan, and deferred CUDA route.
- [x] Implement the owning mutable plan and immutable semantic optimizer output contracts.
- [x] Define and validate the supplied candidate-relation seam for CP-BP-02/03 without duplicating discovery or scoring.
- [x] Connect completed CP-BP-02 canonical pairs through CP-BP-03's exact scored-relation adapter; synthetic and hand-authored exact relations continue to exercise the optimizer independently.
- [x] Add deterministic optimization/refinement and adversarial tests.
- [x] Benchmark configurable widths through the repository mutex.
- [x] Profile the CPU oracle and open `cellpack-packing-plan-cuda-evaluator` as a separate deferred CUB-backed workstream after oracle share exceeded the routing threshold.

## Implementation Progress

- 2026-08-14T13:40:50Z: Claimed CP-BP-04 for implementation as `codex-cp-bp-04`. Beginning Milestone 1 with the authoritative feature-membership mutable state, supplied-candidate contract, and immutable semantic plan contract. Existing CP-BP-01/02/03 work and provisional CellPack files will be preserved.

## Blockers

- No blocker remains for the CP-BP-04 v1 supplied-candidate optimizer contract.
- CP-BP-02 canonical pair discovery and CP-BP-03 exact scoring/adaptation are
  complete. Production candidate-quality tuning remains separate from the
  supplied-candidate optimizer and CP-BP-05 frozen-plan handoff.
- Persistence ABI details remain deliberately deferred to CP-BP-13.

## Progress Notes

- 2026-08-14: Preliminary coordination reconciled the completed `cellpack-packing-plan-evaluator` and temporarily marked this optimizer claimed; the later audit below supersedes that pickup state, not the dependency analysis.
- 2026-08-14: The preliminary claim was released after an optimizer-facing architectural audit. CP-BP-04 is now `planned/ready`; no optimizer, inference, or refinement code was added by the audit.
- 2026-08-14: Audit verdict: Step 5 optimizer implementation may begin with `prepared_csr_support` plus caller-owned evaluation buffers and `packing_plan_view` candidates. CP-BP-05 remains blocked until this stream emits an owning, versioned, read-only plan/mapping contract.
- 2026-08-14: Completed conceptual Step 5 planning after CP-BP-01 closed. Selected feature-first deterministic constrained coarsening plus bounded move/swap refinement, proxy batches judged by CPU exact-oracle checkpoints, identity/fixed-width rows, and a separate immutable semantic plan. No implementation file was changed.

## Next Actions

- CP-BP-05 may consume `frozen_packing_plan`, its lifetime-bound `packing_plan_view`, feature block/local lookups, and `validate_compatibility()`; it remains responsible only for applying the semantic plan.
- CP-BP-03 now combines completed CP-BP-02 canonical pairs with exact evidence
  into `candidate_relation_view` without moving discovery into CellPack.
- The separate CUDA evaluator workstream may be picked up for profiling/acceleration, but it is not a prerequisite to CP-BP-05.

## Done Criteria

- Plan mappings and inverses round-trip every canonical gene exactly once.
- Every block respects configured width and all compatibility/version fields are validated.
- Fixed inputs/configuration yield deterministic plans; every published plan is exactly evaluated and never worsens the configured evaluator objective relative to its exact baseline.
- Width 8/16/32 behavior is testable and benchmarkable without assuming 32 always wins.

## 2026-08-14 Post-evaluator optimizer audit

This audit was performed after the reference-exact PackingPlan evaluator was implemented. It is the handoff from exact Step 4 geometry evaluation into Step 5 inference/refinement; it did not implement search.

### Semantic contract and mutation constraints

- `packing_plan_view` losslessly expresses row and feature permutations plus independent row-group and feature-block boundaries. For both axes, `permutation[execution_position] = canonical_id` and the inverse maps canonical id back to execution position.
- Boundaries are arbitrary strictly increasing partitions of their execution axes. Unequal and uneven final groups/blocks are legal; empty groups/blocks are not. Widths are not fixed by the evaluator.
- Arbitrary valid bijections are accepted. Row and feature geometry validate independently and couple only when source coordinates are mapped to tiles.
- Moving a boundary does not require rebuilding either permutation or inverse. Changing order requires rebuilding the affected axis's inverse map, but not the other axis.
- `make_packing_plan_view(const static_plan&)` is a zero-copy adapter. Its counts come from `row_groups`/`modules`, while its boundaries come from separate vectors. The current builder keeps these synchronized, but mutating only one representation can create an invalid or semantically stale adapter. Optimizer candidates should therefore own coherent permutation/boundary arrays and expose a view; they should not mutate `static_plan` piecemeal.

No correctness defect was found, so evaluator code and tests were not changed.

### Exact cost decomposition

Let `R` be rows, `F` features, `N` structural NNZ, `G` row groups, `B` feature blocks, and `K` occupied tiles.

- One-time source preparation: validate canonical CSR offsets, bounds, and strictly increasing per-row feature ids in `O(R + N)` time and `O(1)` extra owned memory. `prepared_csr_support` is a non-owning view and validation flag; the source arrays remain caller-owned and immutable.
- Reusable source-only state: canonical `row_offsets` and `feature_ids`, `O(R + N)` source storage. No CSC, incidence bitsets, per-feature row lists, or row/block histograms are cached by the evaluator.
- Per complete plan: validate permutations and boundaries in `O(R + F + G + B)`; clear row/group outputs in `O(R + G)`; map rows and NNZ in `O(R log G + N log B)`; comparison-sort `(tile_id, execution_row)` entries in `O(N log N)`; reduce and emit statistics in `O(N + R + G + K)`.
- Per-plan memory: `O(N)` caller-owned scratch, plus `O(R + G + min(N, G*B))` caller-owned output. Plan arrays are `O(R + F + G + B)`.
- Row-only mapping work is inverse-row lookup plus row-group binary search, `O(R log G)`, but a changed row geometry currently feeds a complete `O(N log N)` evaluation.
- Feature-only mapping work is inverse-feature lookup plus feature-block binary search per incidence, `O(N log B)`, but a changed feature geometry currently feeds the same complete sort/reduction.
- Changing only `packing_cost_model` is `O(1)` over an existing occupancy result. Within-group row reorder, within-block feature reorder, and whole-group/block relabeling are occupancy-invariant for the current position-independent metrics and cost model.

The `O(N log N)` term is specifically forced by `std::sort` over one mapped record per NNZ. Canonical CSR order does not group arbitrary row/feature-permuted coordinates by `(tile_id, execution_row)`, and that ordering is then used for exact tile NNZ and distinct participating-row counts.

### Exact local-delta feasibility

- Move one feature: only old/new feature-block tile columns receive NNZ/participation changes, but changing block widths also changes dense-slot, density, and padding terms for every occupied tile in those columns. Exact update needs CSC feature incidences, per-row block counts, per-tile state, and distribution maintenance; cost is roughly `O(deg(feature) + affected column tiles)`.
- Swap two features: same-block swaps are occupancy-invariant. Cross-block swaps touch rows incident to either feature and the two block columns; widths stay fixed. With the same cache, exact cost is `O(deg(f1) + deg(f2) + affected tiles)` and is a plausible later incremental fast path.
- Merge feature blocks: tile NNZ add directly, but participating-row unions require per-row block counts or row membership. Cost is `O(active row/block references in the two blocks + G)`, `O(R)` worst case.
- Split a feature block: all incidences of features assigned to the split block must be repartitioned. Cost is `O(N_block + affected row/block references)` and may approach a partial/full evaluation.
- Move one row: only old/new row-group tile rows receive incidence changes, but changed group heights alter dense-slot, density, and padding terms for all occupied blocks in both groups. CSR plus cached row/block counts and tile state gives `O(nnz(row) + affected group tiles)`.
- Swap two rows: same-group swaps are occupancy-invariant. Cross-group swaps retain group widths and touch the union of both rows' active blocks, `O(nnz(r1) + nnz(r2) + affected tiles)` with cached state.
- Merge row groups: per-block NNZ and participating rows combine because source rows are disjoint; exact update is `O(B)` or the two groups' occupied-tile counts.
- Split a row group: every row/block incidence in the split group must be reassigned, `O(N_group)` without a sparse row/block cache.
- Reorder complete groups or blocks: only labels/execution positions change under the current position-independent evaluator, so the score delta is zero after the permutation remains valid.

An exact incremental cache would add CSC `O(N + F)`, sparse row/block counts up to `O(N)`, occupied-tile state `O(K)`, and nontrivial distribution bookkeeping. That is justified only after profiling shows cross-block swaps/moves dominate and full-oracle calls are too expensive; it is not a prerequisite for the first optimizer.

### Recommended Step 5 architecture and readiness

Use a proxy-plus-oracle hybrid. Let CP-BP-02/03 support evidence and exact candidate gains propose many feature operations; maintain a lightweight owning candidate plan; invoke `evaluate_packing_plan()` on high-quality candidates, accepted batches, and refinement checkpoints to validate/rerank globally. Exploit known zero-delta reorder cases, but do not begin with a full incremental exact cache.

The current host benchmark (20,000 rows, 5,000 features, 640,000 NNZ, 22.1785 ms/evaluation) makes the CPU evaluator adequate to start Step 5 on representative/sampled support. At that measured size, 100 evaluations are about 2.2 seconds and 1,000 about 22 seconds. Conservative extrapolation, including comparison-sort growth, puts 10 million NNZ near 0.4-0.6 seconds/evaluation and 50 million NNZ near 2-3 seconds/evaluation, with 16 bytes of evaluator scratch per NNZ before allocator/container overhead. These are planning estimates, not production measurements.

CUDA acceleration is deferred, not prerequisite. Reconsider a CUB radix-sort/run-length/reduce evaluator when a typical optimizer epoch exceeds roughly 100-300 full evaluations, a full evaluation exceeds about one second, or aggregate oracle work reaches approximately `10^9` mapped NNZ records and is measured to dominate. Any device path must account for source residency and CUB temporary storage; on the native 16 GB V100, naïvely materializing one 16-byte record per NNZ plus sort scratch will constrain large inputs. This operation is regular sparse CUDA, not Tensor Core work.

### Stable Step 5 handoff

- Stable semantic inputs: `csr_support_view`; `prepare_csr_support()`/`prepared_csr_support` with immutable source lifetime; and `packing_plan_view` with explicit two-sided permutations and boundaries.
- Stable evaluator calls: `validate_packing_plan_view()`, `query_packing_evaluation_requirements()`, and `evaluate_packing_plan()`.
- Stable caller responsibility: own and reuse `packing_evaluation_workspace_view` and `packing_occupancy_buffers`; do not allocate inside a candidate loop.
- Stable result semantics: `packing_occupancy_result`, exact tile/group/row occupancy, and conservation totals.
- Stable cost seam: `estimate_packing_cost()` consumes occupancy without redefining/recomputing the plan.
- Primary optimization targets: `packing_cost_estimate::score`/`total_bytes`, `dense_padding`, `occupied_tile_count`, and `row_active_block_references`, selected explicitly by policy.
- Diagnostics/invariants: `total_nnz`, logical/empty tiles, distributions, participating rows, feature-block reuse, and per-row/per-group activity. These can inform search but are not an implicit universal objective.
- Provisional representations: `static_plan` as mutable/durable optimizer output; the numeric defaults and fields of `packing_cost_model`; the exposed host `packing_evaluation_entry` scratch encoding; and any future physical codec/version ABI.

### Readiness gate

- **Step 5 optimizer:** ready to begin. The next implementation task is deterministic proxy-plus-oracle optimization over supplied CP-BP-03-scored candidate edges, initially using the CPU evaluator at checkpoints.
- **CP-BP-05 apply-frozen-plan:** not ready. It remains blocked until CP-BP-04 publishes an owning, versioned plan containing canonical/execution mappings, block membership/local coordinates, boundaries, and compatibility validation.

## 2026-08-14 Conceptual Step 5 implementation plan

This is the authoritative CP-BP-04 v1 plan. It was written after the exact evaluator and its optimizer-facing audit were complete. No source implementation was performed during this planning pass.

### V1 scope and sequencing decision

CP-BP-04 v1 is feature-first. This is not automatic deference to the initial suggestion; it follows the available evidence and scale:

- the feature axis is expected to be roughly 20,000-60,000 while rows may number in the millions;
- CP-BP-01 now exposes gene-major sampled-cell support, detected-cell counts, and provenance directly;
- CP-BP-02/03 naturally emit feature relations and exact feature-support evidence;
- row signatures based on inferred feature-block activity are more meaningful than signatures over singleton features; and
- optimizing both axes simultaneously would multiply the search space before either proxy quality or evaluator cadence is measured.

V1 therefore implements: canonical candidate ingestion; singleton initialization; variable-width feature-block coarsening under a hard maximum width; bounded feature moves and swaps; explicit proxy and oracle accounting; identity row order with fixed-width row groups; and freeze to an immutable semantic plan. It deliberately defers block splitting, feature-order tuning inside a block, row reordering, row-group inference, incremental exact evaluator deltas, CUDA evaluator/search, physical packing, execution kernels, and serialization.

### Candidate relation contract

The shared input is a pointer-plus-count sequence of canonical unordered feature pairs. The planned minimal record contains:

- `feature_a`, `feature_b`: canonical feature IDs;
- signed score numerator and positive denominator: an exact rational evidence score rather than an optimizer-critical floating value;
- `score_kind`: a small enum such as exact merge gain, exact support intersection, exact Jaccard, MinHash similarity, or deterministic opaque structural rank;
- flags identifying exact versus approximate evidence; and
- optional support counts/intersection counts guarded by explicit validity flags.

Only endpoints are absolutely required. Optional scores nominate and diagnose candidates; CP-BP-04 recomputes one globally comparable integer structural proxy from support evidence before mutation. Candidate scores need only be comparable within the same declared score kind and scoring contract. Mixed score kinds are never numerically compared as though their scales match.

Ingestion canonicalizes each pair to `(min_feature, max_feature)`, rejects any out-of-range endpoint as a whole-input error, discards and counts self edges, stable-sorts by canonical pair, and collapses reversed/duplicate pairs. Conflicting exact duplicates under the same score kind are invalid. Otherwise exact evidence takes deterministic precedence over approximate evidence; heterogeneous evidence retains a representative by fixed score-kind precedence for diagnostics only. Input order must not affect the normalized edge list.

Floating candidate scores are not required in v1. If an adapter receives them, it must reject non-finite values and deterministically quantize them under an explicit score contract before entering the optimizer.

### Owning mutable optimizer plan

The authoritative feature state is block membership in canonical IDs. Each active block has a stable logical key equal to its minimum canonical member; members are kept in ascending canonical order. The implementation should use bounded, preallocated block/member storage suitable for the configured maximum width rather than expose a vector-based hot public API.

Derived mutable caches are:

- canonical feature -> active block slot;
- canonical feature -> local coordinate;
- block size and generation;
- block union-support bitset and support count, lazily rebuilt only for touched blocks;
- feature execution permutation, inverse permutation, and feature-block offsets; and
- row permutation/inverse and row-group offsets required by `packing_plan_view`.

Membership is authoritative. Maps and execution geometry are never independently mutable. Mutation methods are the sole writers, mark derived execution geometry dirty, and update or invalidate affected support caches. An explicit `materialize_execution_geometry()` rebuilds permutations, inverses, boundaries, and local coordinates. `view()` is valid only for a clean materialized state; it must not hide an expensive rebuild. `validate()` checks exact feature coverage, uniqueness, width, stable block keys/order, cache agreement, permutation round trips, and boundary validity. Batch rollback may use a complete host snapshot in v1 because `O(F + block_capacity * max_width)` is small relative to exact evaluator scratch and is simpler than a mutation journal.

Planned operations are `merge_blocks`, `move_feature`, `swap_features`, `materialize_execution_geometry`, `view`, `validate`, and `freeze` or semantically equivalent names. No caller receives mutable pointers into authoritative membership.

Rows use a separate authoritative v1 specification: row count, identity-order tag, and explicit configured row-group width. Identity permutation is losslessly represented through the evaluator's null permutation pair; row-group offsets are materialized, including an uneven final group. Row and feature geometry remain independent.

### Initial feature-block construction

Use deterministic constrained greedy coarsening in repeated sweeps:

1. Start with one canonical feature per block, including isolated and empty-support features.
2. Scan the normalized sparse candidate graph, map endpoints to current blocks, and retain a bounded deterministic fanout of promising cross-block relations per block.
3. Deduplicate nominated block pairs and compute the exact structural block proxy from current block union-support bitsets for only that bounded set.
4. Sort legal positive proposals by proxy gain descending, then optional same-contract CP-BP-03 gain descending, combined width ascending, stable block-key pair ascending.
5. Select a deterministic non-overlapping merge batch in that order, capped by the configured oracle batch size.
6. Apply the batch, checkpoint with the exact evaluator, accept or roll back, and repeat until no positive legal merge remains or a configured pass/evaluation budget is reached.

Block widths are variable from one through `maximum_block_width`; widths such as 8, 16, and 32 are configurations, not distinct codecs and not mandatory fill targets. A merge exceeding the maximum width is illegal. Isolated features stay singleton. Low-confidence/approximate edges may nominate work but cannot bypass the recomputed support proxy or exact oracle. Conflicting strong edges are resolved by the global deterministic proposal order; the first selected proposal consumes its two blocks for that batch.

For `E` candidate edges, bounded fanout `k`, sampled support word count `W`, nominated block pairs `Q <= kB`, and `P` coarsening sweeps, expected work is `O(E log k + QW + Q log Q)` per sweep, not `O(F^2)`. Implementations must not compute an all-pairs block graph.

### Proxy objective and source-side cache

For each feature block `b`, let `U_b` be the bitwise union of sampled-cell supports of its member features. Then:

`active_block_references = sum_b |U_b|`.

This has a direct relationship to the evaluator's `row_active_block_references`. The v1 proxy uses exact integer reduction in sampled active-block references as its primary structural quantity; upstream candidate scores only nominate or break ties within a compatible score domain.

- Merge `A,B`: `gain = |U_A| + |U_B| - |U_A union U_B| = |U_A intersection U_B|`.
- Move feature `f` from `A` to `B`: compare `|U_A| + |U_B|` against `|U_(A-f)| + |U_(B+f)|`.
- Swap `f` and `g` across `A,B`: compare the two current union counts against both recomputed union counts.

Optional secondary integer terms may count eliminated blocks or consume a common exact CP-BP-03 gain policy, but physical mask/codec bytes must not be baked into the optimizer. Zero primary support gain is not accepted by default merely to reduce block count.

CP-BP-04 should maintain lazy host block union-support bitsets after singleton initialization. Singleton support aliases CP-BP-01 data; active merged blocks materialize their own union bitsets. Removing a feature requires recomputing the source block union from at most `maximum_block_width` gene bitsets. This is bounded and much simpler than maintaining a full row/block incidence table or incremental exact evaluator cache. Candidate aggregation first limits proposals; exact bitset proxy computation is performed only for nominated merges/moves/swaps, not every edge in a large candidate graph.

### V1 refinement operations

Must-have operations are:

- block merge, both for initial coarsening and later newly profitable block pairs;
- feature move to a candidate-neighbor block when the destination has capacity, including deletion of an emptied source block; and
- cross-block feature swap, especially when both blocks are at maximum width.

Move destinations and swap partners come only from bounded top candidate-neighbor relations; no dense feature/block neighborhood is generated. A cheap aggregate of incident candidate evidence nominates operations. Exact block-support proxy deltas are computed only for the bounded shortlist. Legality requires canonical coverage, valid distinct blocks/features, maximum width, and no empty active block after mutation unless the source is atomically removed. Ties use proxy delta, same-contract exact evidence, stable source/destination block keys, then canonical feature IDs.

Explicit split is deferred. Feature order inside a block remains ascending canonical ID because current occupancy is position-insensitive; optimizing it would be meaningless until a later codec exposes position-sensitive cost. Reorder-only mutations are not implemented.

### Exact-oracle cadence and rollback

1. Evaluate and retain the singleton baseline exactly.
2. Apply a deterministic proxy-ranked batch of at most the configured batch size.
3. Materialize a clean `packing_plan_view`, run `evaluate_packing_plan`, then `estimate_packing_cost`.
4. Accept only an exact improvement under an explicitly configured exact objective. V1 supports integer `packing_cost_estimate.total_bytes` and exact `row_active_block_references`; weighted floating `score` is optional, must be finite, uses documented absolute/relative tolerance, and treats near ties as no improvement. `total_bytes` is legal only when the supplied cost model contains a geometry-sensitive term (dense occupied slots, occupied-tile metadata, or row-active-block metadata); the evaluator's compact zero-metadata defaults would otherwise be constant and must be rejected rather than silently produce a no-op optimizer.
5. On regression or tie, restore the complete pre-batch snapshot, halve batch size, and deterministically regenerate from the restored state. At batch size one, reject and blacklist that mutation for the current plan generation.
6. Evaluate at the end of every coarsening/refinement phase and perform one final exact evaluation before freeze/publication.

V1 follows one deterministic trajectory rather than maintaining a beam of top-K plans. It never publishes a proxy-only improvement. The final exact score must be no worse than the exact baseline unless a future explicit exploration mode says otherwise; no such mode exists in v1.

### Row geometry

Choose identity row permutation plus caller-configured fixed-width row groups for v1 (options A and D). The width is explicit configuration, not a hidden default; the last group may be smaller. This yields a valid two-sided `packing_plan_view` while avoiding premature row search over millions of rows.

The optimizer's evaluation source and frozen row domain must be explicit. A plan optimized/evaluated on sampled CSR is sample-scoped and cannot silently claim compatibility with a different full row universe. A CP-BP-05-ready freeze for arbitrary partitions of one dataset must carry that dataset's full row count/domain identity and identity grouping, or be re-frozen/re-evaluated for that target row domain.

The natural next row substage, after feature blocks exist, derives each row's sorted active-block signature/count vector, stable-sorts rows by that signature with canonical row ID as final tie-break, proposes bounded row groups, and uses the same exact evaluator to accept or reject the row geometry. That is deliberately outside v1.

### Frozen semantic PackingPlan

The frozen result is an immutable Cellerator/CellPack semantic object, not a serialized ABI. Its authoritative geometry contains or losslessly derives:

- semantic schema version and dimensions;
- execution feature -> canonical feature permutation;
- canonical feature -> execution feature inverse;
- feature-block offsets;
- canonical feature -> block and local coordinate derived lookup tables for CP-BP-05;
- execution row -> canonical row permutation, or an explicit identity tag;
- canonical row -> execution row inverse, or lossless identity derivation;
- row-group offsets;
- configured maximum block width and row-group width;
- canonical feature-axis compatibility fingerprint plus fingerprint method/version;
- row-domain/dataset identity, row count, and evaluation-source/sampling provenance sufficient to distinguish sample-scoped from full-dataset plans; and
- exact baseline/final evaluator summaries plus the cost-policy identity used for acceptance.

Execution permutation plus boundaries are authoritative. Inverses and feature block/local lookup tables are freeze-time derived acceleration metadata and must validate exactly against that authority. `freeze()` materializes, validates, copies into immutable ownership, and exposes a lifetime-bound `packing_plan_view`. It defines no endianness, offsets-on-disk, alignment, `.cspack`, or codec layout.

CP-BP-05 is unblocked only when this object can reject mismatched feature order/count/fingerprint and row domain, round-trip every canonical coordinate, expose block/local mapping without inference, and pass view/freeze equivalence tests.

### Determinism rules

- Use canonical endpoint ordering and stable sorting; never let input edge order decide output.
- Use exact integer/rational candidate and proxy quantities where possible. Cross-multiply rational scores with checked wide arithmetic; never compare mixed score kinds numerically.
- Do not use unordered-container iteration order in proposal or block ordering decisions.
- Stable block key after merge is the minimum canonical feature member. Active blocks are materialized by stable key; members are ascending canonical IDs.
- Proposal keys fully order proxy delta, compatible exact evidence, sizes, block keys, operation kind, and feature IDs.
- Priority queues, if used internally later, must carry the same total key and block generation counters; v1 planning does not require one.
- Batches are selected from one total proposal order, are non-overlapping, and are replayed identically after rollback.
- Weighted floating exact scores use fixed tolerance and reject near ties. Integer total bytes are preferred when the supplied cost model is geometry-sensitive; exact row-active-block references are the explicit format-neutral fallback, never an undocumented secondary tie-break.
- Fixed input, support provenance, candidate list, configuration, and cost policy must produce byte-identical semantic arrays and metrics on repeated runs.

### Complexity and memory expectations

Let `F` be features, `E` candidate edges, `S` sampled cells, `W=ceil(S/32)` support words per feature, `B` active blocks, `k` proposal fanout, `Q<=kB` nominated block pairs, `P` passes, `N` evaluator NNZ, and `O` exact oracle calls.

- Candidate ingestion: `O(E log E)` time and `O(E)` memory for canonical sorting/deduplication.
- Singleton initialization: `O(F)` state; support remains non-owning CP-BP-01 storage.
- Coarsening pass: `O(E log k + QW + Q log Q)` plus bounded mutation/cache rebuild work.
- Move/swap pass: `O(E log k)` cheap nomination plus `O(M * maximum_block_width * W)` for the bounded exact-proxy shortlist `M`; `M` is an explicit budget.
- Exact checkpoints: `O * O(N log N)` time, with current evaluator scratch of 16 bytes per NNZ plus sparse output.
- Materialization/freeze: `O(F + B + R + G)` conceptually; implicit identity row maps avoid storing `O(R)` permutation arrays in v1, while row boundaries remain `O(G)`.
- Optimizer state: `O(F + E + B*maximum_block_width + B_active_cache*W)`. It must never allocate dense `O(F^2)` relations.

At `S=65,536`, `W=2,048`; CP-BP-01 gene support is about 164 MB at 20,000 features and 492 MB at 60,000 features. A second full `F*W` block-support slab would duplicate that cost, so merged-block support must be lazy/reusable and singleton blocks should alias source support. Candidate graphs must remain sparse. Millions of source rows affect exact CSR oracle cost, not candidate graph size; feature inference should normally use representative support and explicitly scoped oracle sources.

### Test plan

Before implementation acceptance, add focused coverage for:

- candidate canonicalization independent of input order;
- reversed/duplicate/self edges, conflicting exact duplicates, invalid IDs, invalid denominators, and unsupported score kinds;
- no-edge identity plan and all isolated features;
- one obvious profitable merge with hand-computed support intersection;
- conflicting strong edges and deterministic non-overlapping selection;
- variable block sizes and maximum-width rejection at widths 1, 8, 16, and 32;
- low-confidence/approximate evidence that nominates but cannot override exact proxy/oracle truth;
- merge acceptance and rollback;
- feature move acceptance, source-block deletion, and illegal over-capacity move;
- full-block swap acceptance and deterministic tie resolution;
- a proxy-improving batch that the global evaluator rejects, followed by deterministic batch shrink/rollback;
- rejection of a geometry-invariant exact cost configuration and coverage of both integer total-byte and exact row-active-reference objectives;
- final exact cost no worse than singleton baseline;
- canonical feature and row round trips after every mutation class;
- dirty/materialized mutable-state invariants and corrupted-cache detection;
- mutable `view()` versus frozen `view()` occupancy equivalence;
- feature schema and row-domain compatibility rejection;
- repeated runs producing identical frozen semantic arrays and metrics;
- randomized sparse candidate graphs checking coverage, width, inverse, determinism, rollback, and exact-NNZ conservation properties; and
- empty support, ubiquitous feature, disjoint support, identical support, tail support words, empty rows/features, and uneven final row groups.

### Benchmark plan

Add a mutex-serialized optimizer benchmark separate from `cellPackEvaluatorBench`. Report: features; sampled cells/support words; source/evaluator rows and NNZ; candidate edges before/after deduplication; maximum block width and row-group width; initial/final blocks and width histogram; passes; merge/move/swap proposals considered, shortlisted, proxy-accepted, oracle-accepted, and rejected; full oracle evaluations; initial/final exact score and cost components; candidate ingestion, proxy, oracle, freeze, and total wall time; and peak additional optimizer memory.

Sweep sparse graph degree, 20,000-60,000 features, representative support sizes, widths 8/16/32, and oracle batch sizes. Report optimizer quality/time only. Do not claim packed-layout or execution-kernel speedup from these results.

### Deferred but expected CUDA evaluator milestone

GPU evaluator acceleration is explicitly deferred from CP-BP-04 v1, not removed. The current path is CPU/reference-centric and dominated by mapped-record comparison sorting. The selected CUDA route is native Tesla V100 16 GB (`sm_70`), library-backed regular CUDA through CUB; this workload is not Tensor Core eligible.

The future decomposition is: keep canonical support device-resident; generate execution-row/tile keys on the caller's stream; use CUB radix sort with caller-owned double-buffer and temporary storage; use run-length encoding/scan/reduce-by-key for tile NNZ and participating-row statistics; and copy only requested summaries at an explicit boundary. Key packing must remain overflow-checked and may use key/value pairs rather than force one width. Critical intermediates live in device HBM. PCIe upload per oracle call would erase much of the benefit, so persistent source residency is a prerequisite. Start with separate CUB primitives; fusion is optional only after HBM passes and launch count are measured. PTX/SASS work is premature.

Open that workstream when measurement shows any trigger sustained across representative optimizer runs and the oracle consumes roughly 30% or more of optimizer wall time:

- more than about one second per full oracle evaluation;
- roughly 100-300 full oracle evaluations per optimizer epoch;
- approximately `10^9` aggregate mapped-NNZ records per optimization run; or
- host key generation/sorting demonstrably dominates despite reasonable batching.

Current host scratch is 16 bytes per NNZ. A CUDA radix sort normally needs input/output key storage plus CUB temporary storage, so a naïve 16-byte-record translation can consume several times the source-record footprint. On a 16 GB V100, memory-fit accounting and chunk/partition strategy must precede implementation for large `N`. Any deep tuning build should be `sm_70`-specific and benchmark/profiler runs must use the repository mutex.

### Expected implementation files

The next agent should expect narrowly scoped additions such as:

- `components/CellPack/include/CellPack/candidate_relation.hh`: shared canonical relation/evidence contract for CP-BP-02/03/04;
- `components/CellPack/include/CellPack/packing_plan.hh`: immutable semantic frozen plan and compatibility/view contract;
- `components/CellPack/include/CellPack/optimizer.hh`: pointer-first optimizer inputs, configuration, result, and diagnostics;
- `components/CellPack/src/optimizer_state.hh`: private authoritative mutable membership and derived-cache mechanics;
- `components/CellPack/src/optimizer.cc`, split into coarsening/refinement files only if file-size pressure requires it;
- `components/CellPack/tests/optimizer_test.cc`;
- `components/CellPack/bench/cellpack_optimizer_bench.cc`; and
- `components/CellPack/CMakeLists.txt` for source/test/benchmark wiring.

The CP-BP-01 support header should not be edited merely to make CellPack convenient. Add a zero-copy non-owning adapter at the CellPack boundary if direct dependency wiring would otherwise couple the component to sampling ownership. No CUDA source is expected in CP-BP-04 v1.

### Independently testable implementation milestones

1. **Contracts and invariants:** candidate normalization; immutable semantic plan shape; private mutable singleton state; explicit materialization/view/freeze; identity/fixed-width rows; validation and round-trip tests.
2. **Support proxy:** zero-copy CP-BP-01 support adapter; lazy block union-support cache; exact merge/move/swap proxy formulas; hand-computed and randomized proxy tests.
3. **Deterministic coarsening:** bounded fanout, proposal total order, constrained merge batches, isolated/width/conflict tests; no refinement yet.
4. **Oracle checkpointing:** reusable prepared CSR/workspace/output context, exact baseline, batch acceptance, rollback/shrink, final verification, and proxy-versus-oracle rejection fixture.
5. **Move/swap refinement:** bounded proposals, legality, deterministic acceptance, cache invalidation, rollback, and adversarial tests.
6. **Freeze and CP-BP-05 handoff:** compatibility fingerprint/domain checks, derived block/local maps, frozen/view equivalence, sample/full-row scoping documentation, and CP-BP-05 unblock review.
7. **Benchmark and completion evidence:** mutex optimizer benchmark sweeps, memory accounting, exact test/build commands, ledger updates, and a decision against the recorded GPU activation thresholds.

Milestones 1-3 can use synthetic relations/support without waiting for CP-BP-02/03. Milestone 4 needs the completed evaluator, which already exists. Full integration/acceptance of candidate quality requires CP-BP-02/03. Milestone 6 must remain serial because it defines the CP-BP-05 boundary.

## 2026-08-14 implementation completion and CP-BP-05 handoff

This section records what was implemented after the conceptual Step 5 plan above. It does not replace the planning/audit history.

### Completed implementation

- `components/CellPack/include/CellPack/candidate_relation.hh` and `src/candidate_relation.cc` now own the pointer-first supplied-candidate contract and deterministic normalization. Endpoints are canonical unordered pairs; self edges are counted/discarded; invalid endpoints, zero denominators, unsupported flags/kinds, and conflicting exact duplicates are rejected. Reversed/duplicate records collapse only within one score kind, so heterogeneous score semantics remain separate. The optimizer applies a deterministic exact-before-approximate and declared-kind priority without comparing numeric scores across kinds.
- `components/CellPack/include/CellPack/packing_plan.hh` and `src/packing_plan.cc` now own the immutable semantic `frozen_packing_plan`. Its authoritative feature permutation/boundaries and identity row geometry losslessly derive the inverse permutation, canonical feature-to-block/local maps, fixed-width row groups, dimensions, schema version, configured widths, feature-axis fingerprint/version, row-domain/evaluation/sampling identities, exact baseline/final summaries, objective kind, and cost-policy identity. `validate_compatibility()` rejects feature-schema and sample/full-row-domain mismatches. This is not a serialized ABI.
- `components/CellPack/include/CellPack/optimizer.hh` exposes `sampled_feature_support_view`, the zero-copy `make_sampled_feature_support_view(gene_support_bitset_view, ...)` adapter, caller-owned evaluator workspace requirements/view, deterministic configuration, diagnostics, result, and `optimize_packing_plan()`.
- `components/CellPack/src/optimizer_state.hh` implements the private membership-authoritative mutable plan. Block membership is the only authority. Feature-to-slot and execution permutation/inverse/boundary/block/local arrays are derived; mutation marks execution geometry dirty, and `view()` rejects dirty state. Singleton supports alias CP-BP-01 words; merged/mutated blocks lazily materialize or recompute union bitsets. `merge_blocks`, `move_feature`, `swap_features`, materialization, validation, and freeze-time extraction preserve exact feature coverage and canonical/execution round trips.
- `components/CellPack/src/optimizer.cc` implements deterministic bounded-fanout coarsening, exact sampled-support merge/move/swap proxy deltas, candidate-neighbor-only refinement, non-overlapping batches, exact evaluator/cost checkpoints, full-state snapshot rollback, deterministic batch halving, size-one blacklisting, final exact verification, and identity rows with explicit uneven final row groups. Geometry-invariant cost configurations are rejected. No plan is published from proxy evidence alone.
- `components/CellPack/tests/optimizer_test.cc` covers normalization/input-order invariance, duplicate/reversed/self/invalid/conflicting evidence, empty/disjoint/subset/ubiquitous/tail-word support, maximum widths 1/8/16/32, dirty/materialized state, merge/move/source deletion/swap legality and proxy formulas, no-edge identity, uneven row groups, exact-worse rollback, batch shrink, deterministic blacklisting, exact-objective monotonicity, mutable/frozen evaluator equivalence, canonical round trips, compatibility rejection, repeated-plan determinism, and randomized mutation invariants.
- `components/CellPack/bench/cellpack_optimizer_bench.cc` is mutex-serialized and reports optimizer/source sizes, normalized candidates, block histogram, proposal/acceptance counts, oracle calls/objectives, phase timings, oracle fraction, and estimated additional optimizer memory. It measures optimizer behavior only.
- `components/CellPack/CMakeLists.txt` builds the new sources, `cellPackOptimizerTest`, and `cellPackOptimizerBench`.

### Exact oracle cadence and invariants

- Baseline singleton geometry is evaluated exactly. Each selected proxy batch is materialized and evaluated with `evaluate_packing_plan()` plus `estimate_packing_cost()`. Only strict improvement under the configured exact objective is accepted.
- Regression or tie restores the complete pre-batch state. Multi-operation batches halve and regenerate deterministically; a rejected size-one mutation is blacklisted for that unchanged plan generation. The final frozen result receives a fresh exact verification and cannot be worse than baseline.
- Feature permutations obey `permutation[execution] = canonical`; inverse, feature-to-block, and feature-to-local tables are checked against that authority. Rows are implicit identity with explicit fixed-width boundaries. A sampled-row plan is labeled as such and cannot pass full-dataset row-domain compatibility checks.

### Validation evidence

- Passed `cmake --build Cellerator/build -j 4 --target cellPackOptimizerTest cellPackOptimizerBench` and `./Cellerator/build/cellPackOptimizerTest`.
- Passed all host CellPack tests together: `cellPackFormatTest`, `cellPackPlannerTest`, `cellPackMatrixViewTest`, `cellPackReconstructionTest`, `cellPackLayoutMetricsTest`, `cellPackEvaluatorTest`, `cellPackLayoutSelectorTest`, `cellPackGatingTest`, and `cellPackOptimizerTest`.
- Passed `./Cellerator/build/quantizedMatrixTest` and `./Cellerator/build/exactSearchRuntimeTest`.
- A full `cmake --build Cellerator/build -j 4` reached and built every CellPack target, then failed in unrelated parallel distributed/NCCL header work: `include/Cellerator/dist/nccl_communicator.cuh` could not see `local_context` while compiling CellShard `mask_groups.cu`. That extraction was subsequently checkpointed as `b69a168`; CP-BP-04 does not own or modify those files.

### Benchmark evidence and CUDA routing

- Default development run: 5,000 features, 4,096 sampled rows, 20,000 evaluator rows, 320,000 NNZ, 9,997 candidates, width 8, row-group width 128. Result: 5,000 to 4,712 blocks, 38 oracle calls, objective 320,000 to 301,340, 1.587 s total, 1.232 s oracle, 77.6% oracle fraction, 2.95 MB estimated optimizer state/snapshot/proposal memory, and 5.12 MB evaluator scratch.
- Representative feature-scale run: 20,000 features, 4,096 sampled rows, 20,000 evaluator rows, 320,000 NNZ, 39,997 candidates, width 8, row-group width 128. Result: 20,000 to 19,712 blocks, 38 oracle calls, objective 320,000 to 315,392, 2.771 s total, 1.291 s oracle, 46.6% oracle fraction, 8.66 MB estimated optimizer state/snapshot/proposal memory, and 5.12 MB evaluator scratch.
- Oracle share exceeded the 30% routing trigger in both runs and the current evaluator is known to be dominated by host mapped-record comparison sort, so `cellpack-packing-plan-cuda-evaluator` was opened as a separate deferred workstream. Absolute triggers were not reached: about 32-34 ms per evaluation, 38 evaluations, and about 12.2 million aggregate mapped-NNZ records, far below one second/evaluation, 100-300 evaluations, or `10^9` records. CUDA acceleration is therefore expected but remains non-prerequisite.
- The deferred route remains regular library-backed CUDA on native Tesla V100 16 GB `sm_70`: persistent device source, execution-row/tile key generation, CUB radix sort, run-length/scan/reduction, explicit caller stream/scratch, and memory-fit accounting for input/output records plus CUB temporary storage. Tensor Cores, custom execution kernels, PTX/SASS, and GPU optimizer search are not part of CP-BP-04.

### Known limitations and explicitly unimplemented work

- CP-BP-02 canonical pair discovery and CP-BP-03's exact scored-relation
  adapter are complete. Production candidate-quality acceptance remains a
  validation concern; CP-BP-04 tests/bench continue to use supplied synthetic
  or exact-support fixtures. CP-BP-04 does not implement MinHash, LSH,
  candidate discovery, or a codec-specific merge scorer.
- General split, within-block ordering, row optimization/signatures, incremental exact deltas, beam/global search, GPU evaluator/search, physical packing, masks/bitmaps/BELL/SELL selection, execution kernels, `.cspack`, endian/alignment/file offsets, and CellShard changes remain unimplemented.
- The v1 optimizer is deterministic but intentionally bounded. Candidate fanout/shortlist and pass/oracle budgets can stop before all profitable geometry is explored. The final exact plan remains monotonic relative to baseline.
- `packing_plan_semantic_schema_version` is a semantic in-memory version, not a durable ABI. CP-BP-05 may rely on the current mapping/view/compatibility semantics; serialization ownership remains CP-BP-13/CellShard.

### Handoff decision

CP-BP-04 v1 is complete for its supplied-candidate contract. CP-BP-05 is unblocked because `frozen_packing_plan` owns and validates every canonical/execution mapping it needs without defining a physical representation. Completed CP-BP-02 pairs plus CP-BP-03 exact evidence connect through `candidate_relation_view` without changing the optimizer/evaluator/frozen-plan boundary.

### 2026-08-16 reconciliation

Focused rebuilds and `cellPackEvaluatorTest`, `cellPackOptimizerTest`, and
`cellPackReconstructionTest` passed. The mutex optimizer benchmark at 20,000
rows, 5,000 features, 320,000 NNZ, and 4,096 sampled rows reduced the exact
row-active-block objective from 320,000 to 301,340 with 38 oracle evaluations;
oracle time was 1,202.46 ms of 1,552.79 ms total (77.44%). This confirms the
recorded host-v1 contract and deferred CUDA route. The acceptance-complete
implementation, tests, and benchmark wiring were checkpointed in `597a3eb`
after inspection at `HEAD` `1ebb734`.
