# Private native consumers

This file describes implemented candidates, not measured conclusions. The
independent scalar references and frozen bounds live in `reference/`.
The supplied numerical fixtures have no additive baseline term; device
consumers reject `include_baseline=true` instead of silently omitting it.

The byte and direct-plane controls assign one `(cell, group)` to a warp.
Ordered products reuse the existing ordered warp prefix/suffix scans up to
arity32; larger arities retain the division-free fallback. Output incidences
and participant VJPs are distributed across lanes. Linear outputs and local input
VJPs use lanes, with FP32 accumulation and explicit dtype rounding. The
separate weight-gradient kernel reduces over cells before the local dW round.
All overlapping canonical results use FP32 atomic addition. Supplied and
hardened-teacher routes compute no task dq. STE computes dq for every declared
derivative candidate, including forward-dormant candidates.

The materialization control converts condition-major planes into cell-major
64-bit bitmaps, then sorted per-cell group worklists. Its backward worklist is
the union of primal and derivative membership. Conversion and worklist kernels
belong to `prepare`, and their cost must be charged on each membership change.
The numerical launch still reserves one warp for every cell×group capacity
pair. A warp beyond its cell's materialized work count returns early. This
reduces numerical work but is not globally compact dispatch; the cohort grids
dispatch fewer CTAs and loop over more logical work per warp. Any observed
cohort gain needs dispatch-work attribution as well as HBM/cache evidence.

Cohorts are local to a supplied neighborhood and a 32/64/128-cell tile. A
bounded stable insertion sort groups exact `(P,D)` masks; no hash establishes
equality. A warp then owns one equal-program cohort and its lanes own cells.
The O(tile squared) preparation and wasted lanes in small cohorts are explicit
reasons this candidate may lose. Its static feature union is an address map.
Forward loads only features demanded by P; backward loads only P union D.

The retained `cohort_direct` arm owns one cell tile per CTA. The separate
`cohort_direct_two_tile` control owns two successive tiles per CTA, matching
`cohort_shared`'s dispatch and per-tile cohort assignment while reading X and
parameters through ordinary global loads. It uses the same prepared order and
signatures and the same numerical body. A CTA barrier separates its two tile
phases. `stage_count` continues to describe shared staging buffers; it has no
new ownership meaning. Compare the two cached arms to isolate ownership and
dispatch, then compare the two-tile cached arm with shared staging at the same
tile size. The original one-tile cached/shared comparison also changes dispatch.

The `masked_direct_two_tile` arm provides the separate contiguous-cell versus
cohort ablation. It prepares the same local `(P,D)` rows in identity cell order
without sorting. The numerical grid, 128-thread CTA, two tile phases, barrier,
cached global loads and per-cell product/linear helpers match the two-tile
cached cohort arm. Each lane instead owns a contiguous cell and takes its own
group mask; no cohort-start scan or equal-key run traversal occurs. Warp work
distribution is part of this intended scheduling difference. This comparison
must include unsorted-row versus cohort preparation costs; it does not isolate
sorting alone. Both preserve forward P-only reads and full declared STE D work
in backward. `stage_count` allocates no staging buffers in either cached arm.

The shared candidate processes two successive cell tiles per CTA and stages
parameters once for those same two tiles. Both one-buffer and two-buffer
variants have that ownership and reuse boundary. The one-buffer path performs
load, consume, barrier, overwrite, consume. The two-buffer path lets warp zero
load the next demanded tile while the other warps consume the current tile;
warp zero then consumes its assigned cohorts. A CTA barrier precedes consuming
the next buffer. SM70 uses ordinary global loads and shared stores here. There
is no claim of an asynchronous copy instruction, guaranteed overlap, or a
speedup. The candidate is rejected when its feature and parameter stages do
not fit the explicit 48-KiB shared-memory limit.

Condition production is separate from consumer preparation. The direct
root-mapped control evaluates prepared levels within one CTA per cell, using
shared node bytes, and writes coalesced root bytes directly. It creates no row
bitmap or condition plane unless a downstream producer explicitly requests
planes. Cellerator's existing native condition-bitmap evaluator remains an
additional `bitmap_api` control. The plane producer evaluates the same
validated NodeV1 bank level by level.
Both expose the same root identities. Dependency updates compare changed leaf
inputs on the GPU and propagate dirty state through the actual DAG, preserving
unaffected state. The update fixture toggles between two versioned input
states, so repeated updates do not hide a full state reset. Source-identical
shared banks can be deduplicated only after full node and level equality.

The fixed-batch cuBLAS control is a separate private owner with borrowed host
layout metadata. Its activity readback, stream synchronization, zero-filled
packing, GEMMs, output recovery and backward recomputation are all part of its
cost. It never packs dormant source inputs in forward. Backward recomputes
dormant local F only for STE dq. Wholly inactive experts skip parameter reads.

Useful discriminators are complete resident cost, cold preparation, real
condition and parameter updates, memory capacity, and reuse break-even against
the strongest matched control. Passing a fixture or reducing divergence alone
does not establish a useful specialization.

`descriptor_debug.hh` exposes borrowed preparation buffers solely for
out-of-timing acceptance checks. Tests derive cell-major bitmaps, sorted
workset membership, raw group identities, stable cohort permutations and exact
P/D signatures independently from the supplied host fixture. These checks also
cover nonmonotonic raw group IDs, overlapping/repeated mechanisms, tail tiles,
dormant derivatives, poisoned forward inputs, zero arity and an empty batch.
