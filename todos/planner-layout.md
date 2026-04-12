# Current Objective

## Summary
Replace CSR-byte-only shard planning with execution-aware planning that treats Blocked-ELL as the native sparse layout and compressed as a measured fallback.

## Planning Notes
- Current series planner in src/apps/series_workbench.cc estimates standard CSR bytes and uses build_by_bytes(); this is the wrong objective for repeated SpMM execution.
- Raw embryo data still needs measured planning, but the planner should assume Blocked-ELL-first and fall back to compressed only where a caller or algorithm really requires it.

## Assumptions
- Planner is allowed to reshape shard boundaries for execution efficiency because backward compatibility is not required.

## Suggested Skills
- `cuda-v100` - Sparse layout and pair-local distributed planning on V100 topology.

## Useful Reference Files
- `src/apps/series_workbench.cc` - Current plan_series_ingest implementation and CSR-byte estimation.
- `src/ingest/series/series_partition.cuh` - Current greedy byte/nnz partition helpers.

## Plan
- Extend planner outputs with execution-layout decision fields and per-shard execution metrics.
- Replace byte-only shard planning with Blocked-ELL-first scoring plus explicit compressed fallback selection.
- Keep logical row ownership intact while balancing pair-local resident bytes and projected SpMM time.

## Tasks
- [ ] Inspect ingest plan structs and identify minimum interface changes for execution-layout-aware planning.
- [ ] Implement planner-side metrics collection for candidate block sizes and pair-local memory estimates.
- [ ] Update series workbench runtime tests to validate new planner outputs.

## Blockers
_None recorded yet._

## Progress Notes
- Planner metadata now records execution-format choice, Blocked-ELL geometry, fill ratio, execution bytes, and preferred pair; the remaining gap is that the series ingest payload writer still lays out parts by the older byte-only compressed planner.

## Next Actions
- Inspect planner structs in series_workbench headers and the downstream conversion path before editing.
- Replace the compressed-only payload planning/writer in src/ingest/series/series_ingest.cuh with native blocked-ell-aware layout emission.

## Done Criteria
- Planner emits execution-aware shard decisions consumable by downstream conversion and benchmark paths.
