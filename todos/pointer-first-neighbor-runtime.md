# Current Objective

## Summary
Refactor the forward-neighbor runtime to use reusable pointer-first host/device workspaces, capacity-aware CUDA scratch, and a shared host-buffer primitive across neighbor surfaces.

## Planning Notes
- Keep forward-neighbor result semantics unchanged while removing repeated per-block and per-call allocation churn from the search path.
- Use one shared host buffer primitive across neighbor code rather than preserving separate container families.

## Assumptions
- forward_neighbors is the actively built neighbor surface; cuvs_sharded_knn has no current CMake target and can only be validated statically in this pass.

## Suggested Skills
- `cuda-v100` - Keep workspace reuse explicit and device-local on sm_70 without hiding transfer or launch boundaries.

## Useful Reference Files
- `pointer_migration_plan.md` - Phase 1 makes forward neighbors and KNN the first pointer-first migration budget.
- `optimization.md` - Neighbor search still pays host orchestration and repeated query staging overhead.

## Plan
- Introduce a shared pointer-first host buffer utility for neighbor code and make graph device buffers capacity-aware so scratch reuse avoids repeated cudaMalloc.
- Add a forward-neighbor search workspace with reusable host/device scratch and overloads that allow callers to reuse it across repeated queries.
- Refactor hot loop helpers to fill caller-owned scratch instead of constructing fresh interval, embryo, and candidate tables each block.
- Update the forward-neighbor compile/runtime test to exercise the workspace reuse path explicitly.

## Tasks
- [x] Introduce shared neighbor host-buffer storage and capacity-aware device-buffer reuse.
- [x] Add forward-neighbor workspace overloads and route search through reusable scratch.
- [x] Update tests to validate repeated workspace-backed forward-neighbor searches.

## Blockers
- cuvs_sharded_knn still has no active build target in this repo, so the shared host-buffer change there was not compile-validated in this pass.

## Progress Notes
- Added src/compute/neighbors/host_buffer.hh and switched forward-neighbor plus cuvs_sharded_knn headers to the shared pointer-first host buffer primitive.
- Made src/compute/graph/workspace.cuh capacity-aware so reusable device scratch no longer forces cudaFree/cudaMalloc when logical sizes shrink or repeat.
- Added ForwardNeighborSearchWorkspace, new overloads on ForwardNeighborIndex, and reusable host/device scratch for block embryo sets, intervals, ANN metadata, and downloaded candidate tables.
- Updated forwardNeighborsCompileTest to use a reusable workspace across exact and repeated ANN calls; build and runtime verification passed.
- This workstream is complete for the forward-neighbor workspace refactor and shared host-buffer consolidation; the remaining gap is dedicated cuvs_sharded_knn compile coverage.

## Next Actions
- If neighbor work continues, add a compile target for cuvs_sharded_knn or move it under an exercised test target before deeper refactors.

## Done Criteria
- forward_neighbors supports caller-reused workspaces and no longer allocates fresh host/device scratch for every query block.
- neighbor code uses one shared pointer-first host buffer primitive instead of duplicated container implementations.
- forwardNeighborsCompileTest covers the reusable workspace path and passes.
