---
slug: "gpu-prototype-neighbors-trajectory"
status: "in_progress"
execution: "idle"
owner: "codex"
created_at: "2026-04-16T09:18:52Z"
last_heartbeat_at: "2026-04-18T13:18:32Z"
last_reviewed_at: "2026-04-16T09:20:06Z"
stale_after_days: 14
objective: "Prototype GPU-resident forward-neighbor query orchestration and trajectory graph follow-on stages for speed screening on V100"
---

# Current Objective

## Summary
Prototype GPU-resident forward-neighbor query orchestration and trajectory graph follow-on stages for speed screening on V100

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: forward-neighbor search and trajectory build already use GPU kernels in places, but too much orchestration, staging, and merge work still returns to CPU before the pipeline is complete.
- In scope: CUDA prototypes for persistent forward-neighbor query workspaces, resident ANN metadata, device-side top-k merge, and optional follow-on trajectory graph stages that keep candidate tables on device longer.
- Out of scope / dependencies: CPU record sorting, text-based provenance, and pure query/result formatting stay on host unless a numeric phase is clearly dominating end-to-end runtime.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: `optimization.md`, `src/compute/neighbors/forward_neighbors/forward_neighbors.cu`, `src/compute/neighbors/cuvs_sharded_knn.cu`, `src/compute/graph/forward_candidates.cuh`, `src/compute/graph/forward_prune.cuh`, `src/compute/graph/supernode_reduce.cuh`, `src/trajectory/trajectory_build.cuh`.

## Planning Notes
- Treat forward-neighbor query orchestration as the first target because it is already close to the right backend but still pays repeated upload, sync, and D2H merge penalties.
- Trajectory follow-on stages are a second pass: only keep them if candidate-table residency materially reduces end-to-end wall time.

## Assumptions
- A useful prototype may stop at device-side merged top-k plus one resident candidate-table handoff rather than a complete end-to-end graph rewrite.
- Topology-sensitive work should respect the real V100 NVLink pairs and avoid introducing cross-pair chatter without measurement.

## Suggested Skills
- `cuda` - Use the host-device-pipeline and sparse-bio routing first; the main issue is setup and residency, not one isolated hot kernel.
- `todo-orchestrator` - Keep the neighbor and trajectory prototypes coupled so the handoff boundary stays explicit.

## Useful Reference Files
- `src/compute/neighbors/forward_neighbors/forward_neighbors.cu` - Current query-block upload, eligible-list rebuild, sync, and host-merge path.
- `src/compute/graph/forward_candidates.cuh` - Existing GPU candidate scorer and the current full-table D2H boundary.
- `src/compute/graph/supernode_reduce.cuh` - Current CPU tree, supernode, and DAG reducers that would be next if candidate tables remain resident.

## Plan
- Prototype persistent forward-neighbor query scratch on device so block-local uploads and best-array initialization stop paying full setup each block.
- Prototype resident ANN metadata selection and device-side top-k merge for forward neighbors.
- Benchmark exact and ANN search after the orchestration rewrite to classify the new limiter.
- If candidate-table residency wins clearly, prototype GPU prune and supernode follow-on stages for trajectory build.

## Tasks
- [ ] Prototype persistent device query buffers and best-candidate scratch for forward_neighbors.
- [ ] Prototype resident ANN eligible-list metadata and eliminate repeated host centroid rebuilds.
- [ ] Prototype device-side final top-k merge or a much smaller D2H result boundary for forward_neighbors.
- [ ] Prototype analogous setup reductions for cuvs_sharded_knn if its host orchestration remains measurable after the first pass.
- [ ] Prototype GPU prune or supernode-reduce follow-ons for trajectory only if candidate-table residency shows a measurable end-to-end gain.
- [ ] Add focused benchmark coverage for query latency tiers and end-to-end trajectory build tiers on V100.

## Blockers
_None recorded yet._

## Progress Notes
- Forward-neighbor public inputs now use pointer/layout views instead of owned dense batch structs. Added dense owned convenience buffers for callers that need temporary storage.
- Forward-neighbor build/search now accepts dense, host blocked-ELL, and host sliced-ELL inputs via one layout-explicit view contract. The current implementation materializes sparse inputs into dense host staging before the existing GPU search path.
- Focused validation passed: `forwardNeighborsCompileTest` now covers dense, blocked-ELL, and sliced-ELL inputs at runtime, and `quantizeModelTest` still passes against the new API.

## Next Actions
- Start with forward_neighbors exact and ANN query orchestration and benchmark before committing to GPU trajectory follow-on stages.
- Next push the same workstream further toward the original GPU-residency objective: remove the remaining host-side device-group merge in forward_neighbors and decide whether ANN eligible-list filtering should move fully on-device before starting the trajectory graph rewrite.

## Done Criteria
- A working GPU-resident forward-neighbor prototype exists with materially fewer host staging and merge steps and benchmarkable V100 latency data.
- Trajectory GPU follow-ons are either prototyped and benchmarked or explicitly rejected with measurement showing the extra port is not worth it.
