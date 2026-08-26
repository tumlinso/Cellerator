---
slug: "cellpack-packing-plan-evaluator"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-08-14T12:38:27Z"
last_heartbeat_at: "2026-08-14T12:50:37Z"
last_reviewed_at: "2026-08-14T12:50:37Z"
stale_after_days: 3
objective: "Implement reference-exact occupancy and cost evaluation for externally supplied two-sided CellPack PackingPlans."
---

# Current Objective

## Summary

Implement the evaluator in the middle of the data-derived sparse packing compiler pipeline:

`canonical CSR support -> supplied PackingPlan -> exact occupancy -> separate hypothetical cost estimate`.

The evaluated plan is two-sided execution geometry: execution-to-canonical row and feature permutations, their inverses, row-group boundaries, and feature-block boundaries. The evaluator must preserve canonical row/feature identity and must not infer a plan or emit a physical sparse format.

## Quick Start

- Why this stream exists: the previous inspection incorrectly centered a hypothetical `row_block_offsets`/mask/value codec. The required stage is exact geometry evaluation before any codec is selected.
- In scope: canonical CSR support, generic plan geometry, exact occupied-tile statistics, mergeable distributions, a separate reference cost model, static-plan adapter, serious tests, a mutex-serialized evaluator benchmark, and concrete handoff notes.
- Out of scope / dependencies: plan inference, physical packed buffers, `.cspack`, serialization, preprocessing/model changes, warp execution kernels, learned gating, and final codec ABI.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: `AGENTS.md`, `components/CellPack/AGENTS.md`, `style_hint.md`, `scope.md`, `components/CellPack/include/CellPack/{planner,matrix_view,layout_metrics}.hh`, CUDA native-system route, Volta router, sparse-bio layout addendum, CUB/library routing, and Volta benchmark route.

## Planning Notes

- Native system/architecture route: 4x Tesla V100 16 GB, Volta `sm_70`.
- Bottleneck route: sparse structural incidence and repeated plan evaluation. This is not Tensor Core eligible.
- First implementation is CPU/reference-exact and CSR-support-only. It uses caller-owned reusable `O(nnz)` scratch and outputs only occupied tiles, avoiding an `O(row_groups * feature_blocks)` allocation.
- Deterministic ordering is by `(row_group, feature_block, execution_row)` through a stable key definition and deterministic sort; floating values do not participate because support is structural.
- The future CUDA decomposition is a CUB-backed regular-CUDA path: generate device-resident 64-bit tile/row keys, radix sort, run-length/reduce, and retain explicit stream/scratch ownership. No CUDA kernel is implemented in this stage.
- The likely reference limiter is `O(nnz log nnz)` comparison sorting and host memory traffic. The source remains canonical CSR and is not decoded through host COO.

## Assumptions

- CSR feature IDs are canonical dataset feature IDs and strictly increase within each row, matching the current CellPack validator.
- Permutations use the existing CellPack convention: `permutation[execution_position] = canonical_id`; inverse maps canonical ID to execution position.
- Both permutation pointers may be null to represent identity; one without the other is invalid.
- Row-group and feature-block boundaries partition their complete execution axes contiguously, begin at zero, and end at row/feature count.
- Every stored CSR entry is one structural nonzero. Duplicate row/feature coordinates are rejected by existing CSR validation.
- Existing M2-M4 files remain provisional and are preserved; the evaluator does not treat their physical layout labels as realized codecs.

## Suggested Skills

- `cuda`: preserve native Volta, CUB, memory-scaling, determinism, and future stream/workspace constraints.
- `todo-orchestrator`: keep this workstream and root pickup register accurate through implementation and validation.

## Useful Reference Files

- `components/CellPack/include/CellPack/planner.hh`
- `components/CellPack/src/planner.cc`
- `components/CellPack/include/CellPack/matrix_view.hh`
- `components/CellPack/include/CellPack/layout_metrics.hh`
- `components/CellPack/src/layout_metrics.cc`
- `components/CellPack/CMakeLists.txt`
- `components/CellPack/AGENTS.md`
- `bench/benchmark_mutex.hh`

## Plan

1. Add generic two-sided PackingPlan geometry and exact occupancy/cost contracts without a physical format ABI.
2. Add explicit row-group and feature-block boundaries to `static_plan` as derived evaluation geometry and expose a zero-copy adapter.
3. Implement deterministic CSR-support evaluation with sparse occupied-tile output and caller-reusable scratch/output buffers.
4. Implement a separate reference cost model that can score the same occupancy under different metadata/value-storage assumptions.
5. Add identity, nontrivial, equivalent-plan, randomized conservation, pathological sparsity, overflow/validation, and canonical-identity tests.
6. Add a mutex-serialized benchmark reporting source shape, plan widths, evaluation time, scratch/output requirements, occupied tiles, and NNZ per occupied tile.
7. Build/run focused tests and benchmark smoke; update this ledger and CellPack guidance with exact status and limitations.

## Tasks

- [x] Add evaluator contracts and exact implementation.
- [x] Add static-plan boundary arrays and adapter.
- [x] Add focused evaluator tests.
- [x] Add evaluator benchmark and CMake wiring.
- [x] Build and run focused CellPack tests.
- [x] Run benchmark smoke under the repository mutex.
- [x] Append dated architectural correction and handoff details to CellPack documentation.
- [x] Update root ledger/status with exact completed and intentionally deferred work.

## Blockers

_None._

## Progress Notes

- 2026-08-14: Audited the dirty worktree and existing closed M3/M4 ledgers. No existing active stream was available to resume, so this additive workstream was created.
- 2026-08-14: Chose CSR structural support as the deliberate initial input restriction. The evaluator will not require values or COO staging.
- 2026-08-14: Implemented pointer-first caller-owned evaluation buffers and `O(nnz)` reusable key scratch. Only occupied tiles are emitted, so output does not scale with the full logical tile grid.
- 2026-08-14: Added `prepared_csr_support` so canonical CSR validation is a one-time source-side phase rather than repeated for every candidate plan.
- 2026-08-14: Built and passed the complete focused CellPack host suite plus `cellPackGatingCudaTest` on the native V100 checkout.
- 2026-08-14: Mutex benchmark smoke over 20,000 rows, 5,000 features, and 640,000 NNZ reported 0.552438 ms source preparation, 22.1785 ms mean reference evaluation, 10,240,000 temporary bytes, 439,216 output bytes, and 6,224 occupied of 6,280 logical tiles. This is host-reference evidence only.

## Next Actions

- Choose whether the next active stream is data-derived candidate-plan inference/refinement or a CUB-backed device evaluator. Candidate quality should remain the default next stage unless measured evaluator throughput blocks optimizer iteration.

## 2026-08-14 Post-implementation optimizer handoff audit

- The evaluator passed an optimizer-facing semantic and complexity audit without a correctness defect; no evaluator implementation or test was changed.
- `packing_plan_view` is the stable Step 5 semantic input: row and feature permutations use execution-to-canonical direction, inverse maps are exact, and independent strictly increasing boundary arrays permit unequal nonempty groups/blocks without fixed-width assumptions.
- `prepared_csr_support` reuses only validated canonical CSR. Each complete plan still pays `O(R + F + G + B)` validation, `O(R log G + N log B)` mapping, `O(N log N)` comparison sorting, and `O(N + R + G + K)` reduction, using `O(N)` scratch and sparse `O(R + G + min(N, G*B))` output.
- Exact local deltas are possible for feature/row swaps and moves with CSC incidence, sparse row/block counts, occupied-tile state, and distribution bookkeeping. Splits are inherently broader. That cache is deliberately deferred until optimizer profiles justify its multiple-`O(N)` storage and maintenance cost.
- Step 5 should use a proxy-plus-oracle hybrid: propose many operations from support/candidate scoring and use this evaluator to validate/rerank high-quality plans at checkpoints. The CPU oracle is sufficient to start; CUDA is deferred until measured full-evaluation volume dominates.
- At 640,000 NNZ the measured 22.1785 ms reference time implies about 2.2 seconds per 100 complete evaluations. Conservative planning estimates are 0.4-0.6 seconds at 10 million NNZ and 2-3 seconds at 50 million NNZ, subject to source/support distribution and host behavior. Reconsider CUB acceleration around one second per evaluation, 100-300 oracle calls per optimizer epoch, or approximately `10^9` mapped NNZ records of aggregate oracle work.
- CP-BP-04 is ready to begin against `csr_support_view`, `prepared_csr_support`, `packing_plan_view`, caller-owned requirements/workspace/buffers, `packing_occupancy_result`, and `estimate_packing_cost()`. `static_plan` remains a zero-copy adapter source, not yet the durable mutable optimizer output.
- CP-BP-05 remains blocked until CP-BP-04 publishes an owning, versioned frozen plan with canonical/execution mappings and block-local coordinates.

## Done Criteria

- Exact occupancy conserves source NNZ and reports all requested tile/row/group totals and distributions.
- Identical logical geometry evaluates identically whether identity permutations are implicit or explicit.
- Canonical row and feature maps remain explicit/invertible.
- At least identity, nontrivial, randomized, and pathological suites pass.
- A separate cost model produces different estimates from the same occupancy without reevaluation.
- The focused benchmark builds, runs under the mutex, and reports the requested fields without claiming final-kernel performance.
- Documentation records completed, provisional, deferred, and next-stage work precisely.

<!-- todo-orchestrator:v2-managed:start -->
# cellpack-packing-plan-evaluator: Historical CPU packing-plan evaluator

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Preserve completed evaluator/referee evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
_No structured ownership._

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
