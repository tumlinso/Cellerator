---
slug: "cellpack-packing-plan-cuda-evaluator"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T14:09:02Z"
last_heartbeat_at: "2026-08-14T14:09:02Z"
last_reviewed_at: "2026-08-14T14:09:02Z"
stale_after_days: 7
objective: "Accelerate exact PackingPlan occupancy evaluation with a caller-owned CUB/V100 path while preserving the CPU evaluator as the correctness oracle."
---

# Current Objective

## Summary

Add a future device-resident exact evaluator for repeated optimizer checkpoints. This is acceleration of the existing semantic evaluator, not plan inference, a physical sparse codec, or an execution kernel.

## Quick Start

- Why this stream exists: CP-BP-04 benchmarks measured the CPU exact oracle at 46.6%-77.6% of optimizer wall time across 5k/20k-feature fixtures, crossing the recorded oracle-share/host-sort routing trigger.
- In scope: persistent device support, execution-row/tile key generation, CUB radix sort, run-length/scan/reduction, caller-owned stream/scratch, exact CPU/GPU agreement, 16 GB V100 memory-fit accounting, and mutex benchmarks.
- Out of scope: changing `PackingPlan` semantics, optimizer search, physical format selection, packed execution kernels, `.cspack`, PTX/SASS-first tuning, and Tensor Cores.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: `todos/cellpack-packing-plan-evaluator.md`, `todos/cellpack-bp04-packing-plan-optimizer.md`, `components/CellPack/include/CellPack/evaluator.hh`, Cellerator stream/workspace conventions, and `optimization.md`.

## Plan

1. Profile representative production-scale oracle calls and confirm that host key generation/sort, rather than proxy work or source staging, remains dominant.
2. Define a device-resident prepared-support/context and pointer-first stream/scratch requirements without changing `packing_plan_view` or occupancy-result semantics.
3. Implement checked execution-row/tile key generation, CUB double-buffer radix sort, and run-length/scan/reduction with exact CPU agreement.
4. Establish V100 16 GB fit/chunking limits for per-NNZ input/output records plus CUB temporary storage.
5. Benchmark through the repository mutex and retain CPU evaluation as the reference path.

## Tasks

- [ ] Collect representative `N`, oracle-count, mapped-record, and oracle-share profiles beyond the synthetic CP-BP-04 fixtures.
- [ ] Define explicit device residency, stream, scratch, overflow, and synchronization contracts.
- [ ] Implement the CUB-backed exact evaluator without changing plan or codec semantics.
- [ ] Add exact equivalence, determinism, memory-bound, and failure-path tests.
- [ ] Benchmark on native Tesla V100 16 GB `sm_70` and document activation/dispatch policy.

## Blockers

- This work is not prerequisite to CP-BP-05. Before implementation, confirm a caller can keep canonical support device-resident; per-oracle PCIe upload would erase the intended benefit.

## Progress Notes

- 2026-08-14: Opened after CP-BP-04 optimizer benchmarks. At 20,000 features / 320,000 evaluator NNZ, 38 full evaluations consumed 1.291 s of 2.771 s total (46.6%); each evaluation was about 34 ms and aggregate mapped records were about 12.2 million. The relative-share trigger was met, but absolute one-second/evaluation, 100-300-evaluation, and `10^9`-record triggers were not.

## Next Actions

- Profile a larger, representative optimizer workload and design the device-resident prepared-source/workspace boundary before writing kernels.

## Done Criteria

- GPU occupancy/totals/distributions and cost inputs agree exactly with the CPU reference for identity, adversarial, random, and overflow-boundary plans.
- Caller stream/scratch ownership and synchronization are explicit; steady-state evaluation performs no hidden allocation or per-call source upload.
- V100 16 GB record and CUB temporary-storage limits are measured and documented.
- Mutex benchmark evidence justifies dispatch while the CPU reference remains available.

<!-- todo-orchestrator:v2-managed:start -->
# cellpack-packing-plan-cuda-evaluator: Legacy CUDA packing-plan evaluator

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Record the old current-objective CUDA evaluator as superseded by operation-aware planner/objective v2.

## State
- Lifecycle: `superseded`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `superseded`

## Next Action
_None._

## Ownership
_No structured ownership._

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
