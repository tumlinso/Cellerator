---
slug: "cellpack-bp06-11-parallel-execution"
status: "in_progress"
execution: "claimed"
owner: "coordination"
created_at: "2026-08-16T19:45:16Z"
last_heartbeat_at: "2026-08-16T20:14:53Z"
last_reviewed_at: "2026-08-16T20:14:53Z"
stale_after_days: 7
objective: "Coordinate checkpointed single-worktree execution of CP-BP-06 through CP-BP-11 with explicit dependency gates, leases, validation barriers, and fork-ready conditional instructions."
---

# Current Objective

## Summary

Execute CP-BP-06 through CP-BP-11 in one shared worktree without duplicate
contracts, cross-thread overwrites, invalid GPU measurements, or premature
downstream ABI assumptions. This is a coordination ledger, not an implementation
claim.

## Quick Start

- Read `todos.md`, `todo-status.md`, this file, the assigned child ledger, all
  prerequisite child ledgers, `AGENTS.md`, and `components/CellPack/AGENTS.md`.
- Before any claim or shared-file edit, atomically acquire
  `/tmp/cellerator-cp-bp06-11-shared.lock` with `mkdir`. If acquisition fails,
  wait and retry; never remove an unexplained lock.
- Use only the assigned `build-cp-bpNN` directory. Native CUDA work is V100
  `sm_70`; prefer CUB for scan/sort/select and keep caller-owned scratch/stream
  semantics. These mask/rank/irregular sparse operations are not Tensor Core
  workloads.
- Child threads never commit, push, stash, reset, switch branches, amend, or
  update the CellStack submodule pointer. The appointed integrator does that at
  explicit barriers after every participating claim is released.
- Barrier A is integrated. The next forkable assignments are CP-BP-06 Phase B
  and CP-BP-07; later IDs must satisfy their gates below before claiming.

## Planning Notes

- Stable host contracts deliberately open downstream reference work before the
  producing CUDA path is complete; checkpoint barriers then prevent those
  contracts from drifting underneath another thread.
- Parallelism is domain-based, not file-count-based. A downstream thread may
  consume a published view but cannot edit the producing stream to fit its own
  implementation.
- CP-BP-11 is staged and returns to `idle` between dependencies so it never
  holds a multi-week claim while waiting on representation/runtime gates.

## Assumptions

- All implementation forks share this Cellerator worktree and see one another's
  uncommitted files immediately.
- The integrator is explicitly appointed by the coordinating user at each
  barrier; ordinary child assignment does not imply integration authority.
- Matrix semantics are sparse scRNA binary incidence for discovery/validation,
  with rows as cells and columns as canonical genes; expression value bytes are
  preserved but not normalized or transformed by these workstreams.

## Suggested Skills

- `todo-orchestrator` for every claim, gate, release, and barrier.
- `cuda` for CP-BP-06 through CP-BP-10 device work on native V100 `sm_70`.
- `bio-experiments` for CP-BP-10/11 split, leakage, and held-out semantics.

## Useful Reference Files

- `todos/cellpack-data-inferred-block-packing-roadmap.md`
- `todos/cellpack-bp06-cell-block-records.md` through
  `todos/cellpack-bp11-statistical-validation.md`
- `components/CellPack/include/CellPack/packing_plan.hh`
- `components/CellPack/include/CellPack/apply_plan.hh`
- `components/CellPack/AGENTS.md`, `style_hint.md`, and `optimization.md`

## Shared Interlocks

### Claim and lease lock

1. Run `mkdir /tmp/cellerator-cp-bp06-11-shared.lock`.
2. While holding it, reread `git status --short`, `todo-status.md`, this
   coordinator, and every currently claimed CP-BP-06 through CP-BP-11 ledger.
3. Confirm the assigned phase gate is satisfied and the stream is
   `planned/ready` or `in_progress/idle`. If not, make no source or ledger edit.
4. Set only the assigned child to `in_progress/claimed`, record a unique owner,
   heartbeat, exact phase, and every intended file under `File Lease`.
5. New files belong to the first recorded lease. Existing files already leased
   by another stream may not be edited. Transfer ownership only in both child
   ledgers while holding the lock.
6. Synchronize `todos.md` and `todo-status.md`, then release with `rmdir`.

The same lock is mandatory before changing a lease, publishing a handoff gate,
releasing a claim, or editing shared coordination files. It serializes metadata
and shared seams; it is not held during ordinary source work.

### Shared source seams

The following are shared and require the lock plus an explicit non-overlapping
lease before editing: root and CellPack `CMakeLists.txt`, `packing_plan.*`,
`apply_plan.*`, `format.hh`, `pack.*`, common validation headers, `todos.md`,
`todo-status.md`, this coordinator, and the CP-BP-00 parent roadmap. Prefer new
workstream-specific headers/sources/tests/benchmarks. If two streams need the
same source seam, the later stream waits for the earlier lease to release; do
not normalize the collision with simultaneous edits.

### Build and GPU lock

- Build directories are `build-cp-bp06` through `build-cp-bp11`; never share a
  build directory across streams.
- CPU builds/tests may run concurrently in distinct directories. Before any GPU
  runtime test, compute-sanitizer, profiler, or benchmark, acquire
  `/tmp/cellerator-cp-bp06-11-gpu.lock` with `mkdir`; release it with `rmdir`.
  Never remove an unexplained GPU lock.
- Benchmarks additionally use `bench/benchmark_mutex.hh`, record the exact
  command, V100/toolchain, shapes, repeats, and transfer/synchronization scope.
- No PTX/SASS work unless later profiling explicitly requests it.

### Git integration barriers

At a barrier, every participating child must be `in_progress/idle`,
`done/closed`, or still blocked; no implementation stream may remain claimed.
The appointed integrator alone:

1. Acquires the shared lock and rereads all leases/diffs.
2. Runs focused combined validation, `git diff --check`, TODO summary, and
   staleness dry-run.
3. Commits and pushes Cellerator `main`.
4. Updates, commits, and pushes the CellStack Cellerator submodule pointer while
   preserving unrelated root work such as `.vscode/`.
5. Records the checkpoint commit and reopens only the next gate-eligible phase.

If validation fails, the integrator records the failure and leaves the affected
stream `in_progress/idle`; it does not silently repair or rewrite another
stream's implementation.

## Handoff Gates

- [x] `BASE_00_05_READY`: CP-BP-00 through CP-BP-05 compose and are validated.
- [x] `CP06_HOST_ABI_READY`: versioned exact plan-geometry identity, checked
  width-32 record ABI, CPU builder/validator/decoder, and adversarial exact
  reconstruction tests exist.
- [ ] `CP06_DEVICE_READY`: CUDA detect/scan/emit is exactly equivalent, explicit
  about scratch/stream/overflow, sanitized, and benchmarked.
- [ ] `CP07_ORDER_ABI_READY`: bounded local permutation/inverse contract and CPU
  reference/baselines consume CP-BP-06 records without rewriting payloads.
- [ ] `CP07_DEVICE_READY`: CUDA ordering agrees exactly and measured local-union
  metrics justify the selected path.
- [ ] `CP08_HOST_ABI_READY`: versioned tile dictionary/mask/payload/rank view,
  CPU builder/decoder, identity propagation, and adversarial decode tests exist.
- [ ] `CP08_DEVICE_READY`: device view and CUDA tile construction are exact,
  sanitized, and benchmarked.
- [ ] `CP09_REFERENCE_READY`: the first operation is frozen as canonical
  feature-weighted row reduction `y[row] = sum(value * weight[feature])`, with a
  CPU/canonical reference and a direct packed consumer contract.
- [ ] `CP09_RUNTIME_READY`: native V100 consumer executes directly from tiles,
  matches the reference, and has fair CSR/current-layout benchmarks.
- [x] `CP11_FOUNDATIONS_READY`: metric schema, immutable split/bootstrap/null
  provenance, leakage checks, and an exact degree-preserving binary-incidence
  null reference with conservation tests exist.
- [ ] `CP11_HELDOUT_READY`: frozen-plan and available record/tile/runtime metrics
  evaluate unseen identities without relearning; this is CP-BP-10's validation
  prerequisite.
- [ ] `CP10_READY`: CP-BP-07/08 are complete, CP-BP-09 runtime is measurable,
  and `CP11_HELDOUT_READY` is published.

## Integration Barriers

- [x] `BARRIER_A_INTEGRATED`: the combined CP-BP-06 host record contract and
  CP-BP-11 statistical foundations were rebuilt and tested together on V100
  `sm_70`; the checkpoint commit is recorded in Progress Notes.
- [ ] `BARRIER_B_INTEGRATED`: CP-BP-06 CUDA records and CP-BP-07 local ordering
  are closed, jointly validated, committed, and pushed.

## Checkpointed Parallel Phases

1. **Phase A:** CP-BP-06 host ABI/reference and CP-BP-11 foundations run in
   parallel. Each stops at its gate, releases leases, and becomes idle. Barrier A
   integrates and pushes both.
2. **Phase B:** CP-BP-06 CUDA completion and CP-BP-07 host/device ordering run in
   parallel from Barrier A. Barrier B integrates and closes CP-BP-06/07.
3. **Phase C:** CP-BP-08 host ABI/reference runs in parallel with CP-BP-11
   record-level held-out/metric adapters. Barrier C integrates the host tile ABI.
4. **Phase D:** CP-BP-08 CUDA completion runs in parallel with CP-BP-09's fixed
   weighted-row-reduction reference/API. Barrier D closes CP-BP-08 and publishes
   the CP-BP-09 reference gate.
5. **Phase E:** CP-BP-09 native CUDA consumer runs while CP-BP-11 adds tile
   validation/bootstrap work that does not require final runtime measurements.
   Barrier E publishes `CP09_RUNTIME_READY`.
6. **Phase F:** CP-BP-10's bounded storage/held-out controller and CP-BP-11's
   final runtime/stability reporting may run in parallel once `CP10_READY` is
   published. Hardware-aware terms remain deferred to CP-BP-12.

## Conditional Fork Instructions

### If assigned CP-BP-06

- Barrier A is recorded; claim Phase B only. Treat the versioned host ABI,
  exact plan-geometry fingerprint, width-32 mask/rank rules, and CPU reference
  as frozen read-only behavior unless a demonstrated correctness defect is
  first recorded under the shared lock.
- Own the CUDA API/source/test/benchmark additions for CUB-backed
  detect/scan/emit with caller-owned stream and scratch. Preserve exact output
  equivalence for row-to-record offsets, block IDs/masks, record-to-value
  offsets, compact bytes, identities, capacities, and overflow behavior.
- Do not store per-NNZ canonical IDs merely to avoid validating the plan; do not
  implement row ordering, tiles, runtime consumers, persistence, or CP-BP-11.
- Publish `CP06_DEVICE_READY` only after exact CPU agreement, sanitizer
  coverage, and a serialized V100 benchmark. Release every lease, set the
  stream idle, and perform no git operation; the Barrier B integrator closes it.

### If assigned CP-BP-07

- Do not claim until `CP06_HOST_ABI_READY` is checked and Barrier A is recorded.
  If absent, remain read-only and report the gate; do not invent a temporary
  active-block ABI.
- Consume CP-BP-06's row-to-record offsets and sorted block IDs read-only.
  Produce bounded-window permutation and inverse arrays plus identity/config
  metadata; do not physically rewrite CP-BP-06 records or globally reorder the
  dataset.
- Implement deterministic CPU reference and original/random/row-NNZ baselines,
  then library-backed CUDA signature/sort where appropriate. No cell labels or
  gene-plan relearning. Publish host and device gates, release, and close.

### If assigned CP-BP-08

- Do not claim until CP-BP-06 is closed and both CP-BP-07 gates are published.
  If absent, remain read-only; do not freeze a speculative tile ABI.
- Phase C owns the pointer-first tile dictionary, `cell_mask`, per-cell
  `gene_mask`, rank/offset, identity, tail-tile, CPU builder, and exact decoder
  contract. Consume CP-BP-06 records plus CP-BP-07 maps; never materialize zeros.
- Stop at `CP08_HOST_ABI_READY`, release, and wait for Barrier C. Phase D may
  then implement CUB/custom CUDA construction with explicit scratch/stream and
  benchmark evidence. Do not implement runtime dispatch or persistence.

### If assigned CP-BP-09

- Do not claim implementation before `CP08_HOST_ABI_READY` and Barrier C.
  Before `CP08_DEVICE_READY`, Phase D may define only the canonical reference,
  pointer-first consumer contract, capacities, and tests for
  `y[row] = sum(value * weight[canonical_feature])`; it must then publish
  `CP09_REFERENCE_READY`, release, and become idle.
- After Barrier D and `CP08_DEVICE_READY`, reclaim Phase E and implement one
  direct V100 tile consumer. No hidden CSR/BELL reconstruction, per-cell
  launches, universal dispatch framework, or extra operators.
- This irregular single-RHS sparse operation is not Tensor Core eligible.
  Benchmark direct packed execution against the canonical CPU result and
  relevant CSR/current Cellerator GPU path; document tolerances and limiter.

### If assigned CP-BP-10

- Do not claim until `CP10_READY` is checked. If absent, remain read-only and
  report the missing gates; do not substitute CP-BP-04's existing move/swap
  optimizer for the alternating controller.
- Build an offline bounded controller over public CP-BP-04/07/08/09/11 APIs,
  with best-plan checkpoint, deterministic stopping, held-out-only acceptance,
  and rollback. Do not relearn or repack in ordinary minibatches.
- V1 is storage plus measured current-runtime refinement. CP-BP-12 hardware
  prediction terms are optional later scope and must not be fabricated.

### If assigned CP-BP-11

- Phase A is integrated; do not claim CP-BP-11 during Phase B. When the
  coordinator explicitly opens Phase C, assume sparse scRNA binary incidence
  with rows=cells and columns=canonical genes; do not normalize, transform
  magnitudes, densify, use labels to learn packing, or alter CP-BP-01 sources.
- Define metrics with denominators; immutable train/held-out/bootstrap/null
  identities; and an exact row/column-degree-preserving bipartite double-edge-
  swap reference that rejects duplicate edges and records seed, attempts,
  accepted swaps, and conservation checks.
- Splits must use caller-supplied donor/sample/study groups when provided. A
  cell-level split without such metadata must be labeled cell-level structural
  validation and must not claim donor/study generalization.
- Resume only for the phase named by the coordinator: record metrics in C,
  tile/bootstrap metrics in E, and final runtime/stability reporting in F.
  Never edit CP-BP-06/08/09-owned representation or runtime files to make
  validation convenient.

## Tasks

- [x] Define fork-ready claim, lease, build, GPU, and git interlocks.
- [x] Define concrete host/device handoff gates for CP-BP-06 through CP-BP-11.
- [x] Add conditional assignment rules to every CP-BP-06 through CP-BP-11
  child ledger.
- [x] Execute and integrate Phase A.
- [ ] Execute and integrate Phases B through F as their gates open.

## Blockers

- No blocker for Phase B: CP-BP-06 and CP-BP-07 are unclaimed and may be forked
  after reading their conditional instructions and claiming disjoint leases.
- Later phases are intentionally blocked by the unchecked handoff gates above,
  not merely by TODO status labels.

## Progress Notes

- 2026-08-16: Barrier A combined validation passed from fresh
  `build-cp-bp-barrier-a` with CUDA 12.9.86, GNU 13.3.0, and V100 `sm_70`.
  Passed `cellPackCellBlockRecordsTest`, `cellPackStatisticalValidationTest`,
  `cellPackPlannerTest`, `cellPackEvaluatorTest`, `cellPackOptimizerTest`,
  `samplingMaterializationRuntimeTest`, exact CPU/CUDA
  `cellPackMergeCostTest`, and `cellPackInferredPackingPipelineTest`, plus
  `git diff --check`, TODO summary, and staleness dry-run. The integrated source
  checkpoint is commit `25fcb43`.
- 2026-08-16: CP-BP-11 published `CP11_FOUNDATIONS_READY` after isolated CPU
  validation and released its statistical-validation, root-CMake, test, and
  ledger leases without committing or pushing. Both Phase A children are now
  idle; Barrier A is ready for the appointed integrator.
- 2026-08-16: CP-BP-06 published `CP06_HOST_ABI_READY` and released its record,
  plan-geometry, component-CMake, and ledger leases without committing or
  pushing. CP-BP-11 remains claimed on its disjoint Phase A files/root CMake;
  CP-BP-06 Phase B and CP-BP-07 remain blocked until Barrier A.
- 2026-08-16: Both Phase A streams are claimed with disjoint leases. CP-BP-06
  owns record/plan-geometry files and component CMake; CP-BP-11 owns isolated
  statistical-validation files and root CMake.
- 2026-08-16: CP-BP-06 Phase A was claimed by
  `codex-cp-bp06-phase-a` at pushed base `8773f87`; CP-BP-11 Phase A remains
  ready for a disjoint parallel claim.
- 2026-08-16: Created the checkpointed one-worktree protocol after the serial
  CP-BP-00→05 audit. The first legal fork pair is CP-BP-06 host ABI/reference
  plus CP-BP-11 validation foundations.

## Next Actions

- Fork CP-BP-06 Phase B and CP-BP-07 as the next parallel pair. Neither stream
  is claimed yet; both must use the shared/file/GPU interlocks and stop at their
  recorded gates for Barrier B.

## Done Criteria

- Every fork assignment can determine claim eligibility, scope, files, build
  directory, validation, stop condition, and forbidden neighboring scope from
  this coordinator plus its child ledger.
- Shared edits, GPU execution, and git integration are serialized; new files
  and active leases cannot be overwritten by another thread.
- Downstream gates depend on working tested contracts, not merely planned names.
- CP-BP-06 through CP-BP-11 retain their architectural ownership and no phase
  enters CP-BP-12/13 implementation.
