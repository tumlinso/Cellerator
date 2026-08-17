---
slug: "cellpack-bp06-11-parallel-execution"
status: "in_progress"
execution: "claimed"
owner: "coordination"
created_at: "2026-08-16T19:45:16Z"
last_heartbeat_at: "2026-08-17T09:46:57Z"
last_reviewed_at: "2026-08-17T09:46:57Z"
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
- Barrier C is integrated at source checkpoint `ebe0509`. CP-BP-08 and CP-BP-09
  are idle at `CP08_DEVICE_READY` and `CP09_REFERENCE_READY`; the appointed
  Barrier D integrator is the sole next actor.

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

### Phase C completed lease map

These released leases record the inputs integrated at Barrier C; they are not
available for a new claim.

- **CP-BP-08 Phase C implementation lease:** new
  `components/CellPack/include/CellPack/warp_tiles.hh`,
  `components/CellPack/src/warp_tiles.cc`, and
  `components/CellPack/tests/warp_tiles_test.cc`; only CP-BP-08-labelled target
  blocks in `components/CellPack/CMakeLists.txt`; and CP-BP-08/coordinator/index
  ledger entries while holding the shared lock.
- **CP-BP-11 Phase C implementation lease:** new
  `components/CellPack/include/CellPack/record_statistical_validation.hh`,
  `components/CellPack/src/record_statistical_validation.cc`, and
  `components/CellPack/tests/record_statistical_validation_test.cc`; only
  CP-BP-11-labelled target blocks in root `CMakeLists.txt`; and
  CP-BP-11/coordinator/index ledger entries while holding the shared lock.
- CP-BP-08 treats `cell_block_records.*`, `local_cell_ordering.*`,
  `packing_plan.*`, `apply_plan.*`, and every statistical-validation file as
  read-only inputs. CP-BP-11 treats those representation/ordering/plan files,
  all `warp_tiles.*`, the evaluator/optimizer, and Phase A
  `statistical_validation.*` as read-only inputs.
- A correctness defect in a frozen input is a stop condition: record evidence
  under the shared lock and return the stream to idle. Do not silently expand a
  child lease or patch the producing contract.

### Phase D exact lease map

CP-BP-08 is claimed by `codex-cp-bp08-phase-d` and CP-BP-09 by
`codex-cp-bp09-phase-d`, both at pushed base
`fe095fb6d6592a0194b0a86f13f0421e23081cd0` under the exact leases below.

- **CP-BP-08 Phase D implementation lease:** new
  `components/CellPack/include/CellPack/warp_tiles_cuda.hh`,
  `components/CellPack/src/warp_tiles_cuda.cu`,
  `components/CellPack/tests/warp_tiles_cuda_test.cu`, and
  `components/CellPack/bench/warp_tiles_bench.cu`; only clearly labelled
  CP-BP-08 Phase D target blocks in `components/CellPack/CMakeLists.txt`; and
  CP-BP-08/coordinator/index ledger entries while holding the shared lock.
- **CP-BP-09 Phase D implementation lease:** new
  `components/CellPack/include/CellPack/feature_weighted_row_reduction.hh`,
  `components/CellPack/src/feature_weighted_row_reduction.cc`, and
  `components/CellPack/tests/feature_weighted_row_reduction_test.cc`; only
  clearly labelled CP-BP-09 Phase D target blocks in root `CMakeLists.txt`; and
  CP-BP-09/coordinator/index ledger entries while holding the shared lock.
- CP-BP-08 consumes `warp_tiles.hh/.cc`, CP-BP-06 records, CP-BP-07 ordering,
  plan/apply-plan, and every CP-BP-09 or validation file read-only. CP-BP-09
  consumes the frozen `warp_tiles.hh` ABI and canonical sparse inputs read-only;
  every CUDA tile file, component CMake file, plan/record/order file, and
  statistical-validation file is read-only.
- Phase D CP-BP-09 defines a CPU/canonical reference plus pointer-first direct
  packed-consumer contract only. V1 interprets payloads as configured
  `cellerator::real::storage_t`, accepts configured `real::compute_t` feature
  weights, and accumulates/emits configured `real::accum_t`; it rejects a tile
  value width that does not match the configured storage type. It must not
  change the arbitrary-byte storage ABI, reconstruct CSR/BELL as a runtime
  contract, add a CUDA consumer, benchmark runtime dispatch, or implement
  another operator.
- A frozen-input defect or need to edit the other stream's lease is a stop
  condition. Record the evidence under the lock, release to idle, and do not
  normalize the collision by expanding scope.

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
- [x] `CP06_DEVICE_READY`: CUDA detect/scan/emit is exactly equivalent, explicit
  about scratch/stream/overflow, sanitized, and benchmarked.
- [x] `CP07_ORDER_ABI_READY`: bounded local permutation/inverse contract and CPU
  reference/baselines consume CP-BP-06 records without rewriting payloads.
- [x] `CP07_DEVICE_READY`: CUDA ordering agrees exactly and measured local-union
  metrics justify the selected path.
- [x] `CP08_HOST_ABI_READY`: versioned tile dictionary/mask/payload/rank view,
  CPU builder/decoder, identity propagation, and adversarial decode tests exist.
- [x] `CP08_DEVICE_READY`: device view and CUDA tile construction are exact,
  sanitized, and benchmarked.
- [x] `CP09_REFERENCE_READY`: the first operation is frozen as canonical
  feature-weighted row reduction `y[row] = sum(value * weight[feature])`, with a
  CPU/canonical reference and a direct packed consumer contract.
- [ ] `CP09_RUNTIME_READY`: native V100 consumer executes directly from tiles,
  matches the reference, and has fair CSR/current-layout benchmarks.
- [x] `CP11_FOUNDATIONS_READY`: metric schema, immutable split/bootstrap/null
  provenance, leakage checks, and an exact degree-preserving binary-incidence
  null reference with conservation tests exist.
- [x] `CP11_HELDOUT_READY`: frozen-plan record metrics evaluate unseen identities
  without relearning; later tile/runtime adapters remain CP-BP-11 acceptance
  work but do not block this CP-BP-10 validation prerequisite.
- [ ] `CP10_READY`: CP-BP-07/08 are complete, CP-BP-09 runtime is measurable,
  and `CP11_HELDOUT_READY` is published.

## Integration Barriers

- [x] `BARRIER_A_INTEGRATED`: the combined CP-BP-06 host record contract and
  CP-BP-11 statistical foundations were rebuilt and tested together on V100
  `sm_70`; the checkpoint commit is recorded in Progress Notes.
- [x] `BARRIER_B_INTEGRATED`: CP-BP-06 CUDA records and CP-BP-07 local ordering
  are closed, jointly validated, committed, and pushed.
- [x] `BARRIER_C_INTEGRATED`: CP-BP-08 host tiles and CP-BP-11 record-level
  held-out adapters are jointly validated, committed, pushed, and recorded at
  one source checkpoint before Phase D opens.
- [ ] `BARRIER_D_INTEGRATED`: CP-BP-08 CUDA construction and CP-BP-09's
  CPU/reference API are jointly validated from one fresh tree, committed,
  pushed, and recorded before Phase E opens.

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
- Claim only the exact Phase C lease above. The ABI must use trivially-copyable,
  device-ready pointer/count views over caller-owned buffers with explicit
  capacities and checked offsets; no owning pointer forest or hidden
  allocation may become the semantic contract. Bind the plan-geometry,
  local-order configuration, source row domain, and canonical row identity.
- Each tile covers at most 32 locally ordered rows. Its sorted unique global
  feature-block dictionary stores one `uint32_t cell_mask` per block, one
  `uint32_t gene_mask` per participating row/block, and only real compact value
  bytes. Offsets/rank rules must decode every byte and canonical feature/row
  identity exactly, including tail tiles, empty rows, and bit 31.
- Focused tests must cover empty partitions/tiles, tail rows, shared/disjoint
  blocks, full and sparse masks, arbitrary value bytes and widths, deterministic
  rebuilds, capacity/offset overflow, and tampered identity/offset/mask rejection.
  Run the CP-BP-06/07 host regressions plus relevant plan/evaluator regressions
  from `build-cp-bp08`.
- Stop at `CP08_HOST_ABI_READY`, release, and wait for Barrier C. Phase D may
  then implement CUB/custom CUDA construction with explicit scratch/stream and
  benchmark evidence. Phase C must not add `warp_tiles_cuda.*`, a CUDA
  benchmark, runtime dispatch, persistence, CP-BP-09, or CP-BP-11 behavior.
- Barrier C is now integrated. If assigned **CP-BP-08 Phase D**, claim only the
  exact Phase D lease above. Freeze `warp_tiles.hh/.cc` as the CPU oracle and
  preserve its identities, capacities, sorted dictionary, mask/rank, tail,
  arbitrary-value-byte, overflow, and deterministic-output behavior exactly.
- Provide an asynchronous caller-stream API with queryable caller-owned device
  scratch and output buffers. Prefer CUB scan/sort/select where it matches the
  irregular construction; keep custom kernels narrow and `sm_70`-native. This
  mask/rank/scan pipeline is not Tensor Core eligible.
- Publish `CP08_DEVICE_READY` only after exact CPU/CUDA byte agreement,
  adversarial capacities/empty/tail/bit-31/value-width cases, CUDA error
  propagation, compute-sanitizer memcheck and racecheck, and one serialized V100
  benchmark reporting shapes, repeats, scratch, transfer/synchronization scope,
  build throughput, bytes/NNZ, metadata/NNZ, and tile-union size. Release every
  lease, return idle, and stop without git for Barrier D. Do not implement
  CP-BP-09 consumers, runtime dispatch, persistence, or CP-BP-11 metrics.

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
- Barrier C is now integrated. If assigned **CP-BP-09 Phase D**, claim only the
  exact Phase D reference/API lease above. Freeze the operation to
  `y[row] = sum(value * weight[canonical_feature])`; define explicit supported
  configured storage/compute/accumulator types, canonical row-domain output
  ordering, capacities, identities, error behavior, deterministic accumulation
  order, and numerical comparison rules.
- Implement allocation-free canonical CSR/record and direct host tile reference
  evaluation sufficient to prove the consumer contract without making host tile
  decode or CSR reconstruction part of the future device hot path. Test empty
  rows/tiles, tail tiles, canonical-feature recovery, non-identity local order,
  multiple supported numeric values, zero NNZ, capacity/identity tampering, and
  deterministic numerical agreement.
- Publish `CP09_REFERENCE_READY`, release every lease, return idle, and stop
  without git for Barrier D. Do not create CUDA consumer/benchmark files, edit
  CP-BP-08 CUDA or host representation files, broaden to SpMM/operators,
  introduce per-cell launches/dispatch, or claim runtime performance.

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
- Phase C is now open. Add frozen-plan and CP-BP-06 record-level held-out/null
  metric adapters only, publish `CP11_HELDOUT_READY`, release, and stop. Do not
  absorb CP-BP-08 tile construction or claim donor/study generalization without
  caller-supplied grouping identities.
- Claim only the exact Phase C lease above and consume the integrated Phase A
  schema/splits/null generator without changing them. Evaluate one frozen plan
  on immutable held-out rows through the CP-BP-06 record view; never relearn,
  mutate, or retune the plan on held-out or null inputs.
- Report versioned denominator-preserving record metrics available today
  (including bytes/NNZ, metadata/NNZ, and blocks/cell), real-versus-null
  comparison, split/null/source/plan identities, and exact reconstruction. Do
  not fabricate tile-union, runtime-throughput, or hardware metrics before
  CP-BP-08/09 exist.
- Tests must cover disjoint and group-aware splits, cell-level-only labeling,
  zero denominators, empty rows/partitions, exact metric arithmetic,
  real-versus-null separation, and rejection of tampered or overlapping
  identities. The binary sparse-incidence matrix is rows=cells and
  columns=canonical genes; do not normalize, log-transform, densify, change
  feature order, or transform expression values.
- `CP11_HELDOUT_READY` in Phase C means the frozen-plan record-level held-out
  contract is ready for CP-BP-10; Phase E/F still add tile/bootstrap/runtime
  stability evidence required to close CP-BP-11. Use `build-cp-bp11`, run the
  statistical-foundation and relevant plan/record regressions, then publish the
  gate, release every lease, and perform no git operation.

## Phase D Fork and Stop Protocol

1. Fork both children from the same current pushed Cellerator `origin/main`
   containing this protocol and Barrier C source checkpoint `ebe0509`. The claim
   must record the full current pushed hash. Assignment text is exactly either
   “You are assigned CP-BP-08 Phase D” or “You are assigned CP-BP-09 Phase D”;
   no addendum is required.
2. Each child first acquires the shared lock, verifies that its stream is still
   unclaimed and the other lease does not overlap, records the claim/base/exact
   lease in its child ledger plus `todos.md` and `todo-status.md`, then releases
   the lock. A failed gate or collision means no source edit.
3. Children work only in `build-cp-bp08` and `build-cp-bp09`, respectively.
   CPU work may overlap. CP-BP-08 must acquire the GPU lock for every GPU test,
   sanitizer, profiler, or benchmark; CP-BP-09 must also acquire it if an
   upstream GPU regression is invoked. Only CP-BP-08 benchmarks in Phase D, and
   every benchmark additionally uses the repository benchmark mutex.
4. At completion, each child reacquires the shared lock, records exact tests and
   evidence, publishes only its named gate, releases all leases, returns to
   `in_progress/idle`, releases the lock, and stops without commit/push/stash,
   branch changes, or CellStack pointer updates.
5. The appointed Barrier D integrator waits for `CP08_DEVICE_READY` and
   `CP09_REFERENCE_READY` with both streams idle, rereads the combined
   diff/leases, and validates from fresh `build-cp-bp-barrier-d`. It runs both
   new focused tests, exact host/CUDA tile agreement, CP-BP-06/07, host tile,
   plan/apply-plan/evaluator/optimizer, inferred-pipeline, and record/statistical
   regressions, plus `git diff --check`, TODO summary, and staleness dry-run.
   Only then may it commit/push Cellerator, record the source hash and
   `BARRIER_D_INTEGRATED`, update/push the CellStack pointer, and stop. It must
   not begin Phase E in that integration turn.

## Tasks

- [x] Define fork-ready claim, lease, build, GPU, and git interlocks.
- [x] Define concrete host/device handoff gates for CP-BP-06 through CP-BP-11.
- [x] Add conditional assignment rules to every CP-BP-06 through CP-BP-11
  child ledger.
- [x] Execute and integrate Phase A.
- [x] Execute and validate Phase B for integration.
- [x] Execute and integrate Phase C.
- [ ] Execute and integrate Phases D through F as their gates open.

## Blockers

- `CP08_DEVICE_READY` and `CP09_REFERENCE_READY` are published with both streams
  idle. Barrier D integration is the sole next action; Phase E remains closed.
- Later phases are intentionally blocked by the unchecked handoff gates above,
  not merely by TODO status labels.

## Progress Notes

- 2026-08-17: CP-BP-08 published `CP08_DEVICE_READY`, released its CUDA tile,
  component-CMake, and coordination leases, and returned idle without git.
  Caller-stream/caller-scratch CUDA construction matches the CPU tile oracle
  exactly across empty/tail/bit-31/multiple-value-width/capacity/identity cases;
  CUDA 12.9 memcheck found zero errors and racecheck zero hazards. Required
  record/order/tile/plan/apply/evaluator/optimizer/pipeline regressions passed.
  The serialized V100 benchmark measured 0.756 ms median for 2,097,152 NNZ
  versus 31.954 ms CPU, with transfers excluded, 8,664,075 scratch bytes,
  2.775 GNNZ/s, and exact byte agreement. Both Phase D gates are now published
  and idle; the appointed Barrier D integrator is the sole next actor.
- 2026-08-17: CP-BP-09 published `CP09_REFERENCE_READY`, released its new host
  `feature_weighted_row_reduction` and root-CMake/ledger leases, and returned
  idle without git. Its versioned configured-precision pointer-first contract,
  canonical CSR reference, compact-record reference, and direct tile traversal
  preserve immutable identities and canonical output row order without
  CSR/BELL reconstruction. Focused and required regressions passed from fresh
  `build-cp-bp09`; CP-BP-08 remains actively claimed and untouched. Barrier D
  now waits only on `CP08_DEVICE_READY`.
- 2026-08-17: `codex-cp-bp08-phase-d` joined the active Phase D pair at pushed
  base `fe095fb6d6592a0194b0a86f13f0421e23081cd0`, owning only new
  `warp_tiles_cuda.hh/.cu/_test.cu`, `warp_tiles_bench.cu`, labelled component-
  CMake blocks, and CP-BP-08 coordination entries. CP-BP-09 retains its
  disjoint host reference/API and root-CMake lease. Both stop without git at
  their named gates for Barrier D.
- 2026-08-17: `codex-cp-bp09-phase-d` claimed CP-BP-09's exact Phase D
  reference/API lease at pushed base
  `fe095fb6d6592a0194b0a86f13f0421e23081cd0`: new
  `feature_weighted_row_reduction.hh/.cc/_test.cc`, labelled root-CMake blocks,
  and CP-BP-09 coordination entries only. CP-BP-08 remains idle/unassigned with
  a disjoint CUDA tile/component-CMake lease and may still be claimed in
  parallel. CP-BP-09 stops at `CP09_REFERENCE_READY` without git.
- 2026-08-17: `BARRIER_C_INTEGRATED` is published at pushed Cellerator source
  checkpoint `ebe0509`. Fresh `build-cp-bp-barrier-c` used CUDA 12.9.86, GNU
  13.3.0, Torch models disabled, and V100 `sm_70`; the new warp-tile and
  record-validation tests, Phase-A statistical validation, CP-BP-06 host/CUDA
  records, CP-BP-07 CUDA ordering, apply-plan, reconstruction,
  planner/evaluator/optimizer, and inferred-pipeline regressions all passed.
  `git diff --check`, TODO summary, and staleness dry-run passed before commit.
- 2026-08-17: Published complete Phase D fork instructions without claiming
  either child. CP-BP-08 owns only new CUDA tile API/source/test/benchmark files
  and labelled component-CMake blocks; CP-BP-09 owns only new
  `feature_weighted_row_reduction` host reference/API/test files and labelled
  root-CMake blocks. Both bind the same pushed coordinator base, serialize GPU
  use, publish distinct gates, release, and stop without git for Barrier D.
- 2026-08-17: `CP08_HOST_ABI_READY` is published. CP-BP-08 Phase C passed its
  focused host and upstream/downstream regressions, released its `warp_tiles`
  and component-CMake/ledger leases, and returned idle without git. Together
  with already-published `CP11_HELDOUT_READY`, this makes Barrier C integration
  the sole next action; Phase D and CP-BP-09 remain closed.
- 2026-08-17: `CP11_HELDOUT_READY` is published. CP-BP-11 Phase C passed its
  focused and required regressions, released its new record-validation and
  root-CMake/ledger leases, and returned idle without git. The gate is explicitly
  record-level today; Phase E/F still own tile/bootstrap/runtime stability.
  CP-BP-08 remains actively claimed on its disjoint host-tile lease, so Barrier C
  must wait for `CP08_HOST_ABI_READY` and must not disturb that work.
- 2026-08-17: `codex-cp-bp11-phase-c` claimed the exact CP-BP-11 Phase C lease
  at pushed base `3925c15`: new `record_statistical_validation` files, labelled
  root-CMake blocks, and CP-BP-11 coordination entries only. Concurrent
  CP-BP-08 retains its disjoint `warp_tiles` and component-CMake lease. Both
  Phase C streams are now claimed and must stop/release at their named gates.
- 2026-08-17: `codex-cp-bp08-phase-c` claimed the exact CP-BP-08 host-only
  lease at pushed base `3925c15`. It owns new `warp_tiles.hh/.cc/_test.cc`,
  labelled component-CMake blocks, and CP-BP-08 coordination entries only.
  CP-BP-11 remains idle/unassigned with its disjoint record-validation and
  root-CMake lease and is still safe to claim in parallel.
- 2026-08-17: Made Phase C fork-ready without claiming implementation. The
  CP-BP-08 host tile ABI/reference and CP-BP-11 record-held-out adapter leases
  are exact, disjoint, and bound to the same current pushed `origin/main`;
  shared-input defect,
  GPU serialization, gate publication, child stop, and Barrier C integration
  rules are explicit. Both streams remain unassigned until the user assigns
  them.

- 2026-08-17: `BARRIER_B_INTEGRATED` is published. Combined CP-BP-06/07 source
  and validation landed on Cellerator `main` as `eeb8c39`; both children are
  `done/closed`. Phase C is now open for unclaimed CP-BP-08 host tiles in
  parallel with idle CP-BP-11 record-level held-out adapters. Neither stream is
  claimed by the integrator.
- 2026-08-17: Barrier B combined validation passed from fresh
  `build-cp-bp-barrier-b` with CUDA 12.9.86, GNU 13.3.0, and V100 `sm_70`.
  Passed exact CPU/CUDA cell-block records and local ordering, CUDA apply-plan
  and merge-cost, planner/evaluator/optimizer, inferred-pipeline,
  statistical-validation, and sampling-materialization tests. Matching CUDA
  12.9 memcheck/racecheck runs reported zero errors/hazards for both new paths;
  serialized benchmarks reproduced 0.395 ms record-build and 0.23344 ms
  local-order CUDA medians with exact agreement. `git diff --check`, TODO
  summary, and staleness checks must pass again immediately before commit.
- 2026-08-17: CP-BP-07 published `CP07_ORDER_ABI_READY` and
  `CP07_DEVICE_READY`, released all leases, and became idle without git
  operations. Its versioned bounded-window host/CUDA order maps agree exactly;
  CUDA 12.9 memcheck reports zero errors and racecheck zero hazards. The
  serialized V100 benchmark at
  65,536 rows measured 0.233472 ms CUDA median versus 22.7307 ms CPU and reduced
  group-union metadata from 4,194,304 original/row-NNZ bytes and 2,701,568
  deterministic-random bytes to 131,072 bytes. Serialized 256- and 4,096-row
  window variants also preserved exact agreement and baseline improvements.
  Both Phase B streams are now
  ready for the appointed Barrier B integrator; CP-BP-08 must not start yet.
- 2026-08-17: CP-BP-06 published `CP06_DEVICE_READY` after exact CPU/CUDA
  reconstruction, memcheck and racecheck with zero findings, downstream
  regressions, and a serialized V100 benchmark. Its CUB scan plus narrow
  regular kernels measured 0.393 ms median for 2,097,152 NNZ versus 13.055 ms
  for the CPU builder with transfers excluded and exact byte agreement. Every
  CP-BP-06 lease is released; the uncommitted implementation remains Barrier B
  input and CP-BP-07 remains actively claimed and untouched.
- 2026-08-17: CP-BP-07 was claimed by `codex-cp-bp07` at pushed base
  `1e25e11`, owning new local-order host/CUDA files and root-CMake target blocks.
  CP-BP-06 retains its new CUDA record files and component-CMake blocks; both
  streams share only coordination ledgers under this lock.
- 2026-08-17: CP-BP-06 Phase B was claimed by `codex-cp-bp06-phase-b` at
  pushed base `1e25e11`. It owns new CUDA record API/source/test/benchmark files
  and CP-BP-06 component-CMake blocks. CP-BP-07 remains ready and must consume
  the integrated host ABI read-only through disjoint files/build wiring.
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

- Appoint one Barrier C integrator to freshly validate both idle Phase C
  implementations, commit/push Cellerator, record the checkpoint, update/push
  the CellStack pointer, and stop. Do not begin CP-BP-08 CUDA Phase D or
  CP-BP-09 in the integration turn.

## Done Criteria

- Every fork assignment can determine claim eligibility, scope, files, build
  directory, validation, stop condition, and forbidden neighboring scope from
  this coordinator plus its child ledger.
- Shared edits, GPU execution, and git integration are serialized; new files
  and active leases cannot be overwritten by another thread.
- Downstream gates depend on working tested contracts, not merely planned names.
- CP-BP-06 through CP-BP-11 retain their architectural ownership and no phase
  enters CP-BP-12/13 implementation.
