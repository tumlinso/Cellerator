# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
- Cellerator owns the accelerated scRNA preprocessing backbone, native workbench implementation, and preprocessing benchmarks.
- Native Cellerator GPU policy remains V100 `sm_70` and Blocked-ELL-first for Cellerator execution. Cellerator preprocessing treats Blocked-ELL and Sliced-ELL as first-class preprocessing layouts, with compressed / CSR as fallback.
- The CP-BP data-inferred packing roadmap is a distinct compact block grammar,
  not ordinary Blocked-ELL/BELL with a smarter permutation. Its codec cost,
  variable-payload offsets, canonical identity maps, and direct runtime path
  must be explicit.
- Cellerator owns packing discovery, plan semantics, transformation, CUDA
  representation, and native consumption. CellShard may later own durable
  `.cspack` serialization/validation/fetch/upload integration without learning
  or defining the plan.
- CP-BP-01 through CP-BP-05 satisfy their recorded acceptance criteria and are
  closed. CP-BP-01/02/04 are checkpointed in commit `597a3eb`; CP-BP-03/05 are
  complete in the current integration. A serial public-API integration test now
  proves the complete sampled-support→plan-application chain. Barrier A now
  integrates CP-BP-06's host record contract and CP-BP-11's statistical
  foundations. CP-BP-06 and CP-BP-07 satisfy their acceptance criteria, are
  closed, and are integrated in pushed Barrier B source checkpoint `eeb8c39`.
  Barrier C integrates CP-BP-08 host tiles and CP-BP-11 record-level held-out
  adapters. Barrier D integrates the CUDA tiles plus CP-BP-09 reference and
  closes CP-BP-08. Barrier E integrates CP-BP-09 at `0334f95`; Barrier F
  integrates and closes CP-BP-10/11 at `2cfa5c8`. CP-BP-12/13 are complete.
- The CP-BP-03/05 protocol below is preserved as completed history; it is no
  longer an active implementation interlock. CP-BP-03 and CP-BP-05 ran
  concurrently in one worktree only under this
  cooperative protocol. Before claiming, changing a lease, or editing a shared
  file, atomically acquire `/tmp/cellerator-cp-bp-shared.lock` with `mkdir`.
  While holding it, reread both child ledgers, record the claim and exact file
  lease in the assigned ledger, and synchronize this index and
  `todo-status.md`; then release it with `rmdir`. If acquisition fails, do not
  edit ledgers or shared files. Never remove an unexplained lock without the
  coordinating user's approval.
- Shared files are `CMakeLists.txt`, `components/CellPack/CMakeLists.txt`,
  `todos.md`, and `todo-status.md`, plus any pre-existing path either child has
  leased. New files belong exclusively to the first recorded lease. A stream
  may not touch the other stream's lease; transfer it through both ledgers
  while holding the lock. Use `build-cp-bp03` and `build-cp-bp05`, respectively,
  and serialize all benchmarks with the repository benchmark mutex.
- Neither implementation thread may commit, push, stash, reset, switch branch,
  or rewrite another thread's changes while either child is claimed. The final
  integrator waits until both ledgers are idle or done, acquires the lock,
  rereads leases and diffs, validates the combined tree, and alone performs
  integration git operations.
- Active CP-BP-06 through CP-BP-11 concurrency is governed by
  `todos/cellpack-bp06-11-parallel-execution.md`. Claims, leases, handoff gates,
  and shared edits use `/tmp/cellerator-cp-bp06-11-shared.lock`; all GPU tests,
  sanitizers, profilers, and benchmarks additionally use
  `/tmp/cellerator-cp-bp06-11-gpu.lock`. Child threads use distinct
  `build-cp-bp06` through `build-cp-bp11` directories and never perform git
  state changes. One integrator validates, commits, pushes, and updates the
  CellStack submodule pointer at every recorded barrier.
- `BARRIER_A_INTEGRATED`, `BARRIER_B_INTEGRATED`, `BARRIER_C_INTEGRATED`,
  `BARRIER_D_INTEGRATED`, `CP06_DEVICE_READY`,
  `CP07_ORDER_ABI_READY`, and `CP07_DEVICE_READY` are published. CP-BP-06/07
  are closed; `CP08_HOST_ABI_READY` and `CP11_HELDOUT_READY` are integrated in
  source checkpoint `ebe0509`. Barrier D pushed CP-BP-08 CUDA construction and
  CP-BP-09's reference API at `0bf9acf`; CP-BP-08 is closed.
  `BARRIER_E_INTEGRATED`, `CP10_READY`, and `BARRIER_F_INTEGRATED` are
  published. CP-BP-09/10/11/12/13 are closed.

## Workflow Routing
- Use the `coding-workflow` MCP interface for substantial implementation.
- Use direct todo-orchestrator and CUDA interfaces only for fallback/debugging,
  recovery, or work on those skills themselves.
- Preserve scRNA raw-count, QC, normalization, and double-processing semantics
  when biological experiment guidance applies.

## Useful Reference Files
- `AGENTS.md`: repository structure, testing, Blocked-ELL, and Volta policy.
- `optimization.md`: current V100 bottleneck and sparse preprocessing guidance.
- `docs/pipeline/preprocess/README.md`: existing preprocessing documentation to update if behavior moves.
- `docs/pipeline/README.md`: pipeline-level documentation surface to keep in sync.
- `todos/cellpack-data-inferred-block-packing-roadmap.md`: CP-BP-00 parent
  architecture, dependency graph, invariants, and child IDs.
- `todos/cellpack-packing-plan-evaluator.md`: completed exact supplied-plan
  evaluator to reuse; it is not plan inference or a physical codec.
- `components/CellPack/AGENTS.md`: CellPack compiler/runtime boundaries and
  static-packing rules.

## Workstreams
- `cellpack-bp00-05-integration-audit` | status: done | owner: codex-cp-bp00-05-integration | file: `todos/cellpack-bp00-05-integration-audit.md` | objective: serially proved the completed CP-BP-01→05 public contracts compose end to end without entering CP-BP-06.
- `cellpack-bp06-11-parallel-execution` | status: done | owner: codex-cp-bp10-11-serial | file: `todos/cellpack-bp06-11-parallel-execution.md` | objective: Barrier F is integrated and the CP-BP-06 through CP-BP-11 execution wave is closed.
- `cellpack-data-inferred-block-packing-roadmap` | status: done | owner: coordination | file: `todos/cellpack-data-inferred-block-packing-roadmap.md` | objective: completed CP-BP-00 parent roadmap for the offline compiler, compact tile format, native runtime, validation, autotuning, and persistence.
- `cellpack-bp01-support-extraction` | status: done | owner: parallel-agent-step-1 | file: `todos/cellpack-bp01-support-extraction.md` | objective: CP-BP-01 representative sampling, binary support, per-gene bitsets, counts, provenance, and exact reconstruction.
- `cellpack-bp02-candidate-discovery` | status: done | owner: codex-cp-bp-02 | file: `todos/cellpack-bp02-candidate-discovery.md` | objective: CP-BP-02 deterministic sketch/LSH candidate generation and deduplication; approximate similarity proposes only.
- `cellpack-bp03-exact-merge-cost` | status: done | owner: codex-cp-bp-03-fork | file: `todos/cellpack-bp03-exact-merge-cost.md` | objective: CP-BP-03 exact bitset overlap and replaceable codec-cost/merge-gain scoring, reconciled with the completed evaluator.
- `cellpack-bp04-packing-plan-optimizer` | status: done | owner: codex-cp-bp-04 | file: `todos/cellpack-bp04-packing-plan-optimizer.md` | objective: CP-BP-04 supplied-candidate deterministic optimizer, exact-oracle rollback, and immutable semantic PackingPlan are complete.
- `cellpack-bp05-apply-frozen-plan` | status: done | owner: codex-cp-bp-05-fork | file: `todos/cellpack-bp05-apply-frozen-plan.md` | objective: CP-BP-05 host/CUDA remap and segmented packed-coordinate ordering are complete and closed.
- `cellpack-packing-plan-cuda-evaluator` | status: planned | owner: unassigned | file: `todos/cellpack-packing-plan-cuda-evaluator.md` | objective: deferred native V100 CUB acceleration of exact PackingPlan evaluation; opened by measured oracle share and not prerequisite to CP-BP-05.
- `cellpack-bp06-cell-block-records` | status: done | owner: codex-cp-bp06-phase-b | file: `todos/cellpack-bp06-cell-block-records.md` | objective: exact compact host/CUDA cell-block records are complete and closed.
- `cellpack-bp07-local-cell-ordering` | status: done | owner: codex-cp-bp07 | file: `todos/cellpack-bp07-local-cell-ordering.md` | objective: bounded local active-block-signature host/CUDA ordering is complete and closed.
- `cellpack-bp08-warp-tiles` | status: done | owner: codex-cp-bp08-phase-d | file: `todos/cellpack-bp08-warp-tiles.md` | objective: compact host/CUDA warp tiles are integrated at Barrier D and closed.
- `cellpack-bp09-native-runtime-consumers` | status: done | owner: codex-cp-bp09-phase-e | file: `todos/cellpack-bp09-native-runtime-consumers.md` | objective: direct packed weighted-row reduction is validated, benchmarked, integrated at Barrier E, and closed.
- `cellpack-bp10-alternating-refinement` | status: done | owner: codex-cp-bp10-11-serial | file: `todos/cellpack-bp10-alternating-refinement.md` | objective: bounded held-out alternating refinement is integrated and closed.
- `cellpack-bp11-statistical-validation` | status: done | owner: codex-cp-bp10-11-serial | file: `todos/cellpack-bp11-statistical-validation.md` | objective: held-out/null/bootstrap plus relearned mapping/runtime stability validation is integrated and closed.
- `cellpack-bp12-hardware-cost-autotune` | status: done | owner: codex-cp-bp12 | file: `todos/cellpack-bp12-hardware-cost-autotune.md` | objective: the replaceable V100 execution-cost policy and configurable storage/runtime selector are validated and closed without entering the later aggressive optimization pass.
- `cellpack-bp13-persistence-integration` | status: done | owner: codex-cp-bp13 | file: `todos/cellpack-bp13-persistence-integration.md` | objective: versioned CellShard CSPACK envelope and Cellerator pointer-free image now load/upload for direct no-repack execution.

## Global Blockers
- No CP-BP child blocker remains; CP-BP-00 through CP-BP-13 are closed.

## Progress Notes
- 2026-08-18: CP-BP-13 and its CP-BP-00 parent closed. CellShard `197d268`
  owns the generic CSPACK envelope, compatibility, publication, fetch, and
  contiguous upload; Cellerator owns the pointer-free plan/order/tile image,
  semantic validation, relocation, and direct CP-BP-09 handoff. The complete
  archive-to-device no-repack test and CUDA 12.9 memcheck passed on V100
  `sm_70`. Aggressive optimization remains a deliberately separate later pass.
- 2026-08-18: `codex-cp-bp13` completed the mandatory audit and claimed a
  serial cross-repository lease. CellShard owns a generic checksummed CSPACK
  envelope and contiguous async staging; Cellerator owns the inner
  plan/order/tile image, semantic validation, and direct execution rebinding.
- 2026-08-17: CP-BP-12 completed its versioned hardware-cost model,
  deterministic lambda/width-constrained selector, adversarial tests, and
  serialized 60-scenario/120-observation V100 campaign. Direct-tile and CSR
  held-out MAPE were 5.15105% and 5.87580% with zero correctness mismatches.
  CP-BP-13 is now `planned/ready`; aggressive optimization remains explicitly
  deferred to a later workflow pass.
- 2026-08-17: `BARRIER_F_INTEGRATED` pushed source checkpoint
  `2cfa5c8d26f0c973dfef4659d72ea5f635201835`. CP-BP-10/11 are complete and
  closed after combined host/CUDA/sanitizer/serialized-benchmark validation.
  CP-BP-12 is now the single primary `planned/ready` continuation and remains
  unclaimed; CP-BP-13 was not started.
- 2026-08-17: `BARRIER_E_INTEGRATED` at pushed Cellerator source checkpoint
  `0334f954b1b9e04366f2e2ce191e098c1d476597`. Fresh CUDA 12.9.86/GNU 13.3.0
  V100 `sm_70` combined validation, CP-BP-09 memcheck/racecheck, and serialized
  benchmark passed. CP-BP-09 is closed and `CP10_READY` is published. Exact
  unclaimed leases, labelled CMake seams, gates, separate builds, GPU/benchmark
  locks, and no-git stop rules now make CP-BP-10/11 Phase F fork-ready without
  addendum.
- 2026-08-17: CP-BP-11 Phase E published `CP11_TILE_BOOTSTRAP_READY`, released
  every lease, and returned idle without git. The allocation-free host adapter
  provides exact frozen-plan held-out/null tile reconstruction and raw physical
  metrics plus multiplicity-bound repeated-row bootstrap summaries with
  deterministic min/mean/max/sample-SD and explicit zero-denominator handling.
  Both Phase E gates are now ready for Barrier E combined integration.
- 2026-08-17: CP-BP-09 Phase E published `CP09_RUNTIME_READY`, released every
  lease, and returned idle without git. The asynchronous one-launch direct tile
  consumer is zero-scratch and allocation/transfer/synchronization-free, writes
  canonical rows, passed focused/adversarial/reference/regression tests and CUDA
  12.9 memcheck/racecheck, and was benchmarked under both locks. Packed V100
  medians were 0.017/0.041/0.117 ms versus existing Cellerator CSR at
  0.075/0.079/0.095 ms across high/medium/low sharing; no low-occupancy packed
  specialization met the 5% retention rule. CP-BP-11 remains actively claimed.
- 2026-08-17: `codex-cp-bp11-phase-e` claimed CP-BP-11's exact host-only tile-
  validation/root-CMake lease at pushed coordinator `b76a861`. CP-BP-09 remains
  concurrently claimed under disjoint CUDA consumer/component-CMake/benchmark
  ownership. Both children stop without git at separate gates for Barrier E.
- 2026-08-17: `codex-cp-bp09-phase-e` claimed the exact CP-BP-09 direct CUDA
  consumer/test/benchmark and labelled component-CMake lease at pushed
  coordinator `b76a861a5c21a908b1ed9368fa1f4961dbf42c3b`. CP-BP-11 remains
  independently idle/unassigned and must not be disturbed.
- 2026-08-17: `BARRIER_D_INTEGRATED` pushed Cellerator source checkpoint
  `0bf9acf` after a fresh CUDA 12.9.86/GNU 13.3.0 V100 `sm_70` build. Both new
  focused tests, exact host/CUDA tiles, CP-BP-06/07, plan/apply/evaluator/
  optimizer, inferred-pipeline, and Phase A/C statistical regressions passed;
  tile memcheck/racecheck reported zero findings. CP-BP-08 is closed.
- 2026-08-17: Published complete, unclaimed Phase E fork instructions. CP-BP-09
  owns new direct weighted-row-reduction CUDA/test/benchmark files plus labelled
  component-CMake blocks. CP-BP-11 owns new host tile-statistical-validation
  files plus labelled root-CMake blocks. Both consume Barrier D source checkpoint
  `0bf9acf` through the current pushed coordinator and record that full hash at
  claim; they use separate builds and shared locks, publish separate gates, and
  stop without git for Barrier E.
- 2026-08-17: CP-BP-08 Phase D published `CP08_DEVICE_READY`, released its
  CUDA tile/component-CMake/ledger leases, and returned idle without git. Its
  asynchronous caller-stream/caller-scratch V100 constructor exactly matches
  the host tiles, passed CUDA 12.9 memcheck/racecheck and required regressions,
  and measured 0.756 ms median for 2,097,152 NNZ versus 31.954 ms CPU with
  transfers excluded. CP-BP-09 remains idle and untouched at
  `CP09_REFERENCE_READY`; Barrier D is the sole next action.
- 2026-08-17: CP-BP-09 Phase D published `CP09_REFERENCE_READY`, released its
  exact host reference/API and root-CMake lease, and returned idle without git.
  The configured-precision pointer-first contract plus canonical CSR, compact-
  record, and direct-tile references agree within the documented versioned
  tolerance and preserve canonical feature/row identities. Focused, tile,
  record, plan/apply/evaluator/optimizer, inferred-pipeline, warning, and diff
  checks passed. CP-BP-08 remains actively claimed and untouched.
- 2026-08-17: `codex-cp-bp08-phase-d` claimed CP-BP-08's exact CUDA tile lease
  at pushed base `fe095fb6d6592a0194b0a86f13f0421e23081cd0`. CP-BP-09 remains
  concurrently claimed under disjoint new host files and root-CMake blocks.
  CP-BP-08 must publish `CP08_DEVICE_READY`, release, and stop without git or
  runtime-consumer work.
- 2026-08-17: `codex-cp-bp09-phase-d` claimed CP-BP-09's exact host-only
  reference/API lease at pushed base `fe095fb6d6592a0194b0a86f13f0421e23081cd0`.
  CP-BP-08 remains independently idle/unassigned with disjoint new CUDA files
  and component-CMake blocks. CP-BP-09 must publish `CP09_REFERENCE_READY`,
  release, and stop without git or Phase E runtime work.
- 2026-08-17: `BARRIER_C_INTEGRATED` records pushed Cellerator source checkpoint
  `ebe0509`. A fresh CUDA 12.9.86/GNU 13.3.0 V100 `sm_70` build passed the new
  warp-tile and record-validation tests, Phase-A statistical validation,
  CP-BP-06 host/CUDA records, CP-BP-07 CUDA ordering, apply-plan,
  reconstruction, planner/evaluator/optimizer, and inferred-pipeline
  regressions. Phase D's two exact leases are published but remain unclaimed.
- 2026-08-17: CP-BP-08 Phase C published `CP08_HOST_ABI_READY`, released all
  leases, and returned idle without git. Its host-only tile contract is
  versioned, pointer-first, device-ready, identity-bound, compact/no-padding,
  allocation-free, exactly validated/decoded, and adversarially tested across
  tail/empty/bit-31/value-width/permutation/tamper cases. Focused, record,
  ordering, reconstruction, planner/evaluator/optimizer, inferred-pipeline, and
  diff checks passed. Both Phase C gates now await Barrier C integration.
- 2026-08-17: CP-BP-11 Phase C published `CP11_HELDOUT_READY`, released all
  leases, and returned idle without git operations. Its new versioned
  record-level adapter binds one const frozen plan to immutable group/cell split
  identity, exactly checks canonical support and arbitrary value bytes through
  CP-BP-06 records, preserves raw zero-denominator metrics, and compares real
  versus exact degree-preserving null records. Focused, foundation, record,
  evaluator, optimizer, inferred-pipeline, and diff checks passed. CP-BP-08
  remains independently claimed and untouched.
- 2026-08-17: `codex-cp-bp11-phase-c` claimed CP-BP-11's exact record-held-out
  lease at pushed base `3925c15`. Concurrent CP-BP-08 remains independently
  claimed on new `warp_tiles` files and component-CMake blocks. CP-BP-11 owns
  new `record_statistical_validation` files plus labelled root-CMake blocks and
  must publish `CP11_HELDOUT_READY`, release, and stop without git operations.
- 2026-08-17: `codex-cp-bp08-phase-c` claimed CP-BP-08's exact host-only lease
  at pushed base `3925c15`. CP-BP-11 remains independently idle/unassigned;
  neither stream may edit the other's implementation or CMake seam. CP-BP-08
  must publish `CP08_HOST_ABI_READY`, release to idle, and stop without git.
- 2026-08-17: Published fork-complete Phase C instructions without claiming
  either child. CP-BP-08 owns new host `warp_tiles` files plus labelled
  component-CMake blocks; CP-BP-11 owns new record-statistical-validation files
  plus labelled root-CMake blocks. Frozen producer APIs are read-only, CPU build
  directories are separate, GPU work is serialized, both children stop/release
  at their named gates without git, and the appointed Barrier C integrator alone
  validates/commits/pushes the combined result. Both streams remain unassigned.
- 2026-08-17: `BARRIER_B_INTEGRATED` records pushed source checkpoint
  `eeb8c39`. CP-BP-06/07 are closed, and the integrator opened—but did not
  claim—the disjoint Phase C pair: CP-BP-08 host tile ABI/reference and
  CP-BP-11 frozen-plan/record-level held-out adapters.
- 2026-08-17: Fresh Barrier B combined validation closed CP-BP-06/07. Exact
  record/order CPU/CUDA tests, downstream host/CUDA regressions, and CUDA 12.9
  memcheck/racecheck passed on V100 `sm_70`; serialized benchmarks reproduced
  0.395 ms record-build and 0.23344 ms local-order medians with exact agreement.
  The source checkpoint is pushed before Phase C is opened.
- 2026-08-17: CP-BP-07 published both ordering gates, released every lease,
  and became idle without git operations. Its deterministic bounded-window
  host/CUDA order maps agree exactly and CUDA 12.9 memcheck/racecheck report
  zero errors/hazards.
  The serialized V100 benchmark measured 0.233472 ms CUDA median versus
  22.7307 ms CPU for 65,536 rows and reduced group-union metadata to 131,072
  bytes versus 4,194,304 original/row-NNZ and 2,701,568 random bytes. Both
  256- and 4,096-row window variants also preserved exact agreement and reduced
  metadata against every baseline. Both Phase B streams now await the appointed
  Barrier B integrator.
- 2026-08-17: CP-BP-06 published `CP06_DEVICE_READY`, released every lease,
  and became idle without git operations. Its asynchronous CUB-backed CUDA
  record builder matches the CPU oracle exactly, passes memcheck/racecheck and
  downstream regressions, and measured 0.393 ms median for 2,097,152 NNZ on
  V100 `sm_70` with transfers excluded. CP-BP-07 remains independently active.
- 2026-08-17: `codex-cp-bp07` claimed CP-BP-07 at pushed base `1e25e11`,
  leasing new host/CUDA local-order files, tests/benchmark, root-CMake blocks,
  and ledgers. CP-BP-06 remains independently claimed on disjoint CUDA record
  files and component-CMake blocks.
- 2026-08-17: `codex-cp-bp06-phase-b` claimed CP-BP-06 Phase B at pushed base
  `1e25e11`, leasing only new CUDA record API/source/test/benchmark files,
  component-CMake target blocks, and coordination ledgers. CP-BP-07 remains
  ready for a disjoint parallel claim and consumes host records read-only.
- 2026-08-16: Barrier A jointly rebuilt and validated CP-BP-06 host records and
  CP-BP-11 statistical foundations from fresh `build-cp-bp-barrier-a` on V100
  `sm_70`. Both new tests, planner/evaluator/optimizer, sampling materialization,
  exact merge-cost CPU/CUDA, and the inferred packing pipeline passed. The
  integrated source checkpoint is `25fcb43`; the next unclaimed fork pair is
  CP-BP-06 Phase B plus CP-BP-07.
- 2026-08-16: CP-BP-11 Phase A published `CP11_FOUNDATIONS_READY` with tested
  metric, group-aware split/bootstrap, immutable provenance, leakage detection,
  and exact row/feature-degree-preserving null foundations. Both Phase A streams
  are idle with leases released; Barrier A integration is ready.
- 2026-08-16: CP-BP-06 Phase A published `CP06_HOST_ABI_READY` with the
  versioned feature-block geometry identity and tested width-32 CPU compact
  record build/validate/decode contract. Its leases are released and it is idle
  without git operations; CP-BP-11 remains independently claimed until its
  foundations gate.
- 2026-08-16: Phase A is running as the intended parallel pair:
  `codex-cp-bp06-phase-a` owns record/plan-geometry files and component CMake;
  `codex-cp-bp11-phase-a` owns isolated statistical-validation files and root
  CMake. Both must stop and release at their host/foundation gates.
- 2026-08-16: `codex-cp-bp06-phase-a` claimed CP-BP-06 Phase A at pushed
  Cellerator base `8773f87` with exact new-file, plan-identity, CellPack-CMake,
  and ledger leases. CP-BP-11 Phase A remains independently forkable through
  disjoint implementation files and the root CMake seam.
- 2026-08-16: Added the checkpointed CP-BP-06→11 single-worktree protocol.
  Phase A permits CP-BP-06 host ABI/reference and CP-BP-11 validation
  foundations in parallel; every later assignment is conditionally gated.
  Shared edits, GPU work, and checkpoint git integration are serialized, and
  every CP-BP-06 through CP-BP-11 child ledger now contains fork-time rules.
- 2026-08-16: Closed the serial CP-BP-00→05 integration audit. A permanent
  public-API test now composes sampling, sampled CSR, support bitsets,
  candidate discovery, exact scoring, full-domain optimization, frozen-plan
  application, and exact reconstruction. Sample/mapping identity is versioned
  through the optimizer boundary, and incomplete exact sources cannot claim a
  full row domain. Focused `sm_70` regressions passed; the unrelated committed
  NCCL `local_context` blocker remains limited to rebuilding the dataset-backed
  `samplingRuntimeTest` target.
- 2026-08-16: Started a serial CP-BP-00→05 integration audit after both parallel
  streams were committed. The audit owns only an end-to-end contract test,
  build wiring, and coordination notes; CP-BP-06 remains ready and untouched.
- 2026-08-16: Final single-worktree integration configured a fresh
  `build-cp-bp-integration` for CUDA `sm_70` with Torch models disabled, built
  CP-BP-03/05 together, and passed `cellPackMergeCostTest`,
  `cellPackApplyPlanTest`, `cellPackEvaluatorTest`, `cellPackOptimizerTest`,
  `cellPackReconstructionTest`, and `geneCandidateDiscoveryRuntimeTest`.
  `git diff --check` and the TODO staleness dry run also passed; both child
  leases were already released before integration began.
- 2026-08-16: CP-BP-05 completed and released its lease. The shared worktree
  now contains pointer-first host/CUDA frozen-plan application, CUB segmented
  ordering, exact reconstruction tests, compute-sanitizer coverage, and a
  serialized V100 benchmark. CP-BP-06 is reactivated as ready/unclaimed over
  the new ordered partition view; no CP-BP-06 records were implemented.
- 2026-08-16: CP-BP-03 completed and released its shared-file lease. The shared
  worktree now contains the versioned CPU/CUDA exact scorer, optimizer-valid
  exact relations, focused tests, and a serialized V100 benchmark; final git
  integration remains reserved for the integrator after CP-BP-05 closes.
- 2026-08-16: `codex-cp-bp-05-fork` claimed CP-BP-05 under the shared
  interlock. It leases root CMake plus new `apply_plan` source/test/benchmark
  paths and must not edit CP-BP-03's component-CMake or merge-cost lease.
- 2026-08-16: CP-BP-03 was claimed by `codex-cp-bp-03-fork` under the
  single-worktree interlock with exact merge-cost source/test/benchmark leases
  and the CellPack CMake integration seam. CP-BP-05 was unclaimed at that
  instant and was subsequently claimed under the same protocol.
- 2026-08-16: Added the single-worktree CP-BP-03/05 claim, file-lease, build,
  benchmark, and git-integration interlock. The streams remain unclaimed until
  a fork records one specific assignment under the cooperative lock.
- 2026-08-16: Reconciled CP-BP-00 through CP-BP-13 against source, tests,
  CMake wiring, V100 benchmarks, git history, and the dirty worktree based at
  `1ebb734`; the acceptance-complete CP-BP-01/02/04 implementation was then
  checkpointed as `597a3eb`. CP-BP-03 and CP-BP-05 are genuinely
  unassigned/ready; CP-BP-05 is
  the primary continuation frontier because it unlocks CP-BP-06 through the
  tile/runtime/persistence chain. CP-BP-11 has reusable deterministic split and
  frozen-plan/evaluator foundations but no null/bootstrap validation harness.
- 2026-08-16: Focused rebuilds and runtime tests passed for sampled CSR
  materialization, CPU/CUDA gene support, CPU/CUDA candidate discovery, exact
  plan evaluation, optimizer, and reconstruction. Serialized V100 candidate
  and optimizer benchmarks passed. After checkpointing, a fresh
  `cmake -S . -B build-checkpoint-validation -DCMAKE_CUDA_ARCHITECTURES=70`
  followed by `cmake --build build-checkpoint-validation --target
  samplingRuntimeTest -j 4` failed while compiling CellShard runtime because
  committed `b69a168` `nccl_communicator.cuh` uses `local_context` before its
  definition. The test source separately rebuilt/linked against the prior
  runtime archive and passed. No repair was attempted in this status-only task.
- 2026-08-14: Completed the post-evaluator CP-BP-04 optimizer-readiness audit. The evaluator passed without code changes; CP-BP-04 is ready to begin with a proxy-plus-CPU-oracle architecture, while CP-BP-05 remains blocked on the owning/versioned frozen-plan contract.
- 2026-08-14: Completed the conceptual CP-BP-04 v1 implementation plan without source changes. Selected feature-first deterministic constrained coarsening plus bounded move/swap refinement, integer support-derived proxy batches, CPU exact-oracle checkpoints with rollback, identity/fixed-width row geometry, and an immutable semantic plan. GPU evaluator acceleration remains an expected deferred CUB/V100 workstream with explicit profiling triggers.
- 2026-08-14: Completed CP-BP-04 v1 for its supplied-candidate contract. Added deterministic score-kind-preserving normalization, membership-authoritative mutable blocks, zero-copy sampled-support proxy caches, constrained merge/move/swap search, exact evaluator rollback/shrink/blacklisting, immutable semantic `frozen_packing_plan`, adversarial/property tests, and a mutex optimizer benchmark. CP-BP-05 is ready; completed CP-BP-02 pairs still require CP-BP-03 exact scored-relation adaptation for production candidate quality.
- 2026-08-14: Opened `cellpack-packing-plan-cuda-evaluator` as a separate deferred workstream after CPU oracle share measured 77.6% at 5k features and 46.6% at 20k features. Absolute volume triggers remain low, so it does not block CP-BP-05.
- 2026-08-14: Added CP-BP-00 through CP-BP-13 as the complete data-inferred
  block-packing roadmap. Each executable step has its own pickup ledger,
  dependencies, collision boundary, acceptance criteria, and implementation
  status; no implementation was performed by this coordination pass.
- 2026-08-14: Reconciled CP-BP-01 with new dirty-worktree deterministic sampling
  and sampled-CSR materialization headers/sources/tests. These are evidence of
  active progress, but sampled support bitsets were not found and the step is
  not marked complete.
- 2026-08-14: Preliminary coordination reconciled CP-BP-04 with the completed
  `cellpack-packing-plan-evaluator` and temporarily recorded an active optimizer
  thread. The later post-evaluator audit supersedes that pickup state: exact
  supplied-plan evaluation is complete infrastructure, and CP-BP-04 is ready.
- 2026-08-14: Recorded safe independent pickups CP-BP-02, CP-BP-03, and the
  foundational reference/metric portion of CP-BP-11. Their ledgers forbid
  edits to CP-BP-01/04-owned files and identify the narrow integration seams.
- Started `cellpack-packing-plan-evaluator` on 2026-08-14 after correcting the earlier downstream physical-format interpretation. This stage evaluates a `PackingPlan` defined by row and feature permutations plus row-group and feature-block boundaries. It does not define `gene_masks`, a final sparse codec, `.cspack` serialization, preprocessing changes, plan inference, or execution kernels.
- Finished `cellpack-packing-plan-evaluator`: exact sparse occupied-tile evaluation, reusable prepared CSR support/scratch, separate reference cost policy, static-plan geometry adapter, comprehensive tests, mutex benchmark, and dated CellPack handoff notes are in place. The next stage is plan inference/refinement unless evaluator throughput is first shown to block search.
- Started `sequence-bits-dna2`: requested scope is a new `include/Cellerator/seq/dna2.cuh` GPU-native DNA 2-bit primitive with packed storage words, warp-compute bitplanes, correctness kernels, tests, primitive benchmark, and docs. This work was separate from the historical core sequence port material.
- Finished `sequence-bits-dna2`: configured Cellerator, built `sequenceDna2Test`, `sequenceDna2CudaTest`, and `sequenceDna2Bench`, ran both focused tests, and ran the primitive benchmark with `./build/sequenceDna2Bench 1048576 16 1 10`.
- Migrated `sequence-bits-dna2` out of Cellerator to the sibling Baseplane
  project. Cellerator now consumes `Baseplane::seq` for sequence bit primitive
  smoke coverage and no longer owns the sequence headers, target, tests,
  benches, or docs.
- Preserved post-umbrella Baseplane explicit-width DNA2 and benchmark work in
  sibling Baseplane commit `12c83b6`, with top-level Baseplane validation for
  `baseplaneDna2Test` and `baseplaneDna2CudaTest`.
- Validated the hard cut with sibling Baseplane: `cmake -S Cellerator -B
  Cellerator/build`, `cmake --build Cellerator/build --target
  coreSparseLayoutRuntimeTest quantizedMatrixTest exactSearchRuntimeTest -j 4`,
  and the three resulting runtime smokes passed.
- Extended `sequence-bits-dna2` validation: added `tests/seq/dna2_test_helpers.hh` for deterministic random sequence generation, changed the benchmark to use random DNA input with independent representation selection, and ran a packed-word64 vs warp-planes32 comparison matrix plus Nsight Systems profiles.
- Finished the first CPU SIMD backend pass: Highway now uses SIMD mask extraction/materialization for full 32-base ASCII pack/unpack, the CPU benchmark reports the active backend, and both Highway-enabled and scalar-only builds pass `sequenceDna2Test` plus `sequenceDna2CpuBench`.
- Added the first former-core ownership slice: the then-current core include path exposed
  sparse layout primitives/device views, and CellShard layout headers were
  compatibility shims over those types.
- Verified the former-core/CellShard wiring with CellShard package-consumer
  checks, Cellerator sparse/quantized tests, and a CellShardPreprocess build.
- Started `cellerator-sparse-ml-layout` from the supplied source layout plan. The intended first pass is behavior-preserving except for moving forward-neighbor policy/API ownership into the new the external neighbor-caller package.
- Checkpointed `cellerator-sparse-ml-layout`: moved sparse-operator/model-op code under `src/compute/ml`, moved shared host buffering to `src/compute/core`, moved cuVS/KNN scoring helpers under `src/compute/neighbors/scoring`, and removed Cellerator-owned forward-neighbor compatibility wrappers.
- Implemented `cellshard-multi-assay-archive`: CellShard now has measurement-agnostic assay descriptors and row-map helpers, Cellerator validates those semantics against the biology semantics package, and docs state that multiome execution uses coordinated single-assay CSPACK artifacts.
- Ran `todo-cleanup --partial` and cleared workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, implement-derived-subset-and-reorder-materialization-for-cellshard-and-workbench.
- Started `cellshard-preprocess-gpu-biology-backbone` from the supplied implementation plan.
- Finished `cellshard-preprocess-gpu-biology-backbone`: moved Blocked-ELL/Sliced-ELL native preprocessing and CSR fallback ownership into CellShardPreprocess, moved preprocessing benchmarks there, and removed Cellerator preprocessing APIs and root targets.
- Checkpointed the earlier core layout migration: old `core/sparse` and `quantized` public paths were removed, matrix representation/runtime substrate/quantized packing moved under the then-current core include path, compute owns conversion and CUDA primitives, CellShard shims were updated, and focused Cellerator runtime checks passed. Standalone CellShard `cellShardMaskGroupsRuntimeTest` builds but exits 14 on its row-keep expectation.
- Checkpointed the matrix/compute boundary cleanup: `core/format` became `core/matrix`, conversion and bucket moved to `compute/matrix/convert`, `warp_reduce.cuh` moved under `src/compute/core/primitives`, and `cmake --build build -j 4` plus `./build/coreSparseLayoutRuntimeTest` passed.
- Started `cellerator-preprocess-rehome`: requested scope is a hard cutover that removes the CellShardPreprocess submodule/package and splits the moved implementation into Cellerator compute preprocessing kernels plus the Cellerator preprocessing policy/runtime API.
- Finished `cellerator-preprocess-rehome`: Cellerator now builds preprocessing compute and pipeline targets, ported preprocessing tests and benchmarks, CellShard no longer fetches or installs the old package, and the root no longer tracks the preprocessing submodule. Validation passed for the Cellerator build, focused preprocessing tests, adjacent Cellerator smoke tests, CellShard build, and CellShard package-consumer install check. `cellShardMaskGroupsRuntimeTest` still exits 14 as previously recorded under `cellerator-sparse-ml-layout`.
- Started `cellerator-python-preprocess-runtime`: Python should expose Scanpy-like Cellerator preprocessing entry points, but all omics preprocessing compute must delegate to GPU-native Cellerator/CellShard runtime paths.
- Finished `cellerator-python-preprocess-runtime`: direct `_cellerator` build, source smoke test, wheel build, and installed-wheel import smoke passed. A fixture-backed GPU session test remains a future follow-up because no stable tiny `.csh5` fixture is currently checked in.
- Started `cellerator-preprocess-scanpy-validation`: the PBMC3K `.h5ad` and `.csh5` fixtures now exist under `data/test/reference`, so the missing fixture-backed validation can compare Cellerator GPU-native preprocessing metrics against a Scanpy reference.
- Finished `cellerator-preprocess-scanpy-validation`: added `tests/validate_scanpy_preprocess.py`, exposed missing Python session metrics, fixed direct Python session device-reservation, fixed the `cell_mito_counts` alias, and passed the PBMC3K Scanpy comparison for all metric families.
- Started `cellerator-runtime-autotune`: requested scope is an optional, bounded runtime optimizer that is callable from Cellerator, defaults off in C++ mode, and is exposed in Python preprocessing as `autotune=True`.
- Finished `cellerator-runtime-autotune`: added the reusable light optimizer surface, plan-aware preprocessing compute calls, Python autotune/session metrics, README notes, and focused validation including PBMC3K `.csh5` autotune smoke.
- Finished `cellerator-sparse-ml-layout`: hard-cut the former core identity into direct Cellerator domains with no compatibility headers or `Cellerator::core` alias, updated sibling CellShard shims/package config, and validated Cellerator plus CellShard configure/build and focused Cellerator runtime tests.
- Finished `cellpack-m3-layout-selection`: CellPack now computes per-region padding/fill/index/value/output-byte metrics from packed coordinate plans, applies deterministic V100-oriented hybrid layout selection, and exposes `cellPackLayoutBench` summaries for CSR, current Blocked-ELL, and selected hybrid plan estimates.
- Started `cellpack-m4-oracle-gating`: requested scope is a narrow static-oracle gating experiment over M2/M3 CellPack plans. The prototype should select precompiled region spans through route masks/tapes and run coordinate-based CUDA SpMV forward plus transpose replay without dynamic module assembly, learned gates, real-data sweeps, or production Blocked-ELL/Sliced-ELL kernels.
- Finished `cellpack-m4-oracle-gating`: CellPack now has host route-mask and route-tape contracts, deterministic static oracle scenarios, a separate CUDA coordinate-span runtime target, focused host/CUDA tests, and `cellPackGatingBench` summaries for no-gating versus oracle-gating.
- 2026-08-14: Current thread resumed CP-BP-01 from the completed deterministic sampler and structural sampled-CSR bundle; next implementation is the pointer-first CPU/CUDA gene-support bitset primitive.
- 2026-08-14: Added include/Cellerator/compute/gene_support_bitset.hh and src/compute/dataset/gene_support_bitset.cu. The pointer-first host bundle retains immutable sampling provenance and sampled-position/global-row mapping; the CUDA builder uploads only CSR row pointers and column indices, constructs gene-major u32 support words with atomicOr, and counts unique detected cells despite duplicate entries.
- 2026-08-14: Added tests/gene_support_bitset_runtime_test.cu and normal CMake targets. Passed samplingRuntimeTest, datasetRuntimeTest, samplingMaterializationRuntimeTest, geneSupportBitsetRuntimeTest, and the opt-in geneSupportBitsetRuntimeTest --full-size-gpu smoke on Tesla V100.
- 2026-08-14: For 65,536 cells and 30,000 genes, words_per_gene=2,048 and support_bytes=245,760,000. The current CUDA convenience builder holds host and device support concurrently; peak structural storage is 2*support_bytes + 2*(genes*4) + cells*8 + (cells+1)*4 + nnz*4 + 4 bytes, excluding small provenance allocations.
- 2026-08-14: Claimed by the current serial thread after confirming no live agent owns CP-BP-02. Step 1 sampling, dataset, materialization, support-bitset, and full-size V100 smoke tests passed before edits.
- 2026-08-14: Added the Cellerator pointer-first host candidate contract and fixed SplitMix64-v1 MinHash/LSH provenance. CPU and CUDA paths omit empty genes, use stable CUB radix sorting/scans/unique, cap oversized buckets with a deterministic circular window, and return lexicographically sorted unique canonical pairs.
- 2026-08-14: Focused candidate runtime test passes CPU/GPU exact agreement. The 64-gene exhaustive fixture retained 48/48 deliberately high-overlap pairs, emitted 48/2,016 candidates, and reduced the unordered pair set by 97.619%.
- 2026-08-14: Final Step 1 regressions and CP-BP-02 CPU/CUDA tests passed. The full 65,536-cell support smoke allocated 245,760,000 bytes; no Step 1 API was changed.
- 2026-08-14: Serialized Tesla V100 sm_70 benchmark at 65,536 cells x 30,000 genes produced 105,000 candidates from 449,985,000 exhaustive pairs, retained all 105,000 constructed cluster pairs, and reduced the unordered pair space by 99.9767%. Three timed runs after one warmup measured 59.832 ms minimum and 60.904 ms median.
- 2026-08-14: Benchmark provenance reports 14,060,031 bytes of CUB scratch, 333,424,355 bytes of accounted peak device allocation, and a 534,306,000-byte conservative fixed bound excluding CUB. The result is synthetic correctness smoke, not a production recall threshold.
- 2026-08-14: CP-BP-02 completed deterministic SplitMix64-v1 global-row MinHash, configurable LSH, bounded oversized-bucket handling, CUB grouping/deduplication, canonical host candidate pairs, CPU/GPU exact tests, and a serialized V100 smoke benchmark. CP-BP-03 was not started.

## Next Actions
- Preserve the completed CP-BP roadmap. Any aggressive performance pass should
  begin as a separate measured workstream with these v1 identities and
  ownership boundaries as compatibility baselines.
- Reactivate blocked children only when their recorded representation/API
  prerequisite lands; do not guess a downstream physical ABI.
- CP-BP-02 is closed and CP-BP-03 now consumes `gene_candidate_pair_view` plus
  its immutable provenance without reinterpretation; persistent device-resident
  Step 1 support remains an optional optimization.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.
- CP-BP-01 through CP-BP-13 satisfy their child done criteria, including exact
  decode/numerical equivalence, held-out/null/stability evidence, fair existing
  layout baselines, and a no-repack durable execution lifecycle.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize sparse operator kernels, workbench browse-cache updates, semantic the biology semantics package cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.

<!-- todo-orchestrator:v2-managed:start -->
# Todo Orchestrator v2 Projection

Project revision: `2063`

## Workstreams
- `CE-LIVE-00` | kind: epic | status: planned | parent: - | objective: Turn completed CE-ARCH foundations into one authoritative planner-backed quantitative biological execution path, then activate CelleraTorch as a thin native adapter.
- `CE-LIVE-01` | kind: integration_task | status: done | parent: CE-LIVE-00 | objective: Reconcile the clean post-CE-ARCH repository, apply the complete CE-LIVE plan transactionally, publish the source and contract map, validate the graph, and leave multiple non-overlapping tasks ready.
- `CE-LIVE-10` | kind: validation_task | status: done | parent: CE-LIVE-00 | objective: Record the exact current clean-head host/CUDA build and focused test baseline, distinguish real failures from unavailable optional background evidence, and publish reproducible commands without changing implementation.
- `CE-LIVE-11` | kind: task | status: done | parent: CE-LIVE-00 | objective: Resolve the feature/row orientation seam so forward relations are feature-or-gene source to row-or-cell destination, transpose projections share the same logical edge identity, and swapped axes fail explicitly.
- `CE-LIVE-12` | kind: task | status: done | parent: CE-LIVE-00 | objective: Create a checksum-pinned PBMC3K computational-correctness fixture contract with exact axis and selection identities, verified stored-value semantics, deterministic extraction, a tiny committed schema smoke fixture, a local representative fixture, and independent numerical referees.
- `CE-LIVE-13` | kind: task | status: done | parent: CE-LIVE-00 | objective: Audit every retained native and conventional candidate, projection schema, numeric tuple, persistent-state requirement, output order, preparation helper, graph capability, and measured regime. Define the minimal host-side activation-catalog contract without implementing it or widening operation_candidate.
- `CE-LIVE-14` | kind: task | status: done | parent: CE-LIVE-00 | objective: Design and implement the runtime-side generation-readiness token or record with same-stream fast paths, cross-stream waits, failed-enqueue safety, and no event or stream ownership in the persistent biological ABI.
- `CE-LIVE-15` | kind: task | status: done | parent: CE-LIVE-00 | objective: Update stale implementation guidance to post-CE-ARCH truth, make native Cellerator without Torch the default build, preserve an explicit compatibility build, and prevent revival of retired CP-Math or universal Blocked-ELL assumptions.
- `CE-LIVE-16` | kind: task | status: done | parent: CE-LIVE-00 | objective: Define one V100-relevant dense-fragment or WMMA candidate lane, including density classification, packing, alignment, tails, numeric policy, forward/backward maps, complete planner costs, and rejection criteria, without modifying common semantic ABIs or registering a kernel.
- `CE-LIVE-19` | kind: integration_task | status: done | parent: CE-LIVE-00 | objective: Wire the foundation tests, audit orientation, fixture, catalog, readiness, documentation, and Tensor Core contracts together, run focused host/CUDA gates, freeze only validated interfaces, and publish the next parallel frontier.
- `CE-LIVE-20` | kind: task | status: done | parent: CE-LIVE-00 | objective: Resolve validated CPE2 projection entries into typed non-owning device views for CPK1 row-masked, CSR, FMP1, and CTP1 with exact identity, orientation, schema, location, size, and map validation.
- `CE-LIVE-21` | kind: task | status: done | parent: CE-LIVE-00 | objective: Implement a deterministic host-side catalog over existing operation-core candidates, exposing capability and preparation metadata without changing operation_candidate, owning runtime resources, or introducing virtual dispatch.
- `CE-LIVE-22` | kind: task | status: done | parent: CE-LIVE-00 | objective: Add a session-integrated cuSPARSE CSR SpMV and SpMM candidate for the live width envelope, remove per-run device selection from the custom CSR path, keep descriptor creation and preprocessing in preparation, and provide a strong fair baseline.
- `CE-LIVE-23` | kind: task | status: done | parent: CE-LIVE-00 | objective: Combine one activated typed projection, biological axes, operation request, session-owned persistent allocation, and catalog entry into a prepared_operation without widening the stable operation-core ABI.
- `CE-LIVE-24` | kind: task | status: done | parent: CE-LIVE-00 | objective: Bind the quantitative fixture to exact Cellerator identities, build the forward feature-to-cell relation and mutable generations, generate deterministic dense operands, and compare supported outputs against an independent CPU reference.
- `CE-LIVE-25` | kind: task | status: done | parent: CE-LIVE-00 | objective: Use readiness in the native training slice, preserve topology across updates, expose native parameter descriptors, return explicit next-generation readiness, and prove same-stream and cross-stream correctness without premature publication.
- `CE-LIVE-26` | kind: task | status: done | parent: CE-LIVE-00 | objective: Derive planner-ready structural and quantitative statistics from the fixture, construct exact persistent planning keys and reuse horizons, expose complete candidate phase inputs, and test invalidation without replacing empirical final selection.
- `CE-LIVE-29` | kind: integration_task | status: done | parent: CE-LIVE-00 | objective: Wire and validate typed activation, catalog, conventional fallback, preparation factory, quantitative adapter, readiness, training integration, and planner inputs. Freeze the minimum executable-core interfaces and open the vertical-slice frontier.
- `CE-LIVE-30` | kind: task | status: done | parent: CE-LIVE-00 | objective: Implement one host-side executable_program that enumerates legal activated candidates, prices complete costs, invokes the planner, reserves session-owned persistent state, prepares the winner, binds changing launch state, exposes output order and workspace requirements, and runs without creating a second runtime.
- `CE-LIVE-31` | kind: validation_task | status: done | parent: CE-LIVE-00 | objective: Run the quantitative PBMC3K fixture through CP-BP compilation, projection construction, complete-cost planning, prepared native or conventional execution, repeated value generations, and canonical recovery at declared widths and reuse horizons.
- `CE-LIVE-32` | kind: task | status: done | parent: CE-LIVE-00 | objective: Implement at most one sm_70 dense-fragment or WMMA candidate under the bounded contract, integrate it as an ordinary planner candidate, and either promote on a complete-cost real-fixture win or leave unregistered with reproducible negative evidence.
- `CE-LIVE-33` | kind: task | status: done | parent: CE-LIVE-00 | objective: Wrap the validated FMP1 and CTP1 N=16 training slice as an explicit prepared executable path with forward, epilogue, backward, sparse and bias updates, parameter descriptors, readiness transitions, and a fair persistent CSR/cuSPARSE baseline without rebuilding topology.
- `CE-LIVE-34` | kind: validation_task | status: done | parent: CE-LIVE-00 | objective: Build and reload a CPE2 image containing live projections, carry it through existing opaque CPEXEC01 compatibility delivery, activate typed device views, select and prepare a candidate, and execute quantitatively without moving semantics into CellShard.
- `CE-LIVE-35` | kind: validation_task | status: done | parent: CE-LIVE-00 | objective: Prove two-stream reuse, readiness waits, supported CUDA Graph capture, stale identity and generation rejection, pointer relocation, and absence of forbidden hot-path behavior.
- `CE-LIVE-36` | kind: validation_task | status: done | parent: CE-LIVE-00 | objective: Measure complete preparation-to-consumer costs, memory, bytes, launches, order work, readiness, forward/backward, reuse-one/eight/persistent regimes, and planner regret against every legal candidate.
- `CE-LIVE-37` | kind: integration_task | status: in_progress | parent: CE-LIVE-00 | objective: Audit every Cellerator invariant, publish the supported and unsupported operation matrix and commands, reconcile the ledger, reach live biological readiness, and freeze only the minimal native entry contract required by CelleraTorch.
- `CE-LIVE-40` | kind: task | status: planned | parent: CE-LIVE-00 | objective: Expose native Cellerator dense operands, value planes, and parameter descriptors to Torch with explicit lifetime ownership and correct device, shape, and stride metadata.
- `CE-LIVE-41` | kind: task | status: planned | parent: CE-LIVE-00 | objective: Wrap executable_program as a Torch custom operation using the current Torch CUDA stream, preserving Cellerator planning and ownership and performing no hidden repeated conversion.
- `CE-LIVE-42` | kind: task | status: planned | parent: CE-LIVE-00 | objective: Connect Torch autograd to the native training executable, propagate readiness and current-stream dependencies correctly, expose native learned parameters without copying ownership into Torch, and test forward/backward/update parity.
- `CE-LIVE-43` | kind: integration_task | status: planned | parent: CE-LIVE-00 | objective: Integrate the three adapter lanes, preserve the old copy-based CSR exporter as an explicit compatibility and debug path, validate package consumers and Torch-off native builds, and publish one coherent adapter surface.
- `CE-LIVE-44` | kind: validation_task | status: planned | parent: CE-LIVE-00 | objective: Run the same quantitative fixture through native Cellerator and CelleraTorch forward/autograd paths, prove numerical and identity parity, verify current-stream behavior, and measure adapter overhead.
- `CE-ARCH-01` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Establish the actual repository, ABI, build, benchmark, task, and ownership state before implementation.
- `CE-ARCH-93` | kind: integration_task | status: done | parent: - | objective: Preserve CP-MATH provenance while making every CP-MATH task, checkpoint, gate, and dependency semantically historical and ineligible for current planning or attention.
- `CE-LIVE-45` | kind: integration_task | status: planned | parent: CE-LIVE-00 | objective: Audit Cellerator, the Tensor Core promotion or rejection decision, and CelleraTorch against CE-LIVE. Reconcile and export the ledger, publish exact evidence and limitations, and leave newer CellShard CPEXEC02 and broader Baseplane execution as external follow-ups.
- `CE-ARCH-02` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Contain unsafe experimental paths before architectural expansion, without performance redesign or new kernels.
- `CE-ARCH-10` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Create the minimal Cellerator-owned identity and heterogeneous operand model shared by dense state, sparse relations, and Baseplane sequence structures.
- `CE-ARCH-11` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Make execution order and data lifetime explicit across Cellerator operations.
- `CE-ARCH-12` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Consolidate runtime ownership so CP-Math and biological operations use one explicit Cellerator execution substrate.
- `CE-ARCH-20` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Preserve validated CP-BP v1 behind new identity/lifetime contracts while separating semantic geometry from physical projection.
- `CE-ARCH-30` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Build a reproducible corpus and measurement program capable of disproving the preferred Cellerator architecture.
- `CE-ARCH-21` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Evolve CPK1 pointer-free persistence into an execution IR holding one semantic geometry and multiple physical projections.
- `CE-ARCH-22` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Transform useful CP-Math experiments into Cellerator core operation, projection, planning, and execution contracts.
- `CE-ARCH-31` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Select the fastest correct end-to-end strategy and feed measured costs into versioned semantic-geometry optimization.
- `CE-ARCH-40` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Make Baseplane sequence structures native Cellerator operands without a host, dense-matrix, or generic-SpMM boundary.
- `CE-ARCH-50` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Prove the foundations do not trap Cellerator in forward-only, single-GPU, fp32, Volta-only execution.
- `CE-ARCH-60` | kind: integration_task | status: done | parent: CE-ARCH-00 | objective: Complete migration from experimental CP-Math and direct CP-BP v1 coupling into the validated biological execution architecture.
- `CE-ARCH-70` | kind: epic | status: done | parent: CE-ARCH-00 | objective: Correct the existing Cellerator execution architecture's foundational ABI, lifetime, identity, planning-key, and device-prebinding defects without expanding its feature or projection scope.
- `CE-ARCH-70A` | kind: integration_task | status: done | parent: CE-ARCH-70 | objective: Integrate the existing local predicate-plan work, expose and require one explicit sequence predicate ABI version, make validity authoritative, and fail incompatible sibling checkouts early.
- `CE-ARCH-80` | kind: epic | status: done | parent: - | objective: Finish the implementation and evidence required by roadmap Phases 4 through 11 and the definition of migration completion, without erasing the bounded results of CE-ARCH-40 through CE-ARCH-79.
- `CE-ARCH-81` | kind: validation_task | status: done | parent: CE-ARCH-80 | objective: Replace stale completion language with a source-backed Phase 4 through Phase 11 exit matrix that distinguishes implemented, partial, missing, and externally blocked requirements.
- `CE-ARCH-70B` | kind: workstream | status: done | parent: CE-ARCH-70 | objective: Replace monolithic persistent scratch ownership with fixed-capacity independent stable CUDA allocations while preserving the pre-reserved transient arena and allocation-free sealed launch binding.
- `CE-ARCH-82` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete Phase 4 so CPK1 compatibility, sectioned semantic structure, schema-extensible projections, multiple value planes, opaque relocation, and direct prepared execution are one tested path.
- `CE-ARCH-70C` | kind: workstream | status: done | parent: CE-ARCH-70 | objective: Make prepared operations and planner candidates explicitly depend on a deterministic bounded set of immutable relation structures and validate each value plane against its own relation and epoch.
- `CE-ARCH-83` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete Phase 5 with real and adversarial structures, full end-to-end phase accounting, forward/transpose observability, reproducible artifact identity, and planner-ready structural features.
- `CE-ARCH-70D` | kind: workstream | status: done | parent: CE-ARCH-70 | objective: Add one compact validated output-effect contract per output and declare the sequence gene-state output as accumulation and predicate-mask output as overwrite.
- `CE-ARCH-84` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete the forward half of Phase 6 with measured low-sharing/tail handling and CTA-scale medium-N execution while preserving row-masked, feature-major, and CSR behavior.
- `CE-ARCH-70E` | kind: workstream | status: done | parent: CE-ARCH-70 | objective: Implement host-side persistent identity interning and generation-safe resolution, remove runtime handles from durable planner evidence, and make cached projection selection resolve to current runtime candidates.
- `CE-ARCH-85` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete the reverse half of Phase 6 by sharing logical edge identity and mutable values between forward and transpose projections and executing a native backward/propagation operation without topology reconstruction.
- `CE-ARCH-70F` | kind: workstream | status: done | parent: CE-ARCH-70 | objective: Construct hot projection views from host-validated offsets plus an arbitrary destination image base and prove a CUDA kernel consumes the device-relative payload after one opaque upload.
- `CE-ARCH-86` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete Phase 7 by selecting across connected operations with explicit order, conversion, preparation, communication, and reuse costs while retaining bounded empirical autotuning and durable invalidation.
- `CE-ARCH-70G` | kind: validation_task | status: done | parent: CE-ARCH-70 | objective: Reconcile executable invariants, focused host/CUDA/sanitizer evidence, frozen interfaces, paired repository commits, deliberate deferrals, and the exact unchanged downstream path.
- `CE-ARCH-87` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete Phase 8 by feeding measured workload-weighted total cost into CP-BP alternating refinement with held-out stability, forward/transpose profiles, activity, and partition-cut terms justified by current evidence.
- `CE-ARCH-88` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete Phase 9 with a module-local learned projection, fused bias/activation/normalization, module-major dense state, native backward, mutable learned values, and topology-stable training step that beats a fair CSR baseline in its declared regime.
- `CE-ARCH-89` | kind: integration_task | status: done | parent: CE-ARCH-80 | objective: Complete Phase 10 with common domain/order identities, direct relation-builder output, materialized and fused sequence-to-regulatory execution, no host boundary, and complete-cost planner selection across reused cell states.
- `CE-ARCH-90` | kind: workstream | status: done | parent: CE-ARCH-80 | objective: Complete Phase 11 with shared-value hierarchy indices, nested partition identity, active-module skipping, execution-order communication planning, and cross-device boundary cost while keeping single-GPU execution simple.
- `CE-ARCH-61` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Version the retirement of obsolete CP-Math runtime interfaces and refresh operation-core documentation identity without restoring deleted implementation.
- `CE-ARCH-71` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Register the existing CP-BP v1 native row-masked N=1 kernel as a real operation-core and planner candidate without changing its projection or kernel semantics.
- `CE-ARCH-91` | kind: integration_task | status: done | parent: CE-ARCH-80 | objective: Consume the completed CellShard foundation to persist, validate, place, upload, and directly execute opaque Cellerator images without CellShard interpreting execution semantics or Cellerator owning storage transport.
- `CE-ARCH-72` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Register the existing legal CSR implementation as the first conventional operation-core/planner fallback.
- `CE-ARCH-92` | kind: validation_task | status: done | parent: CE-ARCH-80 | objective: Run the final fair real-data and adversarial evidence campaign, identify both Cellerator wins and fallback regimes, verify every migration exit criterion, and leave documentation and ledgers truthful.
- `CE-ARCH-73` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Measure projection preparation, input ordering, prepared execution, candidate-private output, referee, all total-cost phases, and winner preparation for real registered candidates.
- `CE-ARCH-74` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Bind at least two different value generations to one immutable CP-BP structure and projection and prove correct reuse and stale-generation rejection.
- `CE-ARCH-75` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Implement the second physical execution projection, feature-major small-N, without replacing row-masked CPK1.
- `CE-ARCH-76` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Compare row-masked, feature-major, and CSR end-to-end for N equals 1, 2, 4, 8, and 16 with complete cost accounting.
- `CE-ARCH-77` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Calibrate Objective V2 from measured candidate costs and only then feed it into CP-BP alternating refinement.
- `CE-ARCH-78` | kind: workstream | status: done | parent: CE-ARCH-00 | objective: Materialize a predicate once, cache it by sequence generation, predicate identity, and coordinate order, reuse it, and compare total cost against fused execution.
- `CE-ARCH-79` | kind: integration_task | status: done | parent: CE-ARCH-00 | objective: Start the separate CellShard foundation program against stable persistent identity and opaque Execution Image v2 contracts without interpreting Cellerator tile grammar.
- `CP-BP-00` | kind: epic | status: done | parent: CE-ARCH-00 | objective: Preserve the completed CP-BP v1 coordinator as validated historical evidence.
- `CP-BP-01` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve sampled support bitsets as completed v1 evidence.
- `CP-BP-02` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve MinHash/LSH candidate discovery as completed v1 evidence.
- `CP-BP-03` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve exact row-active-block merge scoring and referee as completed v1 evidence.
- `CP-BP-04` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve optimizer, rollback oracle, and frozen plan ABI as completed v1 evidence.
- `CP-BP-05` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve exact full-partition plan application and reconstruction.
- `CP-BP-06` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve compact record ABI and CUDA/reference evidence.
- `CP-BP-07` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve bounded local row order and inverse maps.
- `CP-BP-08` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve pointer-free warp-tile construction and validation.
- `CP-BP-09` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve N=1 weighted-row-reduction kernel, referees, and fallback evidence.
- `CP-BP-10` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve accepted v1 refinement and rollback evidence.
- `CP-BP-11` | kind: validation_task | status: done | parent: CP-BP-00 | objective: Preserve independent statistical/bootstrap validation evidence.
- `CP-BP-12` | kind: workstream | status: done | parent: CP-BP-00 | objective: Preserve measured V100 cost-model evidence and explicit fallback.
- `CP-BP-13` | kind: integration_task | status: done | parent: CP-BP-00 | objective: Preserve pointer-free CPK1 and direct CellShard archive-to-device loading.
- `baseplane-dna2-benchmark` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve historical benchmark evidence.
- `baseplane-dna2-explicit-widths` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve explicit-width evidence now owned by Baseplane.
- `cellerator-hierarchy-reset` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve cleanup/inventory history without treating it as biological hierarchy architecture.
- `cellerator-runtime-autotune` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve completed calibration and explicit-metric evidence without promoting its callback chooser to core planner.
- `cellerator-sparse-ml-layout` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve completed source-layout history.
- `cellpack-packing-plan-cuda-evaluator` | kind: workstream | status: superseded | parent: CE-ARCH-00 | objective: Record the old current-objective CUDA evaluator as superseded by operation-aware planner/objective v2.
- `cellpack-packing-plan-evaluator` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve completed evaluator/referee evidence.
- `sequence-bits-dna2` | kind: validation_task | status: done | parent: CE-ARCH-00 | objective: Preserve historical exact-sequence evidence now owned by Baseplane.
- `CE-ARCH-00` | kind: epic | status: done | parent: - | objective: Coordinate migration from completed CP-BP v1 plus experimental CP-Math into a domain-aware biological execution core while preserving validated behavior and the CellShard storage boundary.
- `CP-MATH-00` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-00 biological execution recovery.
- `CP-MATH-00A` | kind: integration_task | status: superseded | parent: - | objective: Reconcile the empty live v2 project and audited CP-BP source at historical baseline 8848f62254605025ac6e96f4cc6a8efbcc803d43; freeze the four consumed CP-BP interfaces without performance work or source changes.
- `CP-MATH-01` | kind: workstream | status: superseded | parent: - | objective: Implement backend-neutral SpMM MathRequest/OperationSignature, alpha/beta, transpose, dtype/compute, determinism, workspace, reuse, epilogue, stable identity and pointer-free ExecutionPlan metadata separation with zero/trivial semantics.
- `CP-MATH-02` | kind: workstream | status: superseded | parent: - | objective: Integrate the independent device/runtime foundation with structured capability rejection, SpMMBackend/prepared ownership, and generic unfused epilogue while preserving no-allocation repeated run semantics.
- `CP-MATH-02A` | kind: task | status: superseded | parent: - | objective: Independently implement cached DeviceMathContext, DeviceCapabilities, DeviceFingerprint, and reusable workspace ownership over the existing runtime context/handle/scratch substrate without defining operation-dependent backend policy.
- `CP-MATH-03` | kind: workstream | status: superseded | parent: - | objective: Adapt apply_frozen_plan output to execution-feature CSR, implement order-identity-safe reusable W_packed conversion, prove X_packed W_packed equals canonical math, and design lazy CSR reconstruction from CPK1 without changing CPK1.
- `CP-MATH-04` | kind: workstream | status: superseded | parent: - | objective: Lower unchanged variable-width semantic blocks and local row order into legal BELL8/16/32 candidates, record occupancy/utilization/expansion/storage, reject absurd candidates, and validate via independent decode.
- `CP-MATH-05` | kind: workstream | status: superseded | parent: - | objective: Expose the existing frozen plan/order/warp tiles/CPK1 to math with derived union masks, packed offsets, density/reuse/workload sidecars and an exact decoder; copy no compact values and invent no row permutation.
- `CP-MATH-06` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-02 containment and CE-ARCH-22 core recovery.
- `CP-MATH-07` | kind: workstream | status: superseded | parent: - | objective: Consume each legal BELL8/16/32 view as a separate prepared cuSPARSE candidate with equivalent dtypes/compute semantics and no handwritten BELL kernel.
- `CP-MATH-08` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-30 and CE-ARCH-22.
- `CP-MATH-08A` | kind: validation_task | status: superseded | parent: - | objective: Establish neutral logical reference, numerical metrics, determinism checks, CUDA-event timing with median/spread, memory/expansion accounting, and reusable benchmark reporting before backend fan-out.
- `CP-MATH-09` | kind: workstream | status: superseded | parent: - | objective: Implement trivial interception, legal candidate enumeration, structured hard filtering, epilogue composition, cheap structural pruning, cache lookup hook, and authoritative selection without emitting unimplemented candidates.
- `CP-MATH-10` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-20, CE-ARCH-22, and CE-ARCH-31.
- `CP-MATH-11` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-12 and CE-ARCH-31.
- `CP-MATH-12` | kind: task | status: superseded | parent: - | objective: Superseded and parked behind CE-ARCH-22/30/31 foundations.
- `CP-MATH-12D` | kind: task | status: superseded | parent: - | objective: Superseded and parked behind CE-ARCH-22/30/31 foundations.
- `CP-MATH-13` | kind: task | status: superseded | parent: - | objective: Superseded by projection plurality and measured registry work in CE-ARCH-22/31.
- `CP-MATH-14` | kind: task | status: superseded | parent: - | objective: Superseded and parked behind CE-ARCH-30/31/50 evidence gates.
- `CP-MATH-15` | kind: task | status: superseded | parent: - | objective: Superseded by explicit NumericPolicy in CE-ARCH-22 and compatibility review CE-ARCH-50.
- `CP-MATH-16` | kind: task | status: superseded | parent: - | objective: Superseded and deferred by CE-ARCH-12/22/50.
- `CP-MATH-17` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-60 after replacements validate.
- `CP-MATH-HARD-DATA` | kind: task | status: superseded | parent: - | objective: Superseded by tiered CE-ARCH-30 corpus with explicit data permission and cost controls.
- `CP-MATH-OPT-NOT-PROMOTED` | kind: task | status: superseded | parent: - | objective: Superseded; historical non-promotion evidence remains terminal.
- `CP-MATH-OPT-PROMOTED` | kind: task | status: superseded | parent: - | objective: Superseded by evidence-gated CE-ARCH-31 candidate promotion.
- `CP-MATH-REAL-DATA` | kind: task | status: superseded | parent: - | objective: Superseded by CE-ARCH-30 falsification corpus.
<!-- todo-orchestrator:v2-managed:end -->
