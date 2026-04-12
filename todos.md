# Active Objectives

## Summary
Concurrent optimization of CellShard ingest planning, real-data sparse layout benchmarking, and V100 pair-local SpMM execution.

## Shared Assumptions
- Target hardware is 4x Tesla V100 16 GB with fast NVLink pairs 0<->2 and 1<->3.
- Logical cells x genes identity must remain intact across the native Blocked-ELL layout and any fallback compressed views.
- Disk space is limited, so benchmark and profiler artifacts must be kept compact and cleaned up unless retained as named results.
- All benchmark and profiler runs must acquire the repository benchmark mutex so concurrent workers do not skew GPU measurements.

## Suggested Skills
- `todo-orchestrator` - Root ledger and concurrent workstream coordination for substantial multi-step repo work.
- `cuda-v100` - V100-specific benchmark design, Tensor Core routing, sparse layout, and pair-local multi-GPU execution.
- `cuda-v100` - Sparse layout and pair-local distributed planning on V100 topology.
- `cuda-v100` - Benchmark standardization, large-data stress design, and real-data sparse replay on V100.
- `cuda-v100` - Primary owner for V100 SpMM, topology, Tensor Core routing, and profiler interpretation.
- `cuda-v100` - Guide execution-layout persistence decisions from measured V100 performance.
- `cuda-v100` - Sparse phase boundary and layout advice for omics workflows on V100.

## Useful Reference Files
- `optimization.md` - Repository-level optimization findings and baseline sparse autograd observations.
- `pointer_migration_plan.md` - Pointer-first migration rules for hot paths and GPU-facing interfaces.
- `AGENTS.md` - Repo-specific performance and workflow requirements.
- `src/apps/series_workbench.cc` - Current plan_series_ingest implementation and CSR-byte estimation.
- `src/ingest/series/series_partition.cuh` - Current greedy byte/nnz partition helpers.
- `bench/compute_autograd_bench.cu` - Existing synthetic CSR and Blocked-ELL SpMM benchmark target.
- `/home/tumlinson/.agents/skills/cuda-v100/references/benchmark-real-data.md` - Real-data benchmark contract.
- `src/compute/autograd/kernels/base_sparse.cu` - Current CSR and Blocked-ELL SpMM kernels and cuSPARSE-backed path.
- `src/compute/autograd/kernels/dist_sparse.cu` - Current distributed reference execution helpers.
- `extern/CellShard/src/disk/matrix.cuh` - Existing disk format codes and packed-byte helpers.
- `extern/CellShard/src/sharded/sharded.cuh` - Shard metadata and part-byte helpers for compressed and Blocked-ELL.
- `src/apps/series_workbench_cuda.cu` - Preprocess and browse-cache surfaces that still assume compressed shards.
- `optimization.md` - Current guidance that compressed remains the general sparse path unless repeated SpMM justifies promotion.

## Workstreams
- `planner-layout` | status: planned | owner: unassigned | file: `todos/planner-layout.md` | objective: planner layout
- `real-bench` | status: done | owner: unassigned | file: `todos/real-bench.md` | objective: real bench
- `cuda-v100-spmm` | status: done | owner: unassigned | file: `todos/cuda-v100-spmm.md` | objective: cuda v100 spmm
- `disk-layout` | status: in_progress | owner: codex | file: `todos/disk-layout.md` | objective: Persist execution-format metadata and promoted shard layouts on disk
- `preprocess-guardrails` | status: in_progress | owner: codex | file: `todos/preprocess-guardrails.md` | objective: Retain fallback compressed preprocessing and indexing guardrails where Blocked-ELL is not yet native
- `quantize-autograd` | status: done | owner: codex | file: `todos/quantize-autograd.md` | objective: Move quantizer reconstruction and range gradients into fused sparse autograd kernels

## Global Blockers
_None recorded yet._

## Progress Notes
- Added the repository benchmark mutex rule to AGENTS.md and started wiring benchmark binaries through a shared file lock.
- Wired the shared benchmark mutex into benchmark binaries so concurrent workers cannot overlap GPU measurements.
- Started a dedicated quantize-autograd workstream to move feature offset/scale quantizer gradients into `src/compute/autograd` without disturbing the active planner and benchmark streams.
- Finished the quantize-autograd implementation with fused CSR and Blocked-ELL reconstruction/range kernels, a CUDA sparse CSR quantizer training path, and passing runtime/model tests on the V100s.
- Measured real-data compressed vs Blocked-ELL SpMM on embryo_1 exon/intron, embryo_10 exon, and embryo_18 intron under the repository benchmark mutex.
- Real-data replay required block-aligned row shards and padded dense RHS slabs for cuSPARSE Blocked-ELL; steady-state Blocked-ELL wins ranged from about 1.75x to 4.54x.
- Corrected the real-data Blocked-ELL replay path for cuSPARSE constraints: rows and descriptor cols must be block-aligned, and RHS uploads must be padded to the promoted block width.
- The shared benchmark mutex now honors CUDA_V100_BENCHMARK_MUTEX_PATH so repo-local benches and cuda-v100 wrapper scripts serialize on the same host-global lock.
- Planner metadata now records execution-format choice, Blocked-ELL geometry, fill ratio, execution bytes, and preferred pair; the remaining gap is that the series ingest payload writer still lays out parts by the older byte-only compressed planner.
- Added native Blocked-ELL raw packfile roundtrip support in CellShard disk/matrix plus test coverage in cellShardBlockedEllTest; the generic sharded packfile path is no longer compressed-only.
- Added native Blocked-ELL HDF5 series support in CellShard: create/load/fetch/prefetch now handle Blocked-ELL parts and cache them separately from legacy CSR payloads.
- Rewrote the series ingest writer to emit Blocked-ELL HDF5 payloads directly from MTX windows instead of writing compressed series parts.
- Ported the workbench browse-cache builder and post-write verifier to native Blocked-ELL headers and device views so converted series files no longer reopen through `/payload/standard_csr`.

## Next Actions
- Create or resume a workstream ledger under `todos/` for the next substantial task.
- Populate detailed workstream ledgers and keep ownership boundaries disjoint.
- Inspect planner structs in series_workbench headers and the downstream conversion path before editing.
- Inspect the current benchmark target and choose whether to extend it or add a dedicated real-data replay bench.
- Review current distributed execution APIs and benchmark hooks before editing.
- Review current CellShard disk serialization and sharded metadata before editing.
- Inspect preprocess-facing CellShard and workbench code before editing.
- Finish wiring the benchmark mutex into the active benchmark targets before running GPU measurements.
- Build the updated benchmark targets and fix any compile issues before running real-data replay on the V100 fleet.
- Keep the manifest and benchmark target as the promotion gate for future layout policy changes.
- If Tensor Core counter proof is needed later, profile the real blocked-ell replay target rather than CSR SpMV.
- Finish auditing preprocess-facing paths that still hard-require compressed parts and either port them to Blocked-ELL or mark them as deliberate guardrail fallbacks.
- Refresh the real-data SpMM replay after the native Blocked-ELL persistence rewrite and fold any deltas into optimization.md.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.
- Root and per-workstream ledgers stay synchronized and sufficient for another worker to resume without rediscovering context.
- Planner emits execution-aware shard decisions consumable by downstream conversion and benchmark paths.
- Representative real-data replay results exist for compressed and Blocked-ELL pair-local SpMM and are summarized in optimization.md.
- V100 pair-local and hierarchical SpMM replay paths are benchmarked and their bottlenecks are documented.
- Planner-selected execution format and geometry round-trip through disk metadata.
- Compressed-path preprocessing and logical indexing remain correct and explicitly guarded after planner changes.
