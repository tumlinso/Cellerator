# Active Objectives

## Summary
Finish the pointer-first forward-neighbor runtime by adding time-window shard routing, lazy multi-GPU residency, and a reusable executor shape for exact forward searches on the real 4x V100 topology.

## Shared Assumptions
- `forward_neighbors` is the actively built neighbor surface.
- The host topology is the real 4-GPU V100 layout with preferred pair order `0,2,1,3`.
- Time-window exact search is the priority; ANN remains a refinement path inside the existing API rather than the primary distributed contract.
- Neighbor code should stay pointer-first and workspace-backed in hot paths.
- Developmental-time implementation work should preserve the current module as a separately buildable baseline while adding a separate CUDA-native sibling and an A/B benchmark harness.
- The user explicitly removed compatibility requirements for the deep CellShard terminology rename, so internal APIs and the .csh5 schema may move from part/num_parts/part_* to partition/num_partitions/partition_* without shims.
- The packfile system is being rewritten into a disposable shard-pack cache for .csh5 with no compatibility shim for the old packfile persistence format.
- The packfile-cache rewrite owns extern/CellShard/src/sharded/{shard_paths,series_h5,sharded_host}* and extern/CellShard/src/sharded/disk* until it lands; the partition-rename stream should avoid those files meanwhile.

## Suggested Skills
- `todo-orchestrator` - Keep the root ledger current for substantial multi-step runtime work.
- `cuda-v100` - Keep shard placement, residency, and merge behavior aligned to the real sm_70 host and pair-local communication posture.
- `cuda-v100` - Own the sparse CUDA boundary and topology-aware multi-GPU path.
- `compare-benchmarks` - Create the fair A/B harness and benchmark contract.
- `todo-orchestrator` - Keep the workstream ledger current while implementation and benchmarking proceed.
- `compare-benchmarks` - Use one shared benchmark contract and concise profiler summaries for packfile vs csh5.
- `todo-orchestrator` - Keep the new CellShard split workstream current as implementation proceeds.
- `cuda-v100` - Keep sparse runtime and layout changes aligned to sm_70 V100 Blocked-ELL-first execution.
- `public-omics-intake` - Drive official GEO and SRA metadata discovery and ranking.
- `quarto-manuscript` - Keep manuscript additions source-first and additive relative to docs/main.qmd.
- `todo-orchestrator` - Keep the ledger current while this multi-step dataset and manuscript track evolves.
- `todo-orchestrator` - Keep the deep rename ledger current while the refactor spans multiple layers.
- `cuda-v100` - Validate any touched hot sparse runtime surfaces on the real sm_70 assumptions if the rename interacts with execution code.
- `todo-orchestrator` - Keep the debug stream legible and pickup ready for future crash work.
- `cuda-v100` - Use only if a CellShard debug issue turns into a GPU runtime or sparse execution investigation.
- `todo-orchestrator` - Keep the new cache-rewrite ledger current while the backend, benchmark, and tests move together.
- `cuda-v100` - Validate that the rewritten blocked-ELL fetch path keeps the shard-oriented sm_70 cost model intact if hot-path regressions show up.
- `native-debugging` - Use for host-side crashes or deadlocks in the new reader-thread and queue code.

## Useful Reference Files
- `AGENTS.md` - Repo-specific engineering rules and path expectations.
- `optimization.md` - Neighbor runtime bottlenecks and V100-oriented priorities.
- `pointer_migration_plan.md` - Phase ordering and hot-path pointer-first policy.
- `src/compute/neighbors/forward_neighbors/fn_query.hh` - Public query and result contracts for the active neighbor surface.
- `src/compute/neighbors/forward_neighbors/fn_index.hh` - Index, shard, and executor surfaces for forward neighbors.
- `src/compute/neighbors/forward_neighbors/forward_neighbors.cu` - Runtime implementation for build, routing, residency, and search.
- `src/models/developmental_time/dT_model.hh` - Current baseline surface to adapt.
- `src/compute/autograd/autograd.hh` - Sparse CUDA building-block surface for the new module.
- `bench/benchmark_mutex.hh` - Mandatory benchmark serialization primitive.
- `extern/CellShard/src/sharded/series_h5.cc` - HDF5 series writer/reader/fetch path being optimized.
- `bench/cellshard_fetch_bench.cu` - Synthetic and real-data fetch comparison benchmark.
- `src/ingest/mtx/mtx_reader.cuh` - MatrixMarket row scan, part counting, and window load helpers for real-data conversion.
- `extern/CellShard/README.md` - Defines the intended CellShard scope and the current Blocked-ELL-first posture.
- `optimization.md` - Documents the repo-wide V100 bottlenecks and Blocked-ELL-first execution priorities.
- `pointer_migration_plan.md` - Constrains hot-path code away from vector-heavy GPU-facing abstractions.
- `docs/main.qmd` - Current manuscript scaffold with explicit dataset and figure placeholders.
- `docs/public_omics/ranked/all_candidates.tsv` - Combined shortlist output for all current public candidate pools.
- `docs/public_omics/benchmark_spec.md` - First benchmark contract for the selected datasets.
- `AGENTS.md` - Repo-wide engineering and performance rules.
- `extern/CellShard/src/sharded/sharded.cuh` - Core sharded matrix type and helper surface that still uses part names.
- `extern/CellShard/src/sharded/sharded_host.cuh` - Host-side sharded builders and fetch helpers that still use part terminology.
- `extern/CellShard/src/sharded/series_h5.cuh` - Series HDF5 public constants and view types.
- `extern/CellShard/src/sharded/series_h5.cc` - Series HDF5 writer, reader, and fetch implementation.
- `extern/CellShard/src/sharded/disk.cuh` - Packfile-style disk schema and loader surface.
- `src/ingest/series/series_ingest.cuh` - Main Cellerator series writer path that consumes CellShard layout names.
- `tests/cellshard_series_h5_test.cu` - Existing HDF5 schema and roundtrip coverage that will need coordinated rename updates.
- `AGENTS.md` - Repo-level guardrails for testing and performance-sensitive edits.
- `extern/CellShard/src/sharded/series_h5.cc` - Current crash backtrace lands in create_series_compressed_h5 here.
- `extern/CellShard/src/sharded/series_h5.cuh` - Public series HDF5 layout declarations and constants.
- `src/workbench/series_workbench.cc` - Caller path that triggers the current crashing test setup.
- `tests/series_workbench_runtime_test.cc` - Current reproducer for the active CellShard crash.
- `AGENTS.md` - Repo-wide engineering rules, benchmark mutex policy, and Blocked-ELL-first guidance.
- `optimization.md` - Current V100 bottlenecks and explicit sparse-runtime priorities.
- `pointer_migration_plan.md` - Do not add new vector-heavy hot-path interfaces while rewriting the cache manager.
- `extern/CellShard/src/sharded/shard_paths.cuh` - Current shard_storage and backend-open abstraction to replace.
- `extern/CellShard/src/sharded/sharded_host.cuh` - Current fetch_part/fetch_shard dispatch that still treats packfile as a primary backend.
- `extern/CellShard/src/sharded/series_h5.cc` - Current direct HDF5 materialization and per-part cache helpers to replace with the new manager.
- `bench/cellshard_fetch_bench.cu` - Current packfile-vs-csh5 benchmark assumptions that must be rewritten to cold-fill vs warm-cache semantics.
- `tests/cellshard_blocked_ell_test.cu` - Legacy packfile roundtrip coverage to convert into cache behavior coverage.

## Workstreams
- `rewrite-cudabiotypes-semantic-contracts` | status: done | owner: unassigned | file: `todos/rewrite-cudabiotypes-semantic-contracts.md` | objective: rewrite cudabiotypes semantic contracts
- `pointer-first-neighbor-runtime` | status: done | owner: unassigned | file: `todos/pointer-first-neighbor-runtime.md` | objective: pointer-first host/device workspaces for forward neighbors
- `distributed-time-window-neighbor-runtime` | status: done | owner: unassigned | file: `todos/distributed-time-window-neighbor-runtime.md` | objective: time-window shard routing and lazy multi-GPU forward-neighbor execution
- `developmental-time-cuda-ab` | status: done | owner: codex | file: `todos/developmental-time-cuda-ab.md` | objective: separate libtorch baseline and pure CUDA developmental-time model with matched outputs and A/B benchmarks
- `make-blocked-ell-csh5-fetch-approach-packfile-performance-as-closely-as-possible-with-shard-packed-payloads-reusable-shard-scratch-and-ssd-only-real-data-fetch-benchmarks` | status: done | owner: unassigned | file: `todos/make-blocked-ell-csh5-fetch-approach-packfile-performance-as-closely-as-possible-with-shard-packed-payloads-reusable-shard-scratch-and-ssd-only-real-data-fetch-benchmarks.md` | objective: make blocked ell csh5 fetch approach packfile performance as closely as possible with shard packed payloads reusable shard scratch and ssd only real data fetch benchmarks
- `cellshard-first-class-build-export-python` | status: in_progress | owner: codex | file: `todos/cellshard-first-class-build-export-python.md` | objective: promote CellShard to a first-class build target with optional export and python package surfaces
- `public-omics-shortlist-manuscript-benchmark-seed` | status: in_progress | owner: codex | file: `todos/public-omics-shortlist-manuscript-benchmark-seed.md` | objective: build first-step public omics shortlist and manuscript benchmark seed artifacts
- `cellshard-core-partition-rename` | status: in_progress | owner: codex | file: `todos/cellshard-core-partition-rename.md` | objective: rename CellShard core storage runtime and schema terminology from part to partition without compatibility shims
- `cellshard-debug-thread` | status: planned | owner: unassigned | file: `todos/cellshard-debug-thread.md` | objective: maintain a pickup ready CellShard debugging thread for crash triage and regression isolation
- `cellshard-packfile-cache-rewrite` | status: in_progress | owner: codex | file: `todos/cellshard-packfile-cache-rewrite.md` | objective: rewrite CellShard packfile into a disposable shard-pack cache for .csh5 with one HDF5 reader thread and explicit predictor overrides

## Global Blockers
_None recorded._

## Progress Notes
- Rewrote the active objective around the forward-neighbor runtime rather than the earlier doc-cleanup pass.
- Added a distributed routing surface to forward neighbors with explicit direction, executor configuration, routing plans, and shard summaries.
- Changed forward-neighbor build ordering to time-first row sorting and contiguous time-range shard assignment so routed shards match real eligibility windows.
- Added lazy shard residency and per-device resident-window control instead of eagerly uploading every shard at build time.
- Added block-level route planning that maps query blocks only to overlapping time shards and then groups those shards by participating device.
- Updated the search hot path so each query block is uploaded once per participating device, exact or ANN passes run only on routed resident shards, and merged results track source shard ids.
- Added `ForwardNeighborSearchExecutor` as a reusable multi-query runtime object over the routed search path.
- Updated `forwardNeighborsCompileTest` to exercise shard summaries, routing plans, executor-backed search, and shard-id result propagation.
- Built `forwardNeighborsCompileTest` successfully and ran the binary successfully in a no-GPU environment via an explicit self-skip path.
- Confirmed the host exposes four Tesla V100-SXM2-16GB devices with fast NVLink pairs `0<->2` and `1<->3`.
- Re-ran `forwardNeighborsCompileTest` on visible GPUs and fixed a brittle shard-count assertion so the test validates contiguous time-range sharding rather than assuming the target shard count is always reached.
- Added `forwardNeighborsBench`, a serialized routed-search benchmark that builds synthetic time-sorted data, drives executor-backed exact search, and emits structured routing plus timing summaries for single-shard, one-pair, and cross-pair scenarios.
- Benchmarked the routed runtime on the V100 host with 262144 rows, 8192 queries, and 64-dim latents: `single-shard` 310.09 ms, `one-pair` 285.93 ms, `cross-pair` 259.59 ms steady-state per search at the current benchmark geometry.
- Profiled the `cross-pair` benchmark case with Nsight Systems and Nsight Compute. The timeline is representative, transfer and allocator overhead are negligible, and `exact_search_kernel_` accounts for effectively all steady-state GPU time.
- Nsight Compute classified `exact_search_kernel_` as a mixed limiter on the sampled cross-pair run, with low measured DRAM and SM saturation and 44 registers per thread, so the next optimization step is structural rather than a transfer or allocator cleanup.
- Added workstream developmental-time-cuda-ab for the matched baseline, CUDA sibling, and benchmark effort.
- Adapted the developmental-time baseline to a scalar-time plus auxiliary-bin contract and added a separate `developmental_time_cuda` sibling with its own sparse-stem training path.
- Added `developmentalTimeCudaRuntimeTest`, `developmentalTimeABBench`, and compare-wrapper scripts; verified the build targets plus a compare-harness run in this environment.
- Implemented blocked-ELL shard-bulk HDF5 reads with reusable shard scratch and materialization helpers; the synthetic benchmark now shows warm csh5 fetch beating packfile on the current SSD run.
- Extended cellShardSeriesH5Test with a blocked-ELL roundtrip exercising prefetch-to-cache, two fetch_part calls within one shard, and fetch_shard on the HDF5 backend.
- SSD benchmark results: synthetic warm csh5 slightly beats packfile; embryo_1_exon and embryo_15_exon warm direct csh5 land within ~25-26% of packfile, while cached csh5 lands within ~1-7%.
- User redirected ingest to require a direct COO->blocked_ell path. Added a direct conversion surface in CellShard and removed the series-ingest CSR bridge/workspace setup so ingest no longer stages through compressed.
- Implemented direct COO->blocked_ell conversion for ingest and removed the series-ingest CSR staging call path; ingest-facing targets build cleanly after the change.
- Profiled packfile vs direct csh5 on one prepared embryo_1_exon artifact directory with implementation-filtered fetch bench runs. Warm timing on the shared 218.7 MB shard contract was ~125 ms for packfile vs ~152 ms for direct csh5.
- perf stat showed direct csh5 using roughly 11 percent more task-clock and cycles with similar cache-miss ratios but lower IPC, pointing to more userspace CPU work rather than a different cache regime.
- strace summary showed packfile dominated by bulk read calls while csh5 introduced many more pread64 calls through the HDF5 stack; traced syscall time alone was not higher for csh5, so the remaining delta is mostly HDF5-side userspace work and materialization overhead rather than raw kernel I/O time.
- Added scripts/build_public_omics_shortlists.py to resolve curated GEO and SRA accessions, apply embryo-specific metadata overrides, and emit reproducible shortlist artifacts under docs/public_omics.
- Generated ranked candidate sets for mouse scRNA, mouse secondary modalities, primate scRNA, and primate secondary modalities, plus a combined all_candidates.tsv summary.
- Added additive manuscript append text, benchmark spec text, and two figure spec files without modifying docs/main.qmd.
- Completed the source-facing partition rename for export, Python, and workbench summaries; the remaining deep storage/runtime/schema rename is split into a dedicated CellShard core workstream.
- Added a dedicated CellShard debug workstream so future crash triage can be picked up independently of the rename work.
- Completed the source-facing partition rename in export, Python, and workbench summaries, which now defines the target vocabulary for the remaining core layers.
- Observed state at handoff: seriesWorkbenchRuntimeTest currently crashes with SIGSEGV, and cuda-gdb shows the fault inside cellshard create_series_compressed_h5.
- Inventory pass found roughly 1877 part-related matches concentrated in extern/CellShard/src/sharded/*, extern/CellShard/src/sharded/disk.*, extern/CellShard/src/sharded/series_h5.*, and extern/CellShard/src/convert/blocked_ell_from_compressed.cuh.
- Focused debug note from the side thread: the current seriesWorkbenchRuntimeTest crash is likely an uninitialized series_layout_view.part_aux in the test fixture, not the public partition rename.
- Patched the failing test fixture to zero-initialize the series HDF5 view structs and set layout.part_aux = nullptr before create_series_compressed_h5.
- Found the same uninitialized compressed-series fixture pattern in tests/cellshard_series_h5_test.cu and patched it to zero-initialize the view structs and set layout.part_aux = nullptr.
- Confirmed the crash fix: zero-initializing the series HDF5 view structs and setting layout.part_aux = nullptr removed the create_series_compressed_h5 segfault in tests/series_workbench_runtime_test.cc.
- Patched the same latent fixture bug in tests/cellshard_series_h5_test.cu so the compressed-series HDF5 tests do not rely on uninitialized view state.
- Validation after the fix: /tmp/cellerator_cellshard_split_build/seriesWorkbenchRuntimeTest returned 0 and /tmp/cellerator_cellshard_split_build/cellShardSeriesH5Test returned 0 when run serially.
- One intermediate parallel rerun produced HDF5 file-lock noise; the real validation path for these tests should stay serial.
- Confirmed the current CellShard standalone build hard-requires CUDA via project(... LANGUAGES CXX CUDA), find_package(CUDAToolkit REQUIRED), and CUDA-contaminated host-side headers.
- Confirmed the no-GPU runtime boundary can stay narrow: series_h5, shard_paths, disk, summary loading, fetch_part, and CSR/export materialization are host-side in behavior, while sharded_device, distributed, bucket, and convert kernels remain CUDA-only.
- Implemented CELLSHARD_ENABLE_CUDA with a host-only CellShard::inspect target, conditional CellShard::runtime export, and conditional CUDAToolkit package dependency so standalone CellShard now configures without nvcc when CUDA is disabled.
- Added src/cuda_compat.cuh plus shared-header cleanup so summary/fetch/materialize/export code compiles as pure C++ in no-GPU mode while the default CUDA build still exposes the existing GPU-facing headers.
- Validated standalone no-GPU configure/build/test and package export with /tmp/cellshard_nogpu_build plus a clean consumer that links CellShard::inspect, then revalidated the default CUDA-enabled standalone export test in /tmp/cellshard_cuda_build.
- Paused overlapping storage-file edits while the dedicated packfile-cache rewrite stream owns shard_paths, sharded_host, series_h5, and disk surfaces.
- Planning is complete: shard-pack cache files, auto fingerprint invalidation, on-demand fill plus preload, and caller-overridable predictor behavior are the locked design.
- Initial repo inspection confirmed that the main old-packfile consumers are bench/cellshard_fetch_bench.cu and tests/cellshard_blocked_ell_test.cu, which makes the behavioral rewrite boundary relatively contained.
- Claimed this idle stream and narrowed the next patch to standalone package-component correctness for no-GPU CellShard installs.
- Patched CellShardConfig.cmake.in to validate requested package components against exported targets, added a package-consumer smoke project, and documented the standalone component contract in extern/CellShard/README.md.

## Next Actions
- Measure one-shard, one-pair, and cross-pair query windows to decide whether pair-local merge needs a stronger explicit path beyond the current device-grouped execution.
- Use the new `forwardNeighborsBench` plus the saved Nsight artifacts to test whether pair-local merge, larger query blocks, or CUDA Graph capture moves `exact_search_kernel_` out of its current mixed limiter regime.
- Add dedicated compile/runtime coverage for `cuvs_sharded_knn` if that surface remains part of the active neighbor roadmap.
- If developmental-time work continues, extend the CUDA sibling from the current single-device hot path to the planned pair-local or 4-GPU path after profiler-backed benchmark review on the target V100 host.
- Build the updated series_h5 path, fix compile breaks, then add blocked-ELL runtime assertions and real-data benchmark conversion.
- Compile the expanded fetch bench with MTX conversion support, then run synthetic and one or two embryo exon SSD comparisons.
- None; implementation and SSD validation are complete unless a deeper profiler pass is requested.
- Compile ingest-facing targets and fix any issues in the new direct COO->blocked_ell converter.
- None unless deeper runtime validation on a GPU-enabled session is requested.
- Build the filtered fetch bench, prepare one embryo exon artifact directory, then run perf and syscall summaries for packfile and direct csh5 separately.
- None unless a deeper HDF5 or Nsight Systems trace is requested.
- Implement the standalone CellShard subproject targets and rewire the root build to link them.
- Get a dataset root from the user, then use the canonical anchor set to build layout plans, manifests, and the first processed-file fetch scope.
- After intake starts, turn the figure specs into rendered dataset-landscape and benchmark-readiness figures.
- Start the deep CellShard core rename in extern/CellShard/src/sharded/*, then propagate the schema rename through Cellerator call sites and tests.
- Use the dedicated debug workstream for the current seriesWorkbenchRuntimeTest crash and any future CellShard runtime triage.
- Run a targeted ripgrep inventory over extern/CellShard/src/sharded and extern/CellShard/src/sharded/disk* to establish the rename order.
- Start the first code patch in the core sharded type and helper headers before moving on to the HDF5 schema keys.
- Read the current crash note, run the reproducer, and inspect the create_series_compressed_h5 inputs before changing code.
- Keep the first debug patch narrow and add a focused assertion or regression test if the root cause is confirmed.
- Rename order should start with sharded.cuh and sharded_host.cuh, then disk.cuh and disk.cu, then series_h5.cuh and series_h5.cc, and only then the dependent Cellerator surfaces and tests.
- Run the native-debugging crash capture for /tmp/cellerator_cellshard_split_build/seriesWorkbenchRuntimeTest and inspect the summary files first.
- Rebuild seriesWorkbenchRuntimeTest and rerun the exact reproducer to confirm the fixture hypothesis.
- Rebuild and run seriesWorkbenchRuntimeTest and cellShardSeriesH5Test serially to confirm the fixture-family fix.
- For the next CellShard issue, run native-debugging/scripts/debug_crash.sh first and read summary.txt before choosing a follow-on tool.
- Keep CellShard runtime-test reruns serial when they touch the same temporary HDF5 files.
- Patch extern/CellShard/CMakeLists.txt, CellShardConfig.cmake.in, and the shared headers to introduce the standalone no-GPU inspect/materialize mode without disturbing the default CUDA runtime path.
- If the broader split continues, rewire the root Cellerator build to consume CellShard::inspect and CellShard::runtime explicitly instead of assuming one monolithic CellShard runtime target.
- Implement the new CellShard packfile-cache manager with one HDF5 reader thread, shard-pack cache files, predictor overrides, and focused fetch-path validation.
- Keep the deep partition-rename workstream off the overlapping sharded storage files until the cache rewrite lands, then resume vocabulary cleanup on top of the new backend.
- Resume the deep rename after the packfile-cache rewrite lands, starting from the rewritten storage backend instead of the old packfile code.
- Patch shard_paths.*, sharded_host.cuh, series_h5.*, and disk.* to establish the new cache-manager data model before touching the benchmark and tests.
- Fork the benchmark/test migration on bench/cellshard_fetch_bench.cu and tests/cellshard_blocked_ell_test.cu once the core cache API shape is stable enough to hand off.
- Patch CellShardConfig.cmake.in to validate requested components against exported targets, then add focused standalone no-GPU package-consumer coverage.
- After the overlapping series_h5 redefinition issue is resolved, rerun standalone extern/CellShard with CELLSHARD_ENABLE_CUDA=OFF and build the new cellShardInspectPackageTest target.

## Done Criteria
- The forward-neighbor surface exposes explicit time-window routing and reusable executor contracts.
- Index shards are contiguous in time and can be summarized and routed without touching latent payloads.
- Search only touches routed shards and can lazily resident-upload them onto participating GPUs.
- The result surface preserves source shard ids through exact and ANN searches.
- The active compile target covers the new routing and executor surfaces.
- Both implementations build together, expose the same outputs, and benchmark under one fair contract.
- Blocked-ELL HDF5 fetch uses shard-bulk reads with reusable shard scratch instead of per-part hyperslab reads.
- cellShardSeriesH5Test covers blocked-ELL HDF5 roundtrip, cache prefetch, repeated part fetch within one shard, and fetch_shard.
- cellShardFetchBench supports configurable artifact roots plus SSD-only real MTX comparisons using embryo_1/exon and embryo_15/exon data under /home/tumlinson/embryo_scratch.
- CellShard configures as a standalone project and the root Cellerator build consumes its targets without behavior regressions.
- Optional export and Python packaging surfaces exist without becoming mandatory dependencies of the default build.
- Series creation emits part hard-limit failures and shard limit warnings for u32-sensitive execution boundaries.
- Repo contains a reproducible first-pass shortlist with machine-readable query specs and ranked outputs.
- Repo contains additive manuscript and figure planning artifacts that do not modify docs/main.qmd.
- The next intake action is explicit and blocked only on dataset-root selection.
- Core CellShard sharded structs, helpers, and fetch surfaces use partition terminology instead of part terminology.
- The csh5 on-disk schema uses num_partitions and partition_* consistently in the primary storage metadata.
- Cellerator builds and the focused CellShard runtime tests pass against the renamed partition schema.
- The remaining active CellShard ledgers are about debugging or new features, not lingering part terminology.
- The current CellShard crash has a stable repro, an identified cause, and a recorded fix or explicit blocker.
- The debug stream contains enough context that another thread can resume the next CellShard triage step without rediscovery.
- Any fixed CellShard runtime crash covered by this stream has a focused regression test or assertion.
- Standalone CellShard configures and builds with CELLSHARD_ENABLE_CUDA=OFF on a machine without nvcc, exposing inspect/materialize/export surfaces but not GPU runtime targets.
- The default standalone CUDA-enabled CellShard build still produces the current runtime targets and preserves the existing V100-oriented hot path behavior.
- CellShard uses .csh5 as the durable source and a shard-pack packfile cache with one HDF5 reader thread plus explicit predictor override and invalidation controls.
- Focused CellShard runtime tests and the fetch benchmark pass against the new cache semantics without depending on the old packfile persistence path.
- CellShard no longer treats packfile as a primary durable storage format; .csh5 is durable and packfile is a disposable shard-pack cache.
- Concurrent worker requests for the same missing shard coalesce onto one HDF5 reader-thread materialization job.
- The predictor can retain or evict shards under a bounded cache budget, but explicit caller commands always override it.
- Focused CellShard tests and benchmark code build and validate the new cold-fill and warm-cache semantics.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize autograd kernels, workbench browse-cache updates, semantic cudaBioTypes cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
