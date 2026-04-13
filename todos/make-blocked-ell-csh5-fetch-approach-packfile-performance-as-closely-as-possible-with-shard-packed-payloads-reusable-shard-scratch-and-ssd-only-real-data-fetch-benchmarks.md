# Current Objective

## Summary
Implement shard-bulk blocked-ELL csh5 fetch, shard scratch reuse, and SSD-only synthetic plus embryo-MTX benchmarks.

## Planning Notes
- Persisted disk optimization is blocked-ELL-only; compressed remains runtime-compatible but is not an on-disk optimization target.
- Benchmark acceptance is SSD-only for now; real-data inputs come from /home/tumlinson/embryo_scratch and converted artifacts live under /tmp or another temporary artifact root.
- Profiling follow-on uses the same fetch bench contract but now isolates implementations and reuses one prepared artifact directory so perf/strace do not include conversion time.

## Assumptions
- Existing blocked-ELL payload datasets are already contiguous by part and shard ranges are contiguous part spans, so shard-packed fetch can be implemented with shard-level offsets plus bulk hyperslab reads instead of a second payload format.

## Suggested Skills
- `todo-orchestrator` - Keep the workstream ledger current while implementation and benchmarking proceed.
- `compare-benchmarks` - Use one shared benchmark contract and concise profiler summaries for packfile vs csh5.

## Useful Reference Files
- `extern/CellShard/src/sharded/series_h5.cc` - HDF5 series writer/reader/fetch path being optimized.
- `bench/cellshard_fetch_bench.cu` - Synthetic and real-data fetch comparison benchmark.
- `src/ingest/mtx/mtx_reader.cuh` - MatrixMarket row scan, part counting, and window load helpers for real-data conversion.

## Plan
- Finish blocked-ELL HDF5 header loading with shard offset tables and codec state.
- Replace blocked-ELL fetch and prefetch paths with one bulk shard read plus reusable shard scratch materialization.
- Extend the fetch benchmark to support configurable artifact roots and real embryo exon MTX conversion under temporary storage.

## Tasks
- [x] Finish blocked-ELL csh5 shard-bulk reader and materializer
- [x] Extend fetch benchmark with SSD-only real embryo MTX cases
- [x] Add blocked-ELL series H5 runtime coverage
- [x] Profile packfile vs direct csh5 on one shared real-data artifact contract

## Blockers
_None recorded yet._

## Progress Notes
- Implemented blocked-ELL shard-bulk HDF5 reads with reusable shard scratch and materialization helpers; the synthetic benchmark now shows warm csh5 fetch beating packfile on the current SSD run.
- Extended cellShardSeriesH5Test with a blocked-ELL roundtrip exercising prefetch-to-cache, two fetch_part calls within one shard, and fetch_shard on the HDF5 backend.
- SSD benchmark results: synthetic warm csh5 slightly beats packfile; embryo_1_exon and embryo_15_exon warm direct csh5 land within ~25-26% of packfile, while cached csh5 lands within ~1-7%.
- User redirected ingest to require a direct COO->blocked_ell path. Added a direct conversion surface in CellShard and removed the series-ingest CSR bridge/workspace setup so ingest no longer stages through compressed.
- Implemented direct COO->blocked_ell conversion for ingest and removed the series-ingest CSR staging call path; ingest-facing targets build cleanly after the change.
- Profiled packfile vs direct csh5 on one prepared embryo_1_exon artifact directory with implementation-filtered fetch bench runs. Warm timing on the shared 218.7 MB shard contract was ~125 ms for packfile vs ~152 ms for direct csh5.
- perf stat showed direct csh5 using roughly 11 percent more task-clock and cycles with similar cache-miss ratios but lower IPC, pointing to more userspace CPU work rather than a different cache regime.
- strace summary showed packfile dominated by bulk read calls while csh5 introduced many more pread64 calls through the HDF5 stack; traced syscall time alone was not higher for csh5, so the remaining delta is mostly HDF5-side userspace work and materialization overhead rather than raw kernel I/O time.

## Next Actions
- Build the updated series_h5 path, fix compile breaks, then add blocked-ELL runtime assertions and real-data benchmark conversion.
- Compile the expanded fetch bench with MTX conversion support, then run synthetic and one or two embryo exon SSD comparisons.
- None; implementation and SSD validation are complete unless a deeper profiler pass is requested.
- Compile ingest-facing targets and fix any issues in the new direct COO->blocked_ell converter.
- None unless deeper runtime validation on a GPU-enabled session is requested.
- Build the filtered fetch bench, prepare one embryo exon artifact directory, then run perf and syscall summaries for packfile and direct csh5 separately.
- None unless a deeper HDF5 or Nsight Systems trace is requested.

## Done Criteria
- Blocked-ELL HDF5 fetch uses shard-bulk reads with reusable shard scratch instead of per-part hyperslab reads.
- cellShardSeriesH5Test covers blocked-ELL HDF5 roundtrip, cache prefetch, repeated part fetch within one shard, and fetch_shard.
- cellShardFetchBench supports configurable artifact roots plus SSD-only real MTX comparisons using embryo_1/exon and embryo_15/exon data under /home/tumlinson/embryo_scratch.
