# Current Objective

## Summary
Own the hot SpMM kernels, pair-local multi-GPU execution model, and profiler interpretation for V100 sm_70.

## Planning Notes
- This is the dedicated -v100 node for the project.
- Primary steady-state target is pair-local execution on 0<->2 and 1<->3, with 4-GPU hierarchical replay as a second stage.
- $cuda-v100 workstream owns benchmark serialization and must respect the repository benchmark mutex for all GPU profiling and timing runs.

## Assumptions
_None recorded yet._

## Suggested Skills
- `cuda-v100` - Primary owner for V100 SpMM, topology, Tensor Core routing, and profiler interpretation.

## Useful Reference Files
- `src/compute/autograd/kernels/base_sparse.cu` - Current CSR and Blocked-ELL SpMM kernels and cuSPARSE-backed path.
- `src/compute/autograd/kernels/dist_sparse.cu` - Current distributed reference execution helpers.

## Plan
- Benchmark compressed and Blocked-ELL SpMM on real shard shapes and representative RHS widths.
- Tune pair-local replay first, then add hierarchical 4-GPU replay only where it wins.
- Use Nsight Systems and Nsight Compute summaries to classify memory-bound vs Tensor-Core-eligible cases.

## Tasks
- [ ] Inspect existing distributed SpMM surfaces and identify missing pair-local replay support for real-data benchmarking.
- [ ] Add or adapt V100 pair-local and hierarchical execution entrypoints as needed by the benchmark.
- [ ] Profile hot kernels and summarize the limiter class in optimization.md.

## Blockers
_None recorded yet._

## Progress Notes
- Wired the shared benchmark mutex into benchmark binaries so concurrent workers cannot overlap GPU measurements.
- Corrected the real-data Blocked-ELL replay path for cuSPARSE constraints: rows and descriptor cols must be block-aligned, and RHS uploads must be padded to the promoted block width.
- The shared benchmark mutex now honors CUDA_V100_BENCHMARK_MUTEX_PATH so repo-local benches and cuda-v100 wrapper scripts serialize on the same host-global lock.
- Refreshed the real-data replay after the native Blocked-ELL persistence rewrite; the steady-state speedup range remained about 1.8x to 4.4x, so the storage rewrite did not regress the hot `SpMM` path.

## Next Actions
- Review current distributed execution APIs and benchmark hooks before editing.
- Build the updated benchmark targets and fix any compile issues before running real-data replay on the V100 fleet.
- If Tensor Core counter proof is needed later, profile the real blocked-ell replay target rather than CSR SpMV.

## Done Criteria
- V100 pair-local and hierarchical SpMM replay paths are benchmarked and their bottlenecks are documented.
