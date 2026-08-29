# CE-PTR before-migration baseline

`baseline_summary.json` is the source-controlled CE-PTR-01 measurement record.
It captures focused representatives of the highest-risk pre-migration owners:
the CPU packing optimizer, the V100 candidate-discovery and merge-score stages,
and geometry gating construction/replay.

The benchmarks acquire `bench/benchmark_mutex.hh`. CPU runs use the repository
global lock because the current guard has no CPU-only mode; GPU runs use the
device-specific lock where the benchmark supplies a device ordinal. The
candidate Nsight Systems trace reports the whole warmup-plus-three-repeat
process, so API, transfer, allocation, synchronization, and launch counts are
process totals rather than per-repeat estimates. Nsight Compute replay timing
is profiling overhead and is not used as end-to-end latency.

The build directory and profiler databases are intentionally ignored. Rebuild
the evidence with:

```text
cmake -S . -B bench/ce_ptr/baseline/build -DCMAKE_BUILD_TYPE=Release -DCELLERATOR_CELLSHARD_SOURCE_DIR=/home/tumlinson/CellShard -DBASEPLANE_SOURCE_DIR=/home/tumlinson/Baseplane -DCELLSHARD_BUILD_EXPORT=OFF
cmake --build bench/ce_ptr/baseline/build -j 20 --target cellPackOptimizerBench geneCandidateDiscoveryBench cellPackMergeCostBench cellPackGatingBench
```

Exact run and profiler commands are stored in the JSON. The configured
`CELLSHARD_BUILD_EXPORT=OFF` avoids a known test-manifest assumption that a
CellShard checkout is physically adjacent to the Cellerator worktree; it does
not alter any benchmark target used here.

Current limitations are explicit in the record. CUPTI reports aggregate CUDA
allocation call counts but not aggregate requested allocation bytes; the
benchmarks independently report fixed, CUB, staged, and accounted peak bytes.
CPU allocation counts come from glibc `libmemusage` and are recorded separately
from uninstrumented latency. Gating does not expose allocation/transfer counters
and therefore remains latency/correctness evidence, not a sealed-execution
claim. CE-PTR consumer lanes must add post-migration paired evidence rather than
interpreting an unavailable field as zero.
