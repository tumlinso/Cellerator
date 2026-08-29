# CE-PTR final acceptance

Date: 2026-08-29 UTC

Integrated input commit: `08ae74f9ff5201cc8139cc7053bf82245f42f33f`

This record is direct final-tree evidence. Quantitative deltas recorded by the
individual CE-PTR lanes remain historical evidence and are not represented as
reruns unless a command appears below.

## Environment

- host: 80 logical CPUs, two NUMA nodes, 64 GiB host memory;
- accelerator: four Tesla V100-SXM2-16GB, compute capability 7.0;
- topology: GPU0-GPU2 and GPU1-GPU3 have NV6 links; same-NUMA peers otherwise
  cross a PCIe host bridge, and cross-NUMA peers cross the system interconnect;
- driver: 580.173.02;
- CUDA/NVCC: 12.9.86;
- CMake: 3.28.3;
- GCC: 13.3.0;
- Clang: 18.1.3;
- build: Release, automatically selected `sm_70`;
- resource isolation: CE-PTR-15 held the workflow
  `cuda-benchmark-mutex`; the optimizer benchmark additionally reported its
  `/tmp/cuda_v100_benchmark.lock` acquisition.

## Source and build gates

```sh
python3 scripts/check_no_inappropriate_core_stl.py --strict-stale
cmake -S . -B build-ce-ptr-final -DCMAKE_BUILD_TYPE=Release
cmake --build build-ce-ptr-final -j "$(nproc)"
```

All commands passed. The final strict inventory contains 169 controlled
spellings in 27 files: 165 vectors, two priority queues, and two shared
pointers. The baseline was 236 spellings in 36 files. No map, set, unordered
map, or unordered set owner remains. The reconciled allowlist deletes zero-count
entries and lowers live ceilings; no ceiling is raised and no family or path is
newly exempted.

## Semantic, persistence, and runtime gates

The acceptance runner executed every discovered executable whose name ended in
`Test` or `RuntimeTest` in the fresh build tree, with a 180-second per-test
timeout. It ran 82 executables with zero failures. The set covers:

- domain, hierarchy, registry, structure-epoch, value-generation, and order
  identity;
- CPI1/CPK1/CPE2/FMP1/CTP1 construction, validation, persistence, relocation,
  upload, rebind, and replay;
- geometry, optimizer, statistical validation, sampling, candidate discovery,
  graph/trajectory, exact-search, forward, projection, and training behavior;
- prepared execution session, workspace, readiness, concurrency, graph-capture,
  allocation, and opaque-artifact contracts;
- Baseplane integration and CellShard storage/transport boundaries.

Representative direct outputs included one CPE2 upload and one hot read,
zero-allocation opaque-artifact binding, two launch bindings over one prepared
session, exact merge-cost CPU/CUDA parity, deterministic CPEXEC01 replay, and
exact candidate recall against the synthetic exhaustive referee.

## CPU evidence

```sh
/usr/bin/time -v ./build-ce-ptr-final/cellPackOptimizerTest
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  ./build-ce-ptr-final/cellPackOptimizerTest
./build-ce-ptr-final/cellPackOptimizerBench
```

The focused optimizer test passed with a 3,200 KiB maximum RSS. The perf smoke
run recorded 4,082,428 cycles, 6,118,092 instructions (1.50 instructions per
cycle), 56,432 cache references, 369 cache misses (0.65 percent), 1,423,846
branches, and 66,321 branch misses (4.66 percent). This small correctness
fixture is not a throughput benchmark and is not used to claim NUMA scaling.

The mutex-protected Release optimizer benchmark used 5,000 features, 4,096
sampled rows, 20,000 evaluator rows, 320,000 evaluator nonzeros, block width 8,
and row-group width 128. It reduced the exact objective from 320,000 to 301,340
in 1,432.77 ms, with 8,925,258 peak additional optimizer bytes and 5,120,000
evaluator-workspace bytes. Its evaluator route remains recorded as
`deferred_cub_sm70`; this run is CPU optimizer evidence, not a GPU speedup claim.

## CUDA sanitizer and profiler gates

The generic HPC SDK wrappers target an uninstalled CUDA 13.1 component, so
acceptance invoked the installed CUDA 12.9 tools explicitly:

```sh
CS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer
$CS --tool memcheck --error-exitcode 99 ./build-ce-ptr-final/celleratorSessionMemoryTest
$CS --tool memcheck --error-exitcode 99 ./build-ce-ptr-final/cellPackExecutionImageV2DeviceTest
$CS --tool memcheck --error-exitcode 99 ./build-ce-ptr-final/geneCandidateDiscoveryRuntimeTest
$CS --tool racecheck --error-exitcode 99 ./build-ce-ptr-final/geneCandidateDiscoveryRuntimeTest

NCU=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/ncu
$NCU --set basic --target-processes all --csv \
  ./build-ce-ptr-final/exactSearchRuntimeTest
```

Compute Sanitizer 2025.2.1 reported zero memcheck errors for all three tests and
zero racecheck hazards for the resident candidate pipeline. Nsight Compute
2025.2.1 completed and collected launch, register, occupancy, cache, and
throughput sections for the exact-search launch family. The tiny correctness
fixture intentionally launches one or two blocks for several kernels; its low
achieved occupancy is not promoted as a representative end-to-end performance
claim. The routed sliced K=1 kernel used 40 registers per thread and 24 bytes of
static shared memory plus an 8,192-byte shared-memory configuration, with no
profiler failure.

## Compiler and source recovery

A Clang 18 Release configure succeeded, and pure C++ identity-registry and
optimizer targets built and passed with `-j "$(nproc)"`. A mixed Clang/NVCC
runtime target failed at final link because the HPC SDK CUDA archive was not
recognized by the Clang link step; GCC/NVCC is the supported full-build pair and
passed the complete tree.

The C++ context compiler's read-only `where` query resolved the prepared
optimizer workspace contract to `src/geometry/optimizer.cc:625`, and a
5,000-token performance slice completed with `sufficient=1`. Its full refresh
was stopped after it spent more than two minutes merging records without
output. Status remains stale/incomplete because two dataset translation units
hit Clang/GCC intrinsic-header parse incompatibilities. Canonical source,
builds, and tests—not the partial index—remain authoritative.

## Bounded limitations

- The strict lexical gate does not control `unique_ptr` or custom buffer clones.
  `host_buffer` and trajectory batch compatibility helpers remain live and are
  not allocation-free prepared-path evidence.
- Legacy `runtime/device_buffer.cuh` retains two shared-pointer spellings,
  allocation/free, blocking copy, and device synchronization under an exact
  compatibility ceiling. New prepared ownership may not use it.
- The resident support-to-candidate lane retains four scalar control
  synchronizations; no bulk intermediate ownership or allocation-sizing round
  trip was reintroduced.
- CE-PTR-03 and the small CE-PTR-10 fixture made no speedup claim. CE-PTR-07 and
  CE-PTR-08 are CPU-only migrations. Historical lane measurements retain their
  original shapes and qualification.
- Historical todo audit discrepancies—legacy interface path prefixes and
  completed tasks without structured gate rows—were not rewritten. Direct
  source/build/test evidence above is the final acceptance authority.

These limits preserve live compatibility infrastructure and the broader
transitional architecture. They do not weaken the CE-PTR source gate, identity
contracts, repository boundaries, or V100-first prepared execution policy.
