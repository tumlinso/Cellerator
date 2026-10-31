# V100 quantization validation and optimization

## Contract and baseline

Validate the low-level CUDA state quantizer in `include/Cellerator/modeling/cuda/quantization.hh` and `src/modeling/cuda/quantization.cu` on Tesla V100 (`sm_70`). Its logical input is row-major `[row, physical_feature]`; `feature_map` associates each physical feature with a sorted profile codebook. Encoding chooses nearest FP32 codebook level (lower ordinal wins ties), packs fixed-width codes (1/2/4/8 bits) into little-endian 32-bit words, and optionally records original FP32 values for exact-mask entries. Decoding reverses codes and applies exact overrides. FP16 entrypoints convert values at the API boundaries. Sparse edits atomically replace a code while retaining neighboring fields in a word; duplicate edit indices are forbidden.

Primary baseline: independent CPU scalar oracle using the public codebooks and exact nearest-level/tie rule. The existing implementation is the performance baseline. Compare resident encode, decode without overrides, decode with empty/sparse/dense overrides, and sparse update with distinct indices. Include CUDA event timing and a strongest available vectorized Torch or CPU-side packing baseline only where the measured operation boundary and transfers are matched. Report setup/transfers separately from resident device times; this is state packing, so W4A16 matrix decode timings are related background rather than a directly equivalent baseline.

## Main risks and unknowns

- Device profile validation and all stream-order prerequisites need explicit coverage.
- Concurrent sparse edits can target neighboring fields in one word; CAS must preserve the union of updates.
- Partial final words and feature-map permutations are easy sources of indexing errors.
- Exact overrides use an atomic sparse sidecar and launch decode application over allocated capacity; sparse density may make this cost visible.
- Nonfinite data and malformed profiles signal asynchronous status bits; tests must inspect status after stream completion and never trust flagged outputs.
- The default source map uses one thread per packed word for encoding, a binary search per element, and one thread per decoded element. Its performance bottleneck and any optimization benefit are unknown until measured.
- Direct shell telemetry is unavailable; root has the working Device Scheduler route and coordinates all admitted V100 launches.

## Candidates and first experiment

Retain the simple current implementation as baseline. First run adversarial correctness with all bit widths, FP32 and FP16, nonuniform codebook sizes, physical/profile feature permutation, non-multiple row and total extents, exact-mask override, edits sharing words, and invalid/nonfinite status. Then time standalone operations across small launch-bound shapes and large streaming shapes, with override density 0%, sparse, and full. Candidate A is capacity-aware host-provided override launch count or a compacting design only if the current sparse apply stage is measured as material. Candidate B is cooperative lane ownership for encode only if profiling shows its per-word serial binary searches/load pattern is limiting; compare conversion and complete encode+decode cost, not isolated encode alone. Do not add a format or universal dispatch rule.

Initial matched decode microbenchmark is `decode_bench.cu`. It measures nine median resident CUDA-event samples for 1K, 1M, and 4M values, comparing no sidecar against an allocated empty sidecar, eight overrides, and a full sidecar. Timing excludes setup and transfers by design and isolates launch scheduling plus device work. This comparison decides whether a sidecar-grid candidate deserves implementation; it is not a full encode-to-decode lifecycle claim.

## Validation and acceptance

Build the focused harness for only `sm_70`. For each operation compare output bytes/codes or decoded values against the independent oracle, exact override identity and values, untouched padding, and status flags. Use explicit exact/tolerant comparisons for FP16 output after reference round-to-nearest conversion. Run admitted Compute Sanitizer memcheck, racecheck, and synccheck after ordinary GPU tests. A candidate is retained only if it matches the oracle and shows repeatable complete-path improvement at a declared shape/regime without regressing a required supported case. User requested validation and optimization; no performance target or production workload distribution was supplied. No GPU job may launch until Device Scheduler grants a UUID-selected exclusive resource.

Compile the timer independently, without changing the project's shared CMake build:

```bash
nvcc -std=c++17 -arch=sm_70 --compiler-bindir=/usr/bin/g++-12 -I include \
  planning/v100-framework-2026-10-29/quantization/decode_bench.cu \
  src/modeling/cuda/quantization.cu -lcudart -o /tmp/cellerator-quantization-decode-bench
```

## Resource limit

One isolated SM70 compiler job (at most 1 build worker); V100 execution, sanitizer, and timing only through root-coordinated scheduler admission. No source edits to shared root CMake/docs. Evidence and any cases stay under this directory.
