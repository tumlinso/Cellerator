# Active Objectives

## Summary
Root coordination ledger for the Cellerator umbrella workspace. Cellerator now
owns the cross-component checkout that previously lived in CellStack.

## Shared Assumptions
- `Cellerator.git` is the umbrella repository for the core genomics GPU stack.
- `components/Baseplane/` owns sequence primitives.
- `components/CellShard/` owns storage, metadata, pack generation, and runtime
  delivery.
- `include/Cellerator/` and `src/foundation/` own only the foundation slice
  needed by components.
- `Archive/current-cellerator/` preserves the previous implementation for later
  subproject classification and is not a default build surface.
- The deprecated CellStack root should remain a wrapper with only the
  `Cellerator/` submodule.

## Suggested Skills
- `todo-orchestrator`: maintain this umbrella ledger for substantial
  cross-component work.
- `cuda`: use for CUDA-sensitive foundation, Baseplane, or CellShard runtime
  work.
- `cuda` - Low-level sequence representation change touches CUDA device helpers and should preserve optional CUDA builds.
- `todo-orchestrator` - Track this multi-file Baseplane API migration through validation.
- `cuda` - Native Volta benchmark route for 4x Tesla V100 sm_70 measurement.
- `todo-orchestrator` - Track benchmark code, commands, outputs, and final interpretation.

## Useful Reference Files
- `AGENTS.md`: umbrella ownership, component boundaries, and git hygiene.
- `components/Baseplane/AGENTS.md`: Baseplane-local implementation rules.
- `components/CellShard/AGENTS.md`: CellShard-local implementation rules.
- `Archive/current-cellerator/`: archived implementation material.
- `components/Baseplane/AGENTS.md` - Baseplane scope, CUDA policy, and style constraints.
- `components/Baseplane/include/Baseplane/seq/dna2.cuh` - Public dna2 API and device-inline helpers.
- `components/Baseplane/tests/seq/test_dna2.cpp` - CPU correctness coverage for representations.
- `components/Baseplane/tests/seq/test_dna2_cuda.cu` - CUDA device helper coverage.
- `components/Baseplane/bench/seq/bench_dna2.cu` - CUDA scanner benchmark driver.
- `components/Baseplane/bench/seq/bench_dna2_cpu.cpp` - CPU sequence primitive benchmark driver.
- `components/Baseplane/src/seq/dna2.cu` - CUDA sequence kernels.

## Workstreams
- `baseplane-dna2-explicit-widths` | status: done | owner: codex | file: `todos/baseplane-dna2-explicit-widths.md` | objective: Implement explicit-width dna2 packed, split-plane, and inline-plane representations in Baseplane
- `baseplane-dna2-benchmark` | status: done | owner: codex | file: `todos/baseplane-dna2-benchmark.md` | objective: Benchmark Baseplane dna2 explicit-width and inline-plane representations comprehensively

## Global Blockers
_None recorded yet._

## Progress Notes
- Completed the umbrella migration from the user-approved restructuring plan.
- Default Cellerator configure/build passed from the deprecated CellStack
  wrapper.
- Focused Baseplane, CellShard, Cellerator foundation, and archive opt-in
  validation passed.
- Added explicit dna2_word32/64, dna2_planes32/64, and dna2_inlplane32/64 representations plus conversion, mismatch, exact-match, and reverse-complement helpers.
- Removed active ambiguous dna2_word and dna2_planes names from Baseplane call sites.
- Validation passed: cmake --build build --target baseplaneDna2Test baseplaneDna2CudaTest -j 4; ./build/components/Baseplane/baseplaneDna2Test; ./build/components/Baseplane/baseplaneDna2CudaTest.
- Large all-GPU warp_planes32 benchmark exposed an int launch-geometry overflow; patched warp scanner to use a grid-stride loop and capped benchmark warp launches.
- Benchmark result files are in build/benchmark-results/baseplane-dna2-explicit-widths/.
- CPU motif lengths 8/16/31 ran over 1,048,576 chunks, 10 iterations, 80 host threads.
- CUDA single-GPU motif lengths 8/16/31 ran over 67,108,864 bases, 20 iterations on GPU0.
- CUDA all-GPU motif length 16 ran over 268,435,456 bases, 10 iterations across four V100s; large hot-path runs used 1,073,741,824 bases, 20 iterations.
- Observed all-GPU large hot path: packed_word64_shifted_count 1.128T overlapping windows/s; inlplane64_aligned_count 335.8G aligned windows/s / 10.745T aligned bases/s.
- Added fair compare-benchmarks contract for packed_word64_aligned_count vs inlplane64_aligned_count; optimized aligned kernels with 4-word/thread unroll; packed aligned won by 2.11% at 1.073B bases across four V100s.

## Next Actions
_None._
- Edit dna2.cuh first, then update implementation and tests until rg finds no active dna2_word/dna2_planes ambiguous names.
- Commit Baseplane submodule changes first, then update the Cellerator umbrella pointer and ledger changes if desired.
- Implement inline-plane64 aligned-window benchmark variants in Baseplane CPU and CUDA benches.
- Rebuild and rerun affected CUDA benchmark matrix after warp benchmark fix.
- Use results to decide whether to add shifted inline-plane window extraction; current inline-plane benchmark only measures aligned 32-base windows.
- If continuing performance work, implement shifted inline-plane extraction and compare it against packed_word64_shifted_count under the overlapping-window contract.

## Done Criteria
- Cellerator contains Baseplane and CellShard under `components/`.
- The previous Cellerator implementation is preserved under
  `Archive/current-cellerator/`.
- Default CMake builds only the umbrella foundation and components.
- CellStack is reduced to a thin `Cellerator/` wrapper.
- No active Baseplane code uses ambiguous dna2_word or dna2_planes names.
- baseplaneDna2Test passes.
- baseplaneDna2CudaTest passes when CUDA is enabled.
- Benchmark output covers packed word64, planes32, inline-plane64, CPU primitive conversion/mask paths, single GPU, and all GPUs where available.
- Results include exact commands and native hardware/toolchain context.
