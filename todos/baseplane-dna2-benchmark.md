---
slug: "baseplane-dna2-benchmark"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-06-05T14:20:13Z"
last_heartbeat_at: "2026-06-05T14:37:28Z"
last_reviewed_at: "2026-06-05T14:20:13Z"
stale_after_days: 3
objective: "Benchmark Baseplane dna2 explicit-width and inline-plane representations comprehensively"
---

# Current Objective

## Summary
Benchmark Baseplane dna2 explicit-width and inline-plane representations comprehensively

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Use Baseplane benchmark mutexes; do not run profiler/benchmark jobs concurrently.
- Native CUDA route is 4x Tesla V100 sm_70; keep benchmark outputs structured and commands exact.
- Add missing inline-plane64 benchmark coverage before running the suite, because current benches only cover packed word64 and warp planes32.

## Planning Notes
_None recorded yet._

## Assumptions
_None recorded yet._

## Suggested Skills
- `cuda` - Native Volta benchmark route for 4x Tesla V100 sm_70 measurement.
- `todo-orchestrator` - Track benchmark code, commands, outputs, and final interpretation.

## Useful Reference Files
- `../Baseplane/bench/seq/bench_dna2.cu` - CUDA scanner benchmark driver.
- `../Baseplane/bench/seq/bench_dna2_cpu.cpp` - CPU sequence primitive benchmark driver.
- `../Baseplane/src/seq/dna2.cu` - CUDA sequence kernels.

## Plan
_None recorded yet._

## Tasks
- [x] Add inline-plane benchmark variants where missing.
- [x] Rebuild CPU and CUDA benchmark targets.
- [x] Run CPU benchmark matrix with exact commands and hardware context.
- [x] Run CUDA single-GPU and all-GPU benchmark matrix with exact commands and hardware context.
- [x] Summarize representation throughput, correctness checksums/hits, and bottleneck interpretation.

## Blockers
_None recorded yet._

## Progress Notes
- Large all-GPU warp_planes32 benchmark exposed an int launch-geometry overflow; patched warp scanner to use a grid-stride loop and capped benchmark warp launches.
- Benchmark result files are in build/benchmark-results/baseplane-dna2-explicit-widths/.
- CPU motif lengths 8/16/31 ran over 1,048,576 chunks, 10 iterations, 80 host threads.
- CUDA single-GPU motif lengths 8/16/31 ran over 67,108,864 bases, 20 iterations on GPU0.
- CUDA all-GPU motif length 16 ran over 268,435,456 bases, 10 iterations across four V100s; large hot-path runs used 1,073,741,824 bases, 20 iterations.
- Observed all-GPU large hot path: packed_word64_shifted_count 1.128T overlapping windows/s; inlplane64_aligned_count 335.8G aligned windows/s / 10.745T aligned bases/s.
- Added fair compare-benchmarks contract for packed_word64_aligned_count vs inlplane64_aligned_count; optimized aligned kernels with 4-word/thread unroll; packed aligned won by 2.11% at 1.073B bases across four V100s.

## Next Actions
- Implement inline-plane64 aligned-window benchmark variants in Baseplane CPU and CUDA benches.
- Rebuild and rerun affected CUDA benchmark matrix after warp benchmark fix.
- Use results to decide whether to add shifted inline-plane window extraction; current inline-plane benchmark only measures aligned 32-base windows.
- If continuing performance work, implement shifted inline-plane extraction and compare it against packed_word64_shifted_count under the overlapping-window contract.

## Done Criteria
- Benchmark output covers packed word64, planes32, inline-plane64, CPU primitive conversion/mask paths, single GPU, and all GPUs where available.
- Results include exact commands and native hardware/toolchain context.
