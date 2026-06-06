---
slug: "sequence-bits-dna2"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-05-04T12:18:57Z"
last_heartbeat_at: "2026-05-04T13:05:00Z"
last_reviewed_at: "2026-05-04T13:05:00Z"
stale_after_days: 3
objective: "Implement Cellerator SequenceBits dna2 packed-word and bit-plane primitive."
---

# Current Objective

## Summary
Add Cellerator's first narrow GPU-native regulatory sequence primitive. The public API exposes packed `uint64_t` storage words and 32-base hi/lo bitplanes, with reverse-complement, mismatch helpers, proof kernels, focused tests, a primitive benchmark, and design docs.

## Quick Start
- Why this stream exists: establish `dna2` as the first SequenceBits primitive for CUDA regulatory sequence kernels.
- In scope: `include/Cellerator/seq/dna2.cuh`, `src/seq/dna2.cu`, CPU/CUDA tests under `tests/seq`, primitive benchmark under `bench/seq`, docs under `docs/SEQUENCE_BITS.md`, and CMake target wiring.
- Out of scope / dependencies: no FASTA parser, PWM scoring, IUPAC ambiguity, minimizers, suffix arrays, motif database, whole-genome indexing, or CSPACK integration.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: `AGENTS.md`, root CellStack `AGENTS.md`, user SequenceBits spec, `CMakeLists.txt`.

## Planning Notes
- Keep the new surface separate from the currently dirty `include/Cellerator/core/sequence/` port material.
- Public active masks are one bit per base. Internal packed-field masks stay hidden.
- Correctness is the first target. The scan kernels are proof kernels, not the final high-throughput scanner.

## Assumptions
- The new API belongs under `cellerator::seq` as requested and does not need to be included by the top-level `Cellerator.hh` in this first version.
- The benchmark target can be wired into the existing root CMake target list without running it until the test targets pass.
- Native CUDA route is regular custom-kernel ballot/popcount logic; this is not Tensor Core eligible.

## Suggested Skills
- `cuda`: CUDA host/device helpers, warp ballot encoding, proof kernels, and validation.
- `todo-orchestrator`: maintain this resumable ledger while implementing and testing.

## Useful Reference Files
- `AGENTS.md`
- `CMakeLists.txt`
- `include/Cellerator/core/sequence/base.cuh`
- `tests/core_sparse_layout_runtime_test.cu`

## Plan
1. Add the dna2 public header with host/device representations and helpers.
2. Add CUDA source wrappers and proof kernels.
3. Add CPU-only tests with reference implementations.
4. Add CUDA tests comparing device helpers, warp ballot, and scan kernels against CPU references.
5. Add the primitive benchmark and CMake targets.
6. Add `docs/SEQUENCE_BITS.md`.
7. Build and run focused tests; run benchmark only after tests pass.

## Tasks
- [x] Add ledger and implementation plan.
- [x] Implement dna2 header/source.
- [x] Add CPU and CUDA tests.
- [x] Add benchmark and docs.
- [x] Wire CMake targets.
- [x] Build and run focused validation.

## Blockers
_None recorded yet._

## Progress Notes
- Workstream claimed on 2026-05-04T12:18:57Z.
- Added `include/Cellerator/seq/dna2.cuh`, `src/seq/dna2.cu`, CPU/CUDA tests, primitive benchmark, docs, and CMake targets. Next step is focused build/test validation.
- Validation passed:
  - `cmake -S . -B build`
  - `cmake --build build --target sequenceDna2Test sequenceDna2CudaTest sequenceDna2Bench -j 4`
  - `./build/sequenceDna2Test`
  - `./build/sequenceDna2CudaTest`
  - `./build/sequenceDna2Bench 1048576 16 1 10`
- Added deterministic random sequence test helpers and comparison-friendly benchmark mode after the initial implementation:
  - `tests/seq/dna2_test_helpers.hh`
  - `./build/sequenceDna2Bench <n> <motif_len> <max_mismatches> <iterations> <packed_word64|warp_planes32|both> <seed>`
- Ran the long random comparison matrix under compare/repo benchmark mutexes. Artifacts are under `build/seq_dna2_compare/20260504T123300Z`; Nsight Systems profiles were collected for the 64M-base motif-16 representative case. Nsight Compute was attempted but this Nsight Compute build reported that profiling is not supported on the V100 device.
- Finished the SIMD backend before CUDA-vs-SIMD comparison:
  - Replaced the Highway ASCII pack/unpack stubs with real 32-base SIMD mask extraction/materialization.
  - Kept scalar fallback for partial chunks and `CELLERATOR_ENABLE_HIGHWAY=OFF`.
  - Rebuilt and ran Highway-enabled `sequenceDna2Test` and `sequenceDna2CpuBench`.
  - Rebuilt and ran scalar-only `build-scalar` `sequenceDna2Test` and `sequenceDna2CpuBench`.

## Next Actions
_None for this workstream._

## Done Criteria
- `dna2.cuh` exposes packed-word and hi/lo plane representations.
- CPU and CUDA tests pass.
- Warp ballot encoding matches CPU pack/unpack.
- Packed-word and bit-plane mismatch counts agree.
- Reverse complement works for lengths 1 through 32 without garbage leakage.
- Simple motif scan kernels agree with CPU reference.
- `docs/SEQUENCE_BITS.md` explains the two-representation design and non-goals.
