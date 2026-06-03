---
status: done
execution: closed
owner: codex
updated: 2026-05-31
---

# Baseplane Sequence Migration

## Objective

Set up Baseplane as the low-level sequence bit primitive owner and migrate the
Cellerator `dna2` sequence bit representations, CPU/SIMD scaffolding, CUDA
kernels, tests, benches, and docs into Baseplane.

## Quick Start

- Read `AGENTS.md`, `Baseplane/AGENTS.md`, `Cellerator/AGENTS.md`, and this file.
- Work in submodules first: Baseplane implementation, then Cellerator hard cut,
  then root pointer/docs/ledger updates.
- Use `$cuda` for CUDA-sensitive sequence kernels and `$todo-orchestrator` for
  ledger continuity.

## Assumptions

- Baseplane public namespace is `baseplane::seq`; CMake target is `Baseplane::seq`.
- CUDA is preferred and hot-path-first, but Baseplane must configure and build
  scalar CPU primitives with `BASEPLANE_ENABLE_CUDA=OFF`.
- Baseplane v1 does not own FASTA parsing, file formats, genome annotation,
  motif databases, PWM scoring, IUPAC ambiguity, or CSPACK integration.
- Cellerator does not keep compatibility sequence headers or `Cellerator::seq`.

## Tasks

- [x] Scaffold Baseplane CMake, docs, local agent guide, and package target.
- [x] Move `dna2` headers, scalar implementation, CUDA kernels, tests, benches,
  and docs into Baseplane with `baseplane::seq` namespace.
- [x] Validate Baseplane CUDA-preferred and CPU-only builds.
- [x] Remove Cellerator-owned sequence source, headers, tests, benches, docs,
  and CMake targets.
- [x] Validate Cellerator configure/build/tests against sibling Baseplane.
- [x] Commit Baseplane, then Cellerator, then root pointer/docs/ledger updates.

## Progress Notes

- Baseplane CUDA build passed with the local HPC SDK CUDA 12.9 toolchain and
  native V100 `sm_70` default.
- Baseplane `baseplaneDna2Test`, `baseplaneDna2CudaTest`, CPU bench smoke, and
  CUDA shifted-count bench smoke passed.
- CPU-only Baseplane configure/build and `baseplaneDna2Test` passed.
- Cellerator configured with sibling Baseplane and built
  `coreSparseLayoutRuntimeTest`, `quantizedMatrixTest`, and
  `exactSearchRuntimeTest`.
- Cellerator runtime smokes passed for `quantizedMatrixTest`,
  `exactSearchRuntimeTest`, and `coreSparseLayoutRuntimeTest`.
- The requested `computeAutogradRuntimeTest` and `quantizeModelTest` targets are
  not present in the current Cellerator CMake graph for this configured build.

## Next Actions

- No root coordination action remains for the Baseplane sequence migration.

## Done Criteria

- Baseplane owns sequence bit primitives and exports `Baseplane::seq`.
- Cellerator no longer owns sequence bit code or public sequence headers.
- Root docs and ledgers show Baseplane as the sequence owner.
- Requested Baseplane and Cellerator validation commands are recorded.

