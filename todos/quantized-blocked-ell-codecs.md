---
slug: "quantized-blocked-ell-codecs"
status: "in_progress"
execution: "idle"
owner: "codex"
created_at: "2026-04-15T13:46:20Z"
last_heartbeat_at: "2026-04-15T15:22:00Z"
last_reviewed_at: "2026-04-15T15:22:00Z"
stale_after_days: 14
objective: "implement a distinct quantized blocked-ell codec family with direct-to-device packed delivery and fused live decode"
---

# Current Objective

## Summary
Implement a separate quantized Blocked-ELL codec stack spanning CellShard codec metadata, a pointer-first packed device view, reusable decode primitives, and a first fused SpMM consumer.

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: the repo now needs a real quantized Blocked-ELL codec family rather than a placeholder quantized-CSR enum or a vague extension of the existing non-quantized Blocked-ELL path.
- In scope: codec-family identifiers, quantized Blocked-ELL metadata contract, packed device-facing view, reusable decode helpers, first fused SpMM consumer, and focused runtime tests.
- Out of scope / dependencies: full persisted pack writer/materializer work is a follow-on slice and depends on the adjacent Blocked-ELL ingest/runtime stream; CUDA mode infrastructure remains owned by the existing dual-mode stream.
- Required skills: `todo-orchestrator` for ledger maintenance and `cuda` for the fused sparse decode path.
- Required references: `AGENTS.md`, `optimization.md`, `extern/CellShard/src/disk/csh5.cuh`, `src/quantized/api.cuh`, `src/compute/autograd/autograd.hh`, and `src/compute/autograd/kernels/base_sparse.cu`.

## Planning Notes
- Treat quantized and non-quantized Blocked-ELL as separate codec families with explicit runtime dispatch rather than fallback forms of the same layout.
- Keep the first consumer path fused: no production unpack-to-float staging kernel unless later benchmarks prove it wins.
- Shape the decode primitive layer so quantized sliced-ELL can reuse it later without baking sliced traversal into this first delivery.

## Assumptions
- The first implementation slice can land the codec contract and fused SpMM path before the full CellShard pack writer exists, as long as the metadata vocabulary and device view are stable.
- Generic CUDA remains the correctness anchor; native and native-extreme specialization can deepen after the first fused path is validated.

## Suggested Skills
- `todo-orchestrator` - Track the codec stream as a standalone resumable workstream that can hand off cleanly to later pack-writer slices.
- `cuda` - Guide the direct-to-device packed decode path and any Volta-specific native-extreme follow-up.

## Useful Reference Files
- `AGENTS.md` - Repo policy for Blocked-ELL-first hot paths and Volta-oriented CUDA work.
- `optimization.md` - Explains why Blocked-ELL SpMM is the right first proving consumer on this host class.
- `extern/CellShard/src/disk/csh5.cuh` - Owns dataset codec family and execution-format metadata definitions.
- `src/quantized/api.cuh` - Current quantized backend umbrella and the natural home for shared decode primitives.
- `src/compute/autograd/autograd.hh` - Public sparse runtime surface for the first fused consumer entrypoint.
- `src/compute/autograd/kernels/base_sparse.cu` - Current custom Blocked-ELL SpMM path and kernel launch boundary.

## Plan
- Add a distinct quantized Blocked-ELL codec-family vocabulary and decode-policy metadata contract.
- Introduce a pointer-first packed quantized Blocked-ELL device view with reusable row/slot decode helpers.
- Implement a first fused quantized Blocked-ELL SpMM path in the autograd runtime.
- Validate the new path with focused runtime tests and then extend the persisted pack writer in a follow-on slice.

## Tasks
- [x] Land the new quantized Blocked-ELL codec-family enums and decode-policy helpers.
- [x] Add the packed quantized Blocked-ELL layout/helper surface under `src/quantized/`.
- [x] Wire a fused quantized Blocked-ELL SpMM entrypoint into the autograd runtime.
- [x] Extend focused tests to cover per-gene-affine and column-scale-row-offset decode inside SpMM.

## Blockers
_None recorded yet._

## Progress Notes
- Started the first implementation slice: new quantized Blocked-ELL codec identifiers, a reusable packed Blocked-ELL quantized helper header, and an initial fused SpMM autograd path are in progress.
- Landed the first working quantized Blocked-ELL slice: `extern/CellShard/src/disk/csh5.cuh` now exposes a distinct `dataset_codec_family_quantized_blocked_ell`, decode-policy helpers live in codec flags, `src/quantized/blocked_ell.cuh` provides a packed device-facing view plus pack/unpack helpers, and autograd now has a fused quantized Blocked-ELL SpMM entrypoint.
- Validated the slice with `cmake --build build -j 4 --target computeAutogradRuntimeTest`, `./build/computeAutogradRuntimeTest`, `cmake --build build -j 4 --target quantizedMatrixTest`, and `./build/quantizedMatrixTest`.
- Extended CellShard with a real `sparse::quantized_blocked_ell` host payload, a distinct quantized Blocked-ELL packfile format, `.csh5` create/append/load/fetch helpers, and shard-pack cache materialization through the existing cached-pack runtime path.
- Added focused validation for the persisted/runtime slice: `cellShardDatasetH5Test` now round-trips quantized Blocked-ELL through `.csh5` plus the cache pack, and `computeAutogradRuntimeTest` now persists a quantized Blocked-ELL partition, reloads it through CellShard, uploads the packed payload to device, and runs fused SpMM without an unpack-to-float staging buffer.
- Validated the persisted/runtime slice with `cmake --build build -j 4 --target cellShardDatasetH5Test`, `./build/cellShardDatasetH5Test`, `cmake --build build -j 4 --target computeAutogradRuntimeTest`, `./build/computeAutogradRuntimeTest`, and `cmake --build build -j 4 --target quantizedMatrixTest && ./build/quantizedMatrixTest`.

## Next Actions
- Add an inspect/runtime summary test that exercises the new quantized Blocked-ELL codec family through `.csh5` metadata and verifies the reported matrix/execution format vocabulary.
- Decide whether CellShard should grow a dedicated device-staging helper for `sparse::quantized_blocked_ell` or keep device upload ownership in Cellerator’s autograd/runtime layer.
- Benchmark the persisted quantized shard-pack path against the existing test-only host packing path once a representative real-data fixture exists.

## Done Criteria
- The repo has a distinct quantized Blocked-ELL codec family and decode-policy vocabulary.
- A packed quantized Blocked-ELL device view exists with reusable decode helpers.
- A fused SpMM path can consume packed quantized Blocked-ELL directly on device with focused test coverage.
