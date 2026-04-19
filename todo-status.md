# Todo Status

## Summary
Use this file as the quick pickup register for `todos.md` workstreams.
- `ready`: planned work that can be started now.
- `claimed`: currently being written; choose another stream.
- `idle`: unfinished but resumable; safe to pick up.
- `closed`: completed or removed from pickup rotation.

## Workstreams
- `dual-cuda-optimization-modes` | status: in_progress | execution: claimed | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | next: Once the active CellShard storage rewrite lands or stabilizes enough to compile `cellerator_compute_autograd` again, extend the distinct extreme path into sparse autograd and the model CUDA surfaces that depend on it.
- `cellshard-first-stable-release` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-first-stable-release.md` | next: Choose and add the CellShard license file, then re-run the documented release checks if the license text changes packaging metadata or release notes.
- `cellshard-blocked-ell-ingest-runtime` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-blocked-ell-ingest-runtime.md` | next: Optimized-only blocked canonical fetch now works again on fresh real artifacts. Next focus is throughput: keep blocked build on CUDA end-to-end and remove host-heavy optimized shard construction.
- `cellshard-file-surface-rename` | status: done | execution: closed | owner: codex | file: `todos/cellshard-file-surface-rename.md` | next: Deleted the compatibility headers after confirming nothing still included them; standalone and package-consumer validation still pass.
- `cellshard-runtime-service-contract` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-runtime-service-contract.md` | next: Generation-qualified execution-pack cache paths and `.csh5` reader hardening are in; next implement the owner-node coordinator/runtime surface, then add append staging and publish/cutover state.
- `cellshard-csr-file-codec-removal` | status: done | execution: closed | owner: codex | file: `todos/cellshard-csr-file-codec-removal.md` | next: Removed the remaining compressed `.csh5` file-codec implementation and tests; only reserved enum ids remain to avoid renumbering persisted codec/execution values.
- `quantized-blocked-ell-codecs` | status: in_progress | execution: idle | owner: codex | file: `todos/quantized-blocked-ell-codecs.md` | next: The persisted/runtime slice and the first transfer-vs-SpMM benchmark are in; next write the missing quantization-semantics contract, add inspect/runtime summary coverage, and only then prototype a tensor-op-oriented sparse tile path.
- `cellshard-user-metadata-annotations` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-user-metadata-annotations.md` | next: Patch csh5 schema/groups first, then update workbench ingest/preprocess persistence, then export/python cold-fetch APIs and focused tests.
- `cuda-porting-audit` | status: done | execution: closed | owner: codex | file: `todos/cuda-porting-audit.md` | next: Finish the static audit of ingest, preprocess, models, neighbors, trajectory, workbench, and graph surfaces, then write the prioritized CUDA port map.
- `gpu-prototype-ingest-blocked-ell` | status: planned | execution: ready | owner: codex | file: `todos/gpu-prototype-ingest-blocked-ell.md` | next: Start by extracting a benchmarkable cold-materialization slice around dataset ingest blocked builder and optimized shard assembly.
- `gpu-prototype-model-sparse-boundaries` | status: planned | execution: ready | owner: codex | file: `todos/gpu-prototype-model-sparse-boundaries.md` | next: Start with DenseReduce because it has both obvious sparse layout churn and the largest avoidable `to_dense()` boundary in current model code.
- `gpu-prototype-neighbors-trajectory` | status: in_progress | execution: idle | owner: codex | file: `todos/gpu-prototype-neighbors-trajectory.md` | next: Start with forward_neighbors exact and ANN query orchestration and benchmark before committing to GPU trajectory follow-on stages.
- `blocked-ell-optimization-study` | status: in_progress | execution: claimed | owner: codex | file: `todos/blocked-ell-optimization-study.md` | next: Start with generated subsets from GSE147520, a dedicated benchmark target, and three algorithm families: exact DP bucketing, stronger column ordering, and alternating local search.
- `gpu-benchmark-sliced-preprocess-campaign` | status: in_progress | execution: claimed | owner: codex | file: `todos/gpu-benchmark-sliced-preprocess-campaign.md` | next: Build and run the focused dataset/runtime targets plus `realPreprocessBench` to validate the new CellShard sliced device cache and measure the warm-path H2D drop.
- `largest-monolithic-files` | status: done | execution: closed | owner: codex | file: `todos/largest-monolithic-files.md` | next: Review the workstream ledger and pick the next concrete step.
- `split-csh5-translation-units` | status: done | execution: closed | owner: codex | file: `todos/split-csh5-translation-units.md` | next: Optional follow-up only: if the private shared include becomes the next maintenance problem, split `csh5_internal.inc` further by domain without changing the public API.
- `split-workbench-runtime-backend` | status: done | execution: closed | owner: codex | file: `todos/split-workbench-runtime-backend.md` | next: Runtime backend now lives under src/workbench/runtime/ with the public workbench API preserved; optional follow-up is a second-pass split of runtime/pipeline.cu by behavior.
- `cellshard-hierarchy-reset` | status: in_progress | execution: claimed | owner: codex | file: `todos/cellshard-hierarchy-reset.md` | next: Start with the public include/install contract and the io/runtime/export/python hierarchy so downstream includes can be migrated cleanly.
- `cellerator-hierarchy-reset` | status: done | execution: closed | owner: codex | file: `todos/cellerator-hierarchy-reset.md` | next: The remaining legacy tree and stale matrix-era targets were deleted; reopen only for optional second-pass helper cleanup or broader app-surface reorganization.

## Staleness Review
- Fresh: 1
- Aging: 0
- Stale candidates: 0
- Stale: 0
- Superseded: 0

## Cleanup Status
- Cleanup mode is explicit only.
- Safe to call `todo-cleanup`: no, active workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, cellerator-hierarchy-reset.
- Partial cleanup is available via `todo-cleanup --partial`; include `stale` in `--scope` only when explicitly intended.
