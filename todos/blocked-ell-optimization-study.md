---
slug: "blocked-ell-optimization-study"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-16T10:19:56Z"
last_heartbeat_at: "2026-04-16T10:19:56Z"
last_reviewed_at: "2026-04-16T10:19:56Z"
stale_after_days: 14
objective: "Research, prototype, and benchmark native-V100 Blocked-ELL optimization algorithms with generated real-data subsets and a persistent study record"
---

# Current Objective

## Summary
Research, prototype, and benchmark native-V100 Blocked-ELL optimization algorithms with generated real-data subsets and a persistent study record.

## Quick Start
- Why this stream exists: The optimized Blocked-ELL dataset artifacts are materially larger than expected relative to CSR, and the current ingest/runtime stream is already claimed. This stream isolates algorithm research, prototype implementation, benchmarking, and profiling so ordering and bucketing experiments do not destabilize the active ingest-runtime work.
- In scope: deterministic subset generation from `GSE147520_all_cells.h5ad`; a dedicated optimization benchmark target; research-backed ordering and bucketing prototypes; native V100 benchmark/profiler runs; and a concise permanent record of results.
- Out of scope / dependencies: no `.csh5` schema changes, no compatibility preservation, no multi-GPU tuning, and no ingestion-contract rewrite in this stream. Production integration should happen later as a follow-up slice once a winner is clear.
- Required skills: `todo-orchestrator`, `cuda`
- Required references: `AGENTS.md`, `optimization.md`, `src/ingest/dataset/dataset_ingest.cuh`, `extern/CellShard/src/disk/csh5.cc`, `bench/benchmark_mutex.hh`

## Planning Notes
- Primary user concern: final blocked dataset size, not just in-memory padded bytes.
- Working rule: always check live GPU availability first and skip rather than waiting blindly on the benchmark mutex.
- Current prototype route keeps the work bench-first with generated MTX subsets and a dedicated study benchmark target.

## Assumptions
- Single-GPU native V100 only for this pass.
- Derived subsets are generated-only and live under `data/test/generated/blocked_ell_optimization/`.
- The smaller reference corpus for v1 is `data/test/reference/GSE147520_all_cells.h5ad`.

## Suggested Skills
- `todo-orchestrator`
- `cuda`

## Useful Reference Files
- `bench/blocked_ell_study_bench.cu`
- `scripts/blocked_ell_subset_h5ad.py`
- `scripts/run_blocked_ell_study.sh`
- `docs/pipeline/blocked_ell_optimization_study.md`
- `src/ingest/dataset/dataset_ingest.cuh`
- `extern/CellShard/src/disk/csh5.cc`

## Plan
- Generate deterministic fast and representative MTX subsets from the GSE147520 raw sparse matrix.
- Compare four algorithm families in a dedicated benchmark target: baseline, exact-DP, overlap clustering, and bounded local search.
- Track both optimized-shard persisted bytes and execution-payload bytes, then run the CUDA SpMM runtime proxy only on the strongest byte winners.
- Record results and algorithm layouts in `docs/pipeline/blocked_ell_optimization_study.md`.

## Tasks
- [x] Create a dedicated benchmark target `blockedEllStudyBench`.
- [x] Create deterministic subset generation tooling for the GSE147520 reference file.
- [x] Add a GPU-aware run wrapper that checks idle devices before benchmarking/profile.
- [x] Record the first fast-subset byte results for `block_size=16` and `block_size=8`.
- [ ] Finish the representative-subset sweep with tighter search bounds.
- [ ] Run the CUDA SpMM runtime proxy and `nsys` on the strongest candidate.

## Blockers
- The representative-subset all-algorithm sweep is much slower than the fast subset because the current local-search prototype is CPU-bound. It needs tighter pruning before broad representative runs become routine.

## Progress Notes
- Confirmed the smaller reference H5AD shape and raw sparse matrix surface: `68008 x 26587`, `nnz=114,455,556`.
- Confirmed all four V100s were idle before the first benchmark runs.
- Added `bench/blocked_ell_study_bench.cu`, which reports CSR bytes, canonical blocked bytes, optimized-shard bytes, execution payload bytes, segment counts, and an optional CUDA SpMM runtime proxy.
- Added `scripts/blocked_ell_subset_h5ad.py`, which writes deterministic `fast` and `representative` MTX subsets plus a manifest at `bench/real_data/generated/blocked_ell_optimization/manifest.tsv`.
- Added `scripts/run_blocked_ell_study.sh`, which picks an idle GPU, records `nvidia-smi` state, and optionally profiles with the native Nsight wrappers.
- Fast subset result summary:
- `block_size=16`: baseline `4.3441x` CSR, exact-DP `4.3054x`, overlap `3.8802x`, local-search `3.8695x`.
- `block_size=8`: baseline `3.6340x` CSR, exact-DP `3.5895x`, overlap `3.0881x`, local-search `3.0812x`.
- Representative subset result summary at `block_size=8`: baseline `3.0430x` CSR, exact-DP `3.0009x`, overlap `2.6193x`, local-search `2.6174x`.
- Early conclusion: exact rebucketing helps modestly, but column grouping and block size choice dominate the byte result.

## Next Actions
- Bound the representative-subset search so the full sweep stays practical.
- Run the CUDA SpMM proxy on the best byte winner from the fast subset.
- Profile the winner with `profile_nsys.sh` on an idle V100.

## Done Criteria
- Generated subsets, benchmark target, and run wrapper exist and build cleanly.
- At least one real-data subset has recorded byte comparisons across all prototype algorithms.
- The study note captures the algorithm layouts, research lineage, and measured outcomes.
- A clear next production candidate is identified for later ingest-path integration.
