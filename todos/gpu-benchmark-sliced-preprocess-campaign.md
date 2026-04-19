---
slug: "gpu-benchmark-sliced-preprocess-campaign"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-16T10:23:08Z"
last_heartbeat_at: "2026-04-16T10:50:01Z"
last_reviewed_at: "2026-04-16T10:50:01Z"
stale_after_days: 14
objective: "Profile native V100 sliced-first ingest and preprocess, identify why GPU utilization is low, and rank GPU-offload priorities"
---

# Current Objective

## Summary
Native V100 profiling campaign for sliced-first ingest and preprocess. Warm-path benchmarking now avoids re-inspecting and replanning the H5AD when a reusable sliced artifact already exists. The measured dominant limiter remains repeated host-to-device upload and host-side orchestration around sliced execution segments, not saturated preprocess kernels.

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Load skills: cuda, todo-orchestrator.
- Primary harness: bench/real_preprocess_bench.cu on GPU 1 in single-device mode first.
- Collect timings-only runs before Nsight Systems, then one targeted Nsight Compute pass after the hot path is identified.

## Planning Notes
_None recorded yet._

## Assumptions
- Official campaign path is sliced-only; blocked compatibility is out of scope for first-pass diagnosis.
- Single-GPU isolate on GPU 1 comes before multi-GPU runs so host starvation is not obscured by coordination overhead.

## Suggested Skills
- `cuda` - Use native V100 profiling and optimization workflow for benchmark design, Nsight interpretation, and sparse CUDA tuning.
- `todo-orchestrator` - Track benchmark matrix, profiler artifacts, bottleneck classification, and next actions in a persistent workstream ledger.

## Useful Reference Files
- `optimization.md` - Repository performance priorities and expected bottlenecks.
- `docs/public_omics/benchmark_spec.md` - Benchmark output contract and phase expectations.
- `bench/real_preprocess_bench.cu` - Primary real-data sliced-first benchmark harness.
- `todos/gpu-prototype-ingest-blocked-ell.md` - Recent prototype ingest/layout findings and artifact-size context.

## Plan
- Run timings-only matrix on prototype and full smaller H5AD for cold and reused sliced-first paths.
- Capture Nsight Systems traces for representative small, large-transfer, and large-compute runs on GPU 1.
- Capture one Nsight Compute report for the representative sliced preprocess hot kernel after Nsight Systems identifies the compute window.
- Summarize measured dominant bottlenecks and rank next GPU-offload candidates.

## Tasks
- [x] Run prototype sliced-only timings on GPU 1 and capture warm/cold deltas.
- [~] Run full smaller H5AD sliced-only timings on GPU 1 and capture cold/warm deltas.
- [x] Capture Nsight Systems traces for representative prototype and full-dataset runs.
- [x] Capture targeted Nsight Compute metrics for the dominant sliced preprocess kernel.
- [x] Write ranked bottleneck analysis and next-port order from measured evidence.

## Blockers
- Fresh cold profiling of the current full direct sliced-ingest path is blocked by a correctness failure on data/test/reference/GSE147520_all_cells.h5ad: realPreprocessBench reports "sliced ingest conversion failed" and "verify: written header does not match the planned layout" when materializing a new full-size sliced artifact.

## Progress Notes
- Prototype plain timings on GPU 1: sliced analyze cold 241.564 ms and warm 57.495 ms on /tmp/cellerator_proto_8192.h5ad.
- Prototype Nsight Systems cold trace: materialize_sliced 3.0069 s (73.7%), load_or_generate 0.8299 s (20.3%), steady_state_compute.sliced 0.2415 s (5.9%). GPU kernels totaled about 12.85 ms across 8 accumulate launches plus 2.37 ms for direct CSR->sliced emit.
- Full smaller H5AD warm timings on GPU 1 from reused sliced artifact: sliced analyze mean 800.739 ms over 10 repeats; 200 ms nvidia-smi sampling showed avg GPU util 14.65%, p95 40%, max 43%, nonzero only 50% of samples.
- Full smaller H5AD Nsight Systems warm trace: load_or_generate 8.4796 s (87.2%), steady_state_compute.sliced 1.2411 s (12.8%), materialize_sliced 6.97 ms. CUDA API time in the warm run was dominated by cudaMemcpy 178.3 ms over 329 calls and cudaStreamSynchronize 45.8 ms over 28 calls.
- Warm full-run memcopy summary: 175 H2D copies totaling 952.1 MB (median size effectively tiny, avg 5.44 MB, max 61.1 MB) and 161 D2H copies totaling 3.39 MB, confirming many small transfers around the preprocess loop.
- Direct Nsight Compute on the dominant accumulate_gene_metrics kernel: prototype grid (4,1,1), full grid (13,1,1), block (256,1,1), 22 registers/thread, 0 shared memory. Full kernel DRAM throughput was 2.48% of peak and SM throughput 2.18% of peak; prototype kernel DRAM throughput averaged 0.86% and SM throughput 0.60%. This is underfilled launch geometry, not a saturated kernel.
- Benchmark front-end fix landed in bench/real_preprocess_bench.cu: when reuse_artifacts is true and the sliced input already exists, the warm path skips inspect_source_entries() and plan_dataset_ingest() instead of redoing H5AD inspection/planning every invocation.
- Post-fix warm full run on GPU 1 with the production sliced execution-segment path restored: sliced analyze mean 782.031 ms over 10 repeats.
- Final clean warm Nsight Systems trace after the benchmark fix and with the execution-segment preprocess path restored: steady_state_compute.sliced 1.275 s (99.4%), materialize_sliced 7.53 ms, no residual load_or_generate bucket in the warm path.
- Final warm trace still shows the main limiter is upload/orchestration: 175 H2D copies totaling 952.1 MB and 329 total cudaMemcpy calls; CUDA API time is dominated by cudaMemcpy 186.3 ms and cudaStreamSynchronize 45.8 ms, while total GPU kernel time is only about 46.7 ms.
- A prototype canonical-partition preprocess path was tested and then reverted. It improved device fill for the dominant accumulate kernel from grid (13,1,1) to (52,1,1) and raised DRAM throughput to 10.42% of peak, but it regressed steady-state wall time because it increased bytes moved and pack decode cost in the hot loop.
- Started the next optimization slice: added a CellShard-owned sliced execution device cache keyed by source path, device, and runtime-service generation so single-GPU warm preprocess can reuse host remap metadata plus uploaded sliced segments instead of open-coding fetch-upload-release on every pass.
- Wired single-GPU sliced preprocess to the new CellShard cache with an explicit fallback knob in `preprocess_config`, added direct CellShard cache reuse/invalidation coverage, and added a workbench correctness check that cached and uncached sliced analysis return identical metrics on the runtime fixture.
- Built `cellShardDatasetH5Test`, `datasetWorkbenchRuntimeTest`, and `realPreprocessBench`; ran `./build/cellShardDatasetH5Test` and `./build/datasetWorkbenchRuntimeTest` successfully after updating the stale H5AD CSC cache-warm branch in the runtime harness to follow the current sliced-vs-blocked layout.

## Next Actions
- Build and run the focused dataset/runtime targets plus `realPreprocessBench` to confirm the new cache compiles cleanly, preserves correctness, and actually cuts warm-path H2D traffic on the reused sliced artifact.
- If warm-path transfer still dominates after partition-level reuse, decide whether the next slice is shard-level staging or a broader CellShard runtime cache surface for multi-GPU preprocess and adjacent consumers.

## Done Criteria
- Representative timing and profiler artifacts exist for prototype and real sliced-first runs.
- Bottleneck class is identified with evidence rather than intuition.
- Workstream ledger records the recommended optimization order.
