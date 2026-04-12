# Current Objective

## Summary
Add real-data benchmark manifests, replay targets, and reporting for compressed vs Blocked-ELL SpMM on representative embryo slices.

## Planning Notes
- Synthetic block-sparse benchmarks already exist, but the real decision needs embryo-derived replay with preserved row skew and feature popularity.
- The benchmark must be profiler-friendly, avoid large temp files, and clean up artifacts aggressively.

## Assumptions
_None recorded yet._

## Suggested Skills
- `cuda-v100` - Benchmark standardization, large-data stress design, and real-data sparse replay on V100.

## Useful Reference Files
- `bench/compute_autograd_bench.cu` - Existing synthetic CSR and Blocked-ELL SpMM benchmark target.
- `/home/tumlinson/.agents/skills/cuda-v100/references/benchmark-real-data.md` - Real-data benchmark contract.

## Plan
- Create a repo-local manifest describing selected embryo exon/intron replay inputs and slice policy.
- Add a real-data benchmark mode that times steady-state SpMM separately from repack cost.
- Integrate profiler-friendly outputs and cleanup policy for Nsight artifacts.

## Tasks
- [ ] Define representative embryo subset and slice policy for deep benchmarking.
- [ ] Implement benchmark manifest and real-data replay target.
- [ ] Capture benchmark summaries and append findings to optimization.md.

## Blockers
_None recorded yet._

## Progress Notes
- Measured real-data compressed vs Blocked-ELL SpMM on embryo_1 exon/intron, embryo_10 exon, and embryo_18 intron under the repository benchmark mutex.
- Real-data replay required block-aligned row shards and padded dense RHS slabs for cuSPARSE Blocked-ELL; steady-state Blocked-ELL wins ranged from about 1.75x to 4.54x.
- Re-ran the same replay after the native Blocked-ELL persistence rewrite and saw the same result class, which rules out the new storage path as a hidden steady-state regression.

## Next Actions
- Inspect the current benchmark target and choose whether to extend it or add a dedicated real-data replay bench.
- Keep the manifest and benchmark target as the promotion gate for future layout policy changes.

## Done Criteria
- Representative real-data replay results exist for compressed and Blocked-ELL pair-local SpMM and are summarized in optimization.md.
