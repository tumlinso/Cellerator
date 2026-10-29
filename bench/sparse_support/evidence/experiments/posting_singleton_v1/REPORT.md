# Singleton reverse-posting query experiment

## Change

For a query row containing one term, the CUDA posting path now derives the
candidate count from the indexed posting offsets and scatters the already
sorted, unique posting with a 128-thread CTA. Queries with zero terms remain
empty; queries with multiple terms retain the existing exact serial merge.
This is a representation-specific path, not a universal sparse-index policy.

## Qualification

- Device: Tesla V100-SXM2-16GB, SM70, CUDA 12.9.86; all work was admitted by
  Device Scheduler on the explicit UUID with exclusive allocation and
  `gpu.quiescence`.
- Comparison: frozen serial-control and candidate binaries, five paired
  AB/BA process rounds per case, 3 warmups, 21 CUDA-event samples per run,
  3 iterations per sample, seed 20261027. No host affinity or clock control.
- Every one of the 30 benchmark invocations passed the independent exact host
  oracle. Control and candidate fixture digests and work counts matched within
  every pair.
- The candidate-linked CUDA worksets test passed (including singleton and
  multi-term merge, duplicate terms, and overflow behavior). Compute Sanitizer
  2025.2.1.0 memcheck reported 0 errors; racecheck reported 0 hazards, errors,
  and warnings on the skewed singleton fixture.
- Scheduler receipts for comparison, sanitizer, and CUDA test attempts are
  retained beside their job specs. Every completed attempt has verified
  cleanup.

## Measured results

Ratios are control time divided by candidate time, so values above 1 favor the
candidate. Per-case ratios below are medians over five paired process rounds;
parenthetical ranges are the observed paired ratios, not confidence intervals.

| Exact fixture | Posting discovery | Posting + bitmap + workset | Candidate posting / candidate rescan | Setup + reuse-16, control → candidate |
|---|---:|---:|---:|---:|
| E=2,176, B=64, ordinary | 67.24 → 35.16 µs (1.91×; 1.86–1.94×) | 157.35 → 124.93 µs (1.26×; 1.25–1.27×) | 1.000× (effectively tied) | 197.77 → 164.88 µs |
| E=2,176, B=64, source skew 0.9 | 1,694 → 40.28 µs (42.1×; 41.35–48.18×) | 3,397.97 → 1,982.46 µs (1.71×; 1.71–1.72×) | 0.966× (candidate posting ~3% faster) | 3,439.86 → 2,024.33 µs |
| E=65,536, B=1, ordinary | 499.71 → 25.94 µs (19.29×; 18.52–19.52×) | 2,078.72 → 1,766.74 µs (1.18×; 1.07–1.24×) | 0.786× (candidate posting ~21% faster) | 3.525 → 3.234 ms |

Cold projection plus transfer was essentially unchanged by the kernel change:
about 0.64 ms for E=2,176 and 23.26 ms for E=65,536 on the candidate. The large
case's one-use setup-plus-pipeline was 25.19 ms for the candidate versus
25.11 ms for the control; 16 uses still did not beat the candidate's
2.247 ms rescan path. The posting index needs sufficiently long reuse to repay
its build and transfer cost at that scale.

The rescan path has no source change. Its paired control/candidate medians were
within 1% on the anchor, effectively tied on the skewed case, and showed a
0.949 control/candidate ratio on E=65,536/B=1 (paired ratios 0.881–1.033).
That spread is a reminder that these five process pairs do not control host or
GPU clocks; the small rescan difference is not attributed to the posting
change.

The full scan remains a valuable explicit control and fallback. In the
candidate binary, ordinary E=2,176/B=64 postings only tie the rescan; source
skew brings postings to near parity; the large steady-state case favors
postings, but cold construction dominates. These fixtures do not establish a
universal dispatch threshold.

## Profile diagnosis

The baseline lineinfo NCU profile showed one active thread per cell and
231,838 global-load instructions in the selected skewed `query_counts_kernel`
launch; source samples concentrated in the serial posting merge/advance loop.
The candidate q=1 count kernel reads posting boundaries instead of walking
posting IDs (1,637 global-load instructions in its selected launch). Its
scatter kernel uses 128 threads to copy the sorted posting directly.

NCU 2025.2.1.0 replay durations were 819 µs for the baseline count launch,
7.4 µs for the candidate count launch, and 10.8 µs for the candidate scatter
launch. These are diagnostic replay values, **not** throughput comparisons;
the unprofiled paired CUDA-event measurements above establish performance.
The candidate scatter still sees global-memory latency, but the first
high-degree posting is copied by a multi-warp CTA rather than traversed twice
by one thread. No Tensor Core arithmetic is involved.

## Limits

The workload is a deterministic synthetic support graph, not a Geneformer or
Karyoplast dataset. The tested query workload has one term per row; multi-term
queries use the unchanged serial merge and have correctness coverage but no
performance optimization here. Five paired rounds expose run-to-run spread,
but are not a formal confidence interval and the host clock/affinity was not
controlled. The benchmark covers indexing, condition bitmap, and workset
construction, not the downstream biological interaction kernel.

## Evidence map

- Production change: `src/compute/indexing/cuda_postings.cu`; exact two-binary
  comparison driver: `scripts/sparse_support/compare_cuda_variants.py`
- Paired raw results: `paired_cuda_v100_20261027.jsonl`
- Comparison job spec and receipt: `paired_cuda_v100_20261027.job.json`,
  `paired_cuda_v100_20261027.scheduler.json`
- Baseline and candidate binary/source/library snapshots: `baseline/`,
  `candidate/`; their `manifest.json` files list SHA-256 identities
- Sanitizer raw output: `sanitizer_v100_20261027_v2.json`; the earlier
  wrapper-only precollection failure and receipt are retained as
  `sanitizer_v100_20261027.json` and `sanitizer_v100_20261027.scheduler.json`
- Candidate NCU reports, source maps, and receipt:
  `../../profiles/ncu_posting_singleton_candidate_v100_20261027/`
- CUDA worksets correctness job and receipt:
  `cuda_worksets_test_v100_20261027.job.json`,
  `cuda_worksets_test_v100_20261027.scheduler.json`
- SHA-256 inventory of the principal source, binary, raw-result, profile, and
  scheduler artifacts: `evidence_manifest.json`
