# Ordered-disjoint postings fast path: restart checkpoint

## Current implementation

The reverse-posting query path retains the existing singleton direct copy and
general exact k-way merge. It adds a concatenation specialization when query
terms are strictly ordered and unique and their posting slices are strictly
increasing and disjoint by stable edge ID. Projection construction already
emits each posting slice sorted and unique. Small query batches prove the
precondition while counting; larger batches use parallel term lookup followed
by a per-row CUB block scan that builds exact destination prefixes. Scatter
uses one CTA per query term, so a small cell batch can still expose many CTAs.
Rows that fail the proof use the exact merge. Capacity is checked before any
output row is written.

The public cursor scratch record now includes a row-relative output prefix and
row mode. Callers allocate it through the public `cursor_v1` type; no caller
should rely on its prior byte layout.

## Validation completed before pause

- Separate `sm_70` build configured at
  `/tmp/cellerator_ordered_postings_build` and target
  `cellerator_sparse_support_cuda_worksets` compiled successfully.
- GPU3 correctness jobs succeeded with exit code 0 and verified cleanup:
  - `ed83d79f-d1d1-41be-8a81-7132078956ff` (initial fastpath)
  - `428d7f4d-1903-4c30-af33-e6bcf297b706` (parallel row scan)
- The focused test covers exact ordered concatenation, unsorted/duplicate/
  overlapping fallback rows, missing singleton terms, stable IDs, and exact
  capacity overflow with unchanged output sentinels.
- A 21-sample same-fixture GPU3 run succeeded with CPU oracle equality for
  q1, q32, q128, q32 duplicate, q1024, and q2048. Run:
  `20235fbe-9a23-43c9-86f9-591c19bf0a32`; cleanup verified. The persistent
  result, manifest, spec/get/logs, and harness snapshot are in
  `bench/sparse_support/evidence/experiments/ordered_postings_fastpath_checkpoint/`.
- q2048 required a temporary copy of the probe harness with only its old
  `q_per_cell <= 128` postings-measurement guard changed to unconditional.
  The old serial q2048 implementation was not timed. Result was exact CPU
  oracle equality, exact overflow/no-partial-write, 21 samples, and verified
  cleanup (`7ae33c76-83e8-4913-8a81-41b9ffd0154c`). The patched harness,
  temporary CMake file, spec/get/logs, and source/binary hashes are persisted
  in the same checkpoint directory.

Median resident query times, in microseconds:

| Case | Postings fast path | Count/scan/concat prototype | Full scan |
|---|---:|---:|---:|
| q1 unique | 22.53 | 21.50 | 513.02 |
| q32 unique | 44.03 | 28.67 | 524.26 |
| q128 unique | 56.32 | 37.89 | 532.48 |
| q32 duplicate4 | 24,476.67 | not applicable | 457.73 |
| q2048 unique | 250.88 | 250.88 | 662.53 |

The q1/q32/q128/q1024/q2048 sample record is in
`bench/sparse_support/evidence/experiments/ordered_postings_fastpath_checkpoint/results/profile_parallel_results.jsonl`;
the q2048 record is in
`bench/sparse_support/evidence/experiments/ordered_postings_fastpath_checkpoint/job_records/q2048.logs.json`.
q1024 candidate postings remained
omitted by the existing probe guard, while concat/full-scan measured 135.17 /
590.85 microseconds. The q32/128 results show a useful reduction from the
previous serial merge and the q2048 result matches the concat prototype at
measurement resolution. Small-query fallback with duplicate terms remains
expensive.

Production source identities at pause:

- `src/compute/indexing/cuda_postings.cu`: `3a6df1882e7feb318aa44b2fc4ba66c42a619f8afa7b452612c1a84fd0f14381`
- `include/Cellerator/compute/indexing/cuda_postings.cuh`: `55fb929f0ce96427253bc2a4132624b503d1f13ce315ee2642dc17cbd0db1bc2`
- `tests/sparse_support/cuda_worksets_test.cu`: `6f8f37668c242465cec88c5d5433822d8a2a3f21a0c5424169d997f51e688ed8`
- Current probe candidate binary: `5e220a8f0bed868cc184ffb00ee1a94624dcfaa0ad70a3f0877105d3ab4e52d3`
- q2048 harness binary: `912f7365bf3af18aacf49ea57ae9ea5049d0014b350fcb193720901c63daed11`

## Resumed validation

After the user resumed work, the latest source was rebuilt in the separate
directory `/tmp/cellerator_ordered_postings_resume_build` with CUDA 12.9.86 and
`CMAKE_CUDA_ARCHITECTURES=70`; target `cellerator_sparse_support_cuda_worksets`
compiled successfully. The focused test passed on GPU3 with exit code 0:
`e6531eec-0397-47e7-a228-ac53832805fb`.

The focused binary also passed all requested Compute Sanitizer checks, each
with exit code 0 and `cleanup_verified: true`:

- memcheck: `cf4b66ab-948c-412b-aeea-933fc9e34f20`, 0 errors
- racecheck: `9b288b3c-c592-4106-ac44-0039a376bee0`, 0 hazards, 0 errors,
  and 0 warnings
- synccheck: `dfc428d3-abe0-4595-be6b-ee4bf6b93f5b`, 0 errors

Their scheduler specs, full `job get` records, and logs are retained under
`bench/sparse_support/evidence/experiments/ordered_postings_fastpath_checkpoint/`.
`job_records/resume_final_gpu3_audit.json` records a final live audit showing
the exact GPU3 UUID available, 0% observed utilization, and no nonterminal jobs
targeting it. Every job launched for this resumed validation is terminal and
cleanup verified. The test binary and current source hashes are in
`resume_source_identities.sha256`.

The earlier performance evidence remains scoped to its recorded fixtures. In
particular, the ordered concatenation precondition is deliberately strict;
duplicate or unsorted query terms and overlapping/non-monotonic posting ranges
fall back to exact k-way merge, where the duplicate-heavy q32 case was much
slower than full scan. The q2048 speed comparison used the concat prototype;
the old serial merge was not timed at that size. Sanitizers validate the focused
test paths, not every possible input size or long-running workload.

After those GPU jobs, a final API review found a theoretical uint64 aggregate
count overflow for shapes where `cell_count * index.item_count` is not
representable. The query now rejects such nonempty shapes before scheduling
kernels; this conservative bound is documented in the public header. Its
host-only preflight test and updated hashes are recorded in
`bench/sparse_support/evidence/experiments/ordered_postings_fastpath_checkpoint/overflow_bound_host_preflight.txt`
and `overflow_bound_source_identities.sha256`. The updated target rebuilt and
the no-allocation preflight test passed locally; the full GPU suite and
sanitizers were not rerun on this final source version because GPU3 was handed
to the indexed-values worker.
