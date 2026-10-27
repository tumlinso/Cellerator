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

## Pending after computer restart

- Compute Sanitizer memcheck and racecheck were not run because the root paused
  new GPU jobs for the restart. No task GPU job is queued or running; all four
  jobs above are terminal with cleanup verified. No local build command is
  still active.
- At checkpoint time a read-only job-list audit found zero nonterminal jobs
  targeting GPU3, and the exact resource was available with zero visible
  processes and 0% utilization. See `job_records/nonterminal_gpu3_audit.json`
  and `job_records/resource_list_final.json`. Four retained job-get records
  report verified cleanup. The initial benchmark job
  (`5d895583-f52c-49fe-bf8e-a2b8239d1828`) had expired from scheduler history
  and now returns `not_found`; its terminal success and cleanup were observed
  during the run, and its spec/raw benchmark output remain preserved. This
  missing archive record is noted in `job_records/bench_initial_expired.txt`.
- Resume by rebuilding the latest source, rerunning the focused test, then
  running memcheck and racecheck through Device Scheduler on the confirmed
  GPU3 UUID `GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`. Keep the measured
  fallback limits above explicit; do not launch new jobs until the pause is
  lifted.
