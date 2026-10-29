# Prepared bitmap retrieval probe

## Decision

Keep the row-CTA bitmap as an experimental, workload-specific candidate. Do not
promote it into the production indexing API from this evidence. The flat global
scan variant is rejected for the tested shapes. Existing sorted postings remain
the safer general path: the bitmap row-CTA result is faster on two small-edge
cases but materially slower on the large-edge single-query and high-skew cases,
and its scratch footprint is larger in every tested case.

The result is deliberately narrower than a full lifecycle comparison. It
measures complete device-resident retrieval queries after prepared structures
exist. One-shot setup and repeated reuse timings in the raw records are single
observations and are too noisy to establish amortized wins or a dispatch rule.

## Method

The baseline is the existing exact CUDA reverse-postings query API. The
candidate prepares term-to-edge-ordinal bitmaps, ORs a query's term bitmaps,
counts set bits, and emits stable logical edge IDs and ordinals. The flat path
uses a global word scan and separate output emission. The row-CTA path assigns
one CTA per query row and performs row-local prefixing and compaction.

All paths were compared on identical frozen inputs regenerated from the official
2026-10-27 indexing fixture. The CPU scalar oracle checks sorted, unique
logical IDs/ordinals. Exact required counts and no-partial-write overflow checks
are part of each bitmap path's validation. The irregular-query checks also
cover duplicate terms, an empty row, an empty query set, and overflow. The paired
run used 3 warmups, 21 CUDA-event samples, 5 queries per sample, and rotated
candidate order on a Tesla V100-SXM2-16GB (SM70, UUID
`6c1cac7f-a360-0aef-ba98-2828bfd1db1a`), CUDA 12.9.

## Paired device-resident query results

Times are p50 microseconds; ranges are p10–p90 across 21 samples. Candidate
counts are total exact output items across the query rows.

| Case | Outputs | Postings | Flat bitmap | Row-CTA bitmap |
|---|---:|---:|---:|---:|
| E2176 / B64 anchor | 1,067 | 22.938 (22.733–23.552) | 38.707 (38.502–39.731) | **15.155 (15.155–15.565)** |
| E65536 / B1 | 563 | **13.722 (13.722–15.770)** | 16.794 (16.589–22.118) | 23.962 (23.757–24.166) |
| E2176 / B64, source skew 0.9 | 33,355 | **30.925 (30.925–31.334)** | 51.814 (51.405–54.477) | 39.322 (39.117–40.960) |
| E2176 / B1 low degree | 11 | 13.312 (13.107–14.541) | 19.251 (19.046–21.094) | **9.011 (8.806–9.830)** |

The row-CTA path is 1.51x faster than postings for the anchor and 1.48x faster
for B1 low degree, but 1.75x slower at E65536/B1 and 1.27x slower under source
skew 0.9. The flat bitmap path loses on all four. These data support a possible
small-edge/low-degree specialization, not a universal bitmap advantage.

## Resident memory and setup limits

Reported standalone bytes include that algorithm's index and workspace/output
capacity (`edges × query rows`); common query buffers are excluded. The
benchmark allocated the variants together, so these figures are per-variant
accounting, not peak process VRAM.

| Case | Postings bytes | Flat bitmap bytes | Row-CTA bytes |
|---|---:|---:|---:|
| E2176 / B64 anchor | 4,586,536 | 4,826,911 | 4,757,536 |
| E65536 / B1 | 5,261,464 | 8,979,295 | 8,945,776 |
| E2176 / B64, source skew 0.9 | 4,584,208 | 4,826,911 | 4,757,536 |
| E2176 / B1 low degree | 192,664 | 298,975 | 297,136 |

The row-CTA scratch penalty is about 3.7% on the two E2176/B64 cases, 69.8%
for E65536/B1, and 54.2% for B1 low degree (about 102 KiB absolute in the last
case). Cold setup includes host projection, CUDA allocation, copies, and common
inputs, but is recorded once. Allocation/first-use noise was large for the
large-edge case, and the reuse-16 estimates were also single measurements.
Those setup and amortization figures are retained in the raw manifest but do
not support a production promotion or break-even claim.

## Correctness and execution evidence

The paired raw records report `scalar_oracle_exact_and_overflow_pass` for all
four fixtures. Fixture digests are `0x82a4635a3cff33dd`,
`0x801e5a5369679941`, `0xf82d50df3b097939`, and `0x68fb8ab9fffb0dc7` in table
order. A first admitted attempt failed in the irregular-query test harness
because its cursor allocation used the number of rows instead of the number of
query terms; the diagnosis and failed receipt are retained at
`evidence/attempt_9b1c1085-c0bf-4eea-a568-45029a03c499.json`. The corrected run
passed the same exactness checks. A scheduled Compute Sanitizer memcheck is
recorded separately in `evidence/memcheck_v100_20261027.json`; it covered the
four-case harness, including bitmap overflow probes, and reported 0 errors.
This is memcheck evidence only; no racecheck claim is made. The JSON helper's
`started_unix_ns` is stamped after the subprocess exits, so use the scheduler
receipt for execution timing.

Paired-run manifest: `evidence/paired_v3/manifest.json`. Raw measurements:
`evidence/paired_v3/results.jsonl`. Scheduler receipt and cleanup:
`evidence/paired_v3/scheduler_job.json` (job
`abcbe645-23c6-4ed7-a024-5da22ad0de62`, cleanup verified). Source identities
are recorded in the manifest; the benchmark binary SHA-256 is
`77096a74fd0d5cdc4da3012b9fa9410c7d5e6d9b9c257c295945ab026355d228`.

## Scope

This probe establishes exact retrieval and a paired query-time comparison for
four synthetic shapes on one V100. It does not establish broad biological
workload coverage, setup break-even, multi-GPU portability, race freedom, or a
production dispatch policy. No production source was changed for this probe.
