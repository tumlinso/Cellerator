# CSR posting-union bitmap probe

## Result

Keep the bitmap as an experimental candidate; do not promote it to a general
backend from this evidence. It returns exact stable-ID work sets and is faster
than the full rescan for q=1 through q=1024 on this fixture, but q=1 is already
handled faster by the existing postings path. More importantly, a structural
count/scan/concatenate path beats the bitmap at every unique query size here,
and the full rescan beats the bitmap at q=2048. The measured topology makes
concatenation applicable: source postings are disjoint and ordered by stable
ID, and query terms are sorted and unique. That path is inapplicable to the
duplicate-term control and is not a generic answer for overlapping postings or
arbitrary ID order.

## Workload and protocol

- Frozen retrieval fixture: 1,000,000 directed edges, 20,000 entities,
  32 cells; zero duplicate endpoint pairs; source degree capped at 512
  (p50 29, p90 91, p99 495). Queries have 1, 32, 128, 1,024, or 2,048
  terms per cell; q32 also has a four-duplicate-term case.
- Fixture seed: `20261027`; generator SHA-256
  `a17c73f2f1e9e8d10e9c73abf4599f6a0df21b86812d59d8665cb6ee13eeee89`;
  fixture manifest SHA-256
  `3e433776b818916bb8650b9ef71a09cef5c8dccb6e9b55e39ac9052d5c9a7091`;
  topology SHA-256
  `5ec1f6bbc958ba62112fcd26a88d4912e5812806bfe0617a1e3df7abc7582bc3`.
  The run's source closure also records the later generator-file snapshot
  `887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`;
  this probe consumes the frozen a17 arrays. The fixture owner and root
  confirmed the later snapshot changed ranked/view fields only and retained
  identical topology/query arrays.
- The bitmap traverses only queried CSR postings, atomically sets per-cell
  edge-rank bits, popcounts words, scans counts, and emits stable-ID-sorted
  results. The strong rescan uses query-membership bits and a parallel
  edge-domain count/scan/emit. Concatenation counts sorted unique posting
  lengths, scans, then scatters postings. All methods share exact result
  capacity and output semantics.
- Paired CUDA-event runs on a scheduler-admitted Tesla V100-SXM2-16GB,
  `sm_70`, GPU UUID `GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`, CUDA 12.9;
  one warmup and 21 measured samples. Table values are per-query method
  latency in microseconds, `p50 [p10–p90]`; common synchronous query upload is
  excluded and listed separately below. Setup and resident memory are also
  reported separately. GPU job `69c1bcf8-380c-46a4-a8ed-486d5dc230ca`,
  attempt `439f79ef-a28c-495b-b138-3a77d5b1dfda`, completed successfully with
  verified cleanup.

## Measurements

| Query | Exact candidates | Existing serial postings | Count/scan/concat | CSR bitmap | Full rescan |
|---|---:|---:|---:|---:|---:|
| q1 unique | 704 | 21.5 [20.5–22.5] | 20.5 [19.5–21.5] | 76.8 [75.8–77.8] | 519.2 [515.1–520.2] |
| q32 unique | 57,365 | 31,946.8 [30,073.9–35,218.4] | 26.6 [24.6–29.7] | 88.1 [85.0–92.2] | 477.2 [454.7–524.3] |
| q128 unique | 212,777 | 302,052.3 [302,012.4–302,102.5] | 36.9 [34.8–57.3] | 106.5 [102.4–116.7] | 469.0 [464.9–486.4] |
| q1024 unique | 1,642,344 | omitted | 128.0 [127.0–129.0] | 305.2 [302.1–310.3] | 520.2 [519.2–524.3] |
| q2048 unique | 3,287,262 | omitted | 234.5 [234.5–235.5] | 661.5 [659.5–668.7] | 589.8 [588.8–592.9] |
| q32 duplicate4 | 51,983 | 27,345.9 [27,344.9–27,347.0] | inapplicable | 86.0 [85.0–87.0] | 454.7 [452.6–456.7] |

The existing serial merge was deliberately not timed at q1024 or q2048. Its
recorded `q × eligible-candidate` work proxies are 1.682 billion and 6.732
billion respectively; these are workload descriptors, not latency or
speedup estimates.

Common synchronous query H2D plus synchronization measured 14.5, 24.9, 74.8,
207.5, 338.7, and 32.1 microseconds for q1, q32, q128, q1024, q2048, and
q32-duplicate queries. Adding that common transfer to each method's p50 gives
complete per-call p50s (concat / bitmap / full rescan): 35.0 / 91.3 / 533.7
microseconds at q1; 51.6 / 113.0 / 502.1 at q32; 111.7 / 181.3 / 543.8 at
q128; 335.5 / 512.7 / 727.7 at q1024; and 573.1 / 1,000.2 / 928.5 at q2048.
At q32 with duplicate terms, bitmap is 118.1 microseconds versus 486.8 for
full rescan; concatenation is correctly rejected by its unique-term
precondition.

At the q2048 high-water output capacity (52,599,344 bytes shared output),
reported total resident bytes were 89,302,871 for bitmap, 70,334,807 for
concatenation, 67,207,631 for full rescan, and 71,177,304 for the existing
postings path. Bitmap's extra scratch excluding output was 20,018,703 bytes,
versus 1,050,639 for concatenation and 2,083,471 for full rescan.

One-time runner setup reported 1,653.99 ms total, including 8.21 ms direct
CSR build, 152.56 ms host-posting preparation, 15.52 ms CUDA posting
projection, and 13.58 ms device allocation/upload. Fixture generation and
query-oracle preparation happened separately before the measured job and took
2,860.38 ms and 441.74 ms in the fixture manifest; they are not in the
per-query timings. These setup costs and the large common output buffer make
amortization and capacity policy part of any eventual choice.

## Correctness and evidence limits

The runner required `cpu_oracle_exact` for every measured case. Duplicate
terms produced a deduplicated union; q2048 overflow probes for bitmap,
fullscan, and concatenation reported the exact required count and left output
sentinels unchanged. The independent CPU preflight also checked stable-ID
ordering when IDs are permuted and rejected concatenation when its ordering
preconditions fail. A scheduler-run Compute Sanitizer memcheck on q32 exercised
the candidate and both comparison paths: exit code 0, `ERROR SUMMARY: 0
errors`, job `c2b8bb5c-514b-4c33-8f2f-f82d3ed7fecf`, attempt
`0d4b187b-c2eb-4721-af11-73cdd44bc518`, cleanup verified.

The bitmap experiment used identity physical layout on this one synthetic
topology. It does not establish a crossover for overlapping hyperedge-style
postings, random logical-ID order, different cell batches, or changing
support. No Nsight profile was needed to answer the adoption question because
the complete-query comparisons already reject bitmap versus the eligible
concat baseline for this workload. Retain the probe as evidence for a possible
general fallback where postings overlap and concatenation cannot apply; that
regime needs its own paired benchmark before promotion.

## Source and receipt identity

- Candidate source SHA-256:
  `4a407ff064e1c46cd9cd461e348f24b0e318eaf0dc885a739f87fca2ecd736b8`
- Runner SHA-256:
  `3c0eff4570574624e8b110bfc374e81c1a84414812d987a150c1ce5b7adfe75b`
- Fixture preparation source SHA-256:
  `d46e50838f10f31e841028773206d1b930910b5d7a9dc8114e37138f98cd19f8`
- Benchmark binary SHA-256:
  `75862a08112973d66fc7dd78a3e1bd5544ae776baa15ca370f1047d1a24e92ec`
- Detailed samples and CUDA stage times: `evidence/gpu3_v3/results.jsonl`.
- Device Scheduler timing receipt: `evidence/gpu3_v3/scheduler_terminal.json`;
  captured logs: `evidence/gpu3_v3/scheduler_logs.json`.
- Sanitizer receipt and logs: `evidence/gpu3_memcheck/scheduler_terminal.json`
  and `evidence/gpu3_memcheck/scheduler_logs.json`.
