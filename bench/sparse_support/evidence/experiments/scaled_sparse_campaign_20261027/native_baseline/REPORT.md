# Scaled sparse-support native baseline

This is the frozen pre-ordered-disjoint CUDA baseline, run on the admitted V100/SM70 GPU2 resource. It compares exact full-rescan discovery with reverse-posting discovery, then includes the same prepared-condition bitmap and workset construction in both complete paths. Each timing is 21 CUDA-event samples with 3 iterations per sample; intervals below are observed p10–p90, not confidence intervals.

## Results

| Case | Graph / batch | Query | Candidate union | Discovery: rescan / postings (ms) | Complete: rescan / postings (ms) | Full gate / workset (ms) | Requested device bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| small_v3 | E=96, N=64, B=8 | 24 terms/cell | 244 | 0.026 [0.026, 0.027] / 0.718 [0.717, 0.719] | 0.139 [0.139, 0.143] / 0.830 [0.830, 0.833] | 0.066 [0.066, 0.067] / 0.048 [0.048, 0.049] | 98,407 |
| large_ranked_v3 | E=262,144, N=4,096, B=32 | 24 terms/cell | 46,704 | 3.172 [2.952, 3.173] / 26.211 [26.201, 26.291] | 9.586 [9.584, 9.590] / 32.100 [32.094, 32.108] | 3.914 [3.914, 3.914] / 1.982 [1.981, 1.982] | 308,109,199 |
| large_events_v3 | E=1,280,000, N=20,000, B=8 | 24 terms/cell | 11,812 | 3.425 [3.415, 3.428] / 21.816 [21.809, 21.822] | 27.860 [27.855, 27.862] / 46.338 [46.331, 46.343] | 22.815 [22.812, 22.819] / 1.494 [1.493, 1.494] | 473,889,167 |

The posting path is slower than rescan in all three pre-fastpath cases. On the scaled ranked fixture, q24 postings take 26.211 ms versus 3.172 ms for discovery; complete paths are 32.100 ms versus 9.586 ms. On large-events, q24 postings take 21.816 ms versus 3.425 ms; complete paths are 46.338 ms versus 27.860 ms. These are evidence against the current serial k-way merge at these shapes, not against reverse postings or the later ordered-disjoint fastpath.

For small-v3, q24 postings also lose: discovery is 0.718 ms versus 0.026 ms, complete path 0.830 ms versus 0.139 ms. At this tiny graph, the measured posting path has much higher overhead than scanning all 96 edges.

## Conditional state and setup

| Case | Local dirty edges | Shared-latent dirty edges | Dirty local update (ms) | Dirty latent update (ms) | Index + projection + root-map setup amortized at 1 / 16 uses (ms) |
|---|---:|---:|---:|---:|---:|
| small_v3 | 9 | 72 | 0.067 [0.067, 0.068] | 0.067 [0.067, 0.069] | 0.850 / 0.832 |
| large_ranked_v3 | 1,874 | 60,352 | 3.360 [3.360, 3.360] | 3.502 [3.502, 3.503] | 95.172 / 36.042 |
| large_events_v3 | 319 | 44,752 | 16.264 [16.264, 16.266] | 16.763 [16.763, 16.764] | 390.121 / 67.824 |

The full and dirty bitmap kernels use the prepared direct root-map overload. The dense root map is 4 bytes per edge (5.12 MB at E=1.28M); its host preparation and H2D transfer are both separately recorded. Workset compaction uses the bitmap-only production interface, so no per-edge binding search occurs in that stage. Setup amortization includes host reverse-index build, device projection packing, root-map preparation/copy, and stream completion; it excludes initial support/query/state uploads shared by the alternatives.

## Correctness and limits

- All three exact host oracles passed: full scan and posting candidate sets matched; gated primal/derivative worksets matched the scalar oracle; full/dirty condition bitmaps matched expected values. CUDA outputs were also checked against those oracles before timing.
- The large-events C++ case uses the fixture’s generic 24-source query per cell. This native run does not measure filtering the event history by its count/time window; the packed training driver has a separately labeled window-filtered event case, still pending the CE-GEO geometry validation fix.
- These are deterministic synthetic performance fixtures, not biological data or evidence of biological validity.
- The frozen executable predates the ordered-disjoint posting fastpath. It is retained as the baseline executable; a later candidate must use a distinct build path and the byte-identical v3 fixtures for a matched comparison.
- Requested allocations include both rescan and posting controls, candidate upper-bound outputs, prepared condition root map, bitmap and workset scratch. This is requested allocation accounting, not measured peak allocator use.
- No Nsight profile was collected in this run. Separate discovery and gating phase timings identify follow-up profiling targets, but do not establish kernel-level causes beyond the measured phase costs.

## Provenance

- Scheduler job: `3f12203b-6104-43a6-aa42-9afcaf590f80`; attempt `7c7e83db-760a-49a3-8660-8c9480f2151e`; exact GPU2 resource `gpu:testlsserver:GPU-d745da9c-7649-8334-41f1-483afa6f3206`; cleanup verified.
- Frozen executable SHA-256: `a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f`.
- Fixture generator SHA-256: `887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`.
- Fixture manifest SHA-256: small `c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be`; large-ranked `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`; large-events `910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8`.
- Full raw result JSON, input hashes, command lines, setup samples, and scheduler receipt are adjacent in `run_001/` and `scheduler_receipt.json`.
