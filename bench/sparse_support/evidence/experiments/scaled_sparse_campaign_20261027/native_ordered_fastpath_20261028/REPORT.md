# Ordered-disjoint postings: matched native CUDA comparison

This report compares the frozen serial reverse-posting merge with the current
ordered-disjoint count/scan/concatenate retrieval path. It reports results for
these exact synthetic fixtures only; it does not rank or eliminate indexing
families for biological inputs.

## Measurement

Both candidates ran through KDA on an admitted Tesla V100-SXM2-16GB (SM70),
resource `GPU-d745da9c-7649-8334-41f1-483afa6f3206`, with CUDA 12.9.86. KDA used
5 warmups, 21 CUDA-event samples, 3 iterations per sample, 2 exploration and 2
confirmation rounds. Each candidate produced 84 samples per case. The exact
host oracle verified candidate IDs/order, capacities, and gated primal and
derivative worksets. Timing intervals below are observed p10–p90 across those
samples, not confidence intervals. KDA's bootstrap decision intervals are
listed separately.

The complete GPU region includes discovery plus conditional bitmap workset
construction. An in-binary full-rescan path provides a control. Host index
construction, device projection, and transfers are measured separately and
reported for context; they are not included in the resident steady-state GPU
region. Allocation figures are requested device bytes, not measured peak
allocator usage.

| Fixture | Shape and query | Candidate union | Serial postings (ms) | Ordered fastpath (ms) | Full rescan control (ms) | Serial / fast | Rescan / fast |
|---|---|---|---:|---:|---:|---:|---:|---:|
| small ranked/event anchor | E=96, N=64, B=8; q=24/cell | 244 | 0.734 [0.710, 0.830] | 0.141 [0.139, 0.162] | 0.119 [0.119, 0.138] | 5.21x | 0.85x |
| large ranked | E=262,144, N=4,096, B=32; q=24/cell; rank view has 512 identities/cell | 46,704 | 32.103 [32.092, 32.111] | 5.953 [5.952, 5.955] | 9.589 [9.586, 9.591] | 5.39x | 1.61x |
| large event topology | E=1,280,000, N=20,000, B=8; q=24/cell | 11,812 | 46.333 [46.325, 46.350] | 24.502 [24.498, 24.507] | 27.858 [27.853, 27.863] | 1.89x | 1.14x |

The ordered path materially improves over the old serial merge in all three
fixtures, but the full rescan is still faster on the tiny fixture. On the
large ranked fixture ordered postings beat rescan; on the large event-topology
fixture they are only about 12% faster in this measured region. This is a
regime-specific comparison: it does not justify automatic dispatch rules or
discarding another mechanism family.

The KDA engine's aggregate geometric comparison was 3.830x (95% bootstrap
interval 3.729–3.925); per-case scores were 5.510x (5.088–5.932), 5.392x
(5.391–5.392), and 1.891x (1.891–1.891). KDA marked the candidate preferred
within this experiment workspace. That state is not a Cellerator production
default or a biological-method selection.

## Setup and memory

The following medians are milliseconds. `combined setup wall` rebuilds the
posting index, projection and condition root map, uploads them, and waits for
the stream. The reuse columns report medians of four per-run values; each run
forms setup-sample-median plus pipeline-sample-median, or setup-sample-median
divided by 16 plus pipeline-sample-median. Because these are medians at
different aggregation levels, a reuse value need not equal the sum of the two
displayed medians. The separately reported host index and projection build
times are diagnostics inside this combined setup measurement; do not add them
again. Fixture loading and host oracle/capacity reservation are excluded from
the amortization because they are benchmark preparation, not the production
request path.

| Fixture | Candidate | Host index build | Host projection prepare | Combined setup wall incl. H2D/sync | Complete postings region | Setup + region at reuse 1 / 16 | Requested device bytes |
|---|---|---:|---:|---:|---:|---:|
| small | serial | 0.005 | 0.001 | 0.018 | 0.734 | 0.788 / 0.771 | 98,407 |
| small | ordered | 0.006 | 0.001 | 0.018 | 0.141 | 0.158 / 0.141 | 104,551 |
| ranked | serial | 57.142 | 1.673 | 64.259 | 32.103 | 95.297 / 36.049 | 308,109,199 |
| ranked | ordered | 57.254 | 1.682 | 65.249 | 5.953 | 72.542 / 10.115 | 308,133,775 |
| events | serial | 326.401 | 18.810 | 356.259 | 46.333 | 402.639 / 68.601 | 473,889,167 |
| events | ordered | 330.321 | 19.361 | 370.643 | 24.502 | 395.649 / 47.698 | 473,895,311 |

On the event fixture, ordered setup-plus-one-use is 395.65 ms vs 402.64 ms for
serial; at sixteen reuses it is 47.70 vs 68.60 ms. The corresponding ranked
values are 72.54 vs 95.30 ms at one use and 10.12 vs 36.05 ms at sixteen. These
are same-fixture posting-implementation comparisons; they do not compare
posting setup against a fully prepared full-rescan product lifecycle.

Fixture validation and condition-state preparation are separate benchmark
stages: 57/31 ms (ranked ordered) and 282/138 ms (event ordered). Host exact
oracle/capacity reservation was 1.30 s and 1.74 s respectively and is
benchmark-only. The requested device allocations differed by about 25 KB on
ranked and 6 KB on events between serial and ordered, atop 308 MB and 474 MB
total allocations.

## Scope and caveats

- Fixtures are deterministic, pair-unique synthetic graphs with controlled
  degree and shared structure. They are not biological datasets and do not
  establish biological validity or behavior on real cell-state distributions.
- The event-topology case exercises its frozen 24-term query workload. This
  report does not measure Karyoplast model training or claim that the query
  generation reproduces a real convoluted event stream.
- No source-line bottleneck claim is derived from the earlier NCU collection;
  Nsight Compute did not import source in that profile.
- Earlier results for the serial merge remain in
  `../native_baseline/REPORT.md`. They are retained as evidence and are not
  evidence against the ordered-disjoint path.
- A first KDA v1 attempt is retained in `kda/workspace/`; its candidate was
  rejected because KDA manifests differed in dependencies, so it supplied no
  paired timing comparison. The accepted matched comparison is v2.

## Reproduction and provenance

- KDA task: `kda/task.json`; aggregate/raw samples: `analysis.json` and
  `kda/workspace_v2/runs/kda-experiments/attempts/c43195bc-b656-43b1-a752-e78e9b0e6bd2/`.
- Scheduler job `c70c3cd4-755f-41ac-8731-8bb4f5e2c842`, attempt
  `e8cbc81b-204f-4a70-9cf3-e33d53cdaf14`; succeeded, lease released, and
  `cleanup_verified=true`. The full receipt is `kda/scheduler_job_v2.json`.
- Frozen serial binary SHA-256:
  `a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f`.
- Ordered fastpath benchmark binary SHA-256:
  `e08c17f5142363ada4d5de298d146570efeff1669c2327d9e7f3bc681502e393`.
- Ordered-postings CUDA source SHA-256:
  `d034ac6815e070b356d08b3f8852cf03d7ab39a91d008ab12008bbf30c92f99c`.
- Benchmark source snapshot hashes are in `source_snapshot_sha256.txt`;
  fixture manifest hashes and all sample records are in `analysis.json` and
  the KDA attempt tree. The native run's driver used the frozen v3 fixture
  generator SHA-256 `887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`.
- KDA workspace promotion applies only to its experiment-local preferred
  reference. No runtime routing or dispatcher was changed by this benchmark.
