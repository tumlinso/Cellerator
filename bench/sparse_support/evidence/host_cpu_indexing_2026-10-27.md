# Host sparse support indexing evidence

This is a **CPU-only** measurement of the current canonical reverse postings
and condition bitmap implementations. It does not establish CUDA performance.
The raw 21-sample records and run manifest are in
[`host_cpu_indexing_2026-10-27.jsonl`](host_cpu_indexing_2026-10-27.jsonl).
That manifest records SHA-256 hashes for the benchmark binary, linked static
library, and each relevant production/header/CMake source file.

## Conditions

- Intel Xeon E5-2698 v4 at 2.20 GHz; 80 logical CPUs; process affinity and CPU
  frequency were not controlled.
- Release build, GCC 13.3.0, CMake 3.28.3; CUDA explicitly disabled.
- Three warmups and 21 measured samples for each phase. A sample averages a
  bounded number of calls targeting about two million edge visits. Raw samples
  plus p10, median, and p90 are retained.
- Inputs are deterministic synthetic supports with 2,176 edges split into a
  stable base and 1/16 mutable delta, 3 ordered participant roles, repeated
  incidences, per-cell queries, four shared gate predicates, and separate
  primal and derivative proposals. Candidate IDs and ordinals, bitmap rebuilds
  and dirty updates, and primal/derivative worksets all passed exact host
  oracles. This is not a biological data replay.
- Source SHA-256 `962b5bff0aee8538f1208fcde0d3bee935cf1eab5647a2e5b48883d813524732`;
  binary SHA-256 `e80d0359e1450a3ede3197e46a624d2a99ca613de27cc5d76881f8f392c562f6`.

## Selected measurements

Times are median microseconds per measured call. `Complete` includes candidate
discovery, full cross-cell bitmap rebuild, and gated workset construction.
`Setup` rebuilds the host reverse posting index. The
amortized column adds `Setup / reuse` to the postings complete path.

| Case | Full scan discovery | Posting discovery | Setup | Complete scan | Complete postings | Postings + setup, reuse 1 / 16 |
|---|---:|---:|---:|---:|---:|---:|
| 2,176 edges, 64 cells | 3,999.5 | 27.0 | 637.3 | 10,085.2 | 6,107.8 | 6,745.2 / 6,147.7 |
| 2,176 edges, 1 cell | 56.8 | 0.3 | 612.4 | 164.6 | 95.4 | 707.8 / 133.7 |
| 2,176 edges, 512 cells | 30,407.1 | 319.0 | 589.7 | 77,174.1 | 44,096.4 | 44,686.1 / 44,133.2 |
| 65,536 edges, 1 cell | 1,629.4 | 8.6 | 21,853.6 | 4,832.8 | 2,962.9 | 24,816.4 / 4,328.7 |
| 90% source skew, hot query | 59.0 | 44.7 | 594.1 | 163.0 | 158.7 | 752.8 / 195.8 |
| 3-term all-of, 2,176 edges, 16 cells | 1,654.9 | 17.6 | 546.4 | 3,173.6 | 1,477.1 | 2,023.5 / 1,511.2 |

At the 2,176-edge/64-cell anchor, complete postings were 1.65× faster than
the full-rescan path before setup; one-use setup amortization still leaves a
1.49× improvement. At one cell, one-use setup loses (707.8 us versus 164.6 us),
while reuse 16 wins (133.7 us). At 65,536 edges, the 21.9 ms setup cost loses
badly for one use and slightly wins at reuse 16 (4.33 ms versus 4.83 ms). Under
90% source skew, 1,931 of 2,176 edges matched the hot query; complete-path
benefit fell to 1.03× before setup, and remained a regression at reuse 16.
This is the measured dense-query crossover warning for this host workload.

The anchor complete-path p10–p90 ranges were 9,763–10,747 us for full rescan
and 5,942–6,887 us for postings. The corresponding discovery ranges were
3,875–4,613 us and 25–29 us. These ranges reflect the unpinned host, not a
confidence guarantee.

The one-cell bitmap comparison matches its state scope: at 1% dirty edges in
the anchor (22 of 2,176), a full one-cell rebuild took 90.0 us and dirty update
2.6 us. At 10% dirty (218 edges), the medians were 79.9 us and 14.6 us. At
65,536 edges and 1% dirty, they were 2,904 us and 109 us. The full 64-cell
bitmap rebuild is also measured separately for complete batch costs. These
numbers are specific to the current host API and fixture.

The probability stress cases produced observed eligible-edge densities of
8.2% and 91.7%; derivative-only eligible work varied from 44 to 469 edges.
They verify variable work and exact derivative union behavior. The complete
pipeline still spends a large share of time rebuilding the full CPU bitmap.
The GPU implementation has different execution geometry and must be measured
independently before choosing a backend or claiming a speedup.
