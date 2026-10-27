# Prepared term-to-ordinal bitmap probe

## Contract

Compare exact any-of retrieval on the frozen sparse-support CUDA fixtures. The
baseline is Cellerator's existing serial k-way merge of queried posting lists.
The candidate prepares one dense `uint32` bitmap per `(participant role,
entity)` and sets bit `logical_edge_ordinal`. At query time it ORs the selected
rows, popcounts words, scans word counts, and emits stable IDs in ascending
ordinal order.

The candidate reports exact required output size and performs no output writes
when the caller's capacity is too small. Preparation, index bytes, common
query upload, workspace/output bytes, and steady complete query latency are
reported separately. Cold setup and reuse counts are compared explicitly.

The follow-up variant keeps the same prepared bitmaps but assigns one CTA per
query row. It combines wordwise OR with row counting, then uses a CTA-local
word scan and row-total prefix to emit the sorted work list in two kernels.
For the bounded matrix it computes row bases from the small row-count array
inside each CTA instead of launching a device-wide scan.

## Validation and comparison

The fixture generator mirrors `bench/sparse_support/cuda_indexing_bench.cu`
(seed `20261027`) and checks the support digest against the official record.
Both GPU algorithms are checked against a separate scalar per-edge oracle.
The matrix covers `E=2176,B=64`, `E=65536,B=1`, source skew `0.9,B=64`, and
`E=2176,B=1`. Both bitmap query kernels are compared against the same
production posting implementation on GPU3. Warmup, samples, iterations, and
input digests are recorded.
No production source or public API is changed by this probe.

## Promotion rule

Retain the candidate as an experiment only if exactness holds and complete
setup-plus-query cost wins in a declared reuse regime without an unacceptable
memory increase. A single-query speedup does not establish a general win.
