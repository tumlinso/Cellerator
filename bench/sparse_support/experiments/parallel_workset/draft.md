# Parallel workset compaction probe

## Decision under test

The production CUDA workset stage currently assigns one CUDA thread to each
cell and serially filters the primal candidates, then merges the eligible
primal and derivative streams. The profile-backed question is whether per-cell
parallel eligibility scans and rank-based stable scatter reduce the complete
gating/workset cost on the V100 workload.

## Candidate

Use one CTA per cell. Tiled CUB block scans produce exclusive eligible-prefix
counts for primal candidates and eligible derivative candidates not already
represented by an eligible primal edge. A second parallel pass scatters primal
items and computes stable derivative-union ranks from those prefixes and
lower-bound searches in the original sorted rows. The candidate has separate
prefix scratch; all of its initialization, workset count, global offsets, and
scatter launches are included in timings. It remains experiment-only.

## Correctness and evaluation

Compare exact `(logical_edge_id, logical_edge_ordinal)` rows and per-cell
offsets with both an independent host set/merge oracle and the production CUDA
builder. Exercise the E=2176/B=64 source-skew fixture, B=1 and B=512 variants,
0.9 source skew, and a derivative-heavy proposal case. Overflow must report
both required counts and must not write public output rows. Use matched,
unprofiled CUDA-event measurements for speed; profiler replay is diagnostic.
Do not promote or alter production sources unless the complete indexed-path
measurements justify that change.
