# Shared condition bitplane result

## Decision

Keep the bitplane DAG path as a specialized experiment for condition programs
with many nodes and cell batches of at least a warp. Do not select it for the
small seven-node DAG tested here: the scalar direct edge-to-root map was faster
in all three small-DAG shapes. A compiled edge-to-root map is a distinct win
for those shapes and should be evaluated independently from bitplanes.

## V100 results

All values are end-to-end host-wall medians in milliseconds. Each row used 11
rotating-order samples; values in parentheses are the paired speedup p10–p90.
The consumer is the same production `cuda_build_condition_worksets_v1` path for
all three methods.

| Edges × cells | Condition program | Current full | Direct-map full | Bitplane full | Current dirty | Direct-map dirty | Bitplane dirty |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2,176 × 64 | 7 nodes | 0.205 | 0.170 | 0.174 | 0.183 | 0.164 | 0.166 |
| 2,176 × 512 | 7 nodes | 0.259 | 0.228 | 0.234 | 0.225 | 0.207 | 0.210 |
| 65,536 × 64 | 7 nodes | 6.251 | 4.805 | 4.839 | 4.686 | 4.148 | 4.192 |
| 2,176 × 64 | 2,304 nodes, weak edge-root reuse | 1.108 | 1.111 | 0.098 | 1.077 | 1.103 | 0.090 |

The direct-map-only arm replaces the binding binary search but retains scalar
per-cell DAG evaluation. Its full-path speedup over current is 1.20×, 1.13×,
and 1.30× on the three small-DAG shapes. Bitplanes are 2–3% slower than that
arm there. In the 2,304-node case, the direct map alone is neutral, while full
bitplane evaluation is 11.3× faster than direct-map-only and 11.3× faster than
current. The dirty bitplane arm is 12.4× faster than direct-map-only; that path
also recomputes only the 19 condition nodes and 19 edges dependent on the
changed predicate, so this gain belongs to the combined bitplane plus
incremental dependency-update mechanism.

The small-DAG paired host-wall speedups for bitplanes over current are
1.17×/1.10×, 1.11×/1.07×, and 1.29×/1.12× for full/dirty. Those gains come
from the edge-to-root map; bitplanes do not improve over the map-only arm in
these regimes.

## Conditions and costs

- One V100 selected by scheduler resource ID
  `gpu:testlsserver:GPU-d745da9c-7649-8334-41f1-483afa6f3206`, SM70, CUDA 12.9.
- Inputs, support bindings, bitmaps, condition planes, and workset staging were
  resident and preallocated during steady timing. Predicate values and the
  changed-predicate mask were fixed from seed `1511508451`.
- Dirty edges were exactly those whose condition dependency includes predicate
  zero: 136 per cell in the seven-node DAG and 19 per cell in the weak-sharing
  DAG. Both paths consumed the same sparse proposal lists and workset compactor.
- Each path had two warmups and eleven samples. The control, direct-map, and
  bitplane order rotated across samples. We recorded CUDA-event timelines and
  host wall time through completion synchronization. Paired speedup p10–p90
  ranges were narrow (typically under 1.5% of the median).
- Per-path additional device metadata was 8,760 bytes for the 2,176-edge,
  64-cell seven-node fixture; 9,152 bytes at 512 cells; 262,200 bytes at 65,536
  edges (about half the 524,288-byte bitmap); and 27,136 bytes for the weak
  2,304-node fixture. These counts include the 32-bit edge-root map and one
  condition-plane array. They exclude the common bitmap and workset buffers.
- The production workset consumer alone took 0.158 ms, 0.199 ms, 3.934 ms, and
  0.073 ms in the four respective shapes. It dominates the first three
  complete paths, limiting the gain from faster gate construction.
- Root-map host compilation took 0.003–0.076 ms. Initial bitplane transpose,
  DAG evaluation, and bitmap materialization took 0.020–0.237 ms; the current
  initial bitmap path took 0.052–1.354 ms. Device setup totals are retained in
  raw rows but include common allocations and first CUDA context startup, so
  they are not treated as candidate-only setup cost.

## Limits

These are synthetic binary predicates and one V100. The weak-sharing fixture
uses 128 predicate leaves and one distinct conjunction root per edge; its large
node count intentionally exposes serial DAG evaluation. The result supports a
specialized bitplane path for large condition programs, not a universal gate
backend. The dirty-path gain combines bitplane packing with dependency-closure
updates; a scalar incremental dependency-closure ablation was not measured.
