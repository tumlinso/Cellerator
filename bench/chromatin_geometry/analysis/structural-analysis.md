# Structural demand and execution costs

These are CPU counts and source analysis, **not GPU timings, HBM traffic, or
measured occupancy**. [structural-counts.csv](structural-counts.csv) contains
160 rows for the frozen ten cases, synthetic and verified 10x prefix inputs,
both input states, both numerical bodies, and Boolean/all-D envelopes. The
source and input hashes are in
[structural-provenance.json](structural-provenance.json).

The all-D rows hold primal membership at the supplied condition decisions.
They describe STE's declared derivative envelope, not a prediction of a trained
score trajectory. Actual teacher decisions and learned STE admissions must be
read from the corresponding run. The replay selects source cell/peak prefixes;
its “after” state applies controlled leaf flips. It is not a measured second
time point. Non-detections do not certify biological exclusions.

## Frozen complete regimes

Every complete case binds every condition edge to a numerical group. The
64-group producer-only diagnostics expose 2,176 conditions and remain separate.
The strong controls for complete cases prune unused node ancestors.

| Complete case | Groups | Source → evaluated nodes | Initial primal occurrences, synthetic / replay | Reuse / leaf changes |
|---|---:|---:|---:|---:|
| Seven-node legacy, sparse, low overlap, frozen | 2,176 | 7 → 7 | 53,357 / 61,338 | 1 / 0% |
| Seven-node coherent, dense, high overlap, updates | 64 | 7 → 4 | 12,160 / 0 | 16 / 1% |
| 2,304-node legacy, sparse, high overlap, frozen | 2,176 | 2,304 → 2,304 | 422 / 629 | 256 / 10% |
| 2,304-node diverse, dense, low overlap, updates | 64 | 2,304 → 192 | 6,545 / 15 | 1 / 100% |

Each case has 256 cells. Thus the sparse legacy all-D envelope contains
557,056 derivative group occurrences even when only 422 supplied primal
occurrences are active. This is a learning-contract cost, not an omission a
correct STE specialization can make. The synthetic activity label is a fixture
configuration; replay replaces those accessibility observations.

## Logical demand and possible reuse

Let P count eligible group occurrences, D the STE derivative occurrences,
I input slots and O local output slots. Counts retain repeated slots. Supplied
and hardened-teacher routes compute mechanism gradients over P and no task dQ.
STE must evaluate the body over D, including dormant groups.

* Product forward needs I participant values per P and a theta value. Native
  scalar/direct/bitmap controls reuse the existing ordered warp prefix/suffix
  scans for I≤32, reading I values per required backward group. The cohort
  scalar VJP instead forms each other-slot product without division: I×D plus
  I×(I−1)×P value uses for STE, or I²×P for a Boolean gate. Arity33 retains the
  fallback. Zeros, repeated slots and FP32 contracts are preserved. The CSV
  separates these algorithms; their difference must not be attributed solely
  to layout or condition packing. Neither count is a cache-miss count.
* Linear forward performs I×O participant/weight uses per P. Backward
  computes local F for STE dQ over D, dX over P, and batch-reduced dW over P,
  preserving the explicit local FP16 rounds and FP32 accumulation. The
  separate dW kernel still scans cells for every local weight.
* Shared staging loads each demanded feature once per cell **per supplied
  execution neighborhood**. It stages each demanded group's parameters once
  across two cell tiles. Shared parameter identities can still cause repeated
  address loads across groups and neighborhoods. Frozen values may already
  benefit from ordinary caches; updating values does not remove within-call
  reuse.

High overlap is extreme controlled reuse: group feature/output patterns repeat
modulo eight. It can favor ordinary cache reuse and cause output atomic
contention. A reduction in logical loads is therefore not an HBM reduction.
Low-overlap cases have less reuse within the selected feature unions. The CSV
separates repeated value uses, unique primal coordinates and staged value loads.

Two current implementation costs deserve specific profiling. Linear hard-gate
backward stages participant X for P even though dX needs only W/dF and its
separate dW kernel reads X again. Also, local F is stored by custom forward but
backward recomputes F for STE. These are candidate-specific extra work, not
requirements of cellular geometry. Do not add variants until profiles establish
whether these costs matter in the complete path.

## Conversion, cohort and capacity costs

The direct root-map control evaluates prepared levels in shared node bytes and
publishes root bytes without a bitmap or plane intermediate. Plane consumers
retain packed conditions; the materialization arm additionally creates
cell-major bitmaps and per-cell P / P∪D group worklists. Its worklist scan and
storage scale with cells×groups even at low activity.

Those worklists compact entries **within each cell**, but the bitmap arm still
dispatches the dense-capacity grid: `ceil(batch * group_count / 4)` CTAs, with
four warps per CTA. `direct_consumer` maps every capacity pair to a cell and
queue slot, then returns when that slot is beyond the cell's work count.
Materialization reduces numerical body work and participant reads; it does
not remove those empty warps from dispatch. This is not a global active-item
array with a count-bounded grid, or a persistent worker queue. Those would be
different, unimplemented dispatch alternatives with their own preparation and
execution costs.

Cohorts add mask
construction, a bounded insertion sort, routing metadata and shared staging.
The retained cached `cohort_direct` and shared grids have different dispatch
granularity: the one-tile cached arm uses
`ceil(batch / tile) * neighborhoods` CTAs, and the shared arm uses
`ceil(batch / (2 * tile)) * neighborhoods`. For the fixture with 256 cells,
64 groups, eight execution neighborhoods and a tile size of 32, this gives
4,096 dense-capacity CTAs versus 64 for `cohort_direct` or 32 for
`cohort_shared` per numerical phase. These
are CTA dispatch counts, not CUDA kernel-launch counts or speedup factors;
cohort warps loop over more logical work. A cohort win may therefore reflect
fewer dispatched warps and amortized eligibility checks before any participant
data is read. Do not attribute it purely to HBM reuse without separate traffic
and cache evidence.

A separate `cohort_direct_two_tile` cached control now matches the shared
arm's two-tile CTA ownership, tile order and cohort assignment while retaining
ordinary global participant/parameter loads and the same numerical body. It
shares the prepared descriptors and uses no staging buffers. This provides the
matched staging ablation; the one-tile cached arm remains the ownership/dispatch
ablation. Added correctness coverage includes 65-cell boundaries, so the second
tile and a trailing CTA are both exercised. Its performance requires its own
qualified source, binary and paired measurements; earlier timings do not
establish its behavior.

The cohort implementation assigns a warp by **sorted cohort start index modulo
four**. One coherent 128-cell cohort uses one warp for four chunks; the other
three warps have no cohort body. Several cohort starts divisible by four can
also concentrate work in one warp. Conversely, tiny diverse cohorts leave many
lanes unused. CSV dispatch lane-fill/warp-presence/chunk-balance ratios quantify
that assignment, including all-inactive cohorts; they are not profiler
occupancy or useful numerical-work measurements. This bounds conclusions about
this candidate, not about all possible cohort schedules.

The implementation rejects shared stages above 12,000 floats (48,000 bytes):

| Maximum neighborhood union | Product, tile128: one / two buffers | Linear, tile128: one / two buffers |
|---:|---:|---:|
| 49 features | 25,120 / **50,208 rejected** | 25,600 / **50,688 rejected** |
| 64 features | 32,800 / **65,568 rejected** | 33,280 / **66,048 rejected** |

Thus every frozen tile128/two-buffer shared case is a host-known capacity
rejection. Tile32/64 with one/two buffers fit. This is the implementation's
conservative shared-memory limit, not a performance result.

## Interpreting measurements

Use measured HBM/L2 bytes, shared conflicts, spills, atomics, synchronization
and useful work to distinguish these explanations. The two-buffer path uses
ordinary SM70 global loads/shared stores, with warp zero loading the next tile;
overlap is an execution hypothesis, not an asynchronous-copy guarantee.

Charge condition/gate updates, packing, routing, clears, output recovery and
training to their actual phase. Current descriptor revision guards add common
launches. The condition fixture allocates all representations for all arms, so
reported cold time and memory describe this harness, not minimum per-arm
storage. The runner's sequence length is not a promise of unchanged gates:
configured leaf changes can be applied at each update step. Break-even reuse
must use actual update counts and complete measured costs. A slower recurring
path cannot be rescued by amortizing its one-time preparation.

Source anchors: [consumer kernels](../backends/consumers.cu),
[condition producers](../backends/conditions.cu),
[library controls](../backends/grouped_linear_library.cu),
[workloads](../workloads.cc), and [complete runner](../campaign_runner.cu).

Reproduce the CPU counts from the repository root:

```sh
g++ -O2 -std=c++17 -Iinclude -Ibench/chromatin_geometry \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/include \
  bench/chromatin_geometry/analysis/structural_counts.cc \
  bench/chromatin_geometry/workloads.cc \
  bench/chromatin_geometry/reference/reference.cc \
  bench/chromatin_geometry/backends/consumer_fixture.cc \
  bench/chromatin_geometry/data/replay.cc -lcrypto \
  -o /tmp/cgs-structural-counts
/tmp/cgs-structural-counts \
  bench/chromatin_geometry/data/cache/pbmc-atac-cgatac01/accessibility.cgatac \
  > bench/chromatin_geometry/analysis/structural-counts.csv
```
