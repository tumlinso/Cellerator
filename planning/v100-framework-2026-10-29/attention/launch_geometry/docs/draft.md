# Attention row-CTA launch geometry experiment

## Objective and source-backed hypothesis

The V100 baseline packet measured resident FP32 forward attention at
`Q=512, N=96, D=32, V=32`: the staged route was `1197.056 +/- 6.513 us`
and the mixed fused/staged route was `100.680 +/- 0.123 us` for all degree-8
rows; with rows of degree 8 and 65, the corresponding results were
`2877.399 +/- 0.041 us` and `2093.793 +/- 0.000 us`. These are baseline
observations from the canonical V100 Autobench snapshot, not candidate results.

In `attention.cu`, the six staged forward and backward kernels assign one row
to a CTA and advance rows by `gridDim.x`. Their launch sites use
`blocks(query_count)`, which computes `ceil(Q/128)`; the fused-small kernel
already uses `query_blocks(Q)`, which computes one CTA per row. Thus at Q=512
the row kernels launch four CTAs and each CTA serially visits 128 rows. The
candidate changes only the six row-kernel grid sizes to
`query_blocks(query_count)`, preserving kernels, row order, arithmetic, API,
and the 65,535-grid cap. This should expose independent rows to the V100 SMs.

## Candidates and first experiment

- `baseline`: frozen `attention.cu` from the task snapshot.
- `one_cta_per_row`: identical source except six launch expressions use
  `query_blocks(Q)`, matching the source's one-row-per-CTA ownership.

Compare both candidates on the exact resident Q=512 attention fixtures already
used by the V100 baseline: degree 8 for every row, and mixed degree 8/65 (with
one degree-65 row per 16). Time both staged and mixed-fused API routes so the
candidate is compared against its strongest existing route. Keep each route's
raw samples and spread rather than claiming a single speedup. The main success
criterion is a paired, repeatable improvement in staged forward latency with
no material mixed-fused regression; the task packet uses per-case promotion
criteria and retains the incumbent if they are not met.

## Correctness contract and reference

The existing attention smoke executable is the trusted device test and
independent host FP64 oracle. It covers staged and fused forward plus backward
for FP16/FP32, CSR degrees 0/1/2/31/32/33/65, value widths 1/5/31/32/33/65,
empty plans, repeated-row/block-stride behavior, and analytic gradients with
finite-difference checks. Run it separately against each candidate.

The timed fixtures also run an independent deterministic host FP64 forward
attention oracle before collecting samples. The validator compares every
output from staged and mixed-fused routes to that reference and to each other;
FP32 uses the existing `3e-5` absolute/relative allowance. This ensures timing
uses nonzero, valid data while keeping the correctness reference outside the
candidate source. Backward correctness remains covered by the full smoke
matrix, since this geometry change is intended to affect the staged backward
row kernels as well.

## Build, execution, and resource boundary

Build each immutable candidate in its own directory with one compiler job,
`CMAKE_CUDA_ARCHITECTURES=70`, and the pinned CUDA 12.9/NVHPC host compiler.
Building is CPU-only. Do not run a device check or timing process directly;
the root owns Autobench registration, UUID-bound Scheduler admission, and all
GPU execution. The new Autobench definition stays under this experiment
directory and is not added to the active registry by this task.

The exact commands will be the task's `probe`, `build`, `validate`, and
`benchmark` adapters. Build uses `kda_experiments prepare` with one job; the
admitted root-owned packet will use `kda_experiments evaluate` on the two
candidates. Each timing sample is a resident CUDA-event interval averaging a
fixed number of repeated calls after warmup; input generation, uploads,
allocations, and output downloads stay outside the timed region. The protocol
will use paired rounds, fresh challenger confirmations, and the KDA engine's
bootstrap decision with a 5% minimum overall improvement and 5% maximum
per-case regression.

## Risks, unknowns, and evidence for a decision

- More CTAs should raise row-level parallelism for staged work, but degree-65
  rows remain scalar in several kernels; this experiment does not change that
  kernel-level work partition.
- Fused-small still launches later kernels for large rows and skips small rows;
  changing those grids may improve or regress mixed workloads, so it is timed
  separately rather than inferred from staged results.
- Backward uses atomics into shared key gradients. Changing CTA count preserves
  the mathematical reduction but can alter accumulation order and timing;
  existing FP16/FP32 backward checks must pass unchanged.
- Q<=0, Q beyond the grid cap, empty rows and block-stride coverage are guarded
  by the existing source and tested by the existing smoke matrix. No tolerance
  is relaxed for this experiment.

Promote only if the admitted paired evaluation passes every candidate's
correctness gates, shows the declared staged improvement, respects all
per-case regression limits, and retains raw samples, source hashes, binary
hashes, environment identity, and the outer Scheduler/Autobench receipt.
Otherwise retain baseline and report the measured limitation.


## Workflow references consumed

The workflow and architecture guidance were taken from KDA at revision `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`, its pinned KernelWiki revision `542ed702ac662422d55b19186562b29abbb5da29`, and the local CUDA skill Volta router and benchmark-standardization guide. These provide process and SM70 context; correctness is governed by the Cellerator host FP64 reference and current device smoke, while measured performance decisions remain fixture-specific.
