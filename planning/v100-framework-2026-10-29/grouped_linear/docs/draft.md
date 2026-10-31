# Grouped-linear V100 validation and optimization

## Objective

Run the grouped expert-major linear forward and reverse-mode implementation on
a V100 (`sm_70`) and compare each result with an independent host oracle. Then
measure the current per-expert cuBLAS launch path over small irregular groups
and dense grouped workloads; retain an optimization only if it preserves the
numerical contract and improves matched end-to-end executor time.

## Contract and workloads

For expert `e`, rows `[offset[e], offset[e+1])` compute `Y = X W^T`,
`dX = dY W`, and `dW = dY^T X`. Weights have independent borrowed allocations
in `[output_features, input_features]` row-major layout. Offsets are ragged and
may repeat. FP16 products accumulate in FP32 before conversion to the declared
output/gradient dtype; FP32 uses FP32 math. Test overwrite and accumulation
semantics, empty experts, repeated empty segments, uneven row counts, identity
preservation across expert boundaries, and dimensions aligned and misaligned
to 8. Use deterministic finite signed inputs and an explicit host formula as
the correctness oracle. FP32 tolerances are 2e-5 absolute and relative for
small fixtures; FP16 allows at most 2e-2 absolute and relative against the
FP32-accumulation oracle, tightened or reported by actual observed ranges.

## Baseline and design choice

Current path launches up to three cuBLAS GEMMs per nonempty expert. cuBLAS is
the correctness and performance baseline because each expert operation is an
ordinary GEMM, and FP16 shapes may use V100 Tensor Cores. First measure the
existing executor separately for forward and backward with CUDA events after
warm-up. Sweep expert count, average rows per expert, feature dimensions, and
equal versus ragged row sizes. Preserve one call per expert as the incumbent.
Consider cuBLAS grouped-batched GEMM, grouping equal or per-expert dimensions
into one library call. CUDA 12.8 notes warn that the grouped-batched API can
have substantial host overhead, so treat it as a measured candidate rather
than an assumed win. Account for descriptor and pointer-array construction,
workspace, and refresh costs when inputs or outputs change between calls. Do
not replace the current cuBLAS path for compute-dense shapes without a matched
faster result and profiler evidence.

## Validation and execution

Extend `grouped_linear_smoke.cu` with an explicit GPU path and host reference.
For every case compare forward, input gradients, and weight gradients under
both overwrite and accumulate policies. Synchronize the submitted stream before
reading results. Run under admitted Device Scheduler execution with the granted
V100 UUID, a `gpu.quiescence` precondition where measuring, and exclusive GPU
mode. Do not execute through a bare CUDA command.

Compile only for `sm_70`. Use Compute Sanitizer memcheck on all correctness
cases, then synccheck/racecheck if the kernel family adds custom kernels. Use
CUDA-event samples for operation latency and Nsight Systems to separate library
launch trains and host-side submission overhead. Do not compare concurrent runs
on one physical GPU.

## Risks and promotion

Most tiny expert segments are probably launch-bound, while larger feature
dimensions are library GEMMs with potential Tensor Core throughput. One
specialized path may not win both. cuBLAS may choose different math pipelines
by shape. Benchmark the exact output/update modes and include preparation cost
when a candidate changes metadata or packing. Promote only a candidate with
passing forward/backward oracle checks and a repeatable latency improvement on
at least one declared workload without material regressions on the others.

## Evidence limits

The requested initial scope is validation plus one bounded performance study,
not a general autotuner or a universal dispatcher. Synthetic fixtures establish
the numerical and GPU mechanism contracts; they do not establish biological
workload relevance.
