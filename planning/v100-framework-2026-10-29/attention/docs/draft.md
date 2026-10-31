# Sparse attention V100 validation and optimization draft

## Contract and baseline

For CSR row `i`, compute `softmax(scale * dot(q_i, k_key)) @ v_key` over its
listed keys. Empty rows output zero. FP16 inputs accumulate scores and values
in FP32 and round the output to FP16; FP32 inputs accumulate and return FP32.
The backward VJP returns FP32 gradients for q, k, and v. Repeated keys may use
atomics in dK/dV, so compare those to a CPU double or Torch float64 oracle with
an explicit tolerance. The current staged path is the incumbent. The optional
fused path handles rows of degree at most 32 and falls back to staged kernels
for larger rows. Use the existing smoke entrypoints as control and candidate.

## Cases and independent reference

Extend the smoke test with deterministic CSR whose row degrees include 0, 1,
2, 31, 32, 33, and 65; include repeated key IDs across and within query rows,
unequal query/key/value dimensions, non-power-of-two D and V, and empty rows
interleaved with nonempty rows. Build a host FP64 forward and analytic backward
oracle from the mathematical contract; check dQ/dK/dV also with finite
differences on a small FP64 fixture. Exercise both staged and fused entrypoints
for FP16 and FP32. Compare FP32 output/gradients at abs 3e-5 + rel 3e-5; FP16
output at abs 3e-3 + rel 3e-3 and gradients at abs 5e-5 + rel 5e-5 against
the oracle over exactly represented half inputs. Test zero and accumulate
gradient modes and undersized workspace/payload capacities.

## First experiments

First run correctness, then Compute Sanitizer memcheck, racecheck, and synccheck
on the smoke executable. Compare staged and fused entrypoints with CUDA events
on the same deterministic, device-resident fixtures. Include low-row-count
launch-sensitive and high-row-count fixtures; degree distributions should be
all-small (1--8), boundary (31/32/33), and mixed with degree 65. Sweep feature
dimensions 16, 64, and 128 and value dimensions 16, 64, and 128. Warm up each
route, collect at least 20 samples, report median and MAD, and never overlap
GPU measurements. Report resident operation latency separately from any
allocation or transfer path. The smallest useful optimization is to keep the
fused route only in regimes with a repeatable win and matching correctness.

## Risk and promotion

Current score and softmax kernels serialize a whole CSR row on thread 0. The
mixed fused entrypoint still launches the three staged kernels so rows above
32 can fall back; all-small invocations therefore launch four kernels versus
three for the staged entrypoint, with small rows skipping their staged work.
The fused route avoids materializing small-row logits, but does not avoid the
launches. Main risk is low device utilization for short rows; the staged route's
row-striding and aggregation geometry may also dominate uneven workloads. A proposed
cooperative edge scoring/online-softmax path should remain a separate candidate
until the admitted baseline identifies a useful regime. Require all correctness
cases and sanitizer tools to pass before comparing timing. Keep a change only
when the same workload has an improvement larger than observed measurement
spread without material regression in the other declared regimes. No performance
or V100-device claim is in scope before receipts are retained.

## First admitted-run correction

The first V100 run passed staged FP32 forward/backward and failed fused FP32
forward for degree 31 with output[15] `-0.0297650509` versus oracle
`-0.0816384167`. The defect was warp participation: aggregation called a
full-warp probability shuffle inside a loop over output dimensions. For
`value_dim=5`, only five lanes entered that loop despite the `0xffffffff`
shuffle mask. The correction stores probabilities and keys in per-CTA shared
arrays, synchronizes the single warp, and has output-owning lanes read those
arrays. Regression dimensions are `V=1,5,31,32,33,65`; a 65,536-row degree-1
case also exercises the block-count cap and a CTA reusing its shared row buffers.
The shared-memory fix is not yet V100-validated or performance-measured.

## Target

Tesla V100, SM70; CUDA 12.9; one scheduler-admitted device per attempt. Source
files: `src/modeling/cuda/attention.cu`,
`include/Cellerator/modeling/cuda/attention.hh`, and
`tests/framework/cuda/attention_smoke.cu`. KDA KernelWiki's Volta attention
page records a separate dense attention family and informs occupancy-aware
case selection only; it is not the numerical oracle or a direct baseline for
this CSR operation.
