# Factorized interaction oracle and workloads

`workloads.json` is the fixed benchmark corpus for the first gated joint law.
`tests/factorized_interactions/workloads.py` materializes each named case with
the specified SplitMix64 indexed generator. It does not use NumPy's mutable
default RNG state. `tests/factorized_interactions/reference.py` evaluates the
logical law in float64 after reading the stored FP16/FP32 values, and returns
zero-safe analytic VJPs plus sums of absolute per-interaction contributions.

## Frozen numerical comparison

The accepted initial domain is finite `x`, `alpha`, `gamma`, and `z` in
`[-1, 1]`, with bias in `[-4, 4]`; corpus values are narrower for the timed
reuse sweep. FP32 outputs and native FP32 VJPs are checked per element using
`abs(actual-reference) <= 2e-6 + 2e-5 * sum_abs_reference_contributions`.
FP16 output is first compared with the correctly rounded float64 oracle, with
one local FP16 ULP added to the same FP32 accumulation bound. Input and
parameter quantization happen before the reference arithmetic. The bound is
part of the root-owned contract; candidate authors do not edit it.

The independent tests cover arities 2/3/4, ordered and repeated roles,
shared/tied parameter IDs, multiple output targets and collisions, context
terms including the empty list, one and multiple zero factors, negative values,
channel summation into gate derivatives, finite-difference checks, saturated
gates, empty mechanisms/cells/channels, and independent activation, parameter,
and output precision choices. `workloads.json` separately fixes the 32-mechanism
participant-cardinality sweep from 16 through 128, plus output fan-in, arity,
context reuse, channel width, cell count, and tied-parameter cases. Additional
fixed adversaries include 8,193-term context cancellation, 8,192 ordered FP32
product-residual terms whose rounded products cancel while exact products do
not, and 100k/1M high-fan-in gradients. The 1M case uses the generic scalar
oracle, not a special analytic shortcut. The serial CUDA correctness packet
runs 29 route/workload invocations on the native runner, then executes 200
native C++ parameter-update steps and the prepared ATen/Python adapter checks.
The Python checks compare outputs and all five gradients to the same independent
oracle, exercise a device owner first uploaded on stream A and reused on stream
B, mutate a parameter between calls, and check saved-tensor mutation and
first-order-only guards. Its dependent sanitizer packet covers both native
routes with memcheck, racecheck, initcheck and synccheck.

## Native runner wire contract

`tests/factorized_interactions/case_io.py` writes the versioned little-endian
`CFINT001` request and reads the `CFOUT001` result. The request consists of a
fixed header, ordered mechanism records, ordered `(context,gamma)` terms, then
contiguous `x`, `z`, `alpha`, `bias`, `gamma`, and FP32 `dy` arrays. Axes match
the native header: `x[occurrence,channel,cell]`, `z[context,cell]`,
`alpha[row,channel]`, `y[target,channel,cell]`. The result contains output in
the requested output dtype followed by `dx`, `dz`, `dalpha`, `dbias`, and
`dgamma` in FP32. The parser rejects wrong extents, dtype codes, truncated
arrays, and trailing bytes.

The request is one logical byte stream. Inputs larger than 32 MiB are written
as ordered files of at most 32 MiB, with the first passed by `--input` and each
remaining file by `--input-shard`. The native host reader concatenates those
bytes before the unchanged `CFINT001` parser runs. The trusted validator hashes
the ordered stream and checks it before and after runner execution. This keeps
the million-mechanism fixture within Autobench's per-file limit without changing
its mechanism records, reference calculation or acceptance bound.

The `validate_runner.py` wrapper generates one request, invokes a supplied
native runner, and compares all six returned tensors against the frozen oracle.
The native runner exercises host prepare/structural validation, uploads the
immutable plan and values, queries workspace, checks output overwrite, then
executes one forward and one VJP in the selected route and gradient mode. It
uses a nonzero FP32 seed to distinguish overwrite from accumulate behavior.
The native runner writes a JSON sidecar that records the actually selected
backend (`packet`, `direct`, or shared `stable` fallback), structural fan-in
maxima, GPU UUID, Scheduler job ID and workspace sizes. Stable fallback runs
remain visible but are not counted as packet-layout comparisons.

The wrapper refuses to launch the CUDA runner without
`AUTOBENCH_SCHEDULER_JOB_ID`; the C++ runner repeats that guard. This environment
marker is only a fail-closed compatibility sentinel. Autobench's Scheduler
grant and matched receipt establish admission. `--allow-oracle-stub` is only
for the CPU protocol smoke test and accepts only `oracle_wire_stub.py`.
Performance packets add `--iterations N --expected-uuid GPU-...`; the runner
emits JSON metrics with preparation, layout upload, input conversion, parameter
initialization/refresh, forward, backward with reductions, resident-step
latency, device UUID, and persistent/scratch memory. It reports output
reduction inside forward and parameter/context gradient reductions inside
backward; they are not falsely reported as separate kernel times. Timing
summaries retain the per-case backend selection and only form packet-versus-
direct layout ratios when those two distinct backends actually ran. Autobench
packet wall time and this phase breakdown are separate evidence.

The runner command contract is:

```text
native_runner --input CASE.bin --output RESULT.bin --route packet|direct \
  [--input-shard NEXT.bin ...] --gradient-mode overwrite|accumulate
```

Build the native executable from this directory so it links the exported
`Cellerator::factorized_interaction_cuda` component. The native runner owns
CUDA events and detailed timing output for the admitted Autobench job. This
Python validator serializes cases and checks outputs; it never submits or times
GPU work itself.
