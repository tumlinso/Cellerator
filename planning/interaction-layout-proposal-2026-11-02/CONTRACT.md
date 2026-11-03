# Factorized interaction implementation contract

The user authorized implementation of the reviewed division-of-labor plan on
2026-11-02. This is the shared implementation contract, not live Todo status.
The canonical cellular geometry definition is unchanged.

## Mathematical and numerical boundary

For mechanism `m`, cell `c`, channel `d`, ordered participant slots `P[m,r]`,
ordered context terms `(Q[m,k], Gamma[m,k])`, and independent canonical
parameter bindings `Alpha[m]` and `Bias[m]`:

```text
g[m,c] = sigmoid(bias[Bias[m]] + sum_k gamma[Gamma[m,k]] * z[Q[m,k],c])
y[T[m],d,c] = sum_{m:T[m]=T} g[m,c] * alpha[Alpha[m],d]
                                      * product_r x[P[m,r],d,c]
```

Output starts at zero; there is no additive base operand. Roles and repeated
context terms retain multiplicity. Repeated uses of the same source occurrence
and independently tied parameter fields accumulate all derivatives. Initial
supported joint arities are 2, 3 and 4. Occurrence ordinals are operation-local
positions, not inferred gene identities. Stable mechanism IDs and structural
epochs survive physical reordering.

Native axes are `x[occurrence,channel,cell]`, `z[context,cell]`,
`alpha[parameter_row,channel]` and `y[target,channel,cell]`.
FP16 storage/default output and explicit FP32 alternatives are supported.
Storage selections remain explicit; native arithmetic/accumulation and the
five VJP buffers (`x`, `z`, `alpha`, `bias`, `gamma`) are FP32. `dy` is FP32.
An optional Torch adapter returns gradients in the original tensor dtype.

FP32 atomic reductions are allowed. The numerical contract is tolerance-based,
not bitwise summation-order reproducibility. Metadata/packing is deterministic.
Initialize sum buffers and include initialization, reduction, and output-cast
costs in the complete operation. Product derivatives never divide by a
participant; gate derivatives include the sum across all channels. Continuous
sigmoid values are not pruning certificates.

## Independent acceptance policy

The oracle evaluates the declared storage-quantized inputs using independent
float64 scalar equations. Per-result FP32 output/VJP error is bounded by
`2e-6 + 2e-5 * sum_abs_reference_contributions`. Report both raw differences
and that contribution scale. FP16 output is compared to the correctly rounded
oracle with one FP16 ULP plus the corresponding FP32 accumulation bound.
Quantization is recorded separately from execution error.

The initial corpus uses finite bounded inputs and parameters, zeros, signs,
cancellation, repeated roles/context terms, tied fields, duplicate targets,
empty cases, saturating sigmoid arguments and larger reduction fan-in. The
oracle owner freezes exact fixtures and their digests before measurement.
Any tolerance/corpus repair must be justified independently and recorded as
a new evidence identity; candidate authors cannot weaken acceptance.

## Ownership and lifetime

The principal owns shared native descriptors and coupled packet forward/VJP.
The preparation executor owns host packing, the direct-control executor owns
its separate CUDA implementation, and the verifier owns the independent
reference and measurement corpus/harness. The integrator owns shared build
wiring and adapters once native interfaces stabilize. Root settles public
policy and final acceptance. Contributors preserve all concurrent edits.

Low-level calls borrow structure, primals, parameters, tape and scratch, use an
explicit stream and do not allocate or synchronize. The caller retains immutable
primals and structure through VJP. Higher-level owners/adapters must make these
obligations usable and retain per-invocation numerical state. Do not overwrite
an outstanding tape during a later forward/update.

The tape retains gates and stable sigmoid slopes. Conditioning and channel
adjoint cancellation use compensated FP32 arithmetic; high-fan-in reductions
use a shared accurate backend rather than weakening acceptance or rejecting
legitimate repeated occurrences. Backend identity is recorded separately from
the requested route; stable/stable comparisons are not layout-win evidence.

## Performance selection policy

Use at least seven independent paired blocks for each compared workload,
alternating packet/direct and direct/packet order within that workload. A
packet regime is eligible for promotion only with median speedup at least
`1.10x` and a paired-bootstrap 95% confidence interval whose lower bound
exceeds `1.0x`, against the strongest qualified direct control for that regime.
Keep source, binary, corpus and sampling identities with the result.

Resident forward/VJP timings include tapes, initialization, output casts,
reductions and dispatch. The primary resident comparison is warm block host
wall time divided by the invocation count, from before the calls through stream
synchronization, without per-step timing events. Separately sampled GPU-event
phases are secondary device metrics. Report preparation, allocation, conversion/upload,
parameter refresh and persistent/scratch storage separately. A setup sample
plus a warmed median is an estimate, not a measured cold step. The numerical
component does not own the caller's optimizer. A direct path charged for full
packet preparation cannot establish a cold-start packet advantage; use the
minimal valid direct preparation/upload control for that comparison.

## Execution and workflow evidence

Only the verifier's coordinated measurement lane submits device qualification,
timing and profiling through the canonical Autobench/Scheduler route. CPU-only
compilation and host checks can run in parallel. The new isolated measurement
workspace is `measurement/`; its coordinator activation is disabled and the
repository's existing paused campaigns are preserved. Device grants and clean
timing admission are established at each finite run, not by physical ordinals.

Project Control's registered selector is `cellerator`. Its initial `next_task`
request returned `workflow_task_already_active` without a workflow handle.
No native Todo IDs, leases, credentials, authority database or existing claims
were changed. Local implementation proceeds under the user's explicit request;
no native workflow completion is asserted without a supported handle.
