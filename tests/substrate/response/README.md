# Requested native responses

`requested_scaled_tanh<float/double>` prepares cold output and derivative demands
for the actual native `tanh(state[i] * parameters[parameter_slots[i]])` program.
The existing prepared host owner executes the primal. Native local multiply and
tanh rules execute the response chain at the saved preactivation. No Jacobian is
formed. The public prepared program remains available to direct native consumers.

Canonical parameters remain caller owned. Every accepted forward copies them into
explicit caller supplied expansion storage. Repeated logical slots accumulate
parameter VJP contributions in ascending state coordinate order. State and
parameter directions have independent role, direction identity and primal owner
generation tags. Saved responses reject later forwards, changed owner generations,
structure epochs, buffer bindings and unsupported rounding policies.

The cold reverse trace includes requested output coordinates that can contribute
to a requested state or canonical parameter derivative. JVP visits requested
outputs. Both preserve requested output order. Trace counts describe this
pointwise response work; the complete primal still executes all coordinates.

`bind_custom_rule` binds an actual compiled owner callback into the existing
prepared program runner. It preserves native capability validation and missing
callback rejection (`unsupported_derivative`). The caller supplies matching
lifetime, binding and numerical rules. This seam adds no registration or automatic
rule discovery.

## Evidence

```sh
python3 -B tests/substrate/response/check.py --build-dir /tmp/ce-is1-response-host --sdk-prefix /tmp/ce-is1-sdk-a
```

134 native checks in one installed host consumer cover float/double primal,
independent and simultaneous state/parameter tangents, VJP adjoint identity,
finite differences, repeated parameters, zero primal with live response,
nonlinear saved chain, demand pruning versus full response, changed coefficients,
stale generations, capacities, aliases and custom compiled callback rejection.
`dependencies.json` records the reused source identities at the frozen base.

## Limits and integration

This slice supports the declared pointwise two operation law and first derivatives.
Coupled matrix/port/frontier adapters, general graph planning, second derivatives
and GPU execution are unavailable. Numerical association and nonfinite propagation
follow the existing f32/f64 host providers under nearest rounding. Buffers and
instance metadata are borrowed and must outlive responses; callers must publish
all value updates through generations and preserve saved intermediate storage.
Unreported mutation and overwrites by another program sharing that storage are
outside this borrowed contract.

For MERGE-B, add `src/math/response`, install the public response directory and
static library, and export component `response` with dependency `prepared_host`.
The scoped CMake target is `cellerator_response` / `Cellerator::response`; the SDK
component export should use `EXPORT_NAME response`. Preserve the existing
prepared_host and local_differential owners and their actual transitive linkage.
Current installed host headers require CUDA SDK headers through local_differential;
validation launches no GPU work. After frontier is integrated, its owner can
supply explicit compiled response rules through the custom seam; this slice does
not import an unintegrated frontier implementation.
