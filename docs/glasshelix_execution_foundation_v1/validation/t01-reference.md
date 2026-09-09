# T01 formula and indexing referees

This suite qualifies independent FP64 test oracles, not a Cellerator backend.
The architectural ownership remains defined by `scope.md` and
`docs/architecture.qmd`. No production calculation, lowering map, prepared plan,
CUDA kernel, or GlassHelix mathematical oracle is included by this test target.

`tests/native_foundation/reference/formulas.hh` supplies explicit scalar formulas:

- A rectangular relation from logical inputs `(A,B,C)` to `(sink,report)`:
  `(2*A-3*C, .5*B+4*A)` and its analytical pullback.
- An ordered nonlinear block `p*x*y + sin(z) + .5*x*x`, its full gradient
  over `(x,y,z,p)`, first directional derivative and second directional derivative.
- Independently expanded test expressions for mixed unary, binary and ternary
  output assembly and repeated-identity inputs.

The test adapter independently declares symbolic endpoints and a differently
ordered edge list. It verifies all six input storage permutations at widths
0, 1, 16 and 33. The scalar referee never receives the adapter's edge list or
slot map. An explicit control shows that two wrong candidate paths can agree
with each other while disagreeing with independently authored logical equations.
This prevents shared-map agreement from being misreported as validation.

All derivative axes are checked against basis finite differences of the scalar
forward expression. Directional expressions are also checked against explicit
gradient contraction, relation JVP/VJP satisfy the adjoint identity, and the
second direction is checked against a three-point finite difference. Repeated
endpoints require both gradient contributions, and a zero product coefficient
retains its nonzero parameter response. No derivative-through-quantization claim
is made; these are smooth real-arithmetic derivatives at stored FP64 values.

Eight deliberately wrong variants must fail comparison: permuted endpoints,
a second shared faulty map, an omitted quadratic derivative, omitted coefficient
derivative, omitted mixed second-direction term, omitted repeated-endpoint
pullback contribution, swapped ordered n-ary roles, and overwrite substituted
for additive output assembly. These controlled faults are not claims of defects
found in production. Future native tests must compare actual production results
against these referees before claiming production conformance.

The `ce_nf1_t01` executable requires IEEE binary64, disables fast-math and uses
finite-only absolute-plus-relative comparisons. Normal tolerance is 3e-13,
first differences use 3e-10, second differences use 2e-8, and negative variants
must exceed 1e-8. Inventory counts and rejection counts are exact. There are no
GPU, sanitizer, timing, production throughput, or unsupported-operation claims.

The scoped `prepare.py` helper is adapted from GlassHelix commit
`93988ed2fc7e179fa47fe71736935718e0d90538`, path
`tests/native_foundation/independent/prepare.py`; only Cellerator's host configure
flags and executable target differ. This shares build provenance plumbing, not
mathematical oracle code. It requires committed clean source, checks the actual
CMake source root, rebuilds `ce_nf1_t01`, and writes immutable external build
logs with source/cache/bindings identities. The native T01 gate separately
executes the exact CTest inventory and checks successful non-skipped JUnit output.
An external build receipt alone does not count as a passed acceptance gate.

The pre-edit ctxpp scan completed with degraded text routing because this
isolated worktree lacks the configured `build-ss1/compile_commands.json`.
Canonical new test source and actual C++ compilation establish this change;
no semantic source rewrite or token-compression claim was made.
