# Executable host mathematical frontier

The public `Cellerator::frontier` component uses existing integrated matrix axes,
four-role generation envelopes and borrowed primal tapes, plus the actual
`ce_moon::mechanisms::Matrix`, multiply, partial-pivot multiple-RHS solve,
condense/compose/recover ports and coarse-direction owners. Elementwise FP64 addition
uses native_numeric. No solver, allocator, optimizer or canonical parameter owner
is copied or replaced. Results own numerical output/scratch; caller inputs remain
borrowed. Outputs cannot overwrite borrowed primal objects.

## Mathematical and capability boundaries

| Family | Label and native behavior |
| --- | --- |
| M02 polynomial | M: declared `LX+XR+XMX` model. FP64 forward, simultaneous X/L/R/M JVP and ordered role VJP. Storage sharing retains separate parameter adjoints. |
| N04 finite delta | E: `LD+DR+DMX+XMD+DMD`, fixed coefficients and unchanged saved generations. The DMD term is present. E is real-arithmetic identity, not bitwise subtraction equality or biological equivalence. |
| Multiple-RHS solve | E for supplied `AX=B`; checked native partial-pivot solve. JVP solves `A dX=dB-dA X` using the same owner. No recursive derivative tape or solve VJP. |
| N05 boundary ports | E for supplied linear regional systems with identical boundary axis/order and distinct independent interiors. Actual condensation, port composition, checked Schur solve and interior recovery; assembled full residual is admitted. |
| N06 multilevel | M by default for declared `D+PKR`; E only when the caller supplies an exact decomposition; A labels an approximate decomposition. Native forward and input VJP use reverse transposed factors. No parameter VJP/JVP claimed. |
| N06 coarse correction | A for one fixed-budget correction toward a general fine solve. Existing coarse_direction owner, explicit alpha and full residual-ratio admission. No convergence promise. |

Every call uses finite FP64 host arithmetic under nearest-even mode. Source/provider
floating association is retained, so algebraically equivalent decompositions are
compared with tolerances. GPU, capture, asynchronous streams, nonlinear jets,
low-rank recompression, Sylvester general solves and universal tensor laws are not
implemented by this component.

## Admission and lifetime

The normalized solve residual is `||AX-B||∞/(||A||∞||X||∞+||B||∞)`. The guard also
bounds observed `||A||∞||X||∞/||B||∞` on the supplied RHS. This is not a global
condition-number estimate or certificate; weak RHS probes can miss bad directions.
The underlying relative pivot guard rejects singular/near-singular pivots.

Standalone checked_solve honors caller pivot tolerance. The preserved port and
coarse owners internally solve at 1e-12, so composed operations explicitly reject
smaller requested tolerances before publication. Policies at or above that floor
are checked on required interior/coarse/Schur solves. No explicit inverse is formed.
Full-system residual checking constructs dense diagnostic scratch; no sparse scaling
or constant-memory claim is made. Raw native provider calls remain available.

The caller must keep every borrowed Matrix/vector/generation owner alive and publish
all state, coefficient, RHS or context changes through the four generation groups.
Tapes save generations and reject changed dependencies. Port solutions expose an
explicit current-generation check. Primal values and metadata must remain unchanged
until responses; unreported writes or concurrent publication are unsupported.

## Native consumers

```
python3 -B tests/substrate/frontier/check.py --build-dir /tmp/ce-is1-frontier-host --sdk-prefix /tmp/ce-is1-sdk-a
```

83 checks cover a native quadratic-delta→multilevel composition and native regional
port solve/recovery→residual correction composition. Independent polynomial scalar
oracles, simultaneous state/parameter finite differences, JVP/VJP duality, shared
parameter roles, multiple RHS, solve JVP, old generations, malformed axes/shapes,
primal aliases, ill-conditioned guards, duplicate interior domains and rejected
residuals are included. Custom-pivot regressions demonstrate the native composition
floor without changing the external solver. The test links accepted installed native
providers and compiles only this new component; it runs no Python/Torch or GPU code.

## Preserved reference directions

- `planning/integrated-substrate-v1/seed/python/is1/operators.py`: N04 QuadraticLedger,
  exact finite delta, N06 multilevel and solve JVP remain sealed mathematical
  references. A persistent ledger/cache and coefficient rebase are not added here.
- `experiments/moonshot-parallel-v1/trajectory/polynomial.py`: original M02 Torch
  polynomial/JVP/VJP remains the reference and adapter path.
- `experiments/moonshot-parallel-v1/seeds/reference/moonref/operators.py`:
  `low_rank_implicit_solve` remains callable M11 Woodbury reference. Native
  diagonal/low-rank dispatch and recompression are deferred; these seeds are not
  reported as implemented native mathematics.
- The integrated `math/matrix/patch.hh` tanh law remains a distinct operation.
  Polynomial APIs reuse its metadata/lifetime seam, not its numerical formula.

## Root MERGE-B SDK recipe

Add known component `frontier`, dependencies `effects;native_numeric`, and dispatch
`_ce_owner(frontier SOURCES src/math/frontier/operators.cc CAPABILITIES
"host_f64_quadratic_delta_jvp_vjp,ports_checked_solve,multilevel_input_vjp")` in the
selected SDK module. Request `frontier` explicitly for its installed component export.
The existing public-header install includes `include/Cellerator/math/frontier`.
The source target from `src/math/frontier/CMakeLists.txt` is
`Cellerator::frontier`. No shared root/SDK file was edited in this task.
