# Composable native host effects

`Cellerator::effects` supplies the C++20 entry point
`<Cellerator/ops/process/effects/affine.hh>`. Existing numerical providers are included
directly from their operation-family headers under `<Cellerator/ops/...>`:
DFA/counted DFA, monomial/block-affine maps, weighted algebras, lifting/residual
trees, finite relations, response jets/checkpoints and port condensation/recovery.
The operation-family headers retain their existing types and mathematical owners.

The new `accumulated_affine<N,Q>` carries precisely
`h'=A h+b`, `q'=q+C h+d`. Composition visits the left region then the right:
`A=AR AL`, `b=AR bL+bR`, `C=CL+CR AL`, `d=dL+CR bL+dR`.
The initial observable may be nonzero. `Q=0` stores no unused readout coefficients.
`hybrid_affine<S,N>` carries fixed finite control transitions and one affine map
per incoming control state. Its right map is selected by the *left output*
control state. Nonbijective transitions are legal; vocabulary mismatches and
out-of-range states are rejected.

New carriers use native persistent axis identities, explicit extents and a
structure epoch for the coordinate/control universe. Composition requires
matching domain/order/geometry/partition identities and versions. Coefficients
are owned immutable-by-convention summaries, not a new canonical parameter
owner. Callers recompose them after changing source parameters/readout maps.
No cache invalidation runtime or scheduler is introduced.

## Exact capability limits

New carriers provide host binary64 application, identity and composition for
their declared affine laws. Finite inputs/coefficients are required; nonfinite
intermediates throw overflow errors. They return values, so failed computation
cannot partially publish a caller output. No numerical VJP/JVP, CUDA, capture,
operator registry, precision adaptation or arbitrary nonlinear closure is
claimed. Continuous affine derivatives can be bound by DIFF later. Integer
control transitions do not acquire an ordinary gradient.

All compositions visit left then right. Counted DFA rejects unsigned overflow.
Lifting/block/port providers retain their existing input and finite arithmetic
checks; monomial results must pass their existing validation before application.
Floating matrix composition is equivalent in real arithmetic and compared with
tolerance; no bitwise associativity promise is made. A jet's right expansion
point must match its left output value. Jet finite-perturbation estimates and
radius admission supply no certified Taylor error bound; changed parameters or
expansion context require recomputation. Port condensation retains the original
linear-model/pivot threshold limits and is not a conditioning certificate.

Native private-port transport lives under `src/ops/process`. The
[original Moonshot campaign](https://github.com/tumlinso/Cellerator/tree/2c2c222f36c3aa1f7f8bb22d97031002c6420978/experiments/moonshot-parallel-v1)
preserves nonlinear matrix and trajectory references, supplied state rewrites,
residual regrowth, and their source-bound receipts.

## Native gate

```sh
python3 -B tests/substrate/effects/check.py --build-dir /tmp/ce-is1-effects-host
```

The consumer executes the actual original C++ numerical providers and both new
carriers. It verifies direct sequential application, noncommuting matrices,
nonzero accumulated observables, empty identity, all control states,
nonbijective routes, three-region composition within tolerance, universe/epoch
mismatch, malformed transitions, finite overflow, unsigned overflow, expansion
point mismatch and original port interior recovery. Legacy header formatting
requires disabling only `-Wmisleading-indentation`; strict warnings, no fast
math and no contraction remain enabled. No GPU or benchmark is launched.

## Shared build and installed ownership

The selectable SDK exports `Cellerator::effects`. Source consumers use the
target defined in `src/ops/process/effects`, and installed consumers select the
`effects` component. The independent source gate exercises host arithmetic.
The installed package has a separate relocation check.
