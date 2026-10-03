# Composable native host effects

`Cellerator::effects` supplies the source-tree C++20 entry points
`<Cellerator/math/effects/providers.hh>` and `affine.hh`. The provider header
aliases the original CE types/functions rather than copying implementations:
DFA/counted DFA, monomial/block-affine maps, weighted algebras, lifting/residual
trees, finite relations, response jets/checkpoints and port condensation/recovery.
Existing `ce_moon` header consumers retain the same types and mathematical owner.

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

The single-cell campaign's native private-port transport is owned by OPS
(`diff/ports/transport.cc`), distinct from the linear condensed-port effect
provider. Its nonlinear matrix-polynomial/trajectory experiments retain their
Torch source at `trajectory/polynomial.py`, `flow.py` and `delta.py`; MATH/DIFF
own promotion. Supplied state rewrites and nonlinear residual regrowth at
`rewrite/` remain ADAPT work. A scalar residual tree or affine carrier is not
an equivalent replacement for those broader functions. Original code and
receipts from both campaigns remain untouched.

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

Add `src/math/effects` and export `Cellerator::effects` through BUILD. Root's
accepted existing component names remain `Cellerator::moonshot_effects`
(`ce_moon/reference.hpp`, `ce_moon/effects.hpp`), `::moonshot_mechanisms`
(`ce_moon/mechanisms.hpp`) and `::moonshot_learning` (`learning.hpp`). No BP
`substrate_*` aliases are requested. BUILD retains the original implementation
headers in the installed SDK and qualifies actual consumer targets; BP's higher
combined consumer waits for that accepted SDK. This host source gate establishes
neither installed package compatibility nor GPU/capture/performance capability.
No existing BP source includes or scientific/sequence semantics were changed.
