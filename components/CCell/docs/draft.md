# Joint modular flow learning research brief

Disposition: propose a conditional residual-pursuit route and compare it
against a strong differentiable structured-sparsity baseline. This is a
research design brief prompted by KDA/KernelBender, not an implementation
or GPU optimization task.

The intended owner is Cellerator: geometry, support discovery, conditioning,
local laws, and reusable fitting mathematics belong there. The canonical
[research proposal](../../Cellerator/docs/research/learning-modular-flow.md)
is connected to CCell's [flow objective](research/compact-flow-methods.md).

## Required result

Specify how to learn participant/output supports, conditioning arguments,
and flow together under calibrated observation assumptions. Return adequate
candidate laws with scope, uncertainty, and evidence of structural usefulness.
The objective is compact dynamics across a supported domain, not noisy-count
reconstruction or unconditional batch mixing.

The current baseline is a methodological proposal, not a running discovery
pipeline. Source inspection found bounded proposal generation, staged
adaptive-support publication, and explicitly biased dormant derivative
sampling in Cellerator. Those facilities are precedents rather than proof
of joint learning.

## Proposed construction

Within a fixed structure, jointly fit state/context, local coefficients,
and smooth gates. Between fits, score bounded support changes through
residual projections and small Gram blocks. Test additive context against
coefficient modulation; use removal/refitting, independent specimens, and
appropriate evidence constraints for structural acceptance.

Use calibrated observation moments or state inference to avoid turning
nonlinear predictor noise into apparent interactions. Poisson factorial
moments illustrate one exact special-case correction. Do not generalize
it to unknown exposure, overdispersion, or arbitrary gates.

The serious alternative initializes a moderate super-support with group
penalties or differentiable selectors. Compare both routes on held-out
dynamical fidelity and complete discovery cost, preserving rare regimes
and pure interactions with weak singleton signals.

## First discriminating action

Design the independent conditional-system fixture in the proposal. It
must distinguish a true gate from an additive covariate, include an
unidentifiable confounded arm, and expose noise-induced false quadratic
terms. Fix evidence permissions, arity/search budgets, and numerical
requirements before implementing candidates.

Reject a route that confidently discovers conditioning under insufficient
excitation, misses the declared pure-interaction control without reporting
search limits, or improves training fit while failing independent dynamics.
There are no executable validation/evaluation commands or performance
results yet. A future implementation task must supply them and the resource
contract before running KDA candidate evaluation.

## KDA inspiration and evidence

Consulted KDA agent entry, starter/agent flow, and KernelBender demand,
summary, lifetime, and refinement bridges. KDA revision:
277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc; KernelBender revision:
2e19a747a3be2abfb12243db809a3e85ae8b5d42.

No profile, kernel, GPU allocation, task registration, or scientific result
is implied. The canonical proposal records source references and the
inspected cellular-geometry identity. KDA remains optional design context,
not a ceiling on mechanism invention.
