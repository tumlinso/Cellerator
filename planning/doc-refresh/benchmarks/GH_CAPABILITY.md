# Optional GlassHelix demonstration: ambiguity, then discrimination

**Purpose:** make the scientific question visible without model training or pretending to discover a molecular mechanism. Reuse current whole-candidate propagation and observation APIs; implement only a small example/test wrapper after checking the pending native repairs.

## Precisely specified synthetic fixture

Two supplied alternatives have scalar state x and controlled input u. At the tested step, A predicts x_next = x + u and B predicts x_next = x - u. Initial x=0. Equal priors 0.5, same Gaussian observation sigma=1. Each candidate travels through the actual qualified Cellerator-backed GH propagation/observation route; do not feed its oracle output directly to a plot.

1. Passive observation u=0, y=0: both predict 0. Support remains 0.5/0.5. This is deliberately non-discriminating evidence.
2. In a fresh pair, controlled input u=1 and y=1: A predicts 1, B predicts -1. Posterior A is 1/(1+exp(-2)), B is its complement. This tests discrimination within the supplied family, not universal identifiability.
3. A control u=1, y=0: equal likelihoods again, so support remains 0.5/0.5. Intervention alone is not enough; the observation must discriminate.
4. Repeat an evidence ID: the current API should reject duplicate evidence. Record the actual exception/status, not just a success flag from this specification.

`gh_reference.py` calculates expected values **only**. Its output is labeled an analytical oracle and must not appear as product results. The actual example emits a small JSON record with observed candidate predictions, weights, evidence IDs and provenance. Compare against the oracle to within 1e-10 on the scalar weight calculation, or explicitly justify a tolerance for the selected numerical route. Do not trigger the known underflow corner to make a broad robustness claim; that repair belongs to its existing owner.

## Implementation surface and output

One example source and one registered existing-style test, plus a tiny result adapter. No general optimizer, trainer, data loader, new numerical backend or generic benchmark harness. Use the current installed GH/CE route even if filenames changed; bind its path at the end. Run with leased resources only where the chosen route requires CUDA.

Produce three rows for the support comparison and a separate duplicate-evidence check. The prepared result/figure template `gh-discrimination.json` remains pending until actual library results exist. The caption must say synthetic supplied alternatives. Document what would remain unknowable with only the passive observations.

Failure is meaningful: if the supplied alternatives or evidence are mishandled, leave the figure unpublished, preserve the witness and hand it to the existing repair owner. Do not silently replace the library with the oracle to finish a pretty chart.
