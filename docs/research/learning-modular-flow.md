# Joint discovery of modular supports and cellular flow

Research proposal, 16 November 2026. This concerns Cellerator's reusable
geometry, modeling, learning, and execution mathematics. CCell supplies a
motivating atlas experiment; it should not become the owner of a parallel
support-learning engine. The proposal extends the
[compact-flow formulation](ccell/compact-flow-methods.md).
It is not an implemented algorithm, a kernel plan, or a biological finding.

## Proposed approach

Use a sparse, revisable structural search around a jointly fitted dynamical
model. Search which inputs matter, which outputs share a mechanism, which
variables modulate it, and where a simple local law suffices. Within a fixed
structure, fit state, coefficients, and smooth conditions together. Between
such fits, propose small structural changes using residual information and
confirm them on independent biological units.

The main expected acceleration is algorithmic: avoid enumerating, storing,
and repeatedly differentiating the unrestricted interaction library. Seek
reusable sufficient statistics and small coupled fits. Dense batched linear
algebra, packed candidate evaluation, and shared preparation can then use
hardware well. None of those advantages is measured here.

KDA and KernelBender inform this approach through demanded information,
mergeable summaries, reuse, and selective refinement. Their current
mechanisms do not delimit the design space. This is research; no GPU run,
experiment-engine registration, or optimization campaign is needed to reason
about it.

## What is learned

One concrete starting family is

$$
f(x,c)=d(x,c)+\sum_m B_m\,
 a_m(x_{C_m},c)
 \sum_k w_{mk}(z_m)\,\phi_{mk}(x_{P_m};\theta_{mk}).
$$

d is a small baseline, initially diagonal affine dynamics where appropriate.
P_m is participant support; sparse rows of B_m determine output support.
C_m specifies inputs that modulate a mechanism. The w_mk are normalized
local approximation weights within a mechanism; a_m is its modeled activity.
Their roles remain distinct. phi_mk may be scalar or low rank; a true
multi-output mechanism need not be forced into rank one.

Unknowns include P_m, C_m, output support, local regions, coefficients,
necessary latent/context state, and permitted observation calibration. Fix
scale/sign conventions for factorizations so that arbitrarily rescaling B,
a, and phi cannot evade penalties. Charge complexity in gates, state, maps,
and regions as well as coefficient count.

Dependency and conditioning roles can be algebraically ambiguous. The same
product can be represented as two participants or as one participant with
a varying coefficient. We can learn useful argument sets and conditional
response structure within a declared family; assigning a unique molecular
role requires more evidence. Retain equivalent explanations when relevant.

Do not begin by partitioning a coexpression graph and declaring the clusters
to be mechanisms. Geometry and correlations may initialize search, but
support should survive tests of the dynamics.

## A concrete learning loop

1. Split independent biological units into discovery, tuning/calibration, and
   final challenge groups. Related cells or overlapping histories stay blocked.
   Freeze targets, allowed evidence, and evaluation conventions.
2. Choose an anchored, modest state/feature domain and calibrated observation
   family. Keep missing context uncertain; do not optimize an unconstrained
   latent vector per cell against its withheld future.
3. Fit the small baseline and a few candidate mechanisms. Build weak dynamical
   constraints appropriate to the evidence. Linked trajectories and population
   snapshots require different equations; inferred velocities are not truth.
4. With structure fixed, alternate state/context inference and coefficient/gate
   fitting. Preserve posterior moments or explicit state alternatives where
   nonlinear features require them. Updating state changes affected statistics.
5. Propose a bounded batch of participant, output, condition, interaction, or
   region changes using the conditional residual scores below. Include
   exploration outside current supports.
6. Fit the additions jointly with retained mechanisms. Compare additive,
   conditioned, split-region, shared-factor, and observation-error explanations.
   Tune on the calibration groups; repeated calibration is model selection.
7. Prune by removing and refitting entire candidate groups, not by thresholding
   one small coefficient. Try merging local laws and sharing conditions.
8. Freeze the chosen candidates before the final challenge. Report fidelity,
   coverage, unresolved alternatives, structural stability, and complete cost.

This is joint learning through coupled fitting and structural search, not a
claim of a globally solved mixed discrete/nonlinear optimization. Use multiple
initializations and a genuinely different baseline. A smooth model can be a
poor optimizer's solution, and a well-optimized model can be scientifically
inadequate.

Observation fit is an evidence constraint on state inference. It must not
dominate the objective into reconstructing measurement noise. Fix or calibrate
the noise family and bound nuisance flexibility: inflating dispersion,
inventing forcing, or warping each assay freely can otherwise conceal a bad
flow.

## Conditional residual pursuit

For a fixed state estimate, support geometry, and linear coefficient family,
write the weak constraints as

$$
b=A\theta+r.
$$

Rows represent actual evidence constraints, not invented cell-to-cell paths.
Use inverse covariance weights W where justified; diagonal or bounded
specimen-block W permits streamed statistics. General dense covariance may
require a different computation.

Consider a proposed block Z, such as a new participant interaction or a
context-dependent coefficient. First residualize Z against the retained
design in the W metric. Call the result Z_perp. For unregularized active
weighted least squares and a ridge penalty mu>0 on the new coefficients, the
decrease in the fitted penalized quadratic objective is

$$
s_Z=(Z_\perp^\mathsf{T}Wr)^\mathsf{T}
 (Z_\perp^\mathsf{T}WZ_\perp+\mu I)^{-1}
 (Z_\perp^\mathsf{T}Wr).
$$

This follows by minimizing over the new block after eliminating the active
linear coefficients. It is a proposal score, not biological evidence or a
validated held-out gain. Other regularization, uncertain state, or changing
gates require the corresponding augmented solve or a labeled approximation.
Do not present this formula as an exact score for every nonlinear fit.

Crucially, the score needs projected residuals and small Gram/cross-Gram
blocks. It need not materialize a gene-by-gene Jacobian or the complete
candidate library. Maintain active-design factorizations and evaluate proposed
blocks in batches. Normalize candidate scales so that scores and complexity
costs are comparable.

Use multi-output residual patterns to propose shared output support. A
candidate explaining coherent residuals in several outputs may become one
shared mechanism. Low-rank residual sketches can nominate such patterns,
but neither shared assay noise nor factorization uniqueness is settled by
that observation. Confirm direct scores and independent transfer.

Candidate sets can draw on existing support neighborhoods, supplied biological
leads, low-order interactions, and residual sketches. Reserve a budget for
random or deliberately diverse multi-input proposals. A pure interaction
can have no useful singleton signal; insisting on strong marginal effects
would systematically miss it. Zero-initialized products can likewise have
zero gradients for every factor. Score a jointly initialized block rather
than relying exclusively on infinitesimal regrowth gradients.

There is no general shortcut that identifies arbitrary high-order
interactions from sublinear evidence or search. The advantage relies on
restricted arity, reusable structure, adequate excitation, and bounded
candidate families. Record searched coverage. Unsearched means unknown,
not excluded.

## Discovering conditioning variables

Suppose a current mechanism contributes phi_m(x_P). To ask whether z changes
its effect, compare at least:

$$
\begin{aligned}
M_0 &: F_{\rm current},\\
M_{\rm add} &: F_{\rm current}+v(z),\\
M_{\rm cond} &: F_{\rm current}+v(z)+q(z)\phi_m(x_P).
\end{aligned}
$$

Use small linear or spline families for v and q initially. Residualize the
interaction candidate against retained terms, the additive z basis, and
relevant existing context. This prevents a simple additive effect from being
misreported as coefficient modulation. Discovery-only conditional centering
of z can help proposal generation, but cannot create independent excitation.

A reproducible independent gain for M_cond over M_add is evidence for
conditional response structure within these families. To distinguish the
two, the data must contain participant variation within overlapping values
of z. If a and z only change together on one trajectory, a gate may be
indistinguishable from a different participant law. Mark that limitation.

After identifying a small argument set, learn a smooth varying coefficient,
gate, or compactly supported local region. Compare one conditioned mechanism
against many unconditioned patches; sharing one condition may explain the
same variation more compactly. Region centers/scales and gate parameters
are refitted along with the mechanism, with coverage and overlap controls.

A sigmoid is never exactly zero at finite inputs. Skipping its small values
is an approximation needing an error policy. An exactly supported smooth
bump gives exact omission relative to that candidate's declared function,
provided all applicable condition inputs and derivatives are retained.
It does not establish biological impossibility outside its support.

If several mechanisms independently need similar residual context, test
one shared latent regulator with a constrained update law or permitted
history, instead of a separate hidden correction per output. Compare against
assay effects and more patches. Its predictive usefulness does not identify
it as a molecule. The latent's complexity, observation requirements, and
inference cost belong in the comparison.

## Preventing nonlinear observation noise from becoming structure

Weak integration can reduce derivative noise while leaving bias in nonlinear
features. Smoothing noisy counts and then evaluating their squares is not
equivalent to observing squared biological state.

For one useful special case, suppose

$$
Y_j\mid x_j\sim{\rm Poisson}(s x_j),\qquad s>0,
$$

with known or independently calibrated exposure s. The falling factorial
(Y_j)_p satisfies

$$
E[(Y_j)_p/s^p\mid x_j]=x_j^p.
$$

In particular, Y(Y-1)/s² estimates x², whereas Y²/s² includes an extra
measurement term. Under conditional independence across channels, products
of falling factorials estimate mixed monomials.

This suggests constructing corrected weak moments and regression statistics
directly for low-degree candidate laws. Correct Gram products using the
combined degree: the product of two separately corrected features computed
from the same count is not generally an unbiased Gram entry. For example,
estimating E[x^4] with (Y)_4/s^4 is different from squaring (Y)_2/s².
Cross-statistics with evidence targets need the same care.

These are observation-model identities, not a general denoiser. Unknown
exposure, overdispersion, coupled capture, or a different latent-state meaning
changes the correction. Corrected statistics can have high variance and a
finite-sample Gram estimate can be indefinite. Use uncertainty-aware
regularization and report weak support rather than interpreting an unstable
solve as a new mechanism.

Derive the uncertainty and fitting criterion for corrected estimating
equations separately. Their finite-sample statistics need not define the
positive weighted least-squares geometry used in the earlier score formula;
that formula is not automatically an exact gain for a corrected-moment fit.

For calibrated additive Gaussian noise with covariance Sigma, another simple
identity is E[YY^T-Sigma|x]=xx^T. Other observation models require their own
estimators or latent-state inference. Preserve E[phi(x)|evidence] and the
needed cross-moments; substituting phi(E[x|evidence]) generally loses them.

Polynomial corrections do not automatically handle a learned nonlinear
gate or normalized membership. Use valid posterior integration, a separately
qualified approximation, or a different estimating equation there. Do not
claim a finite polynomial moment table exactly evaluates arbitrary gates.

## Structural acceptance without mistaking stability for truth

Require each retained addition to improve a declared dynamical criterion
on independent biological groups beyond a meaningful margin, while meeting
absolute adequacy. Compare the same permitted information and nuisance
flexibility. Complexity tuning and candidate search use discovery/calibration;
keep the final challenge out of repeated structural promotion.

Test ablation and refitting. Also test transfer across embryos and relevant
assay overlap, several horizons, and independently selected weak functions.
Bootstrap or subsample biological units to assess whether supports and
predictions are stable. Inclusion frequencies describe the resampling
procedure; they are not automatically probabilities of true mechanisms.
Whole candidate explanations remain intact.

Use prespecified regime-specific criteria alongside aggregate loss. A rare
but valid mechanism can disappear in an average score. A repeatedly observed
assay artifact can be stable across resamples. Include calibrated technical
alternatives and null controls; repeatability alone does not separate them.

For fixed local fits, inspect rank and conditioning of the design. Distinct
supports can be observationally equivalent along a narrow manifold.
Small coefficients, good reconstruction, sparse solutions, or model agreement
cannot turn unexcited directions into identified dependencies.

## Serious alternative: differentiable structured sparsity

A conventional comparison is to initialize a moderate overcomplete set of
mechanisms and learn group-sparse input/output selectors, gates, and
coefficients together. Proximal group penalties or hard-concrete selectors
are candidate tools. Hard-concrete L0 regularization is an established
method for jointly training gates and weights [2], not proof of recovered
cellular structure.

This route can discover combinations missed by greedy pursuit and may use
dense hardware efficiently. Its costs include evaluating dormant structure,
gradient variance or surrogate bias, dead gates, and a potentially large
overcomplete model. Discretizing masks changes the computation: refit and
revalidate after doing so.

The residual-pursuit route avoids broad recurring work and makes structural
changes inspectable, but can be trapped by its candidate generator or greedy
path. A hybrid can use differentiable fitting within small proposed
super-supports, with explicit birth/death/merge outside. Neither route is
presumed superior without comparison.

## Further algorithmic opportunities

**Acquire only the needed statistics.** Generate candidate blocks on demand
and accumulate their residual cross-products and active-design cross-Grams.
Avoid storing every candidate activation for every cell. Use sketches for
nomination, then direct scoring for serious candidates. Approximate screening
needs a valid bound to make exact rejection claims.

**Separate structural and value lifetimes.** Fixed coefficients can be
refitted many times against a fixed design summary. A changed gate, latent
state, observation calibration, or support can invalidate that design.
Cache identities must include those changes; repeated training inputs do
not imply immutable mathematical features.

**Split search by the type of missing information.** Test a conditioned
coefficient before proliferating local regions; test shared context before
giving every output private state. These alternatives can reduce the
description itself rather than merely accelerate one chosen architecture.

**Use coarse-to-fine uncertainty budgets.** Spend more evaluations near
ambiguous support boundaries or competing conditional explanations. Keep
exploration and direct controls: a cheap scorer can be confidently wrong
when its state or observation model is biased.

**Exploit small dense work inside sparse structure.** Bucket compatible
candidate arities and feature families, pack participant/context gathers,
batch Gram updates and local solves, and share evaluations across outputs.
Reuse fit statistics across leave-one-specimen-out calculations only where
the same feature/calibration recipe remains valid. Remove held-out evidence
rows and their weighted contributions from aggregate statistics. Fold-specific
weights and any fold-dependent fitted transforms or state estimates require
their own recomputation; a matching recipe alone does not establish reuse.
Include packing, invalidation, and latent inference in complete cost.

These are algorithmic hypotheses. No sparse format, Tensor Core route,
precision reduction, or cache is mandated. Start with a strong small dense
baseline; keep it when search overhead exceeds avoided work.

## A first experiment that could reject the design

Use a hidden-to-the-fitter system such as

$$
\dot a=-\alpha a,\qquad
\dot b=-\beta b+\gamma q(z)a,\qquad
\dot z=-\lambda(z-u).
$$

Choose a smooth q, several forcing schedules u, independent initial states,
and a declared measurement model. Vary a and z independently enough to
distinguish additive z from modulation of a. Hide the input/output supports
and condition variable from discovery.

Compare shared-support weak regression, differentiable group-sparse fitting,
and conditional residual pursuit. Keep data, observation information,
candidate arity/budget, and held-out constraints comparable. Measure
held-out dynamical fidelity, domain coverage, false/missed support, conditional
variable recovery where identifiable, and complete training/search cost.
Support precision is meaningful in this constructed known-truth system;
for observationally equivalent decompositions compare response behavior.

Required adversaries include:

- A true interaction whose singleton effects vanish.
- Correlated a and z with insufficient excitation: conditioning must remain
  unresolved instead of becoming a confident structural discovery.
- Poisson observation noise that biases uncorrected quadratic features.
- Incorrect exposure or overdispersed measurements: Poisson corrections
  must not be treated as valid by default.
- An unobserved z, with and without informative permitted history.
- A rare gating regime and a stable assay artifact.
- Conflicting local approximations that create a false equilibrium when
  blended.

First establish statistical and structural behavior with an independent
reference. Then compare algorithmic work and admitted hardware execution
under a separate numerical contract. Unknown true dynamics in an embryo
experiment are addressed through independent evidence constraints, not by
substituting this synthetic oracle for biological validation.

## Source and design evidence

Primary precedents establish pieces, not the proposed joint combination:

1. [Discovery of Physics From Data: Universal Laws and Discrepancies](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00025/full):
   group sparsity couples candidate-term support across related systems.
2. [Louizos et al., Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312):
   hard-concrete gates and joint weight/structure optimization.
3. [Hybrid-SINDy](https://arxiv.org/abs/1808.03251):
   separate regime discovery and switching identification.
4. [Ensemble-SINDy](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006119/):
   resampled sparse models and support inclusion statistics; ensembles can
   still fail or worsen a low-data result.
5. [Weak SINDy](https://arxiv.org/html/2005.04339v3):
   weak-form identification; this alone does not remove nonlinear
   errors-in-variables.

KDA was consulted as inspiration on 16 November 2026:

- KDA revision: 277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc.
- KernelBender revision: 2e19a747a3be2abfb12243db809a3e85ae8b5d42.
- [KDA entrypoint](../../../kda/docs/agent-entry.md) and
  [KernelBender skill](../../../kda/skills/KernelBender/skills/kernelbender/SKILL.md).
- Consumed bridges: demanded information, summaries, lifetime/reuse, and
  approximate-then-refine. Their constructions prompted alternatives;
  they do not validate this estimator or set its scientific limits.

Inspected Cellerator source precedents include
[bounded proposal generation](../../include/Cellerator/compute/support/proposal_discovery.hh),
[adaptive support lifecycle](../../include/Cellerator/compute/support/adaptive_support.hh),
and [dormant derivative sampling](../../include/Cellerator/compute/gating/derivative_sampling.hh).
They provide explicit bounded proposals, staged support publication, and an
explicitly biased dormant-sampling contract. They do not implement this
joint support-learning method. Build/install and performance qualification
were not assessed here.

The protected [cellular geometry definition](../../cellular_geometry.md)
remains unchanged. Its inspected SHA-256 is
d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254.
Local cross-repository links assume the current sibling workspace layout.
