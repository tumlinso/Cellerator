# Recovering compact cellular dynamics over an observed domain

Research methodology proposal, 16 November 2026. This is the governing
formulation for the [atlas assessment direction](atlas-quality-direction.md).
The earlier [eight assessments](atlas-quality-methods.md) provide supporting
tests. No implementation, biological result, universal model family, or
low-level development plan is claimed.

The concrete proposal for jointly learning participant/output supports,
conditioning variables, and flow is now in
[Cellerator's modular-flow research note](../../../Cellerator/docs/research/learning-modular-flow.md).
Geometry and reusable support-discovery/learning mathematics belong to
Cellerator. This note retains the atlas motivation and evidence requirements.

## Objective and the unknown truth

Recover the most compact adequate description of biological evolution that
the observed domain supports. Measurements constrain the description; the
target is the dynamics, not a compressed reconstruction of noisy observations.
Integration estimates how observations from different experiments relate to
states evolving under that shared description.

A deterministic working form is

$$
\dot x=f_\theta(x,c),\qquad y_i^{(b)}\sim p_b(\,\cdot\mid x_i,c_i).
$$

c contains declared forcing, environment, or time-dependent context. The
observation family p_b relates biological states to each assay. If needed,
replace the deterministic law by a stochastic generator with drift, diffusion,
or jumps. Technical noise and intrinsic biological variability then have
different roles; suppressing all variability would suppress some dynamics.

The true f is unknown. We can learn candidates, reject candidates using
independent constraints, and characterize unresolved directions. A tested
candidate may approximate the real flow without identifying a unique
molecular mechanism. Finite observations generally support an admissible
family over a stated region and horizon, rather than unrestricted recovery.
This is a useful result if the family agrees on the flow components that the
experiment actually constrains.

Use a fidelity–complexity–coverage comparison. Describe the local laws,
dependencies, supports, shared parameters, necessary context, coordinate
maps, observation calibrations, and state inference. Penalize complexity
moved into an elaborate gate or per-cell hidden code as well as complexity
in f. Measure the biological domain on which predictive constraints hold.
A tiny model covering only an easy corner is not equivalent to one covering
the declared developmental domain.

One model-selection formulation is to minimize total description complexity
subject to prespecified independent dynamical-error margins and coverage
requirements. In practice, compare a frontier of adequate candidates under a
common coding or regularization convention, rather than treating parameter
count alone as a universal number of bits. Preferring the simplest surviving
flow is a modeling choice; it does not remove physical ambiguity that the
measurements leave unresolved.

This bookkeeping matters mathematically: a smooth nonzero vector field can
be made constant locally by a suitable coordinate change [3]. Simple-looking
dynamics in unrestricted coordinates therefore do not establish a compact
description of the system. Biological anchors, physical time, restricted maps,
and the total representation cost prevent a decoder from hiding the problem.

## Which evidence can constrain which flow

| Evidence | What it can constrain | What remains conditional |
|---|---|---|
| Unordered snapshots | Occupied state geometry and observation distributions | Direction, speed, and cross-state couplings |
| Stage-indexed population samples | Changes in distributions and their moments | Individual paths; growth, capture, and unobserved currents |
| Spliced/unspliced or labeling measurements | Selected kinetic and directional restrictions | Kinetic model, rates, units, and measurement calibration |
| Actual repeated or linked temporal evidence | Finite-time predictions under the stated relationship | Clone bags are not repeated measurements of one cell |
| Perturbations or varying environment | Responses and excitation of additional directions | Intervention meaning and comparability of regimes |
| Independent modalities | Hidden-state and observation-model restrictions | Pairing, missingness, and interpretation of inferred coordinates |

There are useful limits we can demonstrate without knowing the biological
truth. A stationary uniform population on a circle is compatible with zero
drift and with constant angular flow of either sign and many speeds.
More generally, with known density rho, growth, and observation process,
temporal marginals constrain the continuity equation. Adding a current u with
div(rho u)=0 leaves that equation unchanged under compatible boundaries.
Thus population marginals alone can leave an entire family of flows unresolved.
Particular process restrictions can make recovery possible; the gWOT theory
is a precedent for such an assumption-dependent result, not general recovery
of arbitrary vector fields from snapshots [4].

Natural developmental variation can nevertheless excite enough directions
to identify restricted coefficients or responses. Check the rank and
conditioning of the actual inference problem. Lack of designed perturbations
is not by itself a proof of non-identifiability.

## Three supports that must remain distinct

1. **Dependency support:** which state components a mechanism reads, including
   the inputs of its conditions. This is Cellerator's central structural idea.
2. **Local validity support:** a region of state/context over which a particular
   approximation to that mechanism is used.
3. **Evidence support:** the states, directions, and temporal scales actually
   constrained by observations.

A learned approximation can be smooth throughout its mathematical validity
region and still have little evidence in part of that region. A compact
support in a small projection can also hide variation in omitted variables.
Claiming that projection suffices is a separate hypothesis to test across
the observed contexts.

A missing observation is not a forbidden interaction. An empty region of a
data cloud is not necessarily a biological barrier. A candidate's exact
support is a computational property of that candidate, not proof of a
natural exclusion.

## A candidate construction with smooth local laws

Distinguish additive mechanisms from alternative local approximations to
the same mechanism. One proposed two-level construction is

$$
f_\theta(x,c)=\sum_m S_m\,a_m(x_{C_m},c)
 \sum_k w_{mk}(z_m)\,
 F_{mk}(x_{P_m},x_{C_m},c;\theta_{mk}).
$$

S_m places a mechanism's output in common state coordinates. P_m and C_m
specify participants and context. The eligibility a_m belongs to the model's
biological hypothesis. The w_mk select or blend local approximations to the
same mechanism; they sum to one on that mechanism's covered domain. They are
not automatically biological gates. z_m contains the declared inputs needed
to locate the approximation, including context where necessary.

Mechanisms may overlap and contribute with opposite signs. Their
contributions add, and physical cancellation can be legitimate. In contrast,
averaging two incompatible estimates of the same drift can create an
artificial equilibrium. These two situations need different tests.

Start with simple local families: affine laws, a few specified interactions,
or low-rank functions with shared factors. Avoid a full polynomial library
over all genes. Sparse equation discovery such as SINDy is an established
precedent for selecting a small set of candidate terms [1]; its library and
sampled domain still determine what it can recover.

For a concrete smooth membership in locating coordinates z, choose a
nonsingular scale matrix L_mk and let

$$
s_{mk}(z)=\|L_{mk}(z-\mu_{mk})\|^2,\qquad
\psi_{mk}(z)=
\begin{cases}
\exp[-1/(1-s_{mk}(z))],&s_{mk}(z)<1,\\
0,&s_{mk}(z)\geq1.
\end{cases}
$$

Set w_mk=psi_mk/sum_l psi_ml where the denominator is positive and
numerically usable. This is one smooth construction compactly supported in z;
partitions of unity provide the general mathematical idea [3]. The closure
of its nonzero region is its support. There is no claim that an ellipsoid
is the correct biological validity region. Pulling it back to a larger state
can give a cylinder over unread variables; this does not make the full-state
support compact or establish independence from those variables.
Keep each closed support inside a region where its local law and required
derivatives are regular, or establish a regular zero extension of the
weighted law. Smoothness strictly inside an open patch alone does not control
singular behavior at its edge.

Coverage and overlap must be explicit. A zero or inadequately small
denominator means unsupported evaluation, not zero biological drift.
Very narrow overlaps can amplify derivatives even when all weights are
smooth. Require usable coverage margins and test sensitivity near seams.
If the latent state is uncertain, retain the union of possibly applicable
supports, or integrate over candidate states; a point estimate does not
justify excluding a plausible mechanism.

This gives a possible scientific and computational advantage: a complex
global law can use simple local formulas, with only a few applicable formulas
per evaluation. Indexing and condition costs must be included. Highly
overlapping or high-dimensional supports can eliminate the advantage.

## Learning local laws without assuming true velocities

The inference route must match the evidence. Do not run a trajectory-based
regression on unrelated destructive snapshots.

A concrete initial algorithm can use biologically anchored coordinates and
an overlapping landmark cover of the training domain, with fixed smooth
memberships. Fit small local candidate laws against the available constraints;
then compare splitting a support, enriching its formula, or adding context.
Require adequate independent specimen support in each region. Landmark
selection and coverage geometry are initialization choices, not recovered
biology. Learning state coordinates, unknown memberships, and nonlinear laws
simultaneously is a later, harder inverse problem; it need not be solved to
test the local-law construction.

**Observed or properly linked trajectories.** For a differentiable state path
of a deterministic ODE and a smooth temporal test function eta that vanishes
at its endpoints,

$$
-\int\eta'(t)x_j(t)\,dt
=\int\eta(t)f_{\theta,j}(x(t),c(t))\,dt.
$$

For stochastic linked paths, the corresponding pathwise relation also contains
martingale and applicable jump terms. Use an expectation/generator identity
or a properly specified martingale estimating relation; do not treat every
random path increment as deterministic drift error.

For a fixed local basis linear in coefficients, these give a small regression
system A theta=b. Weak/integral identification avoids pointwise numerical
differentiation; Weak SINDy provides a direct precedent [2]. It does not remove
errors-in-variables in nonlinear basis evaluations, unknown state, or biased
measurement noise. Infer states through a constrained observation model or
use an appropriately corrected estimator.

**Population snapshots.** For smooth state test functions phi, an appropriate
stochastic generator is

$$
\mathcal L_\theta\phi
=\nabla\phi\cdot f_\theta+
 \tfrac12 B:\nabla^2\phi+
 \int[\phi(x+\xi)-\phi(x)]\,\nu_\theta(d\xi\mid x,c).
$$

The jump term here assumes a finite-activity jump representation. With
population mass measure mu and net growth g,

$$
\int\phi\,d\mu_{t_1}-\int\phi\,d\mu_{t_0}
=\int_{t_0}^{t_1}\int
 [\mathcal L_\theta\phi+g\phi]\,d\mu_t\,dt,
$$

assuming appropriate boundaries or including their flux. Finite stages
require a temporal interpolation/quadrature model and its uncertainty.
Relative capture and normalized proportions do not supply absolute
biological mass. Bound or calibrate sampling/growth and carry uncertainty;
otherwise the weak constraints can become vacuous.

Select test functions involving only a few module coordinates, then
accumulate moments rather than estimate a high-dimensional density or solve
every cell-to-cell correspondence. Separate fitting functions from new
functions used for challenge. Waddington-OT is a precedent for growth-aware
population coupling [5]; the proposed route fits or refutes a structured
generator through selected weak constraints.

**Kinetic evidence.** Splicing or labeling models can constrain some local
derivative components. Jointly estimate a flow consistent with those
restrictions rather than treating a precomputed velocity embedding as the
true field. Shared uncertain rate parameters should stay shared across
constraints. Gene-specific scaling, inconsistent units, or reused graph
targets can change the inferred geometry. RNA velocity remains model-derived
evidence, with assumptions that need examination [6].

With fixed supports and bases, cache local sufficient statistics and use
small constrained fits. When state estimates, supports, or memberships change,
the corresponding design must be updated. Alternating these fits is a
candidate algorithm, not a convexity or convergence guarantee.

## Joining local descriptions and challenging the seams

### Same coordinates and same mechanism

In an overlap, compare independently fitted F_mk and F_ml on withheld
states or common physical readouts, with their state and fit uncertainty.
Look at direction, magnitude in common time units, and finite-time behavior.
Use effect-size margins. Failure of an underpowered disagreement test does
not validate a seam.

If a shared target f is within epsilon of each local approximation, a
nonnegative normalized blend is also within epsilon in a fixed norm at
that state. This elementary bound explains why blending compatible
approximations is attractive. Data must justify the premises; smooth
weights do not make them true.

For example, two alternative scalar fits F_1=1 and F_2=-1 give zero at an
equal-weight seam. That equilibrium is entirely artificial if the true
motion there is unidirectional. Record the disagreement before blending.
Investigate missing context, state aliasing, an inadequate local family,
or an unsupported connection instead of averaging the discrepancy away.

### Different coordinate charts

If u and v describe the same state region with v=tau_ab(u), compatible
deterministic vector fields satisfy

$$
F_b(\tau_{ab}(u))=D\tau_{ab}(u)F_a(u).
$$

This rule assumes time-independent charts and matching clocks. Additional
time/context terms are required for moving charts. Stochastic Itô coordinate
changes also need diffusion and second-derivative corrections; transforming
the drift alone is insufficient.

Fit and challenge the overlap maps using independent anchors. Preserve
local rank and test round trips and cycles on actual overlap. Agreement of
a coordinate cycle alone is not evidence of the biological flow.
Never average vector components in unaligned coordinate systems.

Even for deterministic charts, the transformed Jacobian contains derivatives
of the coordinate map; it is generally not a simple similarity transform.
For a weighted local law,

$$
D(wF)=w\,DF+F(\nabla w)^\mathsf T.
$$

Gate derivatives and shared conditioning contribute too. Dropping these
terms would alter both learning and sensitivity assessment.

### Smoothness and genuine regime changes

On a constrained state manifold, local fields must be tangent to its
admissible directions. Weighted sums of tangent fields remain tangent in
the same state space; unconstrained smooth fits need not. For a justified
linear invariant A x=b, parameterizing f=N v with A N=0 removes forbidden
directions before fitting. This can reduce both inference and execution.
Do not infer biological conservation from library-size normalization or a
coordinate convention, and treat boundary/positivity constraints separately.

On covered regions, locally Lipschitz laws give local uniqueness of
deterministic solutions for specified context. They do not imply global
stability, boundedness, or a valid cellular model. Enforce appropriate
physical constraints separately and stop or broaden uncertainty when the
inferred trajectory exits the supported domain.

A complete deterministic state with the same forcing cannot have two
different instantaneous futures. Apparent branching can indicate unresolved
state, stochasticity, division, or a regime/input change. An observed
projection can fold or cross even when the full-state flow is well defined.
Use augmented state or explicit stochastic/hybrid models where supported;
do not force such evidence into a globally smooth snapshot-only field.

## Shared mechanisms as an integration criterion

Test whether one set of local biological laws, with independently calibrated
assay maps and allowed specimen context, explains new embryos and assays.
Compare it with separately fitted laws using matched inference and fitting
budgets. Successful sharing with adequate independent predictions is evidence
of reusable dynamics and a more compact explanation of the observed domain.

Do not demand identical states, population composition, or forcing in every
embryo. Conversely, arbitrary assay warps, per-cell clocks, hidden identifiers,
or residual forcing can make almost any data appear to share a flow. Restrict
their families, charge their description cost, and test them on new evidence.

A local shared-law discrepancy can arise from an assay calibration error,
missing context, a genuine regime difference, an inaccurate law, or poor
measurement. Compare these explanations. A residual is not a cell-quality
label by itself.

This criterion extends integration assessment beyond proximity and label
preservation. Individual response and predictive-collision tests from the
earlier note remain useful challenges, but they probe consequences of the
flow rather than define the complete objective.

## Determining whether new data follows a learned flow

Use a fit–freeze–challenge protocol at biological-unit level:

1. Discover candidate state coordinates, module contexts, supports, and laws
   using training specimens and permitted evidence.
2. Tune support sizes and complexity on separate calibration specimens.
3. Freeze the biological law. Infer a new specimen's initial state and permitted
   context from only its authorized observations.
4. Evaluate withheld temporal, kinetic, modal, or intervention constraints.
   Observation maps are frozen from training or disjoint calibration controls.
   Any new-specimen calibration uses only prespecified target-free controls;
   withheld challenge values cannot fit the map, initial state, or context.
5. Report discrepancy, uncertainty, coverage, and competing explanations.
   Refinement starts a new fitting cycle with a fresh final evaluation.

For an actual later observation, write h_b for the observation mean when it
exists. A basic innovation is

$$
e=y_{\rm later}
 -h_b(\Phi_{\theta,\Delta}(x_{\rm early},c),c_{\rm later}).
$$

Assess it using the observation model and uncertainty in the early state,
parameters, forcing, and measurement. Use a distributional score where
appropriate. A descendant population needs a population prediction; it
cannot be replaced by an invented individual pair.

One useful localization is to compare the innovation with the predicted
observable flow direction and tangent space. A timing error can resemble
an along-flow displacement, while another state dimension or corruption
can produce a transverse discrepancy. Use an observation-aware metric.
This is a diagnostic decomposition, not a unique causal classification:
law errors and missing context can affect either component.

Challenge multiple time intervals and withheld intermediate stages.
An autonomous sufficient-state model obeys flow composition, but composing
its own numerical solver with itself only checks that implementation.
Compare against independent observations or independently fitted finite-time
constraints. With forcing, compose over the same absolute input schedule;
with stochastic dynamics, compare transition distributions.

A pass supports compatibility with the candidate family over tested regions,
directions, and horizons. A failure refutes that candidate/calibration/context
combination at the stated tolerance. Insufficient support or precision remains
unresolved. Agreement does not prove that the recovered law is the unique true
one, and no finite test bank establishes correctness over an entire continuum
without additional approximation and coverage bounds.

## Refining supports without explaining every residual as noise

Use reproducible residual structure to propose the smallest justified change:

| Pattern on independent calibration data | Candidate change to test |
|---|---|
| Smooth systematic error within a well-sampled patch | Enrich its local law or split the validity region |
| Similar states disagree according to permitted history | Add state/context before adding more snapshot patches |
| Several mechanisms share the same unexplained variation | Try one shared context variable and update law |
| Error tracks assay within genuine biological overlap | Reassess the observation model |
| Overlap fields disagree in otherwise comparable states | Repair the seam, change context, or retain rival connections |
| Little excitation or uncovered region | Preserve uncertainty or acquire evidence |
| Irreducible variation under a deterministic candidate | Compare stochastic, forcing, and measurement alternatives |

These are competing hypotheses, not automatic diagnostic rules. Compare
split, added-context, shared-factor, and noise changes on the same held-out
constraints. Accept improved absolute fidelity at justified additional
complexity; do not select the most elaborate explanation by training fit.

For a fixed local regression, examine small singular values of A. Directions
delta with A delta approximately zero describe coefficient changes weakly
constrained by that evidence. Propagate whole candidate coefficient sets
rather than averaging unrelated edges. For nonlinear state inference this
is only a local sensitivity analysis. Data on a low-dimensional manifold
may constrain tangent behavior while leaving normal derivatives arbitrary.

Try merging patches or tying parameters when independent constraints remain
adequate. Reuse one mechanism across several regions when only its context
changes. This is how compactness can be discovered, rather than prescribed
as a fixed latent dimension or a fixed number of modules.

## Extending a valid domain and bounding approximation

Internal smoothness allows controlled interpolation only with evidence and
regularity conditions. It does not grant biological extrapolation across
an empty region.

For two deterministic flows with matching forcing, initial separation delta_0,
a Lipschitz constant L for one field, and a uniform field discrepancy at most
epsilon on a domain containing both trajectories, a standard differential
inequality gives, for L>0,

$$
\|x(t)-\widehat x(t)\|
\leq e^{Lt}\delta_0+\frac{\epsilon}{L}(e^{Lt}-1).
$$

For L=0 the bound is delta_0+epsilon t. A statistically estimated residual is
not automatically a uniform epsilon bound. Field smoothness, state uncertainty,
measurement calibration, and domain coverage would need to justify it.
Otherwise report empirical horizon-dependent error.

This suggests adaptive allocation of effort: evaluate broad inexpensive
local laws where they suffice, refine near poorly constrained seams and
response-sensitive regions, and stop when uncertainty leaves the supported
domain. Cellerator could propagate numerical or candidate-model bounds with
the same support structure, while scientific evidence determines whether
those bounds relate to biological fidelity.

## Reusable mathematics and scientific ownership

Several central questions belong naturally within Cellerator's reusable
mathematical substrate, even though biological truth does not:

- Local functions with declared participant and context dependencies, smooth
  memberships, distinct eligibility, and explicit coverage.
- Composition of additive mechanisms and alternative local approximations,
  with derivatives through gates, memberships, and coordinate changes.
- Support lookup and conservative active sets under uncertain state.
- Weak constraints, selected derivative actions, and small local fits.
- Coordinate-consistency and overlap-discrepancy calculations.
- Shared preparation, local updates, and explicit numerical approximation
  assessment.

This is a conceptual responsibility map, not an assertion of existing
capabilities or a low-level operation backlog. Cellerator's current
[factorized interactions](../../../Cellerator/docs/factorized_interactions.qmd)
and [incremental computation](../../../Cellerator/docs/incremental_cellular.qmd)
are bounded precedents for explicit joint inputs, conditioning, derivatives,
and invalidation. They do not already discover these supports or recover a
biological flow. Smoothly changing state does not by itself make a cached
result exact or authorize skipping a changed dependency.

CCell owns the atlas observations, data joins, assay conventions, scientific
kinetic methods, and biological validation of this application. Reusable
formal inference and whole-explanation comparison can draw on GlassHelix.
Velocity and history mathematics may develop in several places according
to their scientific and computational meaning. Their absence is not a
prerequisite for choosing a coherent experiment now.

The main computational opportunity is to reuse compact laws, inputs, and
fits across compatible observations while evaluating only applicable local
work. Account for indexing, uncertain-state inference, support maintenance,
overlap, and derivatives. A model restriction and an execution optimization
can share structure while still requiring distinct evidence.

## First experiments that could disprove the construction

Begin with anchored, low-dimensional systems whose laws and observation maps
are known to the evaluator but hidden from fitting. Generate separate
training and evaluation specimens and observation noise realizations.

| System or intervention in the experiment | Required result |
|---|---|
| Uniform circle with snapshots only | Retain ambiguity in direction and speed |
| Same circle with sufficient temporal pairing or kinetic restrictions | Recover the components those measurements constrain |
| One smooth law with two compatible overlapping local fits | Join them without a seam artifact and transfer to new states |
| Oppositely directed alternative fits in an overlap | Expose disagreement before a blend invents a fixed point |
| Two coordinate charts of the same field | Recover agreement after pushforward, not raw component comparison |
| Same observed image with two hidden gate states | Retain ambiguity or recover useful context when evidence supplies it |
| Local approximation with a shared hidden regulator | Compare adding one shared state with proliferating patches |
| A sparsely sampled normal direction off a manifold | Preserve uncertainty rather than claim an identified Jacobian |
| Changing growth/capture with unchanged motion | Avoid inventing drift to explain every population change |
| Constant latent state and expressive decoder | Fail independent dynamical constraints or incur the displaced description cost |
| A real regime switch or stochastic branch | Prefer adequate augmented/hybrid/stochastic explanations over forced smoothing |

A narrow first implementation could fit smooth overlapping laws in anchored
coordinates using known pairing, then add a population-constraint arm that
must explicitly retain its unresolved current. These two arms separate
whether the construction works from whether a particular experiment contains
enough information. Subsequently add unknown assay calibration and hidden
context, one challenge at a time.

For the embryo pilot, inventory actual temporal, kinetic, and relationship
coverage first. Learn a modest shared law on one declared biological regime
and freeze it before new-embryo or held-out-stage challenges. The research
result may be partial recovery, a refuted family, or a specific missing
measurement. None requires pretending the true dynamics were available.

## Sources and proposal boundaries

Sources below establish mathematical or methodological precedents. The
particular synthesis, support-refinement procedure, and proposed embryo
assessment are hypotheses developed here, not claims of priority or validated
methods. Sources were consulted on 16 November 2026.

1. [Brunton, Proctor, and Kutz, sparse identification of nonlinear dynamics](https://arxiv.org/abs/1509.03580):
   sparse candidate-law discovery with library and domain assumptions.
2. [Messenger and Bortz, Weak SINDy](https://arxiv.org/html/2005.04339v3):
   weak-form identification from noisy time-resolved measurements.
3. [Yuchen Bi, Differential Manifolds](https://home.mathematik.uni-freiburg.de/analysis/2025_WiSe_Lehre/2025_WiSe_DG-I/Differential_Manifolds.pdf):
   partitions of unity (§2.6), vector-field coordinate compatibility (§5.1),
   and local straightening (§5.5).
4. [Lavenant et al., Towards a mathematical theory of trajectory inference](https://arxiv.org/html/2102.09204v2):
   recovery of path laws under a restricted stochastic process class; its
   potential-driven assumptions do not identify arbitrary cellular drifts.
5. [Schiebinger et al., Waddington-OT](https://pmc.ncbi.nlm.nih.gov/articles/PMC6402800/):
   growth-aware inference from temporal population samples.
6. [Gorin et al., RNA velocity unraveled](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010492):
   analysis of the biological and statistical assumptions in velocity workflows.

The local foundations remain
[Cellerator cellular geometry](../../../Cellerator/cellular_geometry.md),
[GlassHelix design](../../../GlassHelix/docs/design/overview.md), and
[its evidence and hypothesis distinctions](../../../GlassHelix/planning/learning-v1/references/REVIEW_AND_DESIGN.md).
Local sibling links assume this workspace layout. These are sources of
conceptual constraints, not current-source qualification of a new pipeline.
