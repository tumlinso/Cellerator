# Supporting assessments for compact flow recovery

Methodology brainstorm begun on 15 November 2026 and reframed on 16 November
2026. [Compact flow recovery](compact-flow-methods.md) is the primary method
proposal; the [agreed direction](atlas-quality-direction.md) gives its rationale.
The methods below are plausible calculations to investigate, not established
biological tests or implemented CCell functions. Missing numerical facilities
are implementation questions for later. No universal Cell representation,
new runtime, package interface, or low-level operation inventory is selected.

The governing objective is to recover a compact description of the biological
dynamics supported across an observed domain. Measurements constrain that
description. The eight methods here are diagnostics for observation quality,
state/context sufficiency, or particular consequences of the recovered flow.
Preserving noisy observations or succeeding on one downstream task does not
define adequate integration.

| Assessment | Role under the flow objective |
|---|---|
| Controlled damage curves | Establish which measurements constrain local laws and responses under information loss |
| Conditional module residuals | Challenge a law's dependency set, context, and transfer across specimens |
| History separation gain | Test whether a current-state description needs additional permitted context |
| Predictive collisions | Detect distinctions relevant to evolution lost by an integration |
| Conditional response preservation | Challenge finite-time behavior and gates under a proposed flow approximation |
| Transition versus doublet explanations | Compare biological and technical explanations of mixed observations |
| Population flow feasibility | Constrain or refute candidate laws using temporal distribution changes |
| Distinguishing measurements | Select evidence that separates surviving dynamical explanations |

## Common mathematical and evidential ground

Let x be a modeled biological state, c its relevant environment or forcing,
b an assay or technical batch, and y an observation:

$$
y \sim p_b(\,\cdot\mid x,c), \qquad r=T_b(y,c,H).
$$

H is history actually permitted to the analysis. It can be absent; a known
lineage relation is not a measured molecular history. T may return an embedding,
graph, selected features, or a richer representation. Inferred context and its
uncertainty remain derived information.

For each diagnostic, choose a target Q before scoring: a measured module
readout, a held-out modality, a response over a declared horizon, or a
population outcome. Q probes an aspect of recovery; it does not replace the
flow as the scientific object. One working family of biological dynamics is

$$
\dot x=\sum_m S_m F_m(x_{P_m},x_{C_m},c;\theta_m).
$$

P_m names participants, C_m names conditioning inputs, and S_m places the
mechanism's contribution in state coordinates. Sets may overlap; the
decomposition is a hypothesis, not something a correlation graph identifies.
Other candidates can be finite-horizon predictors, stochastic/jump models,
or history-conditioned functions without exposing this equation.

All comparisons require a declared evidence budget:

- Split by genuinely independent biological units before fitting transforms,
  modules, observation calibrations, targets, or selection rules. Embryos,
  donors, clones, repeated observations, and shared ancestors need appropriate
  blocking. Thousands of related cells are not thousands of independent trials.
- Separate discovery, tuning/calibration, and final evaluation. Transductive
  integration is allowed when declared, but withheld targets cannot enter
  alignment or latent inference. State which new-embryo or fixed-atlas question
  is being evaluated.
- Keep counts, imputed values, fitted velocity, inferred matches, and actual
  measurements distinct. A target estimated from the same graph that is being
  assessed is a self-consistency target.
- Declare meaningful margins and units. For a nonnegative defect d, support a
  pass only when its upper uncertainty bound lies below an adequacy margin;
  support a failure when its lower bound exceeds that margin. Otherwise retain
  an unresolved result. Signed gains use the corresponding directional rule.
- Estimate uncertainty at the biological sampling level. Fix discoveries on
  training data and account for multiplicity in large module/context searches.
  Permutations need defensible exchangeability; shuffling individual cells is
  not a general solution to specimen-level dependence or confounding.
- Check absolute predictive adequacy as well as relative improvement. Two
  representations can agree because both discard the relevant biology.

The comparison is usually an effect-size or predictive-risk comparison.
Failure to reject a null is not a pass, and a model-based response comparison
does not independently validate the biological model.

## 1. Controlled damage curves for biological readouts

### Mechanism

Replace a single count threshold with the measured degradation of a declared
biological answer under controlled loss of information.

For raw UMI counts, construct binomially thinned observations

$$
Y_{ig}^{(q)}\mid Y_{ig}\sim\mathrm{Binomial}(Y_{ig},q).
$$

Use several retention fractions q and independently simulated contamination
levels where a plausible ambient profile is available. Refit or apply the
analysis according to a fixed deployment protocol. Evaluate a module readout
or target Q on separate evidence:

$$
L_m(q)=\operatorname{mean}_{\text{held out}}
  \ell\{Q_m,\widehat Q_m(Y^{(q)})\}.
$$

The curve, its uncertainty, and its crossing of a scientific tolerance are
more informative than a global count cutoff. A sample's unusual module
failure can be compared against depth-matched controls. Use both frozen-model
stress tests and separately labeled retrained tests: robustness of one fitted
model and recoverable information are different questions.

### Meaning and controls

A pass says the selected answer remains adequate under the tested degradation
and domain. A steep curve identifies information fragility. A specimen worse
than depth-matched controls may have additional damage, assay mismatch, or
unmodeled biology; it does not establish which.

Thinning cannot manufacture an independent clean biological truth or recover
information already absent. Split molecule counts can support controlled
denoising experiments under their count model, but are not new biological
replicates. Disjoint Poisson splits are independent conditional on a fixed
intensity under that model, but conditioning on their observed total makes
them dependent. Mixing over biological intensity, normalizing, and reusing
fitted quantities require their own dependence analysis.

Include rare valid states, genuine developmental changes in RNA content, and
whole damaged specimens in calibration. A low-count cell can be adequate for
one question while a high-count cell is inadequate for another. Treat
contamination simulations as tests of the declared corruption process.

### Extension and computational exploit

This extends adaptive QC thresholds toward task-specific information
robustness. The sctk/scAutoQC distribution and multiresolution procedures
remain useful baselines, not biological truth [1].

Use a small fixed dose grid, paired randomization across compared methods,
streamed module summaries, and stratified specimen subsampling. Reuse fitted
models in the frozen arm. Avoid simulating every feature when the target and
its declared dependencies use a bounded subset.

## 2. Conditional module residuals across specimens and assays

### Mechanism

For a proposed program, fit a predictor using its participants and context:

$$
\widehat Q_m=F_m(y_{P_m},c_m).
$$

Q_m can be a held-out expression subset, accessibility readout, or suitably
qualified kinetic quantity. Target channels must not enter the matching
features or fitted representation through an undeclared route.

On independent specimens, assess residual distributions within biological
overlap. Use observation-model standardized residuals; a randomized
probability-integral transform is one option for calibrated discrete-count
predictions. Ask whether residuals retain assay or specimen structure after
conditioning on the biology the comparison intends to preserve.

Separately compare the restricted predictor with one receiving additional
context Z:

$$
G_{m,Z}=L(F_m)-L(F_m^{+Z}).
$$

A useful Z is a measured modality, permitted history, environmental covariate,
or an independently fitted outside-module summary. Compare the same targets,
training budget, and evaluation specimens. Include matched-capacity predictors.

### Meaning and controls

Residual differences above a declared margin show that the tested model and
calibration do not transfer across the compared conditions. They can indicate
technical distortion, missing biological context, or misspecification.
Reproducible positive G identifies extra predictive information; it does not
prove a direct regulatory edge or causality.

A pass requires adequate predictions and small residual discrepancies with
sufficient independent support. Poorly powered tests of a weak model cannot
certify a module. Perfect stage–assay confounding makes the relevant technical
comparison unsupported. Use overlap restrictions and specimen-level controls;
do not erase real stage effects to manufacture invariance.

### Extension and computational exploit

scAtlasTb's covariate and biological-preservation analyses motivate the
comparison [2,3]. The proposed extension localizes it to a program and its
conditioning requirements.

Fit low-dimensional local predictors first. Stream residual means, variances,
and fixed random-feature sketches for distribution comparisons. These can
replace dense all-pairs residual kernels. Cache only preprocessing permitted
by each fit; widening a module's context is a scientific change.

## 3. History separation gain in neighborhood smoothing

### Mechanism

Construct two contributor relations at comparable developmental positions:
W_0 uses current state; W_H also uses permitted history. Keep the candidate
pool, contributor budget, output channels, and coordinate tolerance comparable.

For anchor i and raw channel a,

$$
\widehat a_i^{(H)}=\sum_j W_{H,ij}a_j,\qquad
G_H=L(\widehat a^{(0)})-L(\widehat a^{(H)}).
$$

History determines the relation; only matched anchor observations enter the
sum. Apply the same relation separately to spliced and unspliced channels and
to paired cross-moments where joint support exists. Historical observations
must not silently become extra molecular contributors.

Use independent target evidence, target channels excluded from matching, or a
carefully declared count-splitting experiment. Evaluate on blocked embryos or
histories. Report support by independent groups alongside weight effective
size; 1/sum(w²) does not measure independent biological replication.

Compare ordered histories with current-state, dimension-matched context,
unordered-history, and valid history-shuffle controls. Sweep historical
bandwidth on calibration data to find the shortest useful context.

### Meaning and controls

A positive, reproducible gain means supplied history helps this prediction
beyond current-state matching under the tested controls. Loss of that gain
after integration is evidence to investigate whether useful context or
neighbor structure was discarded. It does not show that temporal ordering
helped unless it beats the unordered control.

Little gain can mean adequate current state, uninformative or uncertain
history, poor matching, or insufficient support. Report which. Prohibit
withheld future targets from entering prospective contexts. Retrospective
reconstruction is a separate valid protocol.

The [Trajectory++ prototype](../../../../trajectorypp/README.md) already embodies
the anchor-only distinction, but its exact search and synthetic examples do
not establish an atlas-scale estimator or biological accuracy. Kinetic
velocity fitting remains separate from local coordinate slopes. Neighborhood
moments in RNA-velocity analysis provide a comparison point [4].
Lineage-informed finite-time transition inference also has established
precedents such as CoSpar [8]. The proposed test concerns the benefit and
failure modes of a contributor relation, rather than a claim to originate
lineage-informed dynamics.

### Computational exploit

Retrieve candidate anchors using stage, compatible coarse state, and actual
relationship constraints, then score histories only on that candidate set.
Index temporal slots and shared context projections. Reuse the sparse
relation across channels and moments. Audit approximate retrieval against
bounded exact searches, especially around rare branches; do not turn absence
from a retrieved candidate set into biological impossibility.

## 4. Predictive collisions introduced by integration

### Mechanism

Search for integrated neighborhoods that combine observations with
distinguishable biological answers. The proposed failure is loss of a useful
distinction, not ordinary coexistence of several observed outcomes.

Choose Q, permitted context, and a forecast horizon if applicable. Fit
capacity-matched readouts from the unintegrated and integrated representations,
with all shared context available to both:

$$
\Delta_Q=L_{\mathrm{test}}(r,c,H)
          -L_{\mathrm{test}}(y,c,H).
$$

Use an appropriate proper score for predicted outcome distributions, or a
declared loss for a deterministic target. Confirm absolute calibration.
Neighborhood pairs then localize where increased risk occurs: flag pairs that
became poolable after integration but have reproducibly different conditional
Q distributions. Use discovery data to choose regions and new specimens to
confirm them.

### Meaning and controls

A confirmed positive risk difference supports loss of accessible Q-relevant
information under the tested readout families and fitting protocol. A failed
readout alone does not prove mathematical information loss; inadequate
capacity or optimization can cause it. Stronger witnesses arise in
known-truth systems where the representation is explicitly many-to-one.

A pass means adequate Q prediction is retained within the tolerance and tested
domain. It does not certify all biological questions. If identical original
inputs already have unresolved futures, integration cannot be blamed for
creating that ambiguity. Legitimate stochastic branching requires comparing
conditional distributions, not demanding the same realized fate from every
neighbor.

RNA recovery, independently withheld accessibility, or permitted clone-level
future outcomes can supply Q. Do not treat unrelated stage-separated cells as
observed parent–child pairs. Correct for the target's observation process;
punishing removal of raw technical variation is not biological preservation.

### Extension and computational exploit

This extends neighborhood, label, and pseudotime preservation toward a chosen
predictive distinction. scIB's trajectory score compares pre/post pseudotime
ordering and depends on curated reference trajectories [5]. It remains a
useful comparator, not independent truth.

Search sparse integrated neighbor relations and a sample of retained/far
controls instead of all cell pairs. Aggregate confirmed effects by module
and independent specimen. Keep the held-out risk comparison primary; a
visually dramatic pair is a localization clue.

## 5. Conditional response and gate preservation

### Mechanism

Challenge a representation using specified model responses rather than only
reconstruction error. For a supplied candidate state x, admissible perturbation
p, horizon tau, and observable R, compare

$$
D_{m,p,\tau}
=d\left(R[\Phi_\tau(x;p)],
         \widetilde R[\widetilde\Phi_\tau(Tx;p)]\right).
$$

This notation applies when T and the reduced model are actually defined on
the candidate state. For an atlas observation encoder T_b(y), require an
explicit inference/observation bridge instead of pretending it maps known
complete state. Keep uncertain state alternatives separate.
Every comparison is between corresponding physical inputs, time units, and
readouts. An embedding alone does not supply an intervention operator.

Use finite responses and selected derivative actions. Include directions
near proposed eligibility boundaries, combinations of participants, gate
context, and random interior controls. A small value error near a gate can
change a response substantially. Joint perturbations can reveal effects
missed by restoring or varying one coordinate at a time.

An optional smooth, deterministic diagnostic is the commutation residual
DE(y)v_y-v_r(E(y)). It is meaningful only with valid velocities and a
differentiable encoder on that domain. A constant encoder can make this
residual zero while destroying all biology, so response/readout adequacy and
the collision test must accompany it. Graph-only outputs need a different
comparison, not fabricated derivatives.

### Meaning and controls

A pass supports preservation of the specified responses for the supplied
candidates and probes. A failure identifies sensitivity to the representation
or approximation, not necessarily false biology. Agreement within one fitted
model cannot independently establish its truth.

Compare physical readouts, not unaligned latent Jacobians. Respect feasible
perturbations and units. Test multiple horizons: a zero instantaneous
a-to-c dependency can coexist with a delayed a-to-b-to-c response. For
stochastic candidates, compare response distributions with simulation
uncertainty; paired random streams can improve precision but one matched
realization is insufficient.

### Extension and computational exploit

Cellerator's [finite response-sensitive precision experiment](../../response_precision.qmd)
provides a direct conceptual starting point. The extension tests biological
readouts under integration, pooling, or feature removal as well as numerical
approximation. These transformations need different controls.
Dynamo already uses reconstructed vector fields, differential geometry, and
perturbation predictions [7]. The proposed extension makes selected response
preservation an assessment of a transformation, with explicit partial-state
and evidence limits.

Evaluate selected Jacobian-vector or adjoint actions and finite probes on
the declared dependency neighborhoods, including all needed multi-hop paths.
Reuse prepared structure across cells. A screen of gate margins can focus
expensive probes, but retain random controls and test calibration outside the
screen. No full dense gene-by-gene Jacobian is required by this design.

## 6. Competing explanations for mixed biological programs

### Mechanism

For observations carrying apparently incompatible programs, fit competing
explanations with a common measurement model:

1. A single cell in a supported transitional or context-dependent regime.
2. An additive molecular mixture of two candidate source cells or profiles.
3. A single-cell profile with bounded ambient contamination or another
   specified technical process.

Fit on a subset of channels, then compare predictions of independent genes,
paired splicing channels, or another legitimately linked modality. A doublet
mixture must operate on the molecular/count scale with explicit capture and
depth assumptions; averaging log-expression or latent coordinates is not its
measurement model.

Use constraints that carry information: restricted source combinations,
nonnegative mixture contributions, calibrated contamination bounds, and
conditional module compatibility. Penalizing a flexible transition model
until the doublet wins would merely encode the desired conclusion. Fit
complexity and priors on separate calibration data.

### Meaning and controls

Consistent out-of-sample preference can support one explanation within the
tested families. Overlapping predictions mean unresolved identity, not an
automatic doublet. Failure of all candidates indicates a deficient model set
or unsupported observation.

Synthetic doublets are useful controls but cannot represent every real
doublet or prove that a rare transition is impossible. Include real
independent doublet evidence where available and known valid transitional
states. Do not interpret marker coexpression alone as incompatibility.

### Extension and computational exploit

This combines familiar mixture-based artifact reasoning with a positive
alternative for legitimate dynamics and conditioning. It is a proposed
comparison protocol, not a claim to have solved transitional-state annotation.

Retrieve a bounded set of plausible source profiles from module signatures,
fit mixture weights in one or two dimensions, and use held-out channels for
discrimination. Share sufficient statistics across candidate pairs. Inspect
retrieval misses on rare states and retain the possibility that no candidate
source is represented in the atlas.

## 7. Population flow feasibility with bounded growth and sampling

### Mechanism

Assess whether module-level dynamics, stage distributions, growth, and sampling
can be mutually consistent. Let rho(x,t) be biological population density, v
its drift, B its diffusion tensor when appropriate, and g net growth:

$$
\partial_t\rho=-\nabla\cdot(\rho v)
 +\tfrac12\sum_{a,b}\partial_a\partial_b(\rho B_{ab})+g\rho.
$$

For a smooth test function phi focused on a module or state region, the weak
form below assumes boundary terms vanish; otherwise retain the boundary flux:

$$
\frac{d}{dt}\int\phi\rho
=\int\left[\nabla\phi\cdot v+
 \tfrac12 B:\nabla^2\phi+g\phi\right]\rho.
$$

Estimate stage-to-stage moment changes using a small bank of fixed smooth
test functions. Include the actual observation/capture process before
comparing sampled cells; normalized observed proportions alone do not
identify biological mass or growth. Use bounded growth and sampling scenarios
when absolute rates are unavailable. For projected non-Markov observations,
augment context or label the closure assumption explicitly.

Compare a restricted dynamic explanation with endpoint-only and
growth/sampling-only controls. Evaluate intermediate stages withheld from
fitting and vary the interval. A nonlocal alternative is sparse unbalanced
transport using the same permitted information.

A sharper version searches for a contradiction witness. Let b contain
observed moment changes and let K be a conservative convex outer bound on
changes attainable by the admitted drift, diffusion, growth, and observation
scenarios. Search for a direction lambda with

$$
\lambda^\mathsf{T}b-\sup_{u\in K}\lambda^\mathsf{T}u
  > \epsilon(\lambda).
$$

Here epsilon covers sampling, measurement, quadrature, and time-discretization
error in that direction; normalize lambda to fix scale. Select lambda on
discovery data and confirm on new specimens, or use simultaneous error
bounds valid over the searched directions. Such a separating
contrast rules out the admitted scenarios under those bounds. It can be
cheaper to find this witness than to reconstruct a flow. Use an outer bound:
an overly restrictive numerical approximation to K can create false
contradictions. Failure to find a witness supplies no general feasibility
certificate.

### Meaning and controls

A discrepancy outside the declared uncertainty and nuisance bounds refutes
that combination of dynamics, observation, and growth assumptions. It does
not uniquely identify an erroneous velocity estimate. A pass establishes
compatibility of selected population moments, not an individual trajectory
or a unique flow. Some circulations or rare branches can be invisible to
the chosen functions.

Arbitrarily flexible g or sampling can explain almost anything. Prespecify
credible bounds and report sensitivity. Sparse or perfectly confounded stages
can make the test unresolved.

### Extension and computational exploit

Waddington-OT is a direct precedent for time-indexed population couplings with
growth/death and held-out distribution prediction [6]. The proposed weak test
can assess a supplied local kinetic law without first solving a dense
transport problem. Its use of overlapping module functions localizes failures.

Accumulate streamed moments and directional derivatives, avoiding high
dimensional density estimation. Gradients of a module-supported phi require
only that module's output drift, though computing that drift still requires
its actual context. Start with functions whose moments are estimable from
the measurement model; do not equate noisy counts with latent state.

## 8. Rival explanations and the smallest distinguishing measurement

### Mechanism

Retain whole explanations that fit the allowed training observations and meet
an absolute adequacy requirement. Compare their answers to Q on the same
physical scale. When answers disagree, search a bounded collection of
plausible additional measurements e: a module-specific accessibility panel,
an earlier permitted history slot, a kinetic measurement, or environmental
context.

For two candidates a and b with calibrated predictive measurement distributions,
one possible ranking is

$$
V(e)=\frac{\operatorname{JS}\{p_a(z_e\mid D),p_b(z_e\mid D)\}}
            {\operatorname{cost}(e)}.
$$

JS is the Jensen–Shannon divergence. It is bounded for probability
distributions, including candidates with differing support. With multiple
candidates, use a declared weighting or a maximin discrimination objective.
For deterministic bounded explanations, separate non-overlapping predicted
measurement ranges instead of inventing probabilities. Include measurement
noise and assay feasibility.

### Meaning and controls

Disagreement among adequate candidates witnesses unresolved answers.
Agreement supports only the searched family and domain. A high V identifies
a proposed measurement that could distinguish those candidates; it is not
a guarantee that the experiment will resolve the real system or that the
candidate set covers all explanations.

Test the ranking by withholding an actually measured channel in a controlled
problem, selecting e without looking at its values, then revealing it on new
evaluation units. Compare against cost-matched random or conventional panels.
Prefer observations that distinguish Q-relevant alternatives rather than
merely maximizing arbitrary latent-state precision.

### Extension and computational exploit

This applies GlassHelix's distinction between predictive sufficiency and
mechanistic interpretation to atlas assessment. It can turn an unresolved
quality question into a specific data requirement. It does not require a
universal mechanistic model or a probability for every hypothesis.

Track disagreement only for selected questions and candidate pairs. Use
response dependency sets to exclude measurements that are structurally
irrelevant under all compared candidates. Where that irrelevance is merely
assumed, retain a broader search/control. Reuse shared forward results and
evaluate small candidate panels before full proposed assays.

## How the methods constrain each other

The useful output is a chain of evidence, not eight votes on whether a cell
is good. A strong case might be: a module is adequately measured (1),
transfers across relevant specimens (2), benefits from real supplied history
(3), and loses independently confirmed predictive performance after
integration (4). Response probes can localize a fragile gate (5), while
competing artifact explanations (6) and population feasibility (7) challenge
the interpretation. Surviving alternatives motivate a measurement (8).

Conversely, a smooth embedding, coherent velocity field, and good model fit
can all arise after erasing a rare branch. Those agreements share assumptions
and are not three independent validations.

Modeling and efficient execution can share the same restricted dependencies,
but scientific restrictions must remain revisable. A learned approximate
exclusion needs an explicit error assessment and fallback. Exact computational
omission needs a valid model contract, including derivative work. Improved
biology and faster execution of an unchanged computation are different
comparisons even when the same structure enables both.

## First discriminating experiments

Use small controlled systems to falsify the candidate tests before expensive
atlas work. These are proposed experiments, not current results.

| Controlled case | Deliberate comparison | Required behavior |
|---|---|---|
| Depth loss without biological change | Original and thinned counts | Quantify task fragility without inventing a new cell state |
| Rare valid program and a predominantly damaged specimen | Adaptive thresholds versus independent evidence | Avoid equating the dominant mode with health |
| Same RNA image, different hidden gate | Snapshot-only versus permitted-context prediction | Preserve unresolved futures when the inputs contain no distinction |
| Useful observed history at an ambiguous anchor | Current, ordered, unordered, and shuffled context | Detect only the information advantage supported by those controls |
| Integration removes a branch distinction | Label/mixing scores versus held-out response loss | Detect information loss even if static scores improve |
| Constant encoder | Local commutation versus independent readout adequacy | Reject the degenerate apparent dynamical pass |
| Delayed a-to-b-to-c effect | Instantaneous and multi-horizon probes | Retain indirect influence without declaring a direct edge |
| True transition and synthetic doublet with similar markers | Whole-candidate held-out predictions | Separate them when evidence permits and abstain otherwise |
| Changing abundance driven only by growth or capture | Transition-only versus bounded nuisance models | Avoid inventing differentiation to explain all fraction changes |
| Two observationally compatible mechanisms | Hidden distinguishing channel, then reveal | Preserve ambiguity first and revise only after new evidence |

For a biological pilot, first inspect which flow constraints are available:
stage distributions, defensible kinetic information, supplied histories,
paired measurements, and perturbations. Follow the learning and independent
challenge procedure in [compact flow recovery](compact-flow-methods.md).
Methods 2, 3, and 7 can help constrain and refine a candidate; methods 4 and 5
challenge dynamical distinctions and behavior. Methods 1 and 6 investigate
technical alternatives, and method 8 targets unresolved flow components.
Their usefulness depends on the available evidence rather than an assumed
low-level implementation sequence.

## Sources and status

The proposed extensions above are author reasoning. The following sources
establish precedents, not novelty or validation of those extensions. Documents
were consulted on 15 November 2026; moving branch documentation can change.

1. [sctk scAutoQC tutorial](https://sctk.readthedocs.io/en/latest/notebooks/automatic_qc.html):
   adaptive per-metric thresholds, QC-space clusters, and multiresolution calls.
2. [scAtlasTb batch analysis](https://scatlastb.readthedocs.io/en/latest/modules/batch_analysis.html):
   pseudobulk, covariate associations, and permutation comparisons.
3. [scAtlasTb integration metrics](https://scatlastb.readthedocs.io/en/latest/modules/metrics.html):
   separate batch correction and biological conservation.
4. [scVelo models](https://scvelo.readthedocs.io/en/stable/about.html):
   neighborhood-moment and kinetic-model precedents, with explicit assumptions.
5. [scIB trajectory conservation](https://scib.readthedocs.io/en/latest/api/scib.metrics.trajectory_conservation.html):
   pseudotime rank preservation based on curated reference trajectories.
6. [Schiebinger et al., Waddington-OT, Cell 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6402800/):
   temporal couplings with growth/death and held-out distribution comparisons.
7. [Qiu et al., dynamo, Cell 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332140/):
   vector-field reconstruction, differential geometry, and perturbation predictions.
8. [CoSpar documentation](https://cospar.readthedocs.io/en/stable/):
   state and lineage information for coherent sparse finite-time transition maps.

Local conceptual references are
[Cellerator cellular geometry](../../../cellular_geometry.md),
[response-sensitive precision](../../response_precision.qmd),
[GlassHelix design](../../../../GlassHelix/docs/design/overview.md),
[GlassHelix evidence and hypothesis design](../../../../GlassHelix/planning/learning-v1/references/REVIEW_AND_DESIGN.md),
[GlassHelix experiment distinctions](../../../../GlassHelix/planning/learning-v2/EXPERIMENT.md),
and [Trajectory++](../../../../trajectorypp/README.md).
The sibling-repository links refer to this workspace layout; historical
plans describe scientific reasoning, not completed learning capabilities.
