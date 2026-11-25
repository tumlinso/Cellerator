# Biological assessment of an embryo atlas

Research direction begun on 15 November 2026 and revised on 16 November 2026.
[Compact flow recovery](compact-flow-methods.md) develops the governing
methodology. The earlier [assessment methods](atlas-quality-methods.md) now
serve as supporting diagnostics.
These are scientific proposals, not implemented capabilities, a product design,
or a commitment to a particular model family.

## The problem

Recover a compact description of the biological dynamics across the observed
domain, as closely as the evidence permits. Measurements constrain that
description. Reconstructing or compressing noisy observations is not the
scientific objective.

Appropriately integrated observations should be compatible with a common
description of state evolution, including the state and context needed to
express it. Integration, state estimation, and learning the flow are therefore
coupled inverse problems. An embedding that preserves cell labels or selected
predictions can still be inadequate if it destroys the dynamical organization.

The working premise is that the underlying dynamics have reusable, compact
structure: limited participant and conditioning sets, shared mechanisms, and
possibly overlapping domains on which simple laws are smooth internally.
Develop and test this premise rather than assuming a particular gene graph,
latent dimension, or unique recoverable mechanism.

Three assessments are related:

1. Measurement quality: which observations and channels are reliable?
2. Representation quality: which useful distinctions survive integration,
   pooling, feature selection, or another transformation?
3. Inferential adequacy: which predictions or explanations are supported by
   the available observations under explicit assumptions?

A sample can constrain one program's dynamics while leaving another
unresolved. Biological questions and response probes test different aspects of
the recovered flow; success on a chosen question is partial evidence, not the
definition of the complete recovery objective.

Compactness concerns the reusable dynamical description, its state/context
requirements, and the maps needed to relate it to experiments. Moving all
complexity into an arbitrary decoder, assay map, or per-cell hidden code
does not achieve this objective. Assess fidelity, complexity, and supported
domain together.

## Smooth modular supports

Distinguish a mechanism's dependency support, the state-space region where a
local law applies, and the region actually constrained by observations.
Smooth local approximations can overlap without requiring a globally simple
polynomial or one universal low-dimensional coordinate system.

Local approximations to the same drift require coordinate-consistent
agreement on overlaps. Contributions from different mechanisms may instead
add. Blending conflicting approximations can manufacture a smooth false
equilibrium; smoothness alone is insufficient. Uncovered or weakly constrained
regions remain explicit rather than becoming zero-flow states.

Geometry, modular supports, and their reusable discovery/learning mechanisms
belong in Cellerator. That includes local laws and conditions, composition,
coordinate and derivative handling, support queries, local fitting, and
selected consistency tests. The
[joint support-learning proposal](../learning-modular-flow.md)
develops those questions there. CCell supplies biological observations and
atlas experiments. Formal inferability and rival-explanation reasoning can
draw on GlassHelix. These ownership decisions do not claim that the needed
facilities already exist.

## Inspiration and attribution

[scAutoQC](https://sctk.readthedocs.io/en/latest/notebooks/automatic_qc.html)
provides sample-specific QC thresholds, groups cells in QC-metric space, and
compares decisions across clustering resolutions. Its distributional and
multivariate reasoning is useful inspiration. Agreement among such decisions
is evidence of stability, not a calibrated probability that a cell is healthy.

[scAtlasTb QC](https://scatlastb.readthedocs.io/en/latest/modules/qc.html)
uses the thresholding component with explicit user alternatives and retained
cell-status annotations. Its
[batch analysis](https://scatlastb.readthedocs.io/en/latest/modules/batch_analysis.html)
and [integration metrics](https://scatlastb.readthedocs.io/en/latest/modules/metrics.html)
provide useful precedents for examining covariates and separating technical
mixing from biological preservation. Its current QC module should not be
described as the complete clustered scAutoQC procedure.

Borrow the questions and documented mathematical ideas with attribution.
Develop independent implementations and comparisons; copying colleagues'
code is not the objective. Extending a published method does not make its
original contribution ours.

## The cellular geometry premise

Cellerator's [cellular geometry](../../../cellular_geometry.md)
distinguishes complete state, its evolution, and the experimental image of
that state. Dependencies may have a reusable admissibility envelope and
conditions determined by restricted context. Joint mechanisms can overlap,
share regulators, and participate in feedback.

Under this premise, useful quality questions include:

- Does an observation expose the inputs needed to assess a particular mechanism?
- Does pooling combine observations with incompatible context or histories?
- Does integration destroy a distinction needed for a biological response?
- Does a proposed restricted module need additional information to predict
  independent observations?
- Which additional measurement could distinguish surviving explanations?

The premise is an assumption to investigate. RNA similarity does not establish
complete-state equivalence. Missing observations and unlisted interactions are
not biological exclusions. Static covariance is not the drift Jacobian.
Instantaneous exclusions do not eliminate indirect finite-time influence.
Stochastic diffusion and jump mechanisms may carry dependencies absent from
the drift. Sparsity depends on meaningful coordinates and admissible
perturbations, not on a preferred latent basis.

The [GlassHelix design](../../../../GlassHelix/docs/design/overview.md) supplies
the complementary question: what description suffices for the behavior of
interest, and what evidence warrants interpreting it as mechanism? Preserve
whole rival explanations, allowed evidence, and unresolved ambiguity.
Identifiability is question- and experiment-relative. Natural dynamical
variation can identify restricted features under appropriate assumptions;
incomplete observation does not imply that nothing can be learned.

## History and velocity

The [Trajectory++ prototype](../../../../trajectorypp/README.md) distinguishes
history-conditioned sharing across trajectories from smoothing along a
trajectory. In the former, context determines compatible neighbors and only
their anchor observations enter the molecular average. A fitted slope along
a history is a coordinate derivative, not automatically an RNA-velocity
kinetic estimate. Inferred history must remain distinguishable from measured
ancestry or actual repeated observations.

Velocity estimates, uncertainty treatment, module-level kinetics, and suitable
history operations may require substantial new mathematics. Their absence
does not prevent specifying an estimator, its assumptions, controls, and
meaning now. Snapshot-only versions of some assessments are useful, provided
they make snapshot claims. An unordered state cloud can constrain geometry
without identifying flow direction or speed. Temporal marginals, kinetic
evidence, supplied histories, and perturbations constrain different aspects;
their combination determines what dynamics can be recovered.

## Modeling and efficient computation

The biological question and its dependency structure can determine both the
calculation and how to execute it efficiently. Restricted participant and
context sets, shared preparation, sparse candidate support, selected response
directions, and reusable partial summaries can avoid unnecessary work.

Cellerator's experimental
[response-sensitive precision](../../response_precision.qmd)
already illustrates this connection for finite supplied probes: preserve
selected responses while changing state precision. A broader proposed use is
to assess what biological answers change under integration, pooling, feature
removal, or numerical approximation. These changes need different scientific
controls even when they share computational machinery.

A justified model restriction and a fast implementation of that restriction
are separate evidence claims. Explicit structure can also increase inference
or maintenance cost. Total cost and fidelity need measurement.

## What a result should establish

A useful result identifies a question, affected observations or mechanisms,
effect size, uncertainty, independent support, and applicability conditions.
Failure of a fitted model is not automatically failure of a cell.

Separate supported adequacy, demonstrated discrepancy, insufficient precision,
missing support, numerical failure, and model-class limitations. Failure to
reject a difference is not evidence of equivalence. Candidate agreement does
not prove uniqueness; candidate disagreement can demonstrate ambiguity.

For development, controlled partial-observation systems can establish that a
calculation recovers a supported flow component, detects a known defect, and
preserves a deliberately unresolved case. Biological usefulness requires
independent real evidence. [Compact flow recovery](compact-flow-methods.md)
sets out a learn–challenge–refine procedure; the
[supporting assessments](atlas-quality-methods.md) challenge observations,
integration, and individual aspects of the inferred flow.
