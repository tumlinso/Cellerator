# First consumer: CellTag reprogramming

## Verified basis, not yet a fitted dataset

The paper describes two biological replicates, time-indexed RNA/ATAC and clone membership, with 1,422 state–fate clones spanning early and at least one later sample. It does not imply complete three-timepoint histories or generally paired same-cell modalities. Its main text uses days 3/12/21 while a methods paragraph says 11; retain this discrepancy in provenance [W4]. GEO's iEP RNA series has 25 samples, including four separate day-14 Zfp281 perturbation/control samples; 21 belong to the nominal-common-forcing time course. The iEP ATAC series has 29 samples. GEO lists processed H5 archives of about 1.0 GB and 2.4 GB respectively [W5–W6]. These archives were not downloaded or fitted during package preparation.

## Minimum data audit

Create deterministic ID joins for sample, replicate, cell, modality and clone. Inventory actual clone × time × modality coverage, missing values, inclusion/exclusion and source hashes. Record the source provenance of clone calls. Prefer released corrected clone assignments when trustworthy; audit whether they used withheld modalities or expression-derived matching. A barcode relation may be authorized metadata even when ATAC values are withheld, but this must be explicit.

Use the released time labels provisionally, including the documented ambiguity, instead of silently changing elapsed intervals. Exclude the day-14 perturbation subseries from the common-regime discovery cohort. It can be a separately declared later experiment, not another sample of identical forcing.

Use existing count/data tools. Preprocessing policy and biological thresholds stay in this experiment (or a suitable existing preprocessing package), not in CE/CT. Do not start by importing a joint RNA/ATAC embedding into an RNA-only inference arm. Fit normalization/feature transformations on training data and record permitted target use.

The default package location is `experiments/celltag_reprogramming` in the GH repository, independently packaged and outside its core dependency graph. It should import installed/pinned public GH and CT packages and ordinary Torch. No third authority or new repository is required merely to begin; it can be extracted later.

## Choose one primary question after coverage

**Blind population-law learning:** fit from time-indexed RNA without clone labels/ATAC values. Test population restrictions with independent lineage/ATAC evidence. This cannot recover individual hidden labels that are exactly indistinguishable in all supplied inputs.

**Retrospective reconstruction:** use permitted early and later RNA siblings to constrain an early distinction; validate with withheld early clone-linked ATAC. It directly tests whether consequences add information about antecedent biology. It is not prospective prediction.

**Prospective family-history prediction:** freeze the shared model trained on other clones; condition a held-out clone using only permitted records through a cutoff; predict later observations. Clone membership is input evidence here, not wholly withheld validation. Require adequate actual coverage.

Do not implement all three as a first gate. The paper concept remains open enough to choose the supported question. A small bag representation, representative set or module-level view may suffice, provided it is described as sampled clone-conditioned evidence rather than a common exact cell state.

## Inference and uncertainty

A common ancestor constrains a family of histories. Descendants can differ through unresolved state, division history and effective forcing. Represent admissible continuations or explicit candidate explanations without automatically treating divergence as an intrinsic fate coin. Observation likelihoods or population discrepancies can be computational tools without making that ontological claim.

Constrain unknown inputs; arbitrary residual forcing can explain any failed model. State the model class, tolerance and selected question. Finite fitted alternatives are a searched subset, not all possible mechanisms. Compare whole explanations, not independent edge averages.

Begin with a controlled partial-observation qualification and a modest biological model. Additional history can be tested only when it is actually supplied; temporal training alone cannot give identical input records different warranted individual labels.

## Fair controls and readouts

Use snapshot-only controls for the information question, and a flexible non-dynamical predictor with the **same allowed inputs** for the architecture question. Compare chronological context with suitable unordered/sibling-pooling controls where meaningful. Keep clones separate across fitting and evaluation; avoid using a clone ID as an arbitrary outcome lookup. Replicate transfer is useful when supported, not a requirement to pretend two replicates represent all contexts.

Hold out ATAC values and any derived structure for an RNA-only arm. A learned mapping from hidden representation to ATAC requires separate calibration and evaluation groups. Predicting an accessibility correlate is not naming a molecular cause. Masked-RNA recovery remains an optional controlled observability arm; freeze target/mask selection on discovery data and hide the chosen channels consistently across permitted times.

Do not collapse mixed clones to modal fate as the only target. A sampled population composition depends on growth, survival and sampling as well as transitions; use explicit outcome definitions and avoid pretending the endpoint fractions are individual transition probabilities.

Separate scientific comparison from computational comparison. A faster native implementation must evaluate the same model, parameter values and error policy as its reference. A biologically constrained model beating a dense model is a different question from its kernel beating a generic library.

## Finish honestly

A bounded real feasibility fit is the target when processed inputs are available. Missing indispensable metadata permits a fixture-tested client, exact missing-input manifest and configured real invocation, with biology marked not_run. It does not license reporting synthetic findings as the first biological experiment. Engineering acceptance requires usable code and semantic tests; the research result may be negative or unresolved.
