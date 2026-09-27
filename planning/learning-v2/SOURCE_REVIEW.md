# Live review and consequential design changes

**Review date:** 27 September 2026. **Method:** current Project Control architecture, source and workflow reads; official Torch integration documentation; CellTag publication and GEO records. Exact locations/identities are in `SOURCES.md` and `validation/source_basis.json`.

## Bottom line

NF1A is a supplied-system execution/interrogation foundation worth retaining. The missing work is not another general ML framework in GlassHelix. It is (1) scientific model and evidence-qualified inference composition in GH, (2) genuinely composable trainable native operations through CelleraTorch, and (3) deliberate biology-purpose learning constructions whose mathematical structure Cellerator can exploit.

The previous L1 package did not adequately separate those outcomes. ML2 replaces the proposal, not existing successful source or historical Todo records.

## Observed facts that change the plan

**1. CelleraTorch does not yet make the inspected native training route ordinary Torch parameter training.** Its combined N16 step updates relation values and bias during backward and returns only the input gradient [S2]. The documentation explicitly forbids also applying a Torch optimizer to those parameters. A zero-copy view and an autograd function are not enough to claim safe `loss.backward(); optimizer.step()` behavior. ML2 therefore gives composable training its own CE/CT outcome.

**2. Native parameter ownership is already a deliberate contract.** The CT README puts native canonical learned buffers in CE and CT owns the view/integration [S3]. The new bridge must coordinate native generations, saved-forward lifetime and legal external optimizer writes. It must not secretly create another authoritative copy. Ordinary Torch-only parameters are unaffected.

**3. NF1A supplies real scientific execution and selected responses.** The completed run and integrated toolbox describe typed supplied systems, direct maps, actual RK4, observations, candidate propagation, alternatives and interrogation [S4,S6]. The current host-JVP binding and replay are limited. ML2 preserves these facilities and requires one same-fitted-component connection, not a rewrite or a claim that all Torch models are already native mechanistic systems.

**4. Both earlier diagnostic/inference defects remain visible.** The right-nullspace extraction still treats column-major `VT` contiguous columns as rows; ordinary candidate weights still lose finite log evidence under underflow [S5]. These are small, independently useful native repairs. They do not need to block an unrelated data audit or model reference.

**5. CE already has more than isolated kernels.** Typed relations, structure sharing, separate value generations, edge gradients and update stamps, indexed ordered/repeated arguments, and a shared source/native calculus are concrete source anchors [S7–S10]. The design should exploit these rather than naming a generic dense layer “biological.” Selected installed paths still need actual qualification; the reviewed indexed evaluator vocabulary is not a complete learned mechanism engine.

**6. Compiler ownership and compiler completeness are different.** `.cell`, semantic/planning/realization layers, library APIs and standard-library construction are CE responsibilities [S10,S11]. The reviewed operational bridge is bounded. Completing the historical compiler epic is not a prerequisite for this consumer, and GH must not build a competing operation IR/JIT.

## Proposed response to those findings

The normal scientific ML path is **GH → Torch + CelleraTorch → CE** where native biological structure helps. A native C++ GH/CE path remains first-class. Generic layers and optimizer arithmetic can stay Torch. Framework registration belongs in CT; reusable structured arithmetic, derivatives, mutation/publication and execution belong in CE.

The worked first candidate in `BIOLOGICAL_ML.md` is a context-modulated shared-support operation. Its purpose is to expose a genuine modeling assumption, exact factorization, trainable activities and gradient contribution structure—not force a new kernel. Typed multi-input assembly and observation-aware hidden-activity inference are serious alternatives, not extra required implementations. A correct composition may be the best first realization. A negative performance result may finish the optimization evaluation honestly.

GH keeps evidence permissions, scientific unknown roles, shared-law fitting versus local conditioning, whole alternatives and interpretation. The experimental client keeps CellTag identifiers, preprocessing, cohort and protocol. The first experiment may be retrospective reconstruction or prospective prediction depending on verified coverage; these cannot be relabeled interchangeably.

## Changes from the old package

- Replace a GH-native learning-session centerpiece with scientific composition over existing framework/native owners.
- Make CE/CT composable training an explicit producer, not an optional numerical afterthought.
- Give biology-purpose learning design a worked mathematical basis and falsifiers, not “find an ML kernel.”
- Separate framework adapter gaps from native operation/derivative gaps and compiler routes.
- Preserve NF1A and two focused repairs; connect learned quantities to existing scientific contracts.
- Keep logical checkpoint state with the numerical owner and add GH scientific provenance, rather than inventing parameter serialization.
- Test reusable math/framework semantics in CE/CT and model/evidence meaning in GH; retain integrated checks.
- Keep six GH outcomes and two CE outcomes, with routine steps inside each and conditional realizations selected by evidence.

## Coverage and limits

This was not a complete runtime requalification, biological fit or benchmark campaign. The source-confirmed old adapter is insufficient for the proposed composable path; this review does not prove that every other historical adapter in the repository is absent or unusable. Reuse any demonstrably adequate existing implementation.

One advisory local-model investigation contradicted the explicit adapter documentation and source; that conclusion was rejected. Direct source was used for the findings above. A second advisory question returned no finished conclusion. Neither supplies acceptance evidence.

The observer's current planning state still includes historical NF1 artifacts. No records were recovered, retired or rewritten. Readiness of a plan is assessed by its target-specific preview, not inferred from an old package validation or another project's authority. The archive's validation status distinguishes native plan preview, CPU algebra tests and package lint from product implementation.
