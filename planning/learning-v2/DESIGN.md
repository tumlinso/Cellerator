# Architecture and smallest useful slice

## 1. Responsibility, not a second ML ecosystem

GlassHelix is the theory-bearing scientific system. It should be written against ordinary PyTorch/libtorch concepts and CelleraTorch's biological extensions in the model-author path. Its existing native NF1A facilities remain a valid direct C++ route. The native and framework routes must share mathematical meaning where they represent the same system; neither must replace the other.

| Owner | Responsibility |
|---|---|
| GlassHelix | Scientific hypotheses and model architecture; state/forcing/observation roles; evidence permissions; fitting and conditioning protocol; competing explanations; identifiability and biological interpretation |
| PyTorch/libtorch | Ordinary tensors, generic layers, autograd orchestration, optimizers and model-state facilities where adequate |
| CelleraTorch | Framework registration, tensor/parameter views, custom-op and module wrappers, autograd integration, update/serialization interoperability and packaging |
| Cellerator | Native biological domains/axes/relations; stable structure and mutable values; CellPack geometry; physical projections; numerical operators and derivatives; training/update primitives; planning/autotuning/fusion; residency/readiness; precision; GPU execution planning; language/compiler/stdlib |
| CellShard | Persistence, ingest, sharding, cache, transport and delivery; not geometry inference or kernel selection |
| Baseplane | Sequence-specialized primitives under the Cellerator computational umbrella; GH interprets sequence evidence scientifically |
| Experiment | Cohort, data adapter, preprocessing choices, model recipes, protocol configurations, controls and figures |

This is supported by the live CE `scope.md` and CT ownership documentation [S1–S3], not by a proposal to turn Cellerator into a generic Torch replacement. Reusable biological computation is more than kernels: an exact compositional representation, lowering rule, parameter-sharing contract or gradient accumulation plan can be the valuable addition.

## 2. What the existing foundation supplies

NF1A's supplied-system assembly, evolution, observations, selected responses, joint candidate comparison and provenance are retained [S4–S6]. It does not yet infer all those systems from measurements. Its scientific contracts should be reused and exposed selectively, not recreated in a new Python ontology with inconsistent native meaning.

The current CT N16 adapter is a combined training step: native updates occur during backward and only the dense input gets a Torch gradient [S2]. It is useful bounded evidence, not the desired composable interface. `TORCH_BRIDGE.md` defines the missing behavior.

The native relation interfaces already expose stable topology, forward/transpose, edge-gradient/update and value-generation structure [S7–S8]. NF1A additionally supplies typed multi-argument incidence and a small executable evaluator vocabulary [S9]. Those are starting points, not evidence that all derivatives and parameter paths needed by a new model already exist.

## 3. Small GH scientific abstractions

Keep these five concerns separable; implement them with as few concrete types/functions as useful.

**Evidence view.** Records IDs, observed values, missingness, time, modality, replicate and actual relations. It specifies which data a particular inference may use. Clone membership is ancestry evidence, not a parent–child edge. A transformed feature carries its fitted-source provenance. A stock tensor and a small semantic wrapper are enough where they preserve this meaning.

**Hypothesis family.** A callable/module plus declared scientific level and capabilities. It can predict at a horizon, use explicit history, evolve augmented state or realize a mechanism. Do not force every family to expose an ODE or named molecular coordinates. A small fixed-structure class is a legitimate first assumption, but its answers must actually be fitted rather than supplied.

**Problem/unknowns.** Separates scientific role from inference treatment. A global law coefficient, case-specific initial condition, hidden input and observation calibration can all be fitted without becoming the same object. Parameters remain ordinary owner-backed numerical values. Observation masks and permitted evidence belong here; optimizer moments do not.

**Inference protocol.** Orchestrates shared-law fitting, freezing, local conditioning, alternative search and evaluation. It delegates numerical optimization to Torch/CE. A protocol may be deterministic with bounded unknown forcing, probabilistic, or based on finite competing fits. No single uncertainty representation becomes universal.

**Fitted explanation.** Combines model state with question/domain/evidence provenance, diagnostics and unresolved alternatives. Finite candidate disagreement witnesses ambiguity. Candidate agreement is not a proof of uniqueness; failed search is not certified infeasibility. Approximation, model misspecification, forcing uncertainty and search incompleteness are separate.

## 4. The first vertical slice

First fit a small partially observed controlled problem with known truth, at least one shared unknown and a local hidden/initial unknown. Include a deliberately unresolved combination and an additional observation that distinguishes it. This qualifies engineering and inferential semantics; it is not biological validation.

In parallel, write the selected biological model in Torch against a modest training-only feature domain. Use CT for a genuinely trained structured component, not a decorative unused import. Connect at least one fitted component to NF1A's typed system/readout/interrogation using the same frozen quantities and semantic IDs. A controlled bridge is sufficient; compiling arbitrary Torch models is not.

Then exercise one coverage-supported CellTag arm. The architecture must support either history-conditioned or snapshot inference, but claims must match the evidence supplied. Population training improves law estimation; it does not give an individual cell a measured history or break exact observation equivalence by itself.

Native work follows the scientifically chosen computation. The initial construction can be a sequence of existing CE operations and ordinary Torch functions. Only add a kernel, native expression or lowering when warranted. The compiler is a first-class CE route, not a mandatory detour for the first Python consumer.

## 5. Ownership at awkward boundaries

**Parameters and optimization:** CE owns its native canonical parameter buffers/identities; Torch owns ordinary Torch-only parameters. CT exposes native views and coordinates legal updates/publication. Standard optimizer arithmetic is reusable; raw writes that bypass readiness are not. Do not create duplicate masters without declaring why and which is authoritative.

**Gradients:** Torch composes the model graph. CT returns mathematical gradients for its op and delegates native primitive derivatives to CE. GH composes scientific objectives and may request selected responses; it must not build another AD engine. Unsupported higher-order or forward-mode actions are explicit, not implied by first-order backward support.

**Persistence:** use `state_dict` or the actual numerical owner's format for values. GH adds a scientific manifest; CT restores logical native identities/values and rebuilds ephemeral prepared state. No portable artifact contains live pointers, CUDA events or a device-specific cache as logical truth.

**Execution:** full packed physical order need not leak to model code, but semantic axis/order assertions remain visible. A model should not choose a state dimension to fit the old N16 adapter. Supported widths, chunking and tail behavior are engineering capabilities.

**Precision:** retain NF1A's scientific/encoding separation. Start with an actually qualified policy, allow supported half storage/execution where justified, and do not force all biological state into FP32 or add a speculative quantization system. Declare what derivative is evaluated when a lower-precision value plane is used.

**Language/compiler:** Cellerator owns `.cell`, parsing/Sema, semantic/planning/realization IR, compiler APIs, standard library and backend selection. The reviewed source bridge is bounded [S10–S11]; planning documents do not prove full compiler readiness. New native semantics should be compatible with that common model without requiring complete compiler work.

## 6. What to defer

No new generic tensor framework, GH optimizer, CT sparse-kernel copy, universal observer, arbitrary mechanism search, mandatory amortized encoder, full multiome generative model, exact division tree, full `.cell` network compiler, JIT, all-Torch-API compatibility, multi-GPU training or paper-scale benchmark is required. Keep callable/capability and evidence boundaries open so these can be added without assigning them the wrong owner.

The outcome is small but real: the system learns unknown quantities, uses them through a qualified biology-aware operation, conditions only on permitted evidence, connects back to NF1A, and can say what it has not established.
