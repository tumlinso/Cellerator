# Execution fields, adaptive relations, and runtime compilation

Date: 2026-10-28
Status: architectural investigation and discussion note; not an adopted implementation plan
Inspected checkout: `c63f591d04ef74b7566937d161c80936dcc4cf05`, including the working-tree state present during the investigation

Follow-up: the subsequently authorized isolated experiments are recorded in the
[bounded experiment decision report](bench/compiler/experiments/adaptive_fields/REPORT.md).
That report includes demonstrated behavior, interface blockers, and the E7 work
left incomplete at the requested wrap-up. This investigation remains the dated
starting inventory; no production JIT or common public API was adopted.

## Purpose and scope

Develop Cellerator as a general-purpose, high-performance mathematical and
execution library that can discover, compile, execute, and adapt computations
using their relational structure. Biological systems are a demanding source of
such structure, rather than a reason to specialize the mathematical library to
one application.

Blastback is a prospective consumer, followed by Karyoplast, Cytoplast,
GlassHelix, and others. Their scientific models, likelihoods, reconstruction
algorithms, and interpretations remain application responsibilities. Reusable
mathematics, compilation, and numerical execution belong in Cellerator.
Cellerator should complement RAPIDS-singlecell through interoperability and
distinct computational responsibilities, rather than duplicate its workflows.

The central opportunity is to use relationships to determine computational
organization: stable potential support, conditional activity, learned values,
derivative requirements, and structural mutations can inform what work is
necessary, how it is represented, and how it executes.

Execution fields remain the optimization-visibility boundary for connected
computations. General JIT is explicitly in scope for investigation; its earlier
deferral to Part Two is no longer a design constraint. JIT should be an execution
strategy for the existing semantic model.

This note records source inspection and retained experimental evidence. The
investigation ran no builds, tests, or GPU workloads and changed no implementation
or workflow state. Source availability, integrated functionality, retained
qualification, and proposed work are distinguished below. Writing this note does
not authorize implementing a JIT, rewriting the language, or consolidating
subsystems.

## Architectural judgment

Preserve execution fields and extend the existing compiler/runtime around them.
The principal gap is integration: substantial compiler, prepared-execution, and
sparse-support mechanisms do not yet form a general system for compiling and
adapting a connected computation during an application's lifetime.

The recommended direction is one semantic model, shared mathematical owners,
multiple physical candidates, and dependency-specific artifact lifetimes. The
native C++ API should provide ordinary useful operations with optional composition
and preparation. Language syntax can add concise fields, static checking,
reflection, and directives while targeting the same mathematical owners. Python
and optional framework adapters should expose those capabilities.

## Actual implementation boundary

| Area | Implemented mechanisms | Boundary or missing integration |
| --- | --- | --- |
| Anonymous and named fields | Parsers recognize fields, bodies, references, source ranges, and planning prologues. | Statement parsing alone does not produce a fully typed mathematical graph. |
| Nested fields and inlining | Field-region IR, captures/results, environment composition, visibility metadata, and a semantic inlining transform. | No integrated caller was found establishing the language's default nested-visibility policy; ordinary C++ bodies are not automatically extracted into semantic IR. |
| Native effects | Versioned contracts bind to resolved functions; opaque-call IR preserves conservative barriers. | Current contracts are narrower than the specification's allocation, transfer, synchronization, readiness, and publication vocabulary. |
| Semantic operations | Versioned numerical operation contracts, recognized families, candidate registration, and CEIR extensions. | The specified general semantic-operation registration protocol is not implemented as one system. |
| Semantic, Planning, Realization IR | Versioned models, validators, transformations, serialization/reflection, profiles, and provenance. | These components do not establish a general source-to-executable pipeline. |
| Discovery and lowering control | Candidate/provider adapters, decomposition portfolios, pass APIs, and replacement/interception mechanisms. | The inspected source-to-selected-plan path is a bounded two-operation slice with supplied profile and cost evidence. |
| Native code generation | Generated C++, ordinary objects, CPU, NVCC, and direct-PTX backend components. | Public SDK facades do not consistently connect them into actual compilation. |
| Prepared execution | Prepared programs, symbolic bindings, persistent projections, value publication, session resources, and reuse contracts. | A general asynchronous compilation and live executable replacement lifecycle is new work. |
| Sparse-support learning | Stable support identity, conditional worksets, derivative exploration, structural mutation, shared parameters, and checkpoints. | These mechanisms are not registered into a common execution-field planning path. |

Concrete field facilities include the
[anonymous parser](src/compiler/frontend/parser/parse_anonymous_execution_fields.cc),
[named parser](src/compiler/frontend/parser/parse_named_execution_fields_and_references.cc),
[field-region IR](src/compiler/ir/semantic/implement_execution_field_operations_and_regions.cc),
and [semantic inliner](src/compiler/ir/semantic/implement_semantic_ir_inlining_and_composition.cc).
The nested-visibility API requires an explicit inline-request boolean; this is an
unresolved integration policy, not demonstrated user-visible divergence from the
specification.

Three source findings are particularly consequential:

- The [source-to-selected-plan path](src/compiler/planning/deliver_source_to_selected_plan_vertical_slice.cc)
  requires exactly two operations and caller-supplied profile and complete costs.
- The [selected-plan delivery bridge](src/compiler/ir/realization/deliver_selected_plan_to_prepared_program_slice.cc)
  executes one host-reference CSR relation using doubles.
- The public [compiler-session `compile()`](src/compiler/api/define_c_compiler_session_api.cc)
  constructs an identity from configuration and source count rather than invoking
  the backend machinery. The public
  [semantic-analysis facade](src/compiler/api/expose_parse_and_semantic_analysis_apis.cc)
  is a token-based placeholder.

These are integration gaps, not reasons to discard the existing compiler. The
language specification also contains conflicting status language: it claims
frozen Part One conformance while retaining a future-implementation disclaimer.
Historical completion statements cannot replace tracing executable paths.

The broader numerical layer includes relation application and transpose,
contraction on support, edge mapping/gating, relation bundles, segment reductions
and normalization, first/second moments, indexed recurrent transitions, and
explicit VJPs. Graph candidate and tree-overlay utilities also exist. See the
[relation algebra](include/Cellerator/compute/operation/relation_algebra.hh),
[segment operations](include/Cellerator/compute/candidate/segment/segment_v2.hh),
and [sparse-training contracts](include/Cellerator/compute/operation/sparse_support/training.hh).

Maturity varies. Current sparse-support training is specialized to ordered pair
edges and width 64, with FP16 inputs/weights and FP32 accumulation, outputs, and
gradients. CE-GEO forward and sparse SIMT coexist; backward remains SIMT. Generic
logical support permits broader participant roles than this numerical consumer.
These restrictions should be candidate eligibility conditions, not the definition
of a relation. The inspection did not establish a general iterative-solver API,
tree dynamic-programming system, or lineage inference library.

## Shared semantic-operation machinery

The natural integration point is the versioned semantic-operation protocol
described in [language specification section 9.10](docs/language/cellerator-language-specification.md#910-extensibility):

- Stable operation identity and typed inputs/outputs.
- Exact effects, numerical requirements, and determinism contracts.
- Data-state transfer functions.
- Candidate and decomposition discovery hooks.
- IR lowering hooks where applicable.

Existing fixed operation schemas, candidate registration, and CEIR extension
acceptance are useful components, but do not collectively implement that general
protocol today.

The proposed flow is:

```text
language fields / native C++ semantic operations / optional framework adapters
    -> shared semantic operations and composition
    -> Semantic IR
    -> Planning IR and candidate providers
    -> Realization IR and prepared artifacts
    -> existing execution session
    -> native kernels, numerical libraries, and external kernels
```

Workload evidence feeds planning. This is a proposed integration, not an existing
end-to-end implementation.

Ordinary functions should have three distinct participation options:

1. Publish effects, retaining the native call with more precise dependencies.
2. Publish mathematical semantics or a semantic body, enabling reasoning across
   the operation, decomposition, fusion, and alternative realizations.
3. Publish an implementation candidate for an existing mathematical operation.

An ordinary C++ function should be able to publish a semantic body without being
declared a language-level `field` function. Explicit registration or annotation
would associate the callable with a versioned definition; automatic recovery of
arbitrary C++ mathematics is a much larger undertaking.

Selected Torch/ATen overloads could map into the same operations. Broadcasting,
aliasing, mutation, dtype, reductions, and derivative behavior must be accounted
for. Torch operator schemas, fake implementations, and autograd registrations
provide useful integration metadata, but do not themselves supply mathematical
lowering semantics. Unsupported operations remain valid native calls, using
effect contracts or conservative barriers.

## Adaptive relations and invalidation

The sparse-support facility already provides stable logical interaction IDs,
ordered participant roles, epochs, reverse postings, occurrence-preserving
temporal indexes, shared condition programs, primal/derivative worksets, gates,
surrogate gradients, bounded dormant sampling, structural proposals and
publication, parameter ownership, and snapshot restoration.

The [adaptive-support manager](include/Cellerator/compute/support/adaptive_support.hh)
prevents structural publication while relevant tapes pin the current epoch. The
[derivative sampler](include/Cellerator/compute/gating/derivative_sampling.hh)
explicitly describes a biased estimator. These contracts must survive compiler
integration.

| Change | Normal response | Possible specialization consequence |
| --- | --- | --- |
| Pointer or stream binding | Rebind runtime inputs. | Reconsider only candidate-specific binding constraints. |
| Numerical value generation | Refresh dependent values or packed operands. | Recompile only if values were explicitly specialized into code. |
| Cell state or conditional activity | Recompute eligibility and worksets. | Sustained workload change may favor another strategy. |
| Active-support generation | Refresh support-dependent artifacts. | Revalidate executables that assume a particular active subset. |
| Logical structure epoch | Revalidate affected indexes, projections, tapes, and coverage. | Rebuild artifacts whose structural assumptions changed. |
| Workload evidence | Reconsider candidate economics. | Switch when expected savings justify the cost. |

A generation change should invalidate only artifacts that depend on it. Global
cache invalidation on every training step would defeat the architecture.

The existing packed-base-plus-sparse-delta behavior is a useful precedent: new
edges can execute on a sparse route while an existing packed image serves covered
edges. Geometry rebuilding can be an economic decision rather than a prerequisite
for every mutation. Gate closure does not automatically authorize structural
pruning or recompilation.

Reduced primal work also does not establish reduced derivative work. Dormant
interactions may still require surrogate derivatives or exploration. Compiler
transformations must preserve the declared estimator, sampling policy, and
reproducibility.

## Duplication and architectural friction

Responsibilities overlap across operation-core schemas, compiler operation
representations, candidate catalogs, decomposition registries, geometry
strategies, prepared-relation state, sparse-support handles, and Python
orchestration.

Several distinctions remain legitimate: logical support versus physical
projection; semantic operation versus executable candidate; indexer versus
numerical kernel; optimizer checkpoint versus executable-artifact cache.

The friction is repeated translation and synchronization of identity, coverage,
generations, ordering, costs, and bindings. The sparse-support path synchronizes
native support state, Python descriptions, numerical endpoint arrays, projection
mappings, and worksets.

Establish shared dependency and binding descriptions before consolidating owners.
Adapt existing candidates into Planning and Realization IR; the existing
[candidate-catalog adapter](include/Cellerator/compiler/planning/adapt_candidate_catalog_v3_providers_v1.hh)
is a useful starting point. A broad registry or runtime rewrite would be premature.

## Runtime compilation options and lifecycle

Three responsibilities should remain separable:

1. Runtime planning and preparation using existing kernels.
2. Runtime code generation for specialized regions.
3. Validation, publication, retention, and retirement of executable plans.

The third responsibility is required even if a replacement only selects different
precompiled kernels.

| Option | Assessment |
| --- | --- |
| Existing generated C++/NVCC plus runtime compilation/loading | Closest initial route to existing backends; measure compilation latency and deployment dependencies. |
| Existing direct PTX emission plus Driver API loading | Attractive for the supported typed subset; requires session-owned module lifetime. |
| NVRTC behind CUDA source generation | Worth investigating for compilation/deployment benefits; does not supply host planning or executable lifecycle. |
| LLVM ORC for host code | Credible for runtime linking, compilation concurrency, and code ownership; defer selection until existing source/object routes are measured. |

The [embedded CUDA image helper](src/compiler/backend/nvptx/implement_fatbinary_object_embedding_for_direct_ptx.cc)
already loads and launches an image, but creates a context and synchronizes around
the operation. It is not a persistent session-integrated module manager.

The [planning cache](include/Cellerator/compiler/planning/implement_planning_cache_and_invalidation_v1.hh)
distinguishes semantic, profile, structure, order, target, toolchain, and planner
dependencies. [Lowering checkpoints](include/Cellerator/compiler/ir/realization/implement_lowering_resumption_checkpoints_v1.hh)
describe reusable artifacts across compilation stages. Both are foundations to
refine, rather than complete incremental-compilation services.

New lifecycle work includes immutable compile requests, cancellation,
duplicate-request suppression, bounded caches, executable dependency manifests,
validation, publication, and retirement. Old plans must remain alive for in-flight
launches and forward/backward tapes. Background results must be checked against
current dependencies before publication.

Existing [value-readiness events](include/Cellerator/runtime/value_readiness.cuh)
and [session resources](include/Cellerator/runtime/session.cuh) are useful
precedents. A value-readiness event alone does not protect executable lifetime.

External references consulted:

- [NVRTC 12.9](https://docs.nvidia.com/cuda/archive/12.9.1/nvrtc/index.html): device
  compilation and module-loading integration.
- [LLVM ORC](https://llvm.org/docs/ORCv2.html): runtime object linking, custom
  compiler layers, concurrent compilation, and removable resources.
- [PyTorch operator documentation](https://docs.pytorch.org/docs/stable/library.html):
  schemas, mutation contracts, fake implementations, and autograd integration.

## Experimental evidence and implications

All experiments below are synthetic. They establish bounded learning or execution
results, not biological validation or universal dispatch thresholds.

| Evidence | Retained result | Architectural implication |
| --- | --- | --- |
| Corrected learned support | Ranked pruned accuracy 56.25%, versus 34.38% shuffled; soft learned accuracy 68.75%. | Preserve learning, corrected gradient scatter, and dormant exploration; no general pruning-benefit claim. |
| Eligible event history | 509/512 correct with history, versus 122/512 with current-event input. | Input selection and visible dataflow can dominate numerical-kernel optimization. |
| Indexed shared values | Sparse event complete step 14.758 to 5.028 ms: 2.935x. Ranked effectively neutral; small control regressed. | Compare full-support and unique primal/derivative-union evaluation using complete costs. |
| Packed forward versus SIMT | Tested complete training generally favored SIMT; large-ranked packed training was 18.8% slower. | Include preparation, memory, value maintenance, and backward; retain packing candidates. |
| Indexing alternatives | Ordered postings improved over serial merge; scanning won the tiny case and remained competitive elsewhere. | Retain multiple indexers and account for preparation and reuse. |

The [corrected learning report](bench/sparse_support/evidence/qualification/learning/v9_corrected/REPORT.md)
records limited dormant sampling, no accuracy confidence intervals, and weak event
learning. Earlier results with incorrect derivative-to-gate scatter remain invalid
estimator evidence.

The [event-history diagnostic](bench/sparse_support/experiments/event_learning_diagnostic/README.md)
uses one controlled fixture and seed. History-route availability alone did not
explain the current-only model's errors; semantic input policy matters alongside
support.

The [indexed-values report](bench/sparse_support/experiments/indexed_values/evidence/qualification/v13/REPORT.md)
retains derivative-only candidates and recomputes values after optimizer
publication. It does not demonstrate cross-step value caching. Event U/E was
5.51%; ranked U/E was 78.31%. These are measured regimes, not dispatch thresholds.

The [complete-cost packing comparison](bench/sparse_support/evidence/experiments/scaled_sparse_campaign_20261027/torch_v13_complete_cost_20261028/REPORT.md),
[geometry-layout results](bench/sparse_support/experiments/geometry_layout/V13_RESULTS.md),
and [packed confirmation](bench/sparse_support/evidence/experiments/packed_training_v9_confirmation/run_001/REPORT.md)
show why tile-count or isolated-forward gains are insufficient. Cold preparation,
allocation, and SIMT backward can erase benefits. The confirmation's separate-
request aggregate interval crossed parity. A packed backward implementation
remains outside the established qualification.

In the [singleton-posting report](bench/sparse_support/evidence/experiments/posting_singleton_v1/REPORT.md),
a 42x discovery gain in a skewed case became a 1.71x discovery-plus-workset gain.
The [ordered-posting report](bench/sparse_support/evidence/experiments/scaled_sparse_campaign_20261027/native_ordered_fastpath_20261028/REPORT.md)
records substantial setup costs and requested storage; it does not establish an
amortized win over a fully prepared scanning lifecycle. Profiler replay timings
remain diagnostic rather than throughput evidence.

Specialization should require expected remaining uses multiplied by expected
per-use saving to exceed compilation, preparation, validation, switching, and
observation costs, with allowance for uncertainty. Hysteresis and minimum evidence
before switching are preferable to compilation on every gate threshold crossing.

FP16 can remain the general default where supported, with storage, multiplication,
accumulation, and output declared independently. Thresholds, reductions, surrogate
gradients, and recurrent state require operation-specific precision evidence.
Candidate selection must not silently change gate admission or the estimator.

## Proposed progressive experiments

These are discussion proposals, not authorized implementation tasks.

1. **Trace one connected computation through both front doors.** Express relation
   application plus reduction or normalization through native C++ and a language
   field. Require equivalent semantic identity, inspectable IR, an existing
   candidate, and prepared CPU execution against an independent reference.
2. **Replay distinct changes.** Change values, then activity, then publish one
   structural mutation. Record exactly which indexes, projections, operands, and
   plans rebuild. Value-only changes must not trigger structural planning; stale
   artifacts must not survive invalidating structural changes.
3. **Register one C++ helper and one external operation.** Supply semantics and
   effects alongside opaque fallbacks. Test planning across the operations while
   preserving aliasing, numerical, and derivative behavior.
4. **Compile one small region during application execution.** Reuse existing code
   generation, load an artifact, rebind values, and demonstrate cache reuse.
   Measure cold/warm costs; begin synchronously before adding replacement
   concurrency.
5. **Exercise replacement and adaptive selection separately.** Pin an old plan
   through a training window while preparing a replacement; reject stale results
   and retire resources safely. Then compare a bounded portfolio under changing
   density, including compilation, preparation, memory, and complete training.

Before broader development, discuss the semantic registration contract, native
C++ composition surface, artifact dependency model, and executable publication
ownership. Defer wholesale Torch lowering, arbitrary C++ semantic inference,
global subsystem consolidation, and gate-driven recompilation.

Cellerator may exploit declared eligibility, learning objectives, and approximation
contracts. A low observed weight alone does not authorize removing an interaction.
Applications supply scientific meaning; Cellerator makes the resulting computation
efficient, adaptable, inspectable, and overridable.
