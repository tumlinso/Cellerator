# Repository work contract

<!-- project-control:start -->
## Project Control and Todo workflow

Use Project Control's existing `overview`, `search`, `delta`, `frontier`, `history`,
`evidence`, and `impact` reads for registered-workspace context and migrated Todo
status/history. The compact profile catalogs are observer 11, coder 12, and
mutator 16 tools. Routine workflow begins with
`next_task(project, repository?, task_id?, run_id?)`; pass its returned handle to
`inspect_task`, `coordinate_task`, and `finish_task`. Use the live host manifest
for exact schemas and profile permissions.

Project Control binds workflow handles to the registered workspace, identity,
profile, credentials, and providers. Routine agents should not manage native
Todo IDs, leases, credentials, or retry keys, and do not pass raw repository
paths as selectors. Privileged plan changes, execution maintenance, and project
metadata stay on their established routes. Observer reads need no task claim and
cannot mutate. The standalone Todo API and CLI remain available for separately
managed authorities. Imported completion is a source assertion, not native
verification; unknown legacy effects require attention and are never replayed
automatically. Existing non-Todo tools remain available for their own purposes.
<!-- project-control:end -->

Read the [architecture](docs/architecture.qmd) for intent, [current implementation record](docs/current_implementation.qmd) for its dated boundary, and [developer reference](docs/developer_reference.qmd) for entry points. The [Python binding guide](docs/development/python-bindings.md) covers the current package surface. For an edit, the current scoped task and inspected source—not a historical plan or README completion sentence—determine what exists and what is authorized.

For source ownership, evidence boundaries, and path routing, see the [contributor routing guide](docs/development/routing.md). The source map linked there is not live task status or capability proof.

## Biological machine learning is the organizing principle

Read [scope](scope.md) and [architecture](docs/architecture.qmd) before proposing
model-facing abstractions. The original biological ML ambition is the governing
design authority: develop computational abstractions around the structure of
the cell and the systems it comprises, making cellular representations,
relationships, and dynamics natural to learn while preserving general
neural-network expressiveness and exploiting biological structure for efficiency.
Evaluate representation, architecture, learning objective, and execution together.
Compiler, scheduler, distribution, and model-sharing machinery serve this
ambition; they do not define the framework's distinctive purpose. Do not organize
it as a cellular simulator with neural learning added through adapters.

- Read the [modeling and neural-network backbone](docs/architecture.qmd#modeling-and-neural-network-backbone)
  before operationalizing model-facing work. Its six concerns are conceptual
  responsibilities, not fixed classes or a required pipeline. Distinguish the
  existing foundations from the remaining modeling and training capabilities;
  neither a kernel nor a namespace establishes the full capability. Await the
  user's operationalization or implementation instructions rather than treating
  this architectural record as permission to develop it.
- Use the [NN foundation guide](docs/nn_foundation.qmd) for the implemented
  generic `Cell`, explicit `Dynamical*` migration, composable neural components,
  biological bindings, prepared relations, labelled queries, explicit supervision,
  auxiliary views, and custom-operation contracts. Keep canonical biological IDs,
  token/model indices, and occurrence/physical positions distinct. Prepare stable
  identity and relation structure at boundaries; value changes do not redefine
  topology. Context metadata is descriptive: supplied numerical conditioning must
  actually enter a registered computation or fail clearly. Preserve biological
  meaning in explicit manifests beside weights, without changing native-owner
  checkpoint contracts or adding executable reconstruction. Current qualification is
  bounded by the implementation record; the first milestone is not the full
  biological modeling ambition or permission for unrelated development.
- Preserve [first-class C++ and Python modeling](docs/architecture.qmd#cpp-and-python-modeling).
  C++ should be at least as expressive for Cellerator's biological and neural
  abstractions, with intuitive model authoring, training, composition, and
  extension. Native kernels beneath a Python-only modeling layer are insufficient.
  Design the public C++ API as an idiomatic native library, not translated Python
  bindings or a cosmetic LibTorch wrapper; Python adapters must not determine
  native types, composition, ownership, or extension conventions.
  Foreground C++ typing, ownership, specialization, and HPC integration, and
  Python experimentation and ecosystem integration. Reuse established backends
  without making LibTorch ergonomics the public modeling API. Distinguish this
  target from current Python-facing NN capabilities; documentation is not
  permission to implement a new backend or duplicate conventional ML machinery.
- Read the [native NN guide](docs/native_nn.qmd) for native model authoring.
  Models and results are ordinary typed structs/callables. Parameters have an
  explicit owner and numeric groups: do not introduce named module registration,
  a member visitor protocol, a mandatory model superclass, or a universal tensor
  facade. Prefer templates, primitive views, explicit extents and lifetimes, and
  separate allocation owners. Use the optional host C++20 module interface
  without making NVCC parse modules or forcing module tooling on header users.
  This C++ interface targets C++ model developers/HPC users; Python supplies its
  own approachable data and experimentation route. Bindings adapt the native API.
- Treat `Cell`, explicit state transitions, histories, and population execution
  as optional capabilities. Static representation learning, dense attention,
  transformers, recurrent networks, conditional models, and hybrids are equally
  valid; no model must adopt a cell lifecycle or biological ontology to participate.
  Dynamics remains a central motivation: optional transition APIs do not demote
  it or privilege static encoders over history-dependent representations.
- Biological structure should add practical expressiveness and computational
  opportunities. Keep supplied and learned relationships possible, preserve
  general neural composition, and treat sparsity as an opportunity rather than
  an architectural requirement. PyTorch is optional infrastructure, not the
  definition or limit of the framework.
- State what a learning objective and observation experiment actually constrain.
  Explicit biological structure does not guarantee causality or identifiability;
  latent-state ambiguity does not imply universal non-identifiability either.
  Frame identifiability relative to the question, admissible model family,
  assumptions, and experiment. GlassHelix owns formal inferability investigation;
  Cellerator owns reusable modeling and learning machinery.
- Distinguish architectural intent from current package maturity. Existing
  state-transition APIs and a small `nn` surface do not define the future scope.
  Do not infer permission for implementation changes from a documentation update.
- Preserve the [cell representation and auxiliary-view design request](docs/architecture.qmd#cells-representations-and-auxiliary-views)
  when revisiting elementary classes. Keep the API minimal: a cell may reference
  model-specific latent features or distributed history/context, and shared
  trainable auxiliary models should derive views such as ATAC without per-cell
  parameter copies or a mandatory state vector. Plan for lazy, batched execution
  and valid representation reuse. The documented syntax is illustrative, not a
  settled API or permission to implement it.
- Keep [model reuse and sharing](docs/architecture.qmd#building-training-and-sharing-cell-models)
  subordinate to the biological ML ambition. Building, training, and sharing
  cell models describes the user experience; the Transformers analogy concerns
  coherence and usability, not a required architecture or backend. Hub integration
  and registration are prospective supporting milestones, not the framework's
  defining objective, existing capabilities, or authorization to implement,
  publish, or register anything.

## Operate within the current authority

Use the installed Project Control/Codex front door and this repository's Todo authority. Take one bounded outcome, inspect only what it needs, preserve other claims/dirty work, and finish with actual evidence. Use configured subagents only for useful bounded work; the owner retains acceptance. Never edit SQLite, generated Todo views, recovery snapshots or generated context indexes by hand. Keep managed workflow blocks intact.

## Preserve the architecture while changing the implementation

- Preserve biological domain/order identity and distinguish immutable structure epochs from mutable value generations.
- Keep discovery/semantic geometry separate from physical realization and actual runtime state. Prediscovered runtime gates are part of the intended design; do not relabel a narrower value update as complete adaptive execution.
- Keep execution order/conversion, allocation, streams and complete costs explicit. Candidate choice is measured, not mandated by a favored sparse format or Tensor Core path.
- Cellerator owns general compilation and numerical execution, including its optional Python and Torch adapters. Baseplane owns sequence meaning, GH owns scientific inference, and CellShard owns persistence/delivery. Preserve actual cross-project consumers.
- Existing code/contract evidence survives cleanup. Resolve a real interface change through its owner; no silent broad ABI rewrite.

External compatibility is not an absolute constraint at this stage. A reviewed internal move may change names/interfaces if it improves development; repair real sibling consumers and relevant tests together. Frozen contracts and overlapping active work still require their owner's explicit reconciliation. Do not use a documentation task to rewrite numerical behavior, introduce another planner, or complete unrelated old epics.

## Hardware and algorithm choice

Use the [CUDA skill](/home/tumlinson/.agents/skills/cuda/SKILL.md) for GPU
computation, porting, correctness, profiling and optimization. Follow its focused
guidance for the actual operation and target hardware, including host execution
rules when running GPU work.

Aim to make full use of the hardware. Reflect on algorithm, representation and
ownership choices before tuning a kernel: consider how layout, locality, reuse,
fusion and precision can make the semantic operation native to the machine.
Consider unconventional algorithms or encodings when they offer a sound fit.
Choose wisely using an independent correctness oracle and complete workload
costs, including preparation, conversion, memory and communication. Compare
with the strongest conventional or library alternative and retain the evidence
that supports or rejects each candidate.

## Software ports and design probes

During active library development, any explicitly user-authorized software
port or design probe with a scoped Project Control task carries standing
permission for additive changes to Cellerator, without renewed permission. Keep
additions general, clean, and maintainable, with explicit contracts and
validation; use small architectural cleanup only to resolve demonstrated
friction. General numerical, state, graph, relation, statistical, and
iterative mechanisms belong in Cellerator; Baseplane should own only
intrinsically sequence-grounded functionality.

Use FP16 as the general computation default, retaining FP32, FP64 and other
supported precision types as explicit alternatives. FP16 execution may use
ordinary CUDA arithmetic or Tensor Cores; choose the implementation by the
operation's contract and measured complete cost. Declare storage, multiply,
accumulation and output types separately, using FP32 accumulation where
appropriate. Respect explicit caller precision and existing operation contracts.
Compare approximate computation with upstream/native computation; a
higher-precision reference may supplement that comparison. Keep numerical claims
within demonstrated evidence and retain suitable alternatives for unsupported
or unsafe regimes. See [Numerical policy](docs/architecture.qmd#numerical-policy)
for the provisional evidence boundary and current implementation distinction.

Prefer templates for algorithms whose behavior is independent of numeric type.
Keep genuinely type-specific arithmetic, numerical constraints and hardware
specializations explicit; do not force them into a shared implementation.

When a basic reusable primitive appears missing or insufficient, report this
promptly to the user. Include the concrete computation and evidence, current
library capability and gap, proposed owner and general contract, and
downstream impact. Discuss major API, granularity, ownership, or architectural
choices with the user before committing to them.

## Validate what changed

Use the [developer reference](docs/developer_reference.qmd) for build and test commands. Run affected source/build/tests after code moves. For prose-only changes check links, status boundaries and rendered pages. Keep scientific claims scoped; record what was not run. Benchmarks require the existing assigned resources and clean timing interval. Include setup/transfer/routing when the claim needs them; no benchmark is launched by document generation.

Current work belongs in live status tools; experimental findings stay with their source-bound records and the dated [implementation record](docs/current_implementation.qmd); durable rationale belongs in design/development docs; superseded plans stay in the archive. Do not recreate the same mutable status table in all four places.

<!-- agent-interface:autobench:start -->
This repository has a canonical Autobench interface.
Route benchmark registration, background measurements and profiling, and registered benchmark evidence requests through `autobench`.
Consider installing the [Autobench skill](https://github.com/tumlinso/autobench)
to simplify development; if installed, read its `SKILL.md` before use.
<!-- agent-interface:autobench:end -->
