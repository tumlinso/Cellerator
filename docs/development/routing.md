# Cellerator contributor routing

This guide routes work to its source owner and records how to make and validate
the source map. It is durable contributor guidance, not a task tracker. Use the
registered Project Control workflow for current task status, ownership claims,
and completion evidence.

## Product boundary

Cellerator is an extensible machine-learning framework organized around learning
biological representations, relationships, and behavior. Route work by the
representation, architecture, objective, and execution needs of the learning
problem. Cellular dynamics, persistent state, and population execution are
capabilities within this purpose; they are not a mandatory foundation for every
network. Static representation learners, dense attention, transformers,
recurrent models, conditional computation, and hybrids all belong in scope.

Native numerical machinery, structured layouts, and specialized kernels can
exploit structure a model contains or discovers. Sparsity and explicit
biological annotations are optional. PyTorch is useful infrastructure for
learned modules, autograd, and tensors; neither a Torch-only interface nor a
universal cell-transition interface defines the framework. See the
[scope](../scope.qmd) and [learning-problem architecture](../architecture.qmd#organizing-the-learning-problem)
for the governing intent.

Consumers retain scientific assumptions and validation. Explicit relationships
can make a question clearer without guaranteeing causal or molecular
identification. Identifiability is relative to the question, model family,
assumptions, and experiment; latent ambiguity is not a universal prohibition.
GlassHelix investigates formal inferability, while Cellerator supplies reusable
learning mechanisms that can reflect the question and its available constraints.

The native method components own preprocessing, velocity, and routed search.
Cellerator owns the reusable modeling and numerical substrate. Callers provide
explicit prepared arrays, tensors, or views. AnnData++ owns native H5AD and
Zarr access. CellShard source is retained as a quasi-deprecated component
detached from supported Cellerator builds and installs. See the
[component ownership record](component-ownership.md).

Keep entity and relation identity, structural order, activity/eligibility,
values and parameters, logical/physical layout, and runtime bindings distinct.
Changes have different invalidation and reuse consequences. Do not infer a
complete adaptive runtime from a narrower gate, a sparse representation, or a
test.

Cellerator is early. Sparse support, indexing, routing and gating are promising
facilities with operation-specific maturity. The small implemented `nn` surface
does not limit the intended range of learned architectures or objectives;
its growth should follow reusable modeling needs and actual implementations.
This guide does not imply a standard biological layer catalogue. Cellerator's
ExecutionField composition remains experimental; its current CPU field has no
JIT or live recompilation and is not a prerequisite for ordinary model use.
Texo owns the language compiler. In Cellerator, numerical geometry work is
planning (`plan_geometry`), while operation and joint-work preparation belong
to execution. New backend-neutral modeling interfaces, state/event
representations, interaction mechanisms, dynamics, and adapters have been
added in rapid development; their implementation maturity and validated
backends differ. The Cellerator AnnData component handles backed buffer reads.
Native method calls take caller-prepared views.

## Source routing

The map routes source to its present subsystem and job; it does not force every path into a new top-level package directory. For native source, owner families follow the actual CMake and header boundaries. For tests, the exact suite path is the current owner, while build registration and test execution remain separate facts. A route may therefore be concrete while maturity remains unqualified.

The [generated source map](source_map/path_map.csv) assigns every indexed path
a current role, owner, target destination, disposition, maturity, consumer and
evidence pointer. Shared metadata lives in
[`routing_rules.json`](source_map/routing_rules.json); per-path Git and worktree
identity plus archive references live in the path map. The map is a navigation
and disposition aid, not proof of implementation, API stability, scientific
validity, or task completion.

| Change | Current source route | Intended owner | Routing notes |
|---|---|---|---|
| Learned representations, neural composition, objectives, and training | `python/cellerator/nn.py`, `ops.py`, `torch`, `adapters`, with relevant interaction/inference and native operation owners | Reusable biological learning capabilities, using existing numerical and ML infrastructure where appropriate | Begin with the representation and learning question. Current namespace placement is not a hierarchy of importance. Do not force a learned component into `Cell` or a state-transition wrapper; do not imply that every intended architecture or objective exists. |
| Optional model instances, dynamics, and population mechanics | `python/cellerator/modeling`, `state`, `events`, `dynamics`, `interactions`, `execution`, and `inference` | Composable mechanisms for model rules, opaque state, event histories, numerical updates, relations, population execution, and inference | Use relevant contracts without requiring every model to adopt dynamics or a lifecycle. Preserve arbitrary representations and general neural composition. Do not place consumer hypotheses in general packages or assume package presence qualifies a native/GPU backend. See the [biological framework](../biological_framework.qmd) for current APIs and [architecture](../architecture.qmd) for their intended role. |
| Learned-model adapters | `python/cellerator/adapters` | Optional bridges to general learning libraries such as Torch, distributed execution, CUDA, and ggml | Keep optional dependencies lazy. Adapters must not define native types, data ownership, or the core state contract. |
| Single-cell preprocessing and RNA velocity | `components/preprocess`, `components/velocity` | The components own method behavior and reference implementations. | Link `Cellerator::preprocess` or `Cellerator::velocity`. Use native views and preserve biological axis identity. |
| Routed neighbor search | `components/neighbors` | Owns CUDA routed exact search and forward-neighbor policy. | Link `Cellerator::neighbors` and use the component headers. |
| AnnData++ file access and buffer reads | `packages/anndatapp`, `components/anndata` | AnnData++ owns data storage and file I/O. Cellerator owns native backed-buffer reads. | Use `import anndatapp;` for standalone access and `import cellerator.anndata;` for native buffer reads. |
| Retained velocity source | `src/ops/velocity`, `tests/velocity` | Inactive source retained for later work | Use `Cellerator::velocity` for active integration. Generic `CelleratorBinary16` remains separately buildable. |
| Retained CellShard component | `components/CellShard/**` | Quasi-deprecated source detached from supported Cellerator builds and installs | Source is preserved; CellShard headers are excluded from host and SDK installs. Retired CMake options and Cellerator package-component requests fail clearly. |
| Native mathematical or model operation | `src/ops/<family>` and matching public headers | Keep arithmetic, derivatives, and specialized backends with the operation. Keep candidate selection and generic binding in execution. | Examples include `Cellerator::ops_numeric`, `Cellerator::csr_multiply`, `Cellerator::private_ports`, and `Cellerator::indexed_mechanism`. Link the concrete owner for the operation. |
| Native data and cellular geometry representations | `structure`, `geometry/dependency`, `state`, `tensor`, `layout`, and `interop/formats` | Structure owns identities. Geometry owns declared relationships. State owns structured model coordinates. Tensor defines encodings and layout materializes physical maps. | The installed `native_data` component exposes geometry, execution, tensor, and host operation targets. See [native data structures and cellular geometry](../native_data.qmd). |
| Identity, support, indexing, and gating | `structure/indexing`, `geometry/dependency`, `geometry/discovery`, `ops/condition`, and the execution dependency owners | Generic postings and occurrences live in structure. Geometry owns support and discovery, condition ops evaluate predicates, and execution prepares worksets and submits dependency programs. | Link the concrete `Cellerator::support`, `Cellerator::indexing`, `Cellerator::gating`, `Cellerator::dependency_geometry`, or `Cellerator::dependency_execution` target. |
| Native sequence and regulatory execution | `components/sequence` and `src/execution/sequence` | The sequence component owns coordinates, borrowed encodings, packing, and predicates. Regulatory execution consumes it. | Link `Cellerator::sequence`. See the [native sequence guide](sequence.md). |
| Prepared model routing | `include/Cellerator/ops/sparse_support/prepared_routing.hh`, `src/ops/sparse_support/prepared_routing.{cc,cu}`, its sparse-support Torch binding and Python adapter | Opt-in prepared entity-to-mechanism route query; current Python query remains the default | Frozen binary passed 16 tests, exact CUDA correctness, and memcheck/racecheck/synccheck. V100 lifecycle timing found a bounded long-reuse win for ranked `[32,8]`, an inconclusive event shape, and dense Torch wins on larger stress shapes. Preserve caller-supplied route and binding generations; see the [qualification report](https://github.com/tumlinso/Cellerator/blob/94c534fc794fe8cc3e0434e4a50f379401ddbb3c/bench/sparse_support/experiments/prepared_routing/REPORT.md). |
| Geometry, physical layouts, values and gradients | `geometry`, `structure`, `tensor`, `layout`, `ops`, and `execution` | Geometry chooses row and feature groups. Tensor views define the numerical encoding. Layout materializes the chosen order and its maps. Packed image readers bind native views through `geometry/persistence`. | `layout::CellPackingGeometryView` carries physical summaries and `execution::CellPackingBinding` carries live association. Measure preparation and conversion when making operation-cost claims. |
| Prepared execution and runtime | `execution/{planning,preparation,program,streaming,tuning,profiling}`, `memory`, and `runtime` | Execution owns candidate plans, bindings, readiness, and submission. Memory owns allocation contracts and copies. Runtime supplies sessions and fleets. | The native-data dependency owner is `Cellerator::dependency_execution`. Optional CUDA composition uses `Cellerator::native_data_cuda`. |
| Torch and Python adapters | `src/interop/torch`, `python/bindings`, `python/cellerator`, `tests/bindings`, `tests/package` | Native LibTorch integration and Python-only extension sources | Compose Torch rather than duplicate its tensor, autograd, or optimizer systems. Current optional build/API limits are in the Python binding guide. The paused CelleraTorch proposal is archived in `docs/history/celleratorch.qmd`; it is not `python/cellerator/torch`. |
| Language compiler | [Texo](https://github.com/tumlinso/Texo) | Texo owns language-compiler source, build/install integration, public APIs, tools, resources, tests, examples, and documentation | Cellerator does not bundle or install the language compiler. Consult Texo for compiler implementation and evidence; the ownership move does not assert a complete compiler pipeline. |
| Model fixtures and single-cell workflows | `examples`, `docs/pipeline` | Examples, testbeds, or consumer-owned applications | Model and trajectory headers live with their workloads under `examples/` and install under `Cellerator/examples/`. Keep useful fixtures and operations. Keep scientific assumptions with Karyoplast, Cytoplast, GlassHelix, or the relevant consumer. |
| Performance, correctness, and research evidence | `bench/**`, qualification records, dated implementation docs | Source-bound evidence index and operation owner | Preserve original reports and rejected alternatives. Scope correctness, biological validation, and performance separately. Some code under an experiment path is linked into installed targets; audit targets and consumers before moving or archiving it. |
| CellPack packing | `geometry/packing`, `geometry/discovery`, `layout/packing`, `layout/projection`, and `ops/reduction` | Geometry chooses groups and scores support. Layout materializes the selected order and physical records. Operations consume those records. | Keep source-linked evidence, biological recovery maps, and the existing checkpoint formats. See the [CellPack pipeline](../cellpack_cp_bp.qmd). |
| Numerical reference and physical projection helpers | `components/reference`, `src/layout/projection`, `src/layout/conversion`, and their public headers | `Cellerator::spmm_reference` owns the independent numerical reference. `Cellerator::physical_projection` owns borrowed CSR and native-tile adapters and BELL lowering. The optional `Cellerator::packed_dense_operand` target owns CUDA operand packing. `Cellerator::compute_matrix_convert` owns CUDA compressed conversion and bucket transforms. | Current tests and benchmarks link their concrete owner. The original `compat` tree and execution-plan metadata are archived byte-for-byte under `docs/history/compat`. |
| Product, package, operation, and contributor guides | Root `README.md`, `scope.md`; `docs/{index,architecture,scope,getting_started,model_building,operations,layout_execution,current_implementation,developer_reference}.qmd`; `docs/development/{operation_registry,implementation_style,python-bindings,routing}.md` | Maintained product and developer documentation | Keep product direction concise; distinguish current API/evidence from proposed ownership and preserve the small, implemented `nn` surface. Former root guide pointers are preserved in `docs/history/repository-notes`; use the maintained operation registry and implementation style guide directly. |
| Experimental and historical design pages | `docs/experiments_and_history.qmd`, retained subsystem chapters, `docs/development/experimental-fields.md` | Exact subsystem owner and evidence record | Keep useful history accessible with currentness and maturity labels. Never rewrite immutable source-bound report bytes to fit a new product description. |
| Source map and agent routing | `docs/development/routing.md`, `tools/maintenance/source_map/`, generated `docs/development/source_map/` | Documentation/maintenance owner with source owners | Generated map gives per-path role, current owner, destination, disposition, maturity, consumers, tests/evidence, and archive reference. Regenerate after source or routing-rule changes; Project Control remains live task status. |
| Plans, Todo state, and repository metadata | `planning`, including `planning/archive/legacy-plans` and `planning/archive/legacy-todo`; current `.todo-orchestrator/project.json` and authority database; `.project-control` | Approved plan, supported Project Control authority, or preserved history | Plans record intent; Project Control owns live status. The [legacy Todo archive index](https://github.com/tumlinso/Cellerator/blob/main/planning/archive/legacy-todo/README.md) resolves archived projections, snapshots, plan inputs and runtime artifacts by former path and content identity. The live project metadata and authority database remain under `.todo-orchestrator`; never hand-edit them or any generated Todo projections to mimic completion. |

Host patch and ragged-port providers live in `src/ops/matrix` and
`src/ops/process` with explicit targets and selectable Substrate SDK components.
Run the current native operator checks to validate those providers. The original
path-based Python proofs and their source-bound receipts are preserved at the
pinned checkout listed in the [Baseplane source history](../history/baseplane.md).

Historical branch closure and artifact holds are documented in the
[archive disposition](https://github.com/tumlinso/Cellerator/blob/94c534fc794fe8cc3e0434e4a50f379401ddbb3c/planning/coherence/archive-disposition.md)
and its linked [branch-closure receipt](https://github.com/tumlinso/Cellerator/blob/94c534fc794fe8cc3e0434e4a50f379401ddbb3c/planning/coherence/branch-closure.json).
These explain preserved refs, bundles, and held historical paths; they do not
replace per-path disposition in the generated source map.

The [legacy plan archive](https://github.com/tumlinso/Cellerator/blob/main/planning/archive/legacy-plans/README.md)
and [repository-note archive](https://github.com/tumlinso/Cellerator/blob/main/docs/history/repository-notes/README.md)
resolve the former root files. Their original bytes and managed historical
references are preserved; archive placement does not make them current guidance.
The older archive-disposition record retains its original path holds as dated
history. Use these newer archive indexes for the reconciled locations.

The [legacy Todo archive index](https://github.com/tumlinso/Cellerator/blob/main/planning/archive/legacy-todo/README.md)
resolves retired Todo projections and runtime artifacts. The current
`.todo-orchestrator/project.json` and authority database remain live workflow
state and are not part of that archive.

Use `build/` as the checkout's local CMake output directory. The IDE uses
`build/`, and separate configurations use subdirectories such as
`build/ctxpp/`. Keep only build trees with a current configured consumer in the
checkout. Retired builds,
local source captures, test receipts and profiler output belong in one verified
durable archive with original-path and content manifests. Historical reports
retain their recorded build paths; recover their original bytes from the archive
when an old build is needed.

## Editing and claim rules

1. Read the repository contract and any applicable subtree instructions. Use
   the active registered task for mutable status and scoped ownership.
2. Before a source change, trace implementation, headers, build targets,
   generated registration, public entry points, Python/native callers, tests,
   examples, and source-bound evidence. Update related consumers with a move.
3. Separate **intended architecture** from **actual implementation**. Cite the
   exact source and relevant tests for a capability. A test pass does not prove
   biological validity or performance; a benchmark does not prove package
   usability or a general speedup.
4. Preserve original bytes and provenance for frozen evidence. Prefer a wrapper
   or index entry for new interpretation; do not edit raw reports to fit the
   current architecture story.
5. Before archival or deletion, inventory tracked, staged, unstaged, untracked,
   ignored, and submodule state. Verify the external recovery manifest covers
   the exact bytes and paths; the source map does not itself archive files.
6. If no rule fits, add an explicit path-specific or owner-family rule before
   accepting the map. Do not blanket-route unknown work into a convenient
   package folder or remove it to silence a validator.

## Generate and validate the map

From the repository root, run:

```sh
python tools/maintenance/source_map/generate.py --repo .
python tools/maintenance/source_map/validate.py --repo .
python tools/maintenance/source_map/lifecycle_test.py
```

The checkout is passed explicitly so the tools do not rely on a machine-specific
root. Run the scripts by absolute path if invoking them from another directory.
The default output is `docs/development/source_map/`; both tools accept
`--output` to select another directory under or outside the checkout. The map
normalizes shared ownership/disposition rules and stores a row per tracked and
non-ignored untracked path. Untracked paths are identified explicitly; generic
unclassified paths fail validation until a rule is added. Ignored paths are
summarized by top-level counts only; ignored payload contents are not opened.
The separate private recovery manifest is authoritative for ignored payload
preservation.

Source identity binds the non-generated Git index, exact worktree content and
mode, non-ignored untracked inputs, and the generator, validator, and shared
snapshot code. Generated map paths are routed as metadata but excluded from
that identity and represented without their own blob/content hash. Output hashes
and a manifest self-digest catch generated-file edits. The captured HEAD is
provenance only, so committing generated outputs does not make validation stale.
Source, rule, or path changes require regeneration; they do not require a second
approval/status file. Validation checks path/rule/content identity and generated
output integrity only. It does not certify source semantics, builds, tests,
science, or performance.

## Current source migration closure

The [active source-cleanup migration manifest](https://github.com/tumlinso/Cellerator/blob/main/planning/active-source-cleanup/source-migrations.json) records path moves, retained destinations, and retired payloads for this cleanup. Use it to resolve old source-bound paths and to verify that every current destination exists. It is the current path-closure record; Project Control remains the authority for live task status.

Run the closure audit with:

```sh
python tools/maintenance/source_map/validate.py --repo . \
  --migration-manifest planning/active-source-cleanup/source-migrations.json
```

The earlier [coherence migration snapshot](https://github.com/tumlinso/Cellerator/blob/9a0353c/planning/coherence/source-migrations.json) remains immutable historical context for its original baseline. It is not the current destination-closure policy after these later moves. Do not edit or use that snapshot for the active-tree closure audit. The source-map migration check verifies declared destinations and retirement accounting; it does not compare file contents or establish capability, build, test, science, or performance status.

A required native evidence name needs a configured producer that records that passing fact. Committed reports, scheduler receipts, and external test runs belong in source references and reviewed assertions; their presence alone does not create a native passing fact. Live task state remains in Project Control.

## Local fixture data

Use `CELLERATOR_DATA_ROOT` for external test and benchmark payloads. The default
is `~/data/Cellerator`. The [fixture staging guide](data-fixtures.md) gives the
expected layout and identifies the tracked manifests used by each caller.

## Optional C++ retrieval maintenance

[ctxpp](https://github.com/tumlinso/ctxpp) is an optional standalone C++ context tool. Ordinary source navigation and coding do not require it or a globally installed skill. `.ctxpp.toml` selects `build/ctxpp/compile_commands.json`. The generated `.ctxpp/index.sqlite` holds the graph. `.ctxpp/observer-export.json` describes the published generation for optional readers. Canonical source files remain authoritative.

Configure a dedicated database for the source configurations you want indexed.
For example, this configuration includes CUDA translation units for SM70.

```sh
cmake -S . -B build/ctxpp -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCELLERATOR_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF \
  -DCELLERATOR_BUILD_TESTS=ON \
  -DCELLERATOR_ENABLE_PYTHON=OFF -DCELLERATOR_ENABLE_TORCH=OFF \
  -DCELLERATOR_ENABLE_TOKENIZATION=OFF \
  -DCELLERATOR_ENABLE_SPARSE_SUPPORT=ON \
  -DCELLERATOR_ENABLE_CONDITIONAL=ON \
  -DCELLERATOR_ENABLE_ANNDATA=OFF
```

Refresh and export through the CLI.

```sh
ctxpp scan
ctxpp export > .ctxpp/observer-export.json
```

Scanning reuses validated translation-unit contributions. Refresh after source-path, include-root, build-flag, or configuration changes; preserve the previous generation until freshness, source coverage, and representative lookups pass. Never hand-edit the database or generated projections. Coverage may remain partial where compilation commands or supported parsers are unavailable; an indexed relationship does not prove runtime behavior.

Project Control uses this provider only for repositories explicitly listed in operator-managed `PROJECT_CONTROL_CPP_EXPORTS` configuration. Without that opt-in, its source retrieval works independently of ctxpp. Keep repository guidance declarations in `.project-context.json`. Historical [retrieval qualification](https://github.com/tumlinso/Cellerator/blob/main/planning/active-source-cleanup/qualification/retrieval-qualification.json) describes the earlier generation and its semantic versus lexical coverage; it is not a claim about the current index.
