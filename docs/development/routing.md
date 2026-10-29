# Cellerator contributor routing

This guide routes work to its source owner and records how to make and validate
the source map. It is durable contributor guidance, not a task tracker. Use the
registered Project Control workflow for current task status, ownership claims,
and completion evidence.

## Product boundary

Cellerator is a general computational and machine-learning framework for
biological modeling, especially cellular dynamics. Its native numerical
machinery, modeling interfaces, structured layouts, and specialized kernels
can express and exploit biological identities, relations, state changes, and
interaction geometry. PyTorch is useful optional integration for learned
modules, autograd, and tensor operations; it does not define the framework's
scope. Consumers retain their scientific assumptions and validation.

Keep entity and relation identity, structural order, activity/eligibility,
values and parameters, logical/physical layout, and runtime bindings distinct.
Changes have different invalidation and reuse consequences. Do not infer a
complete adaptive runtime from a narrower gate, a sparse representation, a test,
or the existence of compiler code.

Cellerator is early. Sparse support, indexing, routing and gating are promising
facilities with operation-specific maturity. The `nn` surface should grow as
real `torch.nn.Module` compositions are implemented; this guide does not imply a
standard biological layer catalogue. Execution fields, compiler/IR integration,
JIT and live recompilation remain experimental and are not prerequisites for
ordinary model use. New backend-neutral modeling interfaces, state/event
representations, interaction mechanisms, dynamics, and adapters have been
added in rapid development; their implementation maturity and validated
backends differ. Classical single-cell analysis remains useful upstream. The
AnnData adapter performs caller-directed data exchange and does not preprocess
or select scientific meaning automatically.

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
| Backend-neutral model instances and dynamics | `python/cellerator/modeling`, `state`, `events`, `dynamics`, `interactions`, `execution`, and `inference` | Distinct open Python package boundaries for model rules, opaque state, event histories, numerical updates, sparse relations, population execution, and inference | These are model-facing interfaces and reference mechanisms under rapid development. Preserve arbitrary state representations. Do not place consumer biological semantics in general packages or assume package presence qualifies a native/GPU backend. See the [biological framework](../biological_framework.qmd). |
| Learned/data adapters | `python/cellerator/adapters` | Optional bridges to general learning libraries and explicit AnnData boundaries | Keep optional dependencies lazy. AnnData identity/layer selection is caller supplied; preserve sparse inputs and exact row/column identities. Do not make Torch the core state contract. |
| Native mathematical or model operation | `src/ops/{numeric,matrix,process,sparse_support}` and the still-active `src/compute/<family>` operations, with matching public headers | Keep ownership with the concrete operation family (`Cellerator::ops_numeric`, `Cellerator::op_matrix_patch`, `Cellerator::op_process_ports`, or the named `src/compute/operation/<family>` target) | `src/ops/numeric`, `matrix`, `process` have explicit CMake owners and Substrate SDK consumers. `src/compute` still owns Product2, native numeric and differential operations, indexed mechanisms, model ops, candidate portfolios, operators, tokenization and training families. The source map records the actual family and evidence boundary; it does not imply each file is built or qualified. Keep derivatives, precision, build registration, and callers together. |
| Identity, support, indexing, and gating | `src/structure`, `src/support`, `src/indexing`, `src/gating`; public identity headers under `include/Cellerator/structure` and support/indexing/gating under `include/Cellerator/compute` | Distinct structure identity, logical eligibility, lookup, and conditional-work owners | CMake currently defines `Cellerator::identity_registry`, `::support`, `::indexing`, and `::gating`, with a compatibility `::sparse_support` umbrella. Host checks include `tests/biological_abi/biological_abi_host_test.cc`, `tests/execution_identity/identity_registry_test.cc`, and focused `tests/sparse_support/*`; CUDA workset checks are separately gated. |
| Prepared model routing | `include/Cellerator/ops/sparse_support/prepared_routing.hh`, `src/ops/sparse_support/prepared_routing.{cc,cu}`, its sparse-support Torch binding and Python adapter | Opt-in prepared entity-to-mechanism route query; current Python query remains the default | Frozen binary passed 16 tests, exact CUDA correctness, and memcheck/racecheck/synccheck. V100 lifecycle timing found a bounded long-reuse win for ranked `[32,8]`, an inconclusive event shape, and dense Torch wins on larger stress shapes. Preserve caller-supplied route and binding generations; see the [qualification report](https://github.com/tumlinso/Cellerator/blob/94c534fc794fe8cc3e0434e4a50f379401ddbb3c/bench/sparse_support/experiments/prepared_routing/REPORT.md). |
| Geometry, physical layouts, values and gradients | `src/geometry/{candidate_discovery,compiler,optimization,optimizer,persistence,strategy,validation}` owns semantic geometry; `src/layout` and physical projection/provider families own realization; `include/CellPack` remains the CellPack native contract | Keep CellPack ABI and `CellPack::*` implementation targets distinct from Cellerator compute, geometry discovery, physical layout, and execution binding | `src/geometry/CMakeLists.txt` and `src/geometry/owned_targets.cmake` declare CellPack targets including `CellPack::cellpack`, `CellPack::semantic_geometry`, and `CellPack::execution_image_v2`; compute, planner, execution, and provider targets consume them. Physical-layout integration and operation maturity remain target-specific. Measure preparation and conversion as part of operation cost. |
| Prepared execution and runtime | `src/execution/{atom_fragment,atom_plane,geometry_acquisition_v2,joint_compiler,native_value_instance,object_binding,prepared_host,program,projection_value_plane,training_program_v2}`, `src/runtime/{memory,multi_gpu}`, and `src/planner` | Prepared execution owns program/value/object binding and preparation; runtime owns resources and multi-device coordination; planner remains its own plan owner | `src/execution/CMakeLists.txt` links `CellPack::execution_image_v2`; route each family by its current headers, target registration, and callers. ExecutionField live recompilation remains experimental; do not treat preparation or gating as proof of adaptive recompilation. Torch should own Torch resources where possible. |
| Torch and Python adapters | `bindings/torch`, `bindings/python`, `python/cellerator`, `tests/bindings`, `tests/package` | Private Torch adapter plus implemented `cellerator` operations/modules | Compose Torch rather than duplicate its tensor, autograd, or optimizer systems. Current optional build/API limits are in the Python binding guide. The paused CelleraTorch proposal is archived in `docs/history/celleratorch.qmd`; it is not `python/cellerator/torch`. |
| Compiler and execution-field subsystems | `src/compiler/{api,ast,backend,composition,diagnostics,discovery,driver,frontend,ir,lto,pass,planning,profile,reflection,sema,tooling}` and matching public compiler headers | Preserve each named subsystem and native boundary. The compiler is an optional native subsystem; language sources and standard-library material remain their own research artifacts | The source map assigns the compiler paths to exact subsystem owners and names their CMake boundaries. The installed host SDK is deliberately narrow: `cmake/compiler/CelleratorHostSdk.cmake` builds `libCellerator` from the downstream compiler invocation and `celleratord` architecture validator, then links the `cellerator` and `celleratord` executables and installs public headers/resources. Those targets do not compile every file under `src/compiler`. Presence or target registration does not prove a full compiler pipeline. ExecutionField, JIT, source-to-device compilation, and live recompilation remain experimental unless a source-bound end-to-end qualification says otherwise. |
| Model fixtures and single-cell workflows | `examples`, `include/Cellerator/examples`, `docs/pipeline` | Examples, testbeds, or consumer-owned applications | Keep useful fixtures and operations. Keep scientific assumptions with Karyoplast, Cytoplast, GlassHelix, or the relevant consumer. |
| Performance, correctness, and research evidence | `bench/**`, qualification records, dated implementation docs | Source-bound evidence index and operation owner | Preserve original reports and rejected alternatives. Scope correctness, biological validation, and performance separately. Some code under an experiment path is linked into installed targets; audit targets and consumers before moving or archiving it. |
| CellPack, compatibility, and sibling interfaces | `include/CellPack`, `compat`, `components`, `modules`, `interop`, and actual sibling callsites | CellPack public contracts stay owned by the CellPack subsystem implemented through `src/geometry` targets; compatibility and sibling APIs retain their concrete native owner | `src/geometry/CMakeLists.txt` and `owned_targets.cmake` provide the implementation and target evidence. Preserve unique code, identifiers, ABI/schema versions, checkpoint behavior, and real sibling users. Do not infer an owner from a planned directory name. |
| Product, package, operation, and contributor guides | Root `README.md`, `scope.md`; `docs/{index,architecture,scope,getting_started,model_building,operations,layout_execution,current_implementation,developer_reference}.qmd`; `docs/development/{operation_registry,implementation_style,python-bindings,routing}.md` | Maintained product and developer documentation | Keep product direction concise; distinguish current API/evidence from proposed ownership and preserve the small, implemented `nn` surface. Former root guide pointers are preserved in `docs/history/repository-notes`; use the maintained operation registry and implementation style guide directly. |
| Experimental and historical design pages | `docs/experiments_and_history.qmd`, retained subsystem chapters, `docs/development/experimental-fields.md` | Exact subsystem owner and evidence record | Keep useful history accessible with currentness and maturity labels. Never rewrite immutable source-bound report bytes to fit a new product description. |
| Source map and agent routing | `docs/development/routing.md`, `tools/maintenance/source_map/`, generated `docs/development/source_map/` | Documentation/maintenance owner with source owners | Generated map gives per-path role, current owner, destination, disposition, maturity, consumers, tests/evidence, and archive reference. Regenerate after source or routing-rule changes; Project Control remains live task status. |
| Plans, Todo state, and repository metadata | `planning`, including `planning/archive/legacy-plans` and `planning/archive/legacy-todo`; current `.todo-orchestrator/project.json` and authority database; `.project-control` | Approved plan, supported Project Control authority, or preserved history | Plans record intent; Project Control owns live status. The [legacy Todo archive index](https://github.com/tumlinso/Cellerator/blob/main/planning/archive/legacy-todo/README.md) resolves archived projections, snapshots, plan inputs and runtime artifacts by former path and content identity. The live project metadata and authority database remain under `.todo-orchestrator`; never hand-edit them or any generated Todo projections to mimic completion. |

One specific preservation constraint: the reusable host patch and ragged-port
providers now live at `src/ops/matrix/patch_provider.cc` and
`src/ops/process/ports_provider.cc`, with explicit native targets and selectable
Substrate SDK components. The legacy Python proofs under
`experiments/moonshot-parallel-v1/diff/{patch,ports}/ops.py` still compile those
current provider files by path. Keep those proofs and provider contracts
connected until the wrappers have a deliberate replacement. The broader
experiment directories still contain separate candidate material; do not infer
its disposition from the extracted providers' new homes.

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

Local generated output has a different lifecycle from source history. Keep only
build trees with a current configured consumer in the checkout. Retired builds,
local source captures, test receipts and profiler output belong in one verified
durable archive with original-path and content manifests. Historical reports
retain their recorded build paths; recover their original bytes from the archive
instead of keeping every old build live. The configured `build/` verification
tree and `build-ctxpp/` compilation database remain local outputs; this repository
cleanup does not change shared retrieval state or its tooling configuration.

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

## C++ retrieval maintenance

`build-ctxpp/compile_commands.json` is the compilation database selected by `.ctxpp.toml`. The `.ctxpp/` index and `.ctxpp/observer-export.json` are generated local artifacts. Use the installed [C++ Context Compiler skill](https://github.com/tumlinso/skills/blob/main/cpp-context-compiler/SKILL.md) to scan and export after source-path, include-root, build-flag, or configuration changes. Preserve the previous generation until the replacement passes freshness and representative lookup checks; never hand-edit generated index state. After file renames or deletions, qualify a clean generation from validated TU caches and audit file/translation-unit references and diagnostic freshness. Incremental retention alone is insufficient evidence of current path coverage.

Project Control consumes the export through operator-managed `PROJECT_CONTROL_CPP_EXPORTS` configuration. Keep repository guidance declarations in `.project-context.json`. The [dated retrieval qualification](https://github.com/tumlinso/Cellerator/blob/main/planning/active-source-cleanup/qualification/retrieval-qualification.json) records the tested configuration and semantic versus lexical coverage. The source map supplies ownership and path routing; semantic coverage comes from the generated index and observer checks.
