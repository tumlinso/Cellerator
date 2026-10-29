# Cellerator contributor routing

This guide routes work to its source owner and records how to make and validate
the source map. It is durable contributor guidance, not a task tracker. Use the
registered Project Control workflow for current task status, ownership claims,
and completion evidence.

## Product boundary

Cellerator is a high-performance PyTorch extension for cellular model building.
PyTorch supplies general tensors, ordinary operations, `torch.nn.Module`,
autograd and optimizers. Cellerator adds operations and execution semantics
that account for structure shared by the biological dynamical system being
modeled. That structure may improve interpretability and performance, but each
performance claim must be measured for the full relevant operation and workload.
Cellerator enables scientific models; it does not supply their scientific
assumptions.

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
ordinary model use. Classical single-cell analysis remains a useful upstream
interface, consumer and testbed. AnnData is a required Python dependency, but
a direct AnnData preparation adapter does not currently exist.

## Source routing

The [generated source map](source_map/path_map.csv) assigns every indexed path
a current role, owner, target destination, disposition, maturity, consumer and
evidence pointer. Shared metadata lives in
[`routing_rules.json`](source_map/routing_rules.json); per-path Git and worktree
identity plus archive references live in the path map. The map is a navigation
and disposition aid, not proof of implementation, API stability, scientific
validity, or task completion.

| Change | Current source route | Intended owner | Routing notes |
|---|---|---|---|
| Native mathematical or model operation | `src/ops/{numeric,matrix,process,sparse_support}`, older `src/compute` and `src/math`, matching `include/Cellerator/ops`, compute, and math headers | Native `ops/<family>`; public Python/Torch surface only where implemented | `src/ops/numeric`, `matrix`, `process` have explicit CMake owners and actual Substrate SDK consumers; sparse-support operations are separately CUDA/feature gated. Keep each operation's derivatives, precision, build target, and callers together. |
| Identity, support, indexing, and gating | `src/structure`, `src/support`, `src/indexing`, `src/gating`; public identity headers under `include/Cellerator/structure` and support/indexing/gating under `include/Cellerator/compute` | Distinct structure identity, logical eligibility, lookup, and conditional-work owners | CMake currently defines `Cellerator::identity_registry`, `::support`, `::indexing`, and `::gating`, with a compatibility `::sparse_support` umbrella. Host checks include `tests/biological_abi/biological_abi_host_test.cc`, `tests/execution_identity/identity_registry_test.cc`, and focused `tests/sparse_support/*`; CUDA workset checks are separately gated. |
| Prepared model routing | `include/Cellerator/ops/sparse_support/prepared_routing.hh`, `src/ops/sparse_support/prepared_routing.{cc,cu}`, its sparse-support Torch binding and Python adapter | Opt-in prepared entity-to-mechanism route query; current Python query remains the default | Frozen binary passed 16 tests, exact CUDA correctness, and memcheck/racecheck/synccheck. V100 lifecycle timing found a bounded long-reuse win for ranked `[32,8]`, an inconclusive event shape, and dense Torch wins on larger stress shapes. Preserve caller-supplied route and binding generations; see the [qualification report](https://github.com/tumlinso/Cellerator/blob/94c534fc794fe8cc3e0434e4a50f379401ddbb3c/bench/sparse_support/experiments/prepared_routing/REPORT.md). |
| Geometry, physical layouts, values and gradients | `src/geometry` remains semantic organization; `src/layout` plus current projection headers are physical owners; legacy `src/packing` and `src/quantized` remain separately routed | Geometry discovery versus layout realization and logical/physical maps | `src/layout/CMakeLists.txt` currently lists the MMA hybrid builder and cover validation sources; other source files under `src/layout` need exact build/caller disposition before being described as integrated. Measure preparation and conversion as part of operation cost. |
| Prepared execution and runtime | `src/execution`, `src/runtime`, `src/planner` | `execution` preparation/reuse and `runtime` resource ownership | Map stream, allocation, executable lifetime, dependency guards, and actual consumers. Torch should own Torch resources where possible. |
| Torch and Python adapters | `bindings/torch`, `bindings/python`, `python/cellerator`, `tests/bindings`, `tests/package` | Private Torch adapter plus implemented `cellerator` operations/modules | Compose Torch rather than duplicate its tensor, autograd, or optimizer systems. Current optional build/API limits are in the Python binding guide. `CelleraTorch/` is a separate documentation experiment, not `python/cellerator/torch`. |
| Compiler, language, and execution-field experiments | `src/compiler`, `include/Cellerator/compiler`, `library`, `stdlib`, `docs/language`, source-bound compiler experiments | Optional experimental compiler/field route | These trees have mixed maturity. Do not claim full source-to-device compilation or live recompilation without a current end-to-end implementation and evidence. |
| Model fixtures and single-cell workflows | `examples`, `include/Cellerator/examples`, `docs/pipeline` | Examples, testbeds, or consumer-owned applications | Keep useful fixtures and operations. Keep scientific assumptions with Karyoplast, Cytoplast, GlassHelix, or the relevant consumer. |
| Performance, correctness, and research evidence | `bench/**`, qualification records, dated implementation docs | Source-bound evidence index and operation owner | Preserve original reports and rejected alternatives. Scope correctness, biological validation, and performance separately. Some code under an experiment path is linked into installed targets; audit targets and consumers before moving or archiving it. |
| CellPack, compatibility, and sibling interfaces | `include/CellPack`, `compat`, `components`, `modules`, `interop`, and actual sibling callsites | Existing native/compatibility owner until consumer map supports a move | Preserve unique code, identifiers, ABI/schema versions, checkpoint behavior, and real sibling users. Do not infer an owner from a planned directory name. |
| Product, package, operation, and contributor guides | Root `README.md`, `scope.md`; `docs/{index,architecture,scope,getting_started,model_building,operations,layout_execution,current_implementation,developer_reference}.qmd`; Python guide | Cellerator documentation spine, updated with source owners | Keep product direction concise; distinguish current API/evidence from proposed ownership and preserve the small, implemented `nn` surface. |
| Experimental and historical design pages | `docs/experiments_and_history.qmd`, retained subsystem chapters, `docs/development/experimental-fields.md` | Exact subsystem owner and evidence record | Keep useful history accessible with currentness and maturity labels. Never rewrite immutable source-bound report bytes to fit a new product description. |
| Source map and agent routing | `docs/development/routing.md`, `tools/maintenance/source_map/`, generated `docs/development/source_map/` | Documentation/maintenance owner with source owners | Generated map gives per-path role, current owner, destination, disposition, maturity, consumers, tests/evidence, and archive reference. Regenerate after moves settle; Project Control remains live task status. |
| Plans, Todo state, and repository metadata | `planning`, `todos`, `.todo-orchestrator`, `.project-control`, root plan snapshots | Approved plan, supported Project Control authority, or preserved history | Plans record intent; Project Control owns live status. Never hand-edit SQLite, generated Todo projections, recovery snapshots, or task ledgers to mimic completion. |

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

## Historical source paths

The [source migration record](https://github.com/tumlinso/Cellerator/blob/9665fcaf34f5e1a1725c78bc7c352e99e72957fc/planning/coherence/source-migrations.json) maps preserved original blobs to current native, header, test, and tool destinations. Use it when a source-bound report or older plan names a translation unit that has moved. Compatibility facades and extracted experiment implementations retain explicit provenance.

For a one-time closure check of the record's declared existing destinations,
run `python tools/maintenance/source_map/validate.py --repo . --migration-manifest planning/coherence/source-migrations.json`.
This opt-in audit checks that each declared destination is tracked; it does not
compare future file contents or make this historical record a permanent path
policy. Ordinary source-map validation continues to check the current Git
index, worktree, routes, and generated outputs without consulting migration
history.

A required native evidence name needs a configured producer that records that passing fact. Committed reports, scheduler receipts, and external test runs belong in source references and reviewed assertions; their presence alone does not create a native passing fact. Live task state remains in Project Control.
