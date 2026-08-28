# Repository Consolidation Program

This document is an execution map for the physical repository remap. It is not
an architecture document. The authoritative semantic baseline remains:

- `scope.md`;
- `docs/architecture.qmd`;
- `docs/current_implementation.qmd`;
- `docs/migration_roadmap.qmd`;
- the frozen CE-ARCH and CE-LIVE decisions and interfaces;
- the implementation at baseline commit
  `e02054fa103551c27f6a41907c49c6d923ce5777`.

The transactional task graph is `cellerator-remap-plan.json` (`CE-REMAP`).
Physical moves must follow that graph and must not reinterpret the architecture.

> **Completion status (2026-08-28):** CE-REMAP completed successfully. The
> observed paths below are a historical pre-remap baseline, not current source
> ownership. Current work belongs in the canonical geometry, runtime, compute,
> planner, preprocess, interop, examples, components, and compat trees described
> by `AGENTS.md` and `docs/current_implementation.qmd`.

## Observed baseline

The baseline was observed on 2026-08-28 at todo revision 2256 before the remap
plan was applied.

- branch: `main`, matching `origin/main`;
- worktree: clean;
- compiler: GNU C++ 13.3.0;
- CUDA compiler: NVIDIA 12.9.86 from the HPC SDK;
- configured architecture: Tesla V100-SXM2-16GB, `sm_70`;
- default configure: successful with native Cellerator, Baseplane, and CellShard;
- default full build: pre-existing failure at 47 percent in the CellShard runtime
  consumer of `include/Cellerator/dist/nccl_communicator.cuh`;
- CE-LIVE: complete at todo revision 2256 with no active claim or ready task;
- historical invalidated CE-ARCH gates: retained as historical audit state, not
  repaired or treated as current architectural failures.

The build failure is caused by the legacy distributed header split:
`nccl_communicator.cuh` defines helpers over `local_context` before
`distributed.cuh` defines that type. It is visible when NCCL is enabled through
the sibling CellShard runtime. The remap does not treat the failure as a reason
to redesign collectives. `CE-REMAP-06` owns the physical consolidation that
must eliminate the duplicate runtime surface while preserving behavior.

Machine-readable baseline evidence is under `bench/repository_remap/baseline/`.

## Authoritative and historical strata

Authoritative current implementation:

- `include/Cellerator/execution/` and `src/execution/`;
- `include/Cellerator/runtime/` and `src/runtime/`;
- the CE-ARCH operation core and preparation factory currently nested under
  `compute/math/operation_core/`;
- current physical projections, native/vendor candidates, native training,
  CPE2 activation, and the planner;
- CelleraTorch's thin native adapter under `components/CelleraTorch/`.

Validated code whose physical ownership is transitional:

- CellPack and CP-BP under `components/CellPack/` plus related packing discovery
  under `src/compute/packing/`;
- runtime helpers under `src/compute/runtime/`;
- hierarchy under `src/distributed/`;
- Baseplane integration under `compute/sequence/`;
- preprocessing under both `compute/preprocess/` and `preprocess/`;
- model and trajectory orchestration under current core-looking paths.

Compatibility or evidence code:

- the retained CP-Math v1 request, structural planner, prepared lifetime, and
  private runtime;
- historical sparse API facades and model/workflow assumptions;
- CPK1 compatibility, which remains a supported input to CPE2 rather than an
  alternate current architecture.

## Observed build ownership

The root `CMakeLists.txt` is 2,306 lines and defines most native libraries,
tests, benchmarks, adapters, and historical targets. CellPack and CelleraTorch
have component-local target definitions, but later CellPack/CE-ARCH targets are
again defined at the root. The remap must move target definitions with source
ownership while preserving public target aliases until their consumers migrate.

The default build is native-only with Torch disabled. CelleraTorch remains
available through `CELLERATOR_ENABLE_TORCH_MODELS=ON` and must not become a
dependency of native targets.

## Mechanical hygiene baseline

The canonical current trees already use `.hh`, `.cc`, `.cu`, and `.cuh`; the
historical extension problem is mostly in sibling dependency sources and paths
that have not yet been moved. New canonical paths will enforce the requested
Unix vocabulary as files enter their final home.

The following public headers currently expose private `src/` headers and are
explicit migration targets:

- `include/Cellerator/compute/runtime.hh`;
- `include/Cellerator/compute/sparse/ops.hh`;
- `include/Cellerator/compute/sparse/project.hh`;
- `include/Cellerator/compute/neighbors/exact_search.hh`;
- `include/Cellerator/models/developmental_time.hh`;
- `include/Cellerator/models/developmental_time_trajectory.hh`;
- `include/Cellerator/models/state_reduce.hh`;
- `include/Cellerator/trajectory/trajectory_tree.cuh`.

The following tests or benchmarks include implementation translation units and
must be converted to linked implementation targets:

- `tests/execution/ce_live_concurrency_test.cu`;
- `tests/math_core/native_training_slice_test.cu`;
- `components/CelleraTorch/tests/quantitative_smoke_test.cc`;
- `bench/architecture_evidence/real_regime_bench.cu`.

Test-only shared fixtures currently included as `.cc` files must become named
fixture libraries or proper fixture headers; they are not exceptions to the
translation-unit rule.

## Cross-project boundaries

Project-control independently observed clean registered workspaces for
Baseplane, CellShard, and GlassHelix. Those observations are not an atomic
cross-repository transaction and are rechecked at the interop phase.

- Baseplane frozen predicate/event contracts are consumed without moving its
  sequence engine into Cellerator.
- CellShard frozen storage, envelope, residency, and delivery contracts remain
  external. Cellerator owns the opaque inner execution image semantics.
- GlassHelix has no frozen Cellerator interface. The remap records a deliberate
  seam but defers `cellerator.glasshelix` rather than inventing scientific
  vocabulary.
- CelleraTorch remains the sole major optional component after geometry moves.

## Phase stop points

1. `CE-REMAP-01`: the program, dependency map, and honest baseline exist; no
   source has moved.
2. `CE-REMAP-02`: canonical directories, local build ownership, layout checks,
   and one host-only module proof work; CUDA has no module dependency.
3. `CE-REMAP-03`: only stable state and execution vocabulary is exported.
4. `CE-REMAP-04`: geometry is the one physical home for CellPack and CP-BP;
   compatibility names do not imply component ownership.
5. `CE-REMAP-05`: modern compute and historical CP-Math/legacy sparse evidence
   are physically distinct; no current target depends on evidence by accident.
6. `CE-REMAP-06`: one runtime owns resources and collectives; distributed
   planning owns hierarchy and communication cost.
7. `CE-REMAP-07`: Baseplane and CellShard seams are explicit; GlassHelix is
   deliberately deferred unless science authority changes.
8. `CE-REMAP-08`: preprocessing has one home and higher-level orchestration is
   visibly example/tool code.
9. `CE-REMAP-09`: obsolete facades and empty historical directories are gone;
   subsystem CMake files own their targets.
10. `CE-REMAP-10`: clean native and compatibility builds, CE-LIVE regression,
    module consumers, layout scans, and cross-project boundary audits pass.

At every stop point, behavior and numerical contracts outrank physical
symmetry. A discovered semantic defect that cannot be fixed as a behavior-
preserving move is recorded as follow-up work rather than folded into this
program.

The module infrastructure probe proved direct Clang 18 precompile and import,
but CMake 3.28 rejected native scanning with Unix Makefiles and the host has
neither Ninja nor `clang-scan-deps`. The native module surfaces are therefore
deferred, exactly as the module stop rule requires. The optional direct compiler
probe leaves the GNU/NVCC pairing untouched and gives no CUDA target a module
import or BMI dependency. The initial layout audit fails new violations
immediately and tracks the observed public/private and implementation-include debts in
`cmake/repository_layout_allowlist.json`; each owning phase must shrink that
allowlist as it removes the corresponding debt.

The foundational border audit is recorded in `modules/BOUNDARIES.md`. Existing
identity, operand, lifetime, order, and launch-binding headers are suitable
future state/execution vocabulary. The current executable-program header is not
a narrow cross-project module surface because it intentionally names internal
operation-core, planner, projection, runtime-session, and readiness types.
CE-REMAP preserves that authority and does not invent an opaque facade merely
to make a module symmetrical.

Geometry consolidation preserves the existing `cellpack` namespace and
`CellPack::` CMake aliases while moving their implementation ownership to
`include/Cellerator/geometry/`, `src/geometry/`, `tests/geometry/`, and
`bench/geometry/`. Temporary headers under `include/CellPack/` contain only
forwarding includes to the canonical declarations. Related gene-candidate
discovery now lives under `src/geometry/candidate_discovery/`; its old public
compute headers and one private test include remain explicit forwarding debt
until consumer-owning phases cut them over.

Compute consolidation gives the CE-ARCH operation core, physical projections,
execution candidates, reusable sparse operators, and native training distinct
homes under `include/Cellerator/compute/` and `src/compute/`. The retained
first-generation request, BELL, packed-dense, and referee implementation is
physically isolated under `compat/cp_math_v1`; historical include paths forward
to either that evidence or the promoted current contract as appropriate.

Runtime consolidation retains the CE-ARCH session as the sole runtime
authority. Local device and NCCL resources are physically owned by
`runtime/multi_gpu`; the older sparse execution-context implementation is a
named runtime compatibility resource rather than a second conceptual runtime.
Hierarchy and communication policy are owned by `planner/distributed`.

The frozen Baseplane predicate/event consumer and the Cellerator-owned
CellShard matrix access adapters now live under `interop/baseplane` and
`interop/cellshard`. Historical paths forward. Host module interfaces remain
deferred with the documented toolchain proof limitation, and no GlassHelix
vocabulary is invented before its scientific seam is frozen.

Preprocessing kernels and runtime surfaces now have one physical home under
`Cellerator/preprocess` and `src/preprocess`; the ncurses application lives in
`tools/preprocess_workbench`. Model and trajectory orchestration is retained as
example/regression workload code, with public compatibility headers forwarding
to explicitly example-owned interfaces.

Compatibility demolition leaves forwarding headers only as external migration
surfaces: current implementation, tests, components, examples, tools, and
benchmarks include canonical or explicitly compatibility-owned paths directly.
The public/private and translation-unit include allowlists are empty. Root
CMake is configuration-oriented and delegates the geometry, compute, planner,
and compatibility target inventories to owning subsystem files. The one
historical preprocessing format benchmark that depends on CellShard private
Blocked-ELL conversion machinery is quarantined under `compat/legacy_sparse`
and named `celleratorLegacyPreprocessFormatCompareBench`; no current Cellerator
target consumes that private seam. The default build and that evidence target
both compile after the remap.

Final validation is recorded in
`bench/repository_remap/final/validation.json`. Fresh native and Torch-enabled
build trees compile completely on CUDA 12.9/Volta; focused modern, geometry,
Baseplane, CellShard, CE-LIVE, and CelleraTorch tests pass. Compute Sanitizer
memcheck reports zero errors for quantitative forward, native training, program
replay, and concurrency. Project-control's final independent cross-workspace
observation reports no contradictory or duplicated ownership across the
Baseplane, CellShard, and GlassHelix seams. Native CMake module scanning,
general install/export validation, and stale standalone CE-LIVE script build
path discovery remain explicitly bounded tooling follow-ups, not architectural
completion claims.
