# Cellerator compiler source layout v1

Interface: `CE-CCP1-I04-SOURCE-LAYOUT`

Version: `1`

Owner: `CE-CCP1-A04-010`

Consumed ownership contract: `CE-CCP1-I03-COMPILER-OWNERSHIP` version 1,
content hash `3478e9787fbee8e66f1c12dba0d69641d01605ef2316420f7d74cac02421b1d0`.

## Public and implementation roots

The installed compiler API is rooted at `include/Cellerator/compiler/`; its
implementation is rooted at `src/compiler/`. Public areas are mirrored by
driver, frontend/source/parser/cxx, ast, sema/field, ir/common/text/semantic/
planning/realization, profile, discovery, composition, program, planning,
reflection, pass, diagnostics, lto, backend/cpu/nvcc/nvptx, tooling, api, and
build-feature contracts. Private support remains below `src/compiler/`.

Executables are `tools/cellerator/main.cc` and `tools/celleratord/main.cc`, both
thin over shared libraries. Language sources live under `stdlib/cellerator/`;
portable fallback data lives under `profiles/reference/`; generated artifacts
live only below `${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/compiler/`.
Compiler validation and measurement use the corresponding `tests/compiler/`
and `bench/compiler/` owner families.

## Current-to-proposed ownership map

| Current responsibility | Proposed destination | Owner/boundary |
| --- | --- | --- |
| runtime identity and execution contracts | `include/Cellerator/execution/`, `include/Cellerator/runtime/`, and matching `src/` | remains Cellerator runtime; compiler consumes public contracts |
| biological geometry compiler | `include/Cellerator/geometry/compiler/` and `src/geometry/compiler/` | remains Cellerator geometry; distinct from source compiler root |
| kernels, candidates, and projections | `include/Cellerator/compute/` and `src/compute/` | remains Cellerator execution machinery |
| measured execution planning | `include/Cellerator/planner/` and `src/planner/` | remains runtime planner; source planning adapts it |
| JBC evidence and discovery | `include/Cellerator/compiler/profile/` and `src/compiler/profile/` | Cellerator compiler |
| JBC certification | `include/Cellerator/compiler/planning/certification/` and matching `src/` | Cellerator compiler |
| JBC atom semantic states | `include/Cellerator/compiler/ir/atom/` and matching `src/` | Cellerator compiler; resident atom instances remain runtime/application data |
| JBC composition and grammar | `include/Cellerator/compiler/planning/extensions/` and matching `src/` | Cellerator compiler |
| JBC basis | `include/Cellerator/compiler/planning/basis/` and matching `src/` | Cellerator compiler |
| JBC superatom composition | `include/Cellerator/compiler/planning/superatom/` and matching `src/` | Cellerator compiler |
| JBC partial algebra/legality versus bytes/recovery | `include/Cellerator/compiler/partial/` plus CellShard persistence | split: Cellerator semantics, CellShard storage/recovery |
| JBC graph compiler | `include/Cellerator/compiler/ir/semantic/` and `ir/planning/` | Cellerator compiler |
| JBC schedule compiler | `include/Cellerator/compiler/ir/planning/` and `ir/realization/` | Cellerator compiler |
| atom store, materialization, placement, residency, transport, leases, recovery, delivery | `components/CellShard/` | CellShard application/storage; no compiler decisions |
| legacy JBC compiler includes and tests | matching `include/Cellerator/compiler/`, `src/compiler/`, and `tests/compiler/` owner | temporary one-way adapters with named retirement proof |
| compiled ruleset export | immutable rules/profile/coverage/realization interface to CellShard | Part Two seam only; deep integration deferred |
| current all-in-one `include/Cellerator/Cellerator.hh` | `compiler.hh`, `runtime.hh`, and full convenience facade | central integration; current header remains until include fixtures pass |
| current `tools/`, `tests/`, and `bench/` | add compiler-specific subtrees | existing evidence stays in place |

CellShard owns storage/application mechanics; it cannot decide biological
semantics, grammar, profiles, certification, exact coverage, planning, or
realization meaning.

## Lane write scopes

- Leaf compiler providers write only their declared public header,
  implementation, focused test, benchmark/evidence, schema, resource, and
  documentation paths.
- Provider fragments live below the provider's exclusive subtree and carry
  stable identity, schema version, owning task, dependencies, and content hash.
- Migration tasks publish source-linked mapping/compatibility artifacts; the
  destination implementation task owns the moved/adapted source.
- Validation lanes may write only their declared tests and evidence.
- Integration lanes alone mutate singleton aggregators, registries, manifests,
  package exports, umbrella headers, normative interface documents, and the
  CellShard gitlink.
- Project Control claims, scopes, locks, barriers, interfaces, rendezvous, and
  integration queues remain authoritative over this descriptive map.

## Central locks

The following are integration-only: `.gitmodules`, `CMakeLists.txt`,
`src/CMakeLists.txt`, `tests/CMakeLists.txt`, `bench/CMakeLists.txt`,
`tools/CMakeLists.txt`, `cmake/compiler/CelleratorCompilerTargets.cmake`,
`cmake/package/CelleratorConfig.cmake.in`,
`cmake/package/CelleratorConfigVersion.cmake.in`,
`include/Cellerator/compiler.hh`, `include/Cellerator/Cellerator.hh`,
`stdlib/manifest.json`, canonical grammar/dialect/backend/pass registries,
generated compiler/resource manifests, and the `components/CellShard` gitlink.

## Dependency direction

Source and parser feed AST and Sema; Sema feeds Semantic IR; profiles enrich
Semantic IR; discovery/composition/program planning produce Planning IR;
planning and lowering produce Realization IR; backends consume Realization IR.
Reflection and passes operate only on their declared public IR stages. LTO owns
cross-TU/object companion work. Tooling and SDK APIs call shared compiler
libraries. Runtime/provider implementations do not flow back into frontend,
AST, Sema, or CEIR common contracts.

No public compiler header includes private `src/` paths. Common IR cannot depend
on concrete backends. CPU, NVCC, and NVPTX providers cannot include one another.
CellShard consumes immutable rules, profile identity, exact coverage, and
realization requirements only through versioned public interfaces.

## Component imports and generated files

Prefer narrow component includes. `compiler.hh` is host-safe, `runtime.hh`
provides the portable runtime surface, and `Cellerator.hh` composes them after
central integration. Generated parser tables, dialect manifests, backend
manifests, embedded resources, and feature/version headers are deterministic
build outputs from checked-in schemas and source fragments; they are not
authoritative checked-in source.

## Source-linked detail

- [Public header tree](freeze_the_public_compiler_header_tree.md)
- [Implementation tree](freeze_the_compiler_implementation_tree.md)
- [Executable locations](freeze_compiler_executable_locations.md)
- [Standard library and resources](freeze_standard_library_and_resource_locations.md)
- [Tests and benchmarks](freeze_compiler_test_and_benchmark_locations.md)
- [Central registries](define_central_registry_and_generated_manifest_ownership.md)
- [Umbrella split](split_umbrella_headers_and_public_component_imports.md)
- [Generated boundary](define_generated_source_and_build_tree_boundaries.md)
- [AGENTS inheritance](define_agents_and_ownership_inheritance.md)

## Compatibility and Part One boundary

This interface adds a destination map; it does not claim absent compiler paths
already exist. Existing runtime and JBC behavior remains available until each
versioned replacement and adapter retirement proof passes. There is no
permanent Clang fork, general JIT, or deep CellShard runtime integration in Part
One.
