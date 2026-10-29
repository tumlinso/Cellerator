# Compiler AGENTS and ownership inheritance

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-009`

This plan specifies future directory-local `AGENTS.md` contracts. This task
does not create those files or change implementation source.

## Inheritance model

The repository-root `AGENTS.md` remains the operational authority for every
compiler path. A nearer `AGENTS.md` adds component-specific invariants, required
reading, validation, and ownership guidance. It may be more restrictive; it
must not weaken Project Control authority, a live Todo scope, architecture
invariants, central-integration ownership, or the root build/test rules.

Task scope and Project Control locks are authoritative for mutations. Directory
guidance explains how to implement inside granted scope; it never grants access
to a path or allows an agent to claim adjacent work.

## Planned local guidance

| Guidance file | Applies to | Required component instructions |
| --- | --- | --- |
| `include/Cellerator/compiler/AGENTS.md` | installed compiler contracts | stable public ABI/schema, ordinary-data boundaries, no raw Clang/LLVM objects, host-safe includes, versioned compatibility |
| `src/compiler/AGENTS.md` | shared compiler implementation | mirror public owners, no executable-owned logic, C++23 compiler baseline with explicit C++17/CUDA17 islands, deterministic diagnostics and generation |
| `src/compiler/ir/AGENTS.md` | CEIR common/text/semantic/planning/realization | public writable IR, stable identities, level boundaries, verification before/after passes, no backend ownership in common IR |
| `src/compiler/backend/AGENTS.md` | backend ABI and providers | backend-agnostic IR, explicit capability/cost, CPU/NVCC/NVPTX isolation, no NVCC source parsing, accelerator dependencies remain provider-local |
| `src/compiler/tooling/AGENTS.md` | driver services and celleratord implementation | shared-library ownership, host-only protocol/queries, bounded incremental state, no compiler logic in executable mains |
| `docs/compiler/migration/AGENTS.md` | JBC provenance and migration receipts | preserve history and reusable code, source-link every disposition, no deletion before replacement proof, CellShard remains consumer not compiler authority |

The compiler IR guidance inherits both repository rules and the shared compiler
implementation rules. Provider-specific backend guidance may be added below
`src/compiler/backend/<provider>/`, but cannot change the public backend ABI or
another provider. Migration guidance governs receipts and compatibility mapping,
not the destination implementation once code moves into its compiler owner.

## Required reading by area

Every local file points back to `AGENTS.md`, `scope.md`,
`docs/architecture.qmd`, `docs/current_implementation.qmd`, and
`docs/migration_roadmap.qmd`. Compiler work additionally reads the language
specification and programming guide. IR, backend, tooling, and migration files
list their nearest public contracts, relevant tests, and benchmark/evidence
requirements rather than duplicating the architecture spine.

## Central-integration exclusions

None of the planned local guidance grants leaf ownership of:

```text
.gitmodules
CMakeLists.txt
src/CMakeLists.txt
tests/CMakeLists.txt
bench/CMakeLists.txt
tools/CMakeLists.txt
cmake/compiler/CelleratorCompilerTargets.cmake
cmake/package/CelleratorConfig.cmake.in
cmake/package/CelleratorConfigVersion.cmake.in
include/Cellerator/compiler.hh
include/Cellerator/Cellerator.hh
stdlib/manifest.json
canonical grammar, dialect, backend, or pass registries
components/CellShard gitlink
```

Those singleton surfaces remain integration-lane owned. Local instructions tell
providers to publish isolated fragments and request integration through Project
Control.

## Compatibility and deferred work

The current repository has root guidance and component guidance for CellPack
under `src/geometry/AGENTS.md`; those contracts remain in force. The compiler
guidance will complement them without reassigning geometry ownership. No
current runtime or JBC behavior changes, and no Part Two JIT or deep CellShard
runtime work is authorized.

## Validation evidence

`tests/compiler/a04/define_agents_and_ownership_inheritance_test.cc` checks the
inheritance model, every planned local guidance path and required instruction,
the complete central-integration exclusion set, existing guidance continuity,
and the explicit no-source-change boundary.
