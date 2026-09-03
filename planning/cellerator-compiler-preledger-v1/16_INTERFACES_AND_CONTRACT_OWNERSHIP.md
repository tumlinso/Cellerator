# Interfaces and contract ownership

Each shared compiler interface has exactly one proposed owner task. Interfaces begin in `draft` when the apply-ready plan is applied and are frozen only by their owner checkpoint after validation.

| Interface | Name | Owner task | Direct consumers | Contract paths |
| --- | --- | --- | --- | --- |
| `CE-CCP1-I01-AUTHORITY-BASELINE` | Compiler authority baseline | `CE-CCP1-A01-009` | `CE-CCP1-A02-012` | `planning/cellerator-compiler-preledger-v1/evidence/live_snapshot.json`, `docs/compiler/architecture/PART_ONE_CHARTER.md` |
| `CE-CCP1-I02-JBC-MIGRATION-MANIFEST` | JBC migration manifest | `CE-CCP1-A02-012` | `CE-CCP1-A03-014`, `CE-CCP1-E02-018` | `docs/compiler/migration/JBC_MIGRATION_MANIFEST.md`, `docs/compiler/migration/jbc_migration_manifest.json` |
| `CE-CCP1-I03-COMPILER-OWNERSHIP` | Cellerator compiler ownership contract | `CE-CCP1-A03-014` | `CE-CCP1-A04-010` | `docs/compiler/architecture/CELLERATOR_CELLSHARD_COMPILER_SPLIT.md` |
| `CE-CCP1-I04-SOURCE-LAYOUT` | Compiler source layout | `CE-CCP1-A04-010` | `CE-CCP1-B01-012` | `docs/compiler/source-layout/SOURCE_LAYOUT_V1.md` |
| `CE-CCP1-I05-BUILD-GRAPH` | Compiler build graph | `CE-CCP1-B01-012` | `CE-CCP1-B02-014`, `CE-CCP1-B03-015`, `CE-CCP1-D01-014` | `cmake/compiler/CelleratorCompilerTargets.cmake`, `include/Cellerator/compiler/build/features_v1.hh` |
| `CE-CCP1-I06-DRIVER` | Cellerator driver | `CE-CCP1-B02-014` | `CE-CCP1-B04-014`, `CE-CCP1-F02-014` | `include/Cellerator/compiler/driver/driver_v1.hh`, `include/Cellerator/compiler/driver/toolchain_v1.hh` |
| `CE-CCP1-I07-SOURCE-MAP` | Unified source map | `CE-CCP1-B03-015` | `CE-CCP1-B04-014`, `CE-CCP1-C02-012`, `CE-CCP1-I01-014` | `include/Cellerator/compiler/frontend/source/source_map_v1.hh` |
| `CE-CCP1-I08-PRAGMA-DIALECT` | File-local Cellerator dialect activation | `CE-CCP1-B03-015` | `CE-CCP1-B04-014`, `CE-CCP1-C01-016` | `include/Cellerator/compiler/frontend/source/dialect_state_v1.hh` |
| `CE-CCP1-I09-CXX-SEMA-BRIDGE` | Upstream C++ semantic bridge | `CE-CCP1-B04-014` | `CE-CCP1-C02-012`, `CE-CCP1-C03-016`, `CE-CCP1-I01-014` | `include/Cellerator/compiler/frontend/cxx/cxx_sema_bridge_v1.hh` |
| `CE-CCP1-I10-PARSER` | Cellerator parser | `CE-CCP1-C01-016` | `CE-CCP1-C02-012` | `include/Cellerator/compiler/frontend/parser/parser_v1.hh`, `include/Cellerator/compiler/frontend/parser/token_kind_v1.hh` |
| `CE-CCP1-I11-AST` | Cellerator AST | `CE-CCP1-C02-012` | `CE-CCP1-C03-016`, `CE-CCP1-G03-016`, `CE-CCP1-I01-014` | `include/Cellerator/compiler/ast/ast_v1.hh`, `include/Cellerator/compiler/ast/diagnostic_v1.hh` |
| `CE-CCP1-I12-BIOLOGICAL-SEMA` | Biological semantic analysis | `CE-CCP1-C03-016` | `CE-CCP1-C04-016` | `include/Cellerator/compiler/sema/semantic_types_v1.hh`, `include/Cellerator/compiler/sema/operation_resolution_v1.hh` |
| `CE-CCP1-I13-FIELD-SEMANTICS` | Execution field and control semantics | `CE-CCP1-C04-016` | `CE-CCP1-D02-016` | `include/Cellerator/compiler/sema/field/field_semantics_v1.hh` |
| `CE-CCP1-I14-CEIR-COMMON` | CEIR common object model | `CE-CCP1-D01-014` | `CE-CCP1-D02-016`, `CE-CCP1-F01-018` | `include/Cellerator/compiler/ir/common/ir_v1.hh` |
| `CE-CCP1-I15-CEIR-TEXT` | CEIR textual and binary artifact | `CE-CCP1-D01-014` | `CE-CCP1-D02-016`, `CE-CCP1-J01-012` | `include/Cellerator/compiler/ir/text/ceir_text_v1.hh`, `include/Cellerator/compiler/ir/text/ceir_artifact_v1.hh` |
| `CE-CCP1-I16-SEMANTIC-IR` | Semantic IR | `CE-CCP1-D02-016` | `CE-CCP1-D03-015`, `CE-CCP1-E01-016`, `CE-CCP1-G01-016`, `CE-CCP1-H01-016`, `CE-CCP1-H02-016` | `include/Cellerator/compiler/ir/semantic/semantic_ir_v1.hh` |
| `CE-CCP1-I17-PROFILE-ARTIFACT` | Representative profile artifact | `CE-CCP1-D03-015` | `CE-CCP1-H03-018`, `CE-CCP1-J01-012` | `include/Cellerator/compiler/profile/profile_artifact_v1.hh` |
| `CE-CCP1-I18-PROFILE-ENVIRONMENT` | Profile analysis environment | `CE-CCP1-D03-015` | `CE-CCP1-E01-016`, `CE-CCP1-E02-018`, `CE-CCP1-I02-014` | `include/Cellerator/compiler/profile/profile_environment_v1.hh` |
| `CE-CCP1-I19-PLANNING-IR` | Planning IR | `CE-CCP1-E01-016` | `CE-CCP1-E02-018`, `CE-CCP1-E03-018`, `CE-CCP1-E04-018`, `CE-CCP1-G01-016`, `CE-CCP1-G03-016`, `CE-CCP1-I02-014` | `include/Cellerator/compiler/ir/planning/planning_ir_v1.hh` |
| `CE-CCP1-I20-DISCOVERY-ATOM` | Cellerator discovery and atom compiler | `CE-CCP1-E02-018` | `CE-CCP1-E03-018`, `CE-CCP1-E04-018` | `include/Cellerator/compiler/discovery/discovery_v1.hh`, `include/Cellerator/compiler/discovery/atom_v1.hh` |
| `CE-CCP1-I21-COMPOSITION-BASIS` | Composition grammar and basis | `CE-CCP1-E03-018` | `CE-CCP1-E04-018` | `include/Cellerator/compiler/composition/grammar_v1.hh`, `include/Cellerator/compiler/composition/basis_v1.hh` |
| `CE-CCP1-I22-PROGRAM-RULESET` | Portable Cellerator program ruleset | `CE-CCP1-E03-018` | `CE-CCP1-E04-018` | `include/Cellerator/compiler/program/ruleset_v1.hh` |
| `CE-CCP1-I23-PLANNER` | Public planning compiler | `CE-CCP1-E04-018` | `CE-CCP1-F01-018`, `CE-CCP1-G02-018`, `CE-CCP1-H01-016`, `CE-CCP1-H02-016` | `include/Cellerator/compiler/planning/planner_v1.hh` |
| `CE-CCP1-I24-REALIZATION-IR` | Realization IR | `CE-CCP1-F01-018` | `CE-CCP1-F02-014`, `CE-CCP1-F03-015`, `CE-CCP1-F04-013`, `CE-CCP1-G01-016`, `CE-CCP1-G02-018`, `CE-CCP1-G03-016`, `CE-CCP1-H01-016`, `CE-CCP1-H02-016`, `CE-CCP1-I02-014` | `include/Cellerator/compiler/ir/realization/realization_ir_v1.hh` |
| `CE-CCP1-I25-BACKEND-ABI` | Backend provider ABI | `CE-CCP1-F02-014` | `CE-CCP1-F03-015`, `CE-CCP1-F04-013`, `CE-CCP1-H01-016` | `include/Cellerator/compiler/backend/backend_v1.hh` |
| `CE-CCP1-I26-CPU-BACKEND` | CPU/native C++ backend | `CE-CCP1-F02-014` | `CE-CCP1-J02-014` | `include/Cellerator/compiler/backend/cpu/cpu_backend_v1.hh` |
| `CE-CCP1-I27-NVCC-BACKEND` | NVCC backend | `CE-CCP1-F03-015` | `CE-CCP1-J02-014` | `include/Cellerator/compiler/backend/nvcc/nvcc_backend_v1.hh` |
| `CE-CCP1-I28-NVPTX-BACKEND` | Clang CUDA and direct NVPTX backend | `CE-CCP1-F04-013` | none | `include/Cellerator/compiler/backend/nvptx/nvptx_backend_v1.hh` |
| `CE-CCP1-I29-REFLECTION` | Cellerator compiler reflection | `CE-CCP1-G01-016` | `CE-CCP1-G02-018` | `include/Cellerator/compiler/reflection/reflection_v1.hh` |
| `CE-CCP1-I30-PASS-EXTENSION` | Open pass and extension API | `CE-CCP1-G02-018` | `CE-CCP1-H02-016` | `include/Cellerator/compiler/pass/pass_v1.hh`, `include/Cellerator/compiler/pass/extension_v1.hh` |
| `CE-CCP1-I31-SELF-TRANSFORM` | Same-compilation transform staging | `CE-CCP1-G02-018` | none | `include/Cellerator/compiler/pass/self_transform_v1.hh` |
| `CE-CCP1-I32-DIAGNOSTICS-PROVENANCE` | Diagnostics and provenance | `CE-CCP1-G03-016` | `CE-CCP1-H02-016`, `CE-CCP1-I02-014`, `CE-CCP1-J01-012`, `CE-CCP1-J03-013` | `include/Cellerator/compiler/diagnostics/diagnostics_v1.hh`, `include/Cellerator/compiler/diagnostics/provenance_v1.hh` |
| `CE-CCP1-I33-OBJECT-CEIR` | CEIR object companion | `CE-CCP1-H01-016` | none | `include/Cellerator/compiler/lto/object_ceir_v1.hh` |
| `CE-CCP1-I34-CELLERATOR-LTO` | Cellerator program and LTO compiler | `CE-CCP1-H01-016` | `CE-CCP1-J02-014`, `CE-CCP1-J03-013` | `include/Cellerator/compiler/lto/lto_v1.hh` |
| `CE-CCP1-I35-LIBCELLERATOR-COMPILER` | libCellerator compiler API | `CE-CCP1-H02-016` | `CE-CCP1-H03-018`, `CE-CCP1-I01-014` | `include/Cellerator/compiler/api/cellerator_compiler.h`, `include/Cellerator/compiler/api/compiler.hpp` |
| `CE-CCP1-I36-LIBCELLERATOR-RUNTIME` | libCellerator runtime SDK facade | `CE-CCP1-H02-016` | `CE-CCP1-H03-018` | `include/Cellerator/sdk/runtime.hpp` |
| `CE-CCP1-I37-STDLIB-SDK` | Cellerator standard-library resource | `CE-CCP1-H03-018` | none | `stdlib/manifest.json`, `stdlib/cellerator/core.cell` |
| `CE-CCP1-I38-PACKAGE` | Installable Cellerator package | `CE-CCP1-H03-018` | `CE-CCP1-J03-013` | `cmake/package/CelleratorConfig.cmake.in`, `cmake/package/CelleratorConfigVersion.cmake.in` |
| `CE-CCP1-I39-CELLERATORD-CORE` | celleratord core | `CE-CCP1-I01-014` | `CE-CCP1-I02-014` | `include/Cellerator/compiler/tooling/language_server_v1.hh` |
| `CE-CCP1-I40-CELLERATORD-SEMANTIC` | celleratord Cellerator semantics | `CE-CCP1-I02-014` | `CE-CCP1-J02-014`, `CE-CCP1-J03-013` | `include/Cellerator/compiler/tooling/cellerator_queries_v1.hh` |
| `CE-CCP1-I41-PART1-COMPLETE` | Cellerator compiler Part One completion | `CE-CCP1-J03-013` | none | `docs/compiler/PART_ONE_FINAL_AUDIT.md` |

## Interface rules

- A consumer depends on the owner checkpoint and requests frozen version 1.
- A provider task may create an isolated contract fragment, but only the named owner publishes/finalizes the interface.
- Existing frozen runtime/JBC interfaces are consumed or adapted. They are not silently mutated to fit the new source compiler.
- Public compiler contracts do not expose raw Clang/LLVM implementation objects.
- Persistent artifacts store stable identities and schemas, not function pointers or runtime addresses.
- Function pointers and backend callbacks live only in source-linked registries or runtime bindings.
- CellShard consumes public Cellerator compiler/ruleset contracts; standalone Cellerator does not require a CellShard link.
- Later incompatible evolution uses adjacent versions rather than reserved-field reinterpretation.
