# libCellerator, standard library, SDK, and package plan

## libCellerator

The SDK has two first-class constituencies:

1. compiler embedding and tooling;
2. direct use of Cellerator’s runtime/execution substrate without the source language.

Component libraries remain separately usable. A convenience `libCellerator` facade ties them together.

Public compiler APIs do not expose unstable Clang or LLVM objects. Direct low-level runtime headers remain available to expert users.

## Standard library

The standard library is `.cell` source and is compiled by Cellerator itself. It provides:

- low-level semantic constructors and concepts;
- explicit relation/state views;
- operation wrappers;
- profile/persistence helpers;
- planning and expert-control builders;
- reflection and pass helpers.

Compiler-semantic types remain base language concepts.

## Installed layout

```text
bin/cellerator
bin/celleratord
lib/libCellerator*
include/Cellerator/
share/cellerator/stdlib/
share/cellerator/profiles/
share/cellerator/schemas/
share/cellerator/backends/
lib/cmake/Cellerator/
```

The install is relocatable and supports clean CMake and non-CMake consumers.

## Workstream task catalog

### H02: libCellerator

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-H02-001` | Freeze libCellerator component architecture | Define separately linkable compiler, IR, profile, planning, realization, backend, diagnostics, and runtime/execution components plus a convenience libCellerator umbrella. |
| `CE-CCP1-H02-002` | Define C compiler session API | Expose driver configuration, source buffers/files, profiles, target/toolchain, diagnostics callbacks, cancellation, outputs, and lifecycle through opaque handles and versioned records. |
| `CE-CCP1-H02-003` | Define C++ compiler session API | Expose RAII contexts, source manager, AST/Sema snapshots, CEIR builders/readers, profiles, pass pipelines, backends, diagnostics, and compilation results. |
| `CE-CCP1-H02-004` | Expose parse and semantic-analysis APIs | Allow clients to parse, inspect, incrementally update, and semantically analyze source without running planning or code generation. |
| `CE-CCP1-H02-005` | Expose CEIR ownership and mutation APIs | Provide contexts, immutable snapshots, builders, cloning, parsing/printing, validation modes, provenance, and serialization for all three IR levels. |
| `CE-CCP1-H02-006` | Expose profile build/load/query APIs | Provide pointer-plus-count ingestion, binary/text load, named-state lookup, diff, transfer functions, and profile environment binding. |
| `CE-CCP1-H02-007` | Expose planning and candidate APIs | Provide provider registration, discovery, decomposition, costs/evidence, planner replacement, force controls, reports, and selected rulesets. |
| `CE-CCP1-H02-008` | Expose realization and backend APIs | Provide target description, physical IR editing, backend registry, generated artifact access, source maps, native fragment hooks, and object emission. |
| `CE-CCP1-H02-009` | Expose pass, reflection, and extension APIs | Provide pass registration/pipelines, extension schemas/protocols, reflection handles/builders, same-compilation transform ABI, and trust modes. |
| `CE-CCP1-H02-010` | Expose structured diagnostics and explainability APIs | Provide stable diagnostic records, planning reports, provenance queries, reproducer creation, progress, timing, and cancellation. |
| `CE-CCP1-H02-011` | Define thread safety and context isolation | Specify global immutable registries, per-context mutable state, thread-safe readers, explicit non-thread-safe builders, and backend process isolation. |
| `CE-CCP1-H02-012` | Preserve direct runtime/execution access | Keep biological ABI, operation core, relation algebra, geometry, candidates, planner, prepared programs, sessions, readiness, and providers available to users who never compile `. |
| `CE-CCP1-H02-013` | Create narrow public runtime facades | Add stable umbrella/facade contracts over currently broad internal dependency closures without deleting existing lower-level expert headers. |
| `CE-CCP1-H02-014` | Define ABI/version and feature queries | Expose source-language revision, CEIR levels, compiler API ABI, backend/provider versions, optional features, and compatibility checks. |
| `CE-CCP1-H02-015` | Create installed SDK examples | Provide plain runtime API, source compiler embedding, CEIR editing, custom candidate, custom pass, and backend examples that build outside the source tree. |
| `CE-CCP1-H02-016` | Freeze libCellerator SDK acceptance | Publish component libraries and umbrellas, build C/C++ external consumers, preserve current execution functionality, and verify compiler APIs do not leak Clang/LLVM internals. |

### H03: standard library and package

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-H03-001` | Freeze the language versus standard-library boundary | Audit every proposed convenience abstraction: only compiler-reasoned semantic facts remain base language; containers, algorithms, biological constructions, policies, and helpers are `. |
| `CE-CCP1-H03-002` | Define standard-library package structure | Organize `. |
| `CE-CCP1-H03-003` | Implement the minimal core `.cell` library | Provide compiler-recognized bridge declarations, concepts, traits, low-level views, span-like pointer/count types, result/status helpers, and no-allocation facilities needed by examples. |
| `CE-CCP1-H03-004` | Implement relation and state construction helpers | Provide explicit low-level constructors/builders from pointers, extents, identities, orders, support, generations, and numeric types without hiding costs. |
| `CE-CCP1-H03-005` | Implement operation wrappers in Cellerator source | Express relation, transpose, contraction, segment, gate, update, bundle, chain, moments, hierarchy, and exchange constructions as inlineable `. |
| `CE-CCP1-H03-006` | Implement profile and persistence helpers | Provide typed names/state selectors, reuse/lifetime facts, generation helpers, mutation contracts, and explicit generic reference-profile selection. |
| `CE-CCP1-H03-007` | Implement planning and expert-control helpers | Provide source conveniences for preferences, constraints, candidate offers, forced plans, cost records, decomposition builders, and unsafe modes while leaving full CEIR accessible. |
| `CE-CCP1-H03-008` | Implement reflection and inline-IR helper library | Provide concepts, visitors, pattern helpers, builders, and pass adapters in `. |
| `CE-CCP1-H03-009` | Define reference species profile policy | Ship explicitly named, low-performance, testing-oriented profiles for a small audited set such as Homo sapiens, Mus musculus, and a rat reference. |
| `CE-CCP1-H03-010` | Build minimal reference profile artifacts | Generate compact relation/domain/statistical placeholders sufficient for parser, Sema, IR, planning fallback, examples, and CI, not claims of realistic optimized biology. |
| `CE-CCP1-H03-011` | Implement compiler resource discovery | Locate standard library, reference profiles, backend manifests, schemas, and support binaries relative to the installed executable with explicit override flags. |
| `CE-CCP1-H03-012` | Create CMake package exports | Install component targets, CelleratorConfig, version config, compiler helpers, backend/provider targets, feature variables, and dependency discovery without source-tree paths. |
| `CE-CCP1-H03-013` | Create CMake compiler integration helpers | Provide functions/toolchain guidance for `. |
| `CE-CCP1-H03-014` | Define installation layout and RPATH policy | Install bin/cellerator, bin/celleratord, component libraries, headers, stdlib, profiles, schemas, backends, docs/examples, and debug metadata in relocatable platform-appropriate locations. |
| `CE-CCP1-H03-015` | Create pkg-config or equivalent lightweight metadata | Expose libCellerator compiler/runtime compile and link flags for non-CMake consumers where platform conventions support it. |
| `CE-CCP1-H03-016` | Build package manifest and resource hashes | Generate installed-file manifests, CEIR/profile/schema revisions, backend identities, standard-library hashes, and reproducibility metadata. |
| `CE-CCP1-H03-017` | Create package upgrade and coexistence tests | Install two versioned resource trees, test selected compiler/resource pairing, plugin compatibility diagnostics, and no accidental cross-version profile/IR loading. |
| `CE-CCP1-H03-018` | Freeze installable SDK and standard-library foundation | Install from a clean checkout, compile ordinary C++, `. |
