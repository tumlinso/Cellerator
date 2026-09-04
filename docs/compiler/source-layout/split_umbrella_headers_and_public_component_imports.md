# Umbrella header and public component import split

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-007`

## Three umbrella contracts

| Include | Purpose | Dependency boundary |
| --- | --- | --- |
| `#include <Cellerator/compiler.hh>` | Source compilation, CEIR, profiles, planning, realization, reflection, passes, diagnostics, LTO, backend contracts, and compiler API | host-safe compiler contracts; no runtime provider, CUDA, Torch, or CellShard dependency |
| `#include <Cellerator/runtime.hh>` | Execution identities, programs, sessions, geometry, projection, planning, and runtime provider contracts | runtime surface; accelerator-specific types remain in explicit provider component headers |
| `#include <Cellerator/Cellerator.hh>` | Convenience facade for applications intentionally using compiler and runtime | includes `compiler.hh` and `runtime.hh`; does not independently enumerate every leaf header |

Each umbrella is centrally integration-owned. Leaf tasks add component headers
and publish fragments; they do not append includes to an umbrella.

## Small-umbrella rule

An umbrella includes only stable component entry headers, never every
implementation leaf. It must not include tests, private `src/` headers,
generated build-private headers, executable headers, or sibling repository
internals. Adding a component import requires a public dependency review and an
include-only downstream compile fixture.

The compiler umbrella remains usable in a host-only build. It cannot include a
header that requires the CUDA compiler, CUDA runtime headers, a GPU device,
CellShard, or CelleraTorch. Backend capability declarations are host-safe;
concrete accelerator provider entry headers are explicit imports.

The runtime umbrella exposes the portable execution surface and provider-neutral
selection. Consumers that need a concrete CUDA, NCCL, distributed, geometry,
or model facility include that component directly. This keeps ordinary SDK
compilation from paying for unrelated transitive headers.

## Preferred component imports

Downstream libraries should prefer the narrowest owner:

```cpp
#include <Cellerator/compiler/ir/common/ir_v1.hh>
#include <Cellerator/compiler/profile/profile_artifact_v1.hh>
#include <Cellerator/compiler/planning/planner_v1.hh>
#include <Cellerator/compiler/backend/backend_v1.hh>
#include <Cellerator/execution/program.hh>
#include <Cellerator/geometry/semantic_geometry.hh>
#include <Cellerator/runtime/session.cuh>
```

Compiler-only tools include `compiler.hh`; runtime-only applications include
`runtime.hh`; end-user applications may select `Cellerator.hh`. Public
libCellerator C and C++ API consumers may include their dedicated API headers
without any umbrella.

## Include-only fixture contract

Central integration provides three independent downstream translation units:

```text
tests/compiler/sdk/compiler_umbrella_include_test.cc
tests/compiler/sdk/runtime_umbrella_include_test.cc
tests/compiler/sdk/cellerator_umbrella_include_test.cc
```

Each fixture contains only its target include plus a trivial `main`, compiles
against the installed/exported include interface, and links no private target.
The compiler fixture runs in the host-only configuration. The runtime and full
fixtures run in host-only and enabled-provider configurations. A separate
component-import fixture compiles every documented preferred import directly.

## Current compatibility state

The current `include/Cellerator/Cellerator.hh` enumerates many runtime,
geometry, compute, provider, model, and interop leaves. It remains compatible
until central integration introduces the two smaller umbrellas and validates
the downstream fixtures. This task does not edit that header or claim the split
already exists.

No current runtime or JBC behavior changes. The split introduces no permanent
Clang fork, Part Two JIT, or deep CellShard runtime ownership.

## Validation evidence

`tests/compiler/a04/split_umbrella_headers_and_public_component_imports_test.cc`
checks the three umbrella contracts, host-only boundary, explicit component
imports, fixture locations, and the candid current-compatibility statement.
