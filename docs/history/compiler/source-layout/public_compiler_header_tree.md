# Public compiler header tree freeze

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-001`

This receipt fixes where installed compiler contracts will live. It creates no
headers and changes no current runtime API. Later implementation tasks must use
these paths rather than introducing another compiler root or placing public
contracts under `src/`.

## Installed tree

```text
include/Cellerator/compiler/
    driver/
    frontend/
        source/
        parser/
        cxx/
    ast/
    sema/
        field/
    ir/
        common/
        text/
        semantic/
        planning/
        realization/
    profile/
    discovery/
    composition/
    program/
    planning/
    reflection/
    pass/
    diagnostics/
    lto/
    backend/
        cpu/
        nvcc/
        nvptx/
    tooling/
    api/
    build/
```

All directory names are lowercase. Header names use the repository's
`snake_case` convention, with an explicit version suffix for durable ABI and
artifact contracts. `include/Cellerator/compiler.hh` is the centrally owned
convenience umbrella; leaf tasks do not edit it.

## Public and internal boundary

- Files below `include/Cellerator/compiler/` are installed SDK contracts. They
  may depend on other installed Cellerator contracts, the C++ standard library,
  and deliberately public third-party ABI types only.
- Installed headers must not include `src/compiler/`, build-tree generated
  implementation headers, executable-private headers, or raw Clang/LLVM
  implementation objects.
- Implementation-only declarations live below `src/compiler/`. A declaration
  moves into the installed tree only when a supported SDK consumer needs it.
- `driver/` owns reusable driver and toolchain contracts, while executable
  argument parsing and `main` functions remain outside the installed tree.
- `frontend/` owns source coordination. Its `source/`, `parser/`, and `cxx/`
  children separate stable source identity, Cellerator syntax, and the upstream
  C++ semantic bridge.
- `ast/` and `sema/` own language structure and biological meaning. They do not
  acquire runtime execution ownership.
- `ir/` owns the CEIR object families. `common/` supplies shared identity and
  traversal contracts; `text/` owns serialization; `semantic/`, `planning/`,
  and `realization/` are distinct, writable IR levels.
- `profile/`, `discovery/`, `composition/`, `program/`, and `planning/` own the
  compiler's biological evidence, grammar/basis, ruleset, and whole-program
  planning contracts. `ir/planning/` is the data model; `planning/` is the
  compiler service that operates on it.
- `reflection/`, `pass/`, `diagnostics/`, and `lto/` are public extension and
  whole-program surfaces. Provenance stays cold and source-linked.
- `backend/` exposes one backend ABI and isolates CPU, NVCC, and NVPTX provider
  contracts. Backend headers do not become the frontend or CEIR authority.
- `tooling/` is reusable tooling protocol, not executable-owned logic. `api/`
  contains the C and C++ libCellerator compiler facades. `build/` exposes only
  generated, installed feature/version declarations.

## Existing source correspondence

The new tree complements rather than renames the current public runtime:

| Compiler area | Existing source evidence that remains authoritative today |
| --- | --- |
| semantic and realization contracts | `include/Cellerator/execution/` |
| biological geometry compilation | `include/Cellerator/geometry/` |
| candidates and native execution | `include/Cellerator/compute/` |
| measured planning | `include/Cellerator/planner/` |
| sessions and runtime bindings | `include/Cellerator/runtime/` |

Compiler headers may adapt those installed contracts without copying their
ownership. The current `include/Cellerator/geometry/compiler/` subtree remains
the geometry compiler and is not a substitute for, or collision with, the new
source-language compiler root. CellShard remains a consumer at the storage and
application boundary; no CellShard header becomes part of this installed
compiler tree.

## Compatibility and deferred work

This freeze preserves all existing runtime and JBC behavior because it is a
layout contract only. It does not create a Clang fork, change the build graph,
install files, define the final umbrella, or introduce Part Two JIT/runtime
materialization. Those actions remain owned by later tasks and central
integration checkpoints.

## Validation evidence

`tests/compiler/a04/freeze_the_public_compiler_header_tree_test.cc` checks the
complete path set, lowercase naming, the public/internal boundary, the
planning-IR versus planning-service distinction, links to existing installed
source areas, and the explicit compatibility/deferred-work statements.
