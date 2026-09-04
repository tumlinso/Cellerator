# Compiler implementation tree freeze

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-002`

This receipt fixes the implementation-side counterpart of the installed tree.
It adds no implementation files and changes no runtime behavior.

## Implementation tree

```text
src/compiler/
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
    support/
```

Public areas mirror `include/Cellerator/compiler/` closely enough that a
contract and its implementation are discoverable by the same relative area.
Implementation-specific helpers may live in an area's `detail/` directory or
in `src/compiler/support/`; neither location is installed. Generated build
artifacts stay in the build tree rather than under `src/compiler/`.

## Include-direction rules

1. A source file includes its public contract through
   `Cellerator/compiler/...`; it does not reach into the installed tree by a
   relative filesystem path.
2. An installed header under `include/Cellerator/` must never include a path
   beginning with `src/compiler/`, `compiler/detail/`, or
   `compiler/support/`.
3. Private headers use quoted, source-relative includes and stay within the
   implementation target that owns them. They are not exported as a shortcut
   for an incomplete public contract.
4. `ir/common/` is dependency-light and does not depend on frontend, planning
   services, backends, tooling, or executable code.
5. Frontend and Sema may produce AST and Semantic IR, but do not include a
   concrete backend implementation.
6. Planning services consume public profile and IR contracts. They do not
   include backend-private headers; provider selection crosses the public
   backend ABI.
7. Backend implementations may consume common and Realization IR contracts.
   CPU, NVCC, and NVPTX implementations do not include one another and do not
   become frontend or CEIR owners.
8. `tooling/` and `api/` are library implementation. Executable entrypoints
   call those libraries and own no reusable compiler mechanism.

These are layering rules, not permission to duplicate a type merely to avoid a
dependency. Shared stable types move to the narrowest installed owner.

## SDK leak prevention

The install surface is selected only from `include/Cellerator/`, checked-in
public schemas/resources, and explicitly generated public build headers.
Nothing under `src/compiler/` is an install candidate. CMake targets may list
private headers for IDE visibility, but must mark them private and must not add
`src/compiler/` to an installed target's public or interface include paths.

The gate scans the existing installed Cellerator headers to ensure they do not
refer to `src/compiler/`. Later implementation gates must retain that scan as
the compiler tree becomes populated.

## Existing implementation correspondence

Existing `src/execution/`, `src/geometry/`, `src/compute/`, `src/planner/`, and
`src/runtime/` remain the implementations of their current public contracts.
They are adapted by compiler code rather than moved, duplicated, or exposed as
compiler-private headers. In particular, `src/geometry/compiler/` keeps its
biological geometry ownership and is not renamed into the source-language
compiler.

## Compatibility and deferred work

This layout-only freeze preserves current runtime and JBC behavior. It does not
change root or subsystem CMake files, package exports, umbrella headers, or
executable targets. It introduces no Part Two JIT or deep CellShard runtime
work.

## Validation evidence

`tests/compiler/a04/freeze_the_compiler_implementation_tree_test.cc` validates
the mirrored area list, layering rules, SDK leak prohibition, links to the
existing implementation roots, and the absence of `src/compiler/` includes in
the current installed Cellerator headers.
