# Compiler executable location freeze

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-003`

## Entrypoints

The two shipped compiler executables have conventional, exclusive entrypoint
locations:

```text
tools/cellerator/main.cc
tools/celleratord/main.cc
```

`cellerator` is the batch compiler and compiler inspection command.
`celleratord` is the long-lived language/tooling service. Both are thin applications over
installed or private shared compiler libraries.

## Thin-main rule

An executable entrypoint may:

- translate `argc` and `argv` into the shared driver or daemon request;
- construct process-level standard input, output, and error streams;
- invoke the shared library entry function;
- translate its result into a process exit status; and
- install process-scoped signal/termination handling when the shared daemon
  contract requires it.

An entrypoint must not own parsing, AST or Sema logic, CEIR construction,
profiles, biological discovery, planning, realization, backend selection,
diagnostic policy, protocol semantics, or reusable option interpretation.
Those mechanisms belong under `include/Cellerator/compiler/` and
`src/compiler/` so libCellerator, tests, and both executables use the same
implementation.

`tools/cellerator/` and `tools/celleratord/` may contain executable-only smoke
fixtures or packaging metadata, but reusable `.cc`, `.hh`, `.hpp`, `.h`, or
`.cuh` implementation files are forbidden there. The only production C++
translation unit in each directory is `main.cc`.

## Developer tools

CEIR printing, CEIR verification, profile inspection, pass listing, provenance
inspection, and backend discovery default to `cellerator` subcommands. A
separate small developer executable is justified only when it still calls a
shared library API and has its own integration-owned target. It must not become
a second driver, IR library, or profile implementation.

## Dependencies and build boundary

Both executables link compiler/tooling libraries; compiler libraries never link
an executable target. `celleratord` protocol and semantic code stays host-only
and does not link CUDA. Accelerator access occurs through compiler backend or
runtime provider libraries selected by shared contracts.

The existing `tools/` directory currently contains build aggregation,
documentation, and a repository validation script. This freeze neither moves
those files nor changes `tools/CMakeLists.txt`; executable targets are added by
the central integration owner later.

## Compatibility and deferred work

This location contract preserves current runtime and JBC behavior. It creates
no executable, target, umbrella include, Clang fork, Part Two JIT, or deep
CellShard runtime mechanism.

## Validation evidence

`tests/compiler/a04/freeze_compiler_executable_locations_test.cc` checks the two
entrypoint paths, thin-main restrictions, shared-library direction, host-only
daemon rule, developer-tool policy, and the current absence of executable-owned
reusable compiler source.
