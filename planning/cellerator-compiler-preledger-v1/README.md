# Cellerator Compiler Part One pre-ledger package

This directory is a complete, non-authoritative planning and manual Todo-bootstrap package for the first production implementation program of the Cellerator programming language and compiler family.

It contains **557 proposed Todos**, including **502 implementation or validation leaves**, organized into **34 workstreams**, **38 first-class lanes**, **41 proposed interfaces**, **10 milestone barriers**, and **44 task-owned checkpoints**.

## The file to apply

The actual Todo-Orchestrator input is:

```text
machine/cellerator-compiler-part1.todo-plan.json
```

It uses Todo plan schema version **3**.

`machine/proposed_todos.json` is a richer pre-ledger research catalog. It is **not** the file to apply.

## Authority snapshot

This package was generated against:

- Cellerator main: `31efdb245f41263acd4432d78fa9e228e21fd444`
- Cellerator worktree clean at local observation: `false`
- Todo revision: `3894`
- Todo semantic fingerprint: `2247877ebdb63a131bb671b82f789c34bedfe80c2a4e6b741f29a45455bc5899`
- CellShard observation: `unavailable`
- package observation time: `2026-09-03T17:34:41+00:00`

These values are evidence, not eternal preconditions. Revalidate them immediately before any manual apply. See [21_MANUAL_TODO_BOOTSTRAP.md](21_MANUAL_TODO_BOOTSTRAP.md).

## What Part One builds

Part One delivers:

- `cellerator`, an NVCC-like Cellerator compiler driver;
- `libCellerator`, with compiler APIs and direct runtime/execution APIs;
- `celleratord`, with ordinary C++ language-server behavior plus Cellerator semantics;
- the Cellerator source language and file-local `#pragma cellerator`;
- public writable Semantic, Planning, and Realization IR;
- representative-data profile artifacts and multi-state propagation;
- Cellerator-owned discovery, exact certification, atoms, grammar, basis, program IR, schedule/rulesets, decomposition, candidates, cost, planning, and realization;
- CPU, NVCC, Clang CUDA/NVPTX, and direct-PTX-capable backend boundaries;
- reflection, inline IR, custom and replacement passes, bounded same-compilation transforms, trusted/unsafe/raw control;
- cross-translation-unit CEIR and Cellerator LTO;
- a `.cell` standard-library foundation, explicit reference profiles, and a relocatable SDK/package.

General JIT/runtime compilation and deep CellShard materialization/runtime integration are explicitly deferred to Part Two.

## Governing ownership

Cellerator is the compiler. It discovers and compiles reusable rules from biological structure and representative data.

CellShard remains an application/storage/runtime built on Cellerator. It may materialize and serve concrete instances under compiled rules, but it does not own compiler discovery, grammar, basis, program IR, or portable schedule compilation.

Existing JBC work is preserved and rehomed. It is not thrown away.

## Package map

1. [01_LIVE_AUTHORITY_AND_RESEARCH_SNAPSHOT.md](01_LIVE_AUTHORITY_AND_RESEARCH_SNAPSHOT.md)
2. [02_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md](02_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md)
3. [03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md](03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md)
4. [04_SOURCE_OWNERSHIP_AND_DIRECTORY_STRUCTURE.md](04_SOURCE_OWNERSHIP_AND_DIRECTORY_STRUCTURE.md)
5. [05_COMPILER_FRONTEND_DRIVER_AND_CXX_INTEGRATION_PLAN.md](05_COMPILER_FRONTEND_DRIVER_AND_CXX_INTEGRATION_PLAN.md)
6. [06_SOURCE_LANGUAGE_IMPLEMENTATION_PLAN.md](06_SOURCE_LANGUAGE_IMPLEMENTATION_PLAN.md)
7. [07_SEMANTIC_IR_IMPLEMENTATION_PLAN.md](07_SEMANTIC_IR_IMPLEMENTATION_PLAN.md)
8. [08_REPRESENTATIVE_PROFILE_IMPLEMENTATION_PLAN.md](08_REPRESENTATIVE_PROFILE_IMPLEMENTATION_PLAN.md)
9. [09_PLANNING_IR_DISCOVERY_AND_JBC_MIGRATION_PLAN.md](09_PLANNING_IR_DISCOVERY_AND_JBC_MIGRATION_PLAN.md)
10. [10_REALIZATION_IR_AND_BACKEND_PLAN.md](10_REALIZATION_IR_AND_BACKEND_PLAN.md)
11. [11_PROGRAMMABLE_COMPILER_REFLECTION_PASSES_AND_DIAGNOSTICS.md](11_PROGRAMMABLE_COMPILER_REFLECTION_PASSES_AND_DIAGNOSTICS.md)
12. [12_CROSS_TU_LTO_AND_OBJECT_ARTIFACT_PLAN.md](12_CROSS_TU_LTO_AND_OBJECT_ARTIFACT_PLAN.md)
13. [13_LIBCELLERATOR_STDLIB_SDK_AND_PACKAGE_PLAN.md](13_LIBCELLERATOR_STDLIB_SDK_AND_PACKAGE_PLAN.md)
14. [14_CELLERATORD_PLAN.md](14_CELLERATORD_PLAN.md)
15. [15_VALIDATION_BENCHMARKS_AND_VERTICAL_MILESTONES.md](15_VALIDATION_BENCHMARKS_AND_VERTICAL_MILESTONES.md)
16. [16_INTERFACES_AND_CONTRACT_OWNERSHIP.md](16_INTERFACES_AND_CONTRACT_OWNERSHIP.md)
17. [17_DEPENDENCY_BARRIERS_LANES_AND_INTEGRATION.md](17_DEPENDENCY_BARRIERS_LANES_AND_INTEGRATION.md)
18. [18_COMPLETE_PROPOSED_TODO_CATALOG.md](18_COMPLETE_PROPOSED_TODO_CATALOG.md)
19. [19_PART_ONE_COMPLETION_AND_ACCEPTANCE.md](19_PART_ONE_COMPLETION_AND_ACCEPTANCE.md)
20. [20_DEFERRED_PART_TWO_INVENTORY.md](20_DEFERRED_PART_TWO_INVENTORY.md)
21. [21_MANUAL_TODO_BOOTSTRAP.md](21_MANUAL_TODO_BOOTSTRAP.md)
22. [22_PACKAGE_VALIDATION_REPORT.md](22_PACKAGE_VALIDATION_REPORT.md)
23. [23_RESEARCH_PRECEDENT_AND_SOURCE_INDEX.md](23_RESEARCH_PRECEDENT_AND_SOURCE_INDEX.md)
24. [24_SPECIFICATION_RECONCILIATION.md](24_SPECIFICATION_RECONCILIATION.md)

Per-Todo proposed records are under `proposed-todos/`.

Machine-readable catalogs and the apply plan are under `machine/`.

Raw source/authority observations and plan-schema evidence are under `evidence/`.

JBC worktree and migration inventories are under `inventories/`.

## Mutation boundary

Creating this package did not apply the plan, activate the run, claim work, create implementation worktrees, or modify compiler source. The mutating commands in the manual guide are for the user to run later.


## Final package index

This directory is the complete non-authoritative planning and manual-bootstrap
package for **Cellerator Compiler Part One**.

The package contains:

- **557 proposed Todo records**
- **34 workstreams**
- **38 first-class lanes**
- **41 proposed interfaces**
- **10 program barriers**
- **44 task-owned checkpoints**
- **684 explicit task/checkpoint dependency edges**

The file to apply manually to Todo Orchestrator is:

[`machine/cellerator-compiler-part1.todo-plan.json`](machine/cellerator-compiler-part1.todo-plan.json)

It uses Todo plan schema version **3**.

Do **not** apply [`machine/proposed_todos.json`](machine/proposed_todos.json).
That file is a richer pre-ledger research catalog and has no Todo authority.

Primary reading order:

1. [`02_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md`](02_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md)
2. [`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md`](03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md)
3. [`04_SOURCE_OWNERSHIP_AND_DIRECTORY_STRUCTURE.md`](04_SOURCE_OWNERSHIP_AND_DIRECTORY_STRUCTURE.md)
4. [`16_INTERFACES_AND_CONTRACT_OWNERSHIP.md`](16_INTERFACES_AND_CONTRACT_OWNERSHIP.md)
5. [`17_DEPENDENCY_BARRIERS_LANES_AND_INTEGRATION.md`](17_DEPENDENCY_BARRIERS_LANES_AND_INTEGRATION.md)
6. [`18_COMPLETE_PROPOSED_TODO_CATALOG.md`](18_COMPLETE_PROPOSED_TODO_CATALOG.md)
7. [`21_MANUAL_TODO_BOOTSTRAP.md`](21_MANUAL_TODO_BOOTSTRAP.md)
8. [`22_PACKAGE_VALIDATION_REPORT.md`](22_PACKAGE_VALIDATION_REPORT.md)

Per-Todo proposed human records are under
[`proposed-todos/`](proposed-todos/). They are deliberately labeled
pre-ledger and contain no fake managed markers.

Package integrity is anchored by [`MANIFEST.sha256`](MANIFEST.sha256).
