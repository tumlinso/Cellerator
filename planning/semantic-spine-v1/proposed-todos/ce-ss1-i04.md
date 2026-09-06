# CE-SS1-I04: Integrate all four lanes and repository-local targets

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Join the independent native, compiler, algebra and verification artifacts without introducing a second runtime or broad build refactor.

## Prerequisites

`CE-SS1-N06`, `CE-SS1-F04`, `CE-SS1-A04`, `CE-SS1-V04`, `CE-SS1-I03`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `src/compute/operation/prepared_relation.cu`
- `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `src/compiler/sema/implement_operation_kind_resolution.cc`
- `examples/CMakeLists.txt`

## Exclusive write scope

- `CMakeLists.txt`
- `cmake/SemanticSpineV1.cmake`
- `examples/CMakeLists.txt`
- `examples/semantic_spine_v1/CMakeLists.txt`
- `tests/semantic_spine/CMakeLists.txt`
- `docs/semantic_spine_v1/integration.json`

## Implementation actions

1. Integrate all lane artifacts against the same foundation. Resolve common build changes here only; substantive algorithm/semantic fixes go back to the owning lane.

2. Define opt-in CE-SS1 test/demo targets using real objects and existing CUDA providers. Avoid installing/exporting an SDK as part of this change.

3. Link the actual parser/Sema subset needed by the source-origin probe, not every nominal compiler interface.

4. Check that algebra classification changes and compiler relation lowering coexist and every intended target resolves symbols.

## Completion evidence

1. All target definitions refer to real source units, not INTERFACE-only placeholders for computation.

2. No broad root-CMake/CUDA backend restructuring was necessary; any prerequisite exception is documented.

3. No pending artifact is mistaken for integrated code.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
