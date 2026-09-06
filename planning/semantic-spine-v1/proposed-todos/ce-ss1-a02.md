# CE-SS1-A02: Make primitive, composition and effect classifications explicit

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-ALGEBRA`. **Repository:** Cellerator.

## Objective

Prevent high-level constructs from being mistaken for the representative primitive opcode used by an older mapping.

## Prerequisites

`CE-SS1-A01`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `include/Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh`
- `src/compiler/sema/implement_operation_kind_resolution.cc`
- `tests/semantic_spine/algebra/classification_test.cc`

## Implementation actions

1. Replace the meaningful ambiguity in operation_kind_resolution with a tagged primitive/composition/effect result or equivalent non-ambiguous representation.

2. Keep chain, exchange, hierarchy, moments and gradient distinguishable from a primitive. Publication is an effect; it must not silently become sparse arithmetic.

3. Preserve useful source spellings and existing flags only where they do not permit accidental execution as the wrong operation.

4. Mark lowering availability separately from semantic recognition. Do not enumerate an entire future algebra or implement all compositions.

## Completion evidence

1. Asking for a primitive from a composition/effect fails explicitly, rather than returning sparse_axis_update or apply.

2. Existing directly supported apply/transpose resolution remains usable.

3. All 14 currently represented source families receive an explicit, truthful classification.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
