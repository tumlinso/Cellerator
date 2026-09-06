# CE-SS1-F01: Lower existing relation semantic IR through the shared contract

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-FRONTEND`. **Repository:** Cellerator.

## Objective

Replace the selected relation-family duplicate semantic interpretation with the canonical descriptor.

## Prerequisites

`CE-SS1-I02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `include/Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh`
- `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `src/compiler/sema/implement_operation_kind_resolution.cc`

## Exclusive write scope

- `include/Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh`
- `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `tests/semantic_spine/compiler/lowering_test.cc`

## Implementation actions

1. Adapt the existing relation_apply_operation_ir_v1 lowering, not a new demo-only parser. Preserve source locations and symbol identity separately from mathematical equality.

2. Map both orientations to the canonical descriptor. Preserve relation storage separately from state type, order identities, output policy and arithmetic permissions.

3. Delegate common semantic validation to the shared validator; compiler-specific diagnostics may wrap it.

4. Restrict legacy result objects/adapters to required existing consumers, accounting for self-referential view lifetime.

## Completion evidence

1. An existing IR forward and transpose yield the same mathematical descriptors as independently built C++ calls.

2. f16 relation/f32 state does not become an all-f32 or all-f16 descriptor.

3. No duplicate scalar interpreter is added as execution proof.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
