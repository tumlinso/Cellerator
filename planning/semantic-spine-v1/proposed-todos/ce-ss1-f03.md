# CE-SS1-F03: Preserve effects and diagnostics without losing equivalence

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-FRONTEND`. **Repository:** Cellerator.

## Objective

Prevent compiler wrappers from silently weakening the native contract.

## Prerequisites

`CE-SS1-F02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `src/compiler/sema/relation_spine_bridge.cc`
- `tests/semantic_spine/compiler/diagnostic_test.cc`

## Implementation actions

1. Check high identity bits, axis orientation, logical edge order, empty shapes, separate arithmetic types and unsupported output/alias modes.

2. Source provenance may differ between native and source-origin requests; the canonical mathematical key must not depend on where a request was spelled.

3. Record relevant read/write effects and keep publication/generation facts out of the static topology key.

4. Test copy/move of lowered records and returned borrowed views so tests do not succeed through dangling stack references.

## Completion evidence

1. Both origins reject the same mathematical invalidities with source-context diagnostics on the compiler side.

2. A new value binding does not require reparsing or changing canonical topology semantics.

3. No full execution-field/LTO/driver implementation sneaks into this task.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
