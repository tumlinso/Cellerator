# CE-SS1-N02: Prepare reusable forward and transpose structure

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-NATIVE`. **Repository:** Cellerator.

## Objective

Construct one reusable topology asset with both physical views using existing packing/projection builders.

## Prerequisites

`CE-SS1-N01`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `src/compute/projection/physical_feature_major.cc`
- `src/compute/projection/physical_transpose.cc`
- `tests/math_core/transpose_backward_candidate_test.cu`

## Exclusive write scope

- `src/compute/operation/prepared_relation.cu`
- `tests/semantic_spine/native/preparation_test.cu`

## Implementation actions

1. Use host CSR input only as the cold interchange for this slice. Retain edge identity/order and use the existing geometry/packing-to-FMP1 and CTP1 construction facilities.

2. Do not fabricate valid-looking packing headers from the tiny fixture. Preparation must depend on supplied endpoints, dimensions and identities.

3. Share forward value positions with transpose where supported. Allocate all fixed-capacity device buffers and scratch before execution.

4. Handle empty support via a legal device zero-fill/no-op or reject physical unsupported cases explicitly without changing the mathematical law. Check all local-index bounds.

## Completion evidence

1. Two structurally different non-square relations prepare and execute correctly; no fixture-specific canned topology.

2. Zero-degree rows, singleton and empty-support cases are covered. Duplicate endpoint contributions either preserve their exact meaning or are explicitly rejected by this limited physical provider; never silently coalesce weights.

3. Forward/transpose preparation identity is stable across subsequent value publications.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
