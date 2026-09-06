# CE-SS1-N05: Execute transpose with shared topology and value authority

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-NATIVE`. **Repository:** Cellerator.

## Objective

Prove reverse application uses the same biological relation and current values, not a second independent semantic model.

## Prerequisites

`CE-SS1-N04`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `src/compute/operation/prepared_relation.cu`
- `tests/semantic_spine/native/transpose_test.cu`

## Implementation actions

1. Bind destination-axis input and source-axis output to CTP1; do not rewrite the relation IDs to mean the reverse graph.

2. Reuse forward value positions at the current generation. Verify transpose is neither an inverse nor a substitute for the full gradient composition.

3. Reject unsupported directions/policies before launch. Preserve distinct domain identities even when extents happen to match.

4. Handle caller ownership/destruction safely with no hidden global synchronization in repeated execution.

## Completion evidence

1. Non-square transpose matches an independent scatter-add oracle at both generations.

2. The dot-product adjoint identity holds within tolerance for multiple vectors.

3. Forward/transpose topology/projection counters stay unchanged after refresh.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
