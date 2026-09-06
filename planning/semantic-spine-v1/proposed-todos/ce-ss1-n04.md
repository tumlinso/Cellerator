# CE-SS1-N04: Execute canonical forward application on sm70

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-NATIVE`. **Repository:** Cellerator.

## Objective

Make canonical forward requests reach the existing device candidate through the prepared pair.

## Prerequisites

`CE-SS1-N03`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `src/compute/operation/prepared_relation.cu`
- `tests/semantic_spine/native/forward_test.cu`

## Implementation actions

1. Bind typed input/output axes and counts, validate arithmetic/output semantics, and invoke the bound forward candidate on the caller stream.

2. Keep reusable physical layout internal. Do not allocate, hash the full relation, discover candidates or transfer output to host during enqueue.

3. Track evidence from the actual selected and launched implementation, not a catalog label entered by the caller.

4. Test two input states and both value generations, with explicit synchronization only at host observation boundaries.

## Completion evidence

1. GPU forward output matches the independent logical-edge oracle within the declared tolerance.

2. Wrong input domain/order/device, alias or stale generation does not enqueue computation.

3. Existing forward candidate regression tests retain their assertions.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
