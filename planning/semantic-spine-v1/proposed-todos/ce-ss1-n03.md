# CE-SS1-N03: Refresh device values without rebuilding topology

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-NATIVE`. **Repository:** Cellerator.

## Objective

Publish changing values as stream-ordered generations while reusing the prepared structure.

## Prerequisites

`CE-SS1-N02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `src/runtime/value_readiness.cu`
- `src/runtime/value_readiness.cu`

## Exclusive write scope

- `src/compute/operation/prepared_relation.cu`
- `tests/semantic_spine/native/generation_test.cu`

## Implementation actions

1. Bind device f16 values with structure ID/epoch, logical edge order, generation and count. Use a device gather/packing path when mapping to persistent value positions.

2. Refresh only value-dependent storage. Transpose consumes the corresponding forward-position values rather than independently rebuilding support.

3. Reject stale/non-monotonic generations and incompatible keys before enqueue. Advance the accepted-generation metadata only after required enqueues succeed; on partial CUDA failure mark the pair unusable until a documented recovery/destruction, not falsely ready.

4. Constrain v1 to one caller stream; values and operands remain alive until the caller fences. Do not insert a global device synchronization.

## Completion evidence

1. Gen1 and gen2 yield different correct results with one topology preparation.

2. A stale generation or changed edge order is rejected before output mutation; counters/generation remain coherent.

3. Refresh traffic is distinguished from topology construction and no host readback computes relation results.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
