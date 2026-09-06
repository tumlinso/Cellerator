# CE-SS1-V01: Create an independent logical-edge oracle and adversarial fixtures

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-VERIFY`. **Repository:** Cellerator.

## Objective

Make correctness independent of physical packing and compiler lowering.

## Prerequisites

`CE-SS1-I02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/verification/oracle.hh`
- `tests/semantic_spine/verification/fixtures.hh`
- `tests/semantic_spine/verification/oracle_test.cc`

## Implementation actions

1. Use small host double accumulation over logical edges for oracle values. Retain declared input quantization; compare GPU output to the same stored inputs, not an unrelated higher-precision problem.

2. Include the demo fixture plus a second non-square relation, empty destination rows, duplicate endpoint edges, zero support, permutations and cancellation cases.

3. Supply an explicit absolute+relative tolerance rule with finite/nonfinite handling. Nonfinite comparisons never pass just because a NaN comparison is false.

4. Keep fixed arrays/contiguous fixtures; no generic framework or biological dataset download.

## Completion evidence

1. Hand-computed fixture outputs match the oracle and transpose is verified independently of the forward implementation.

2. A deliberately transposed/wrong-index result is rejected.

3. Exact edge/identity checks are not weakened by numerical tolerance.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
