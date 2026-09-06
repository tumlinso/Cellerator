# CE-SS1-N06: Prove native reuse and hand off executable evidence

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-NATIVE`. **Repository:** Cellerator.

## Objective

Close the native lane with actual CUDA tests and a minimal, traceable execution report.

## Prerequisites

`CE-SS1-N05`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/native`
- `docs/semantic_spine_v1/native_validation.json`
- `src/compute/operation/prepared_relation.cu`

## Implementation actions

1. Build a lane-local test target against the integrated foundation and existing provider libraries; do not edit root/central build files.

2. Run the new adapter, preparation, generations, forward and transpose cases on an actual sm70 device under a lease. Include a direct-provider comparison as an overhead control, not a speedup promotion.

3. Record numeric tuple, device, source/toolchain, selected symbols, transfers, synchronizations, preparation/refresh counts and all real test commands.

4. Commit and hand off the artifact. Missing hardware is a blocker, never a skipped successful completion.

## Completion evidence

1. No CPU/reference success is reported as accelerator execution.

2. No out-of-scope MMA, optimizer, JIT, training or broad candidate rewrite was added.

3. I04 receives actual implementation plus tests, not only a descriptive registration.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
