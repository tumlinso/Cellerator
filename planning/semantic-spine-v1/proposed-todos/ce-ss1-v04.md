# CE-SS1-V04: Deliver validation harness and adversarial controls

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-VERIFY`. **Repository:** Cellerator.

## Objective

Give integration a compact test harness and prove it detects wrong output and missing hardware.

## Prerequisites

`CE-SS1-V03`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/verification`
- `docs/semantic_spine_v1/verification_handoff.json`

## Implementation actions

1. Compile/run independent oracle and metadata tests locally. Mark integrated probes as pending linkage rather than passed.

2. Exercise negative result vectors, stale generation, NaN comparison logic and missing implementation/hardware controls.

3. Keep performance observations informational in this epic, while making correctness, no-hidden-work promises and GPU execution non-negotiable.

4. Commit and hand off tests for I04/I05/I06. Do not claim final accelerator validation from this preparatory lane.

## Completion evidence

1. Wrong-index and wrong-generation controls fail.

2. Standalone oracle completion and pending integrated execution are separately recorded.

3. The verification lane has no dependency on the native/frontend tail, so its useful work runs concurrently.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
