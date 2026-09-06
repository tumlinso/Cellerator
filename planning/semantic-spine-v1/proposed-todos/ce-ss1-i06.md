# CE-SS1-I06: Run the supplied biological demo and CUDA memory checks

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Make the exact delivered example work with the implemented provisional API and demonstrate reuse without new performance claims.

## Prerequisites

`CE-SS1-I05`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `examples/semantic_spine_v1`
- `docs/semantic_spine_v1/demo_execution.json`

## Implementation actions

1. Build and run the delivered regulatory_reuse.cc in its normal CUDA mode. Adjust only coordinated API spellings or necessary setup, never weaken assertions, expected values or required execution.

2. Run Compute Sanitizer memcheck on the tiny demo and relevant native tests. A missing sanitizer/device is an explicit acceptance blocker, not a skip pass.

3. Confirm one topology preparation, two value publications, correct forward/transpose results, actual bound candidate names and rejection of a stale call before output changes.

4. Keep the reference-only fixture mode visibly distinct and exclude it from GPU completion. Describe normal run/build commands and the small scope.

## Completion evidence

1. Demo exits zero on real sm70 with no memory errors.

2. The default program never provides a dummy or reference fallback to pass.

3. The demo remains small: no optimizer step, training, new dataset, broad planner inspection or performance promotion.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
