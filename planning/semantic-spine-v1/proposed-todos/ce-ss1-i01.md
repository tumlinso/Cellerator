# CE-SS1-I01: Establish the bounded execution baseline and protected edit inventory

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Capture the actual pre-implementation source, available accelerator/toolchain, user-supplied example, and minimal forward/transpose regression baseline. Protect the two existing library edits.

## Prerequisites

None beyond manual execution authorization and the valid applied run.

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `tests/planner_targets.cmake`
- `tests/math_core/feature_major_small_n_candidate_test.cu`
- `tests/math_core/transpose_backward_candidate_test.cu`
- `library/cellerator/cellerator.ceh`
- `library/cellerator.cell`

## Exclusive write scope

- `docs/semantic_spine_v1/baseline.json`
- `docs/semantic_spine_v1/source_disposition.json`

## Implementation actions

1. Read the delivered review and decisions, and inspect only the affected execution, compiler and decomposition paths. Record precise hashes and current HEAD; do not interpret unavailable Todo health as an empty ledger.

2. Verify the example and package have been committed as the reviewed bootstrap input. Read and hash every current .ceh and the existing dirty-library content before any future conversion.

3. Resolve the installed CUDA 12.x toolchain and an actual sm70 device. Build and run existing feature-major and transpose tests under an exclusive device lease; record genuine pre-existing failures rather than disabling tests.

4. Record the anticipated C01/I02 contract publication and non-overlapping lane ownership. Escalate a missing GPU or broken prerequisite without starting a broad repair campaign.

## Completion evidence

1. Baseline names actual compiler, GPU capability, test command/exit status, source hashes and any pre-existing failure.

2. The provided demo is present at examples/semantic_spine_v1/regulatory_reuse.cc; no .ceh content has been lost.

3. No implementation dependency is marked complete from campaign status or capability-name tests.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
