# CE-SS1-I05: Run cross-origin semantic and real sm70 conformance

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Prove the epic’s central assertion with one complete execution witness.

## Prerequisites

`CE-SS1-I04`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `docs/semantic_spine_v1/origin_execution.json`
- `tests/semantic_spine/integration`

## Implementation actions

1. Run host contract/algebra checks and the source-origin/native-origin descriptor comparisons.

2. Execute both origins through the same prepared accelerator implementation for forward and transpose at two value generations.

3. Verify dimensions, edge identity, result numerical tolerances, stale failure, projection reuse and source/launch provenance.

4. Record actual GPU/toolchain and compare direct native candidate baselines to ensure the adapter has not silently moved work to CPU or added full scans/synchronizations.

## Completion evidence

1. Real GPU arithmetic results pass, not merely descriptor serialization.

2. Wrong source bindings and stale generations fail; valid high-width unsupported cases do not fake success.

3. The full .cell driver is still reported deferred, even though source-derived descriptors execute via the native API.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
