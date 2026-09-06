# CE-SS1-A04: Publish bounded algebra conformance and migration disposition

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-ALGEBRA`. **Repository:** Cellerator.

## Objective

Close the semantic repair lane with executable tests and an explicit list of remaining operation work.

## Prerequisites

`CE-SS1-A03`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/algebra`
- `docs/semantic_spine_v1/algebra_validation.json`
- `docs/semantic_spine_v1/algebra_disposition.md`

## Implementation actions

1. Run focused algebra tests, plus the affected existing decomposition/compiler tests after correcting their invalid mathematical assumptions.

2. Document each scoped changed symbol, retained implementation and rejection case.

3. Explicitly defer full contraction candidate plumbing, general composition lowering and algebra redesign.

4. Hand I04 a committed artifact that does not require edits to the native/frontend lane-owned files.

## Completion evidence

1. The tests compute and compare actual partial results, not text labels.

2. All intentional behavior changes are tied to counterexamples.

3. The foundational apply interface remains unchanged or any necessary revision has been coordinated with every consumer.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
