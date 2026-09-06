# CE-SS1-F04: Publish a source-origin conformance test artifact

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-FRONTEND`. **Repository:** Cellerator.

## Objective

Provide a real origin-parity test route and candid language capability status.

## Prerequisites

`CE-SS1-F03`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/compiler`
- `docs/semantic_spine_v1/compiler_validation.json`

## Implementation actions

1. Create the lane-local compile/run test harness and record actual parser, Sema and lowering symbols exercised.

2. Compare fieldwise native/source descriptors for forward and transpose using a non-square relation fixture.

3. Provide I05 with a function that returns the source-derived descriptor for execution through the same native prepared pair. Do not add an independent reference launch path.

4. Record full .cell executable compilation, implicit .cell mode and cross-TU composition as deferred, separately from this passing embedding test.

## Completion evidence

1. Meaningful altered-source and wrong-binding tests fail rather than only checking operation names.

2. No language-complete or installed-SDK-complete claim appears in the receipt.

3. The I04 artifact contains all necessary source-facing changes and focused tests.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
