# CE-SS1-V03: Specify integrated origin and GPU acceptance probes

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-VERIFY`. **Repository:** Cellerator.

## Objective

Prepare cross-origin and hardware probes against the published interfaces while implementation lanes progress independently.

## Prerequisites

`CE-SS1-V02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/verification/integrated_probe.cc`
- `tests/semantic_spine/verification/evidence_requirements.json`

## Implementation actions

1. Write tests that consume compiler-returned and native-constructed descriptors, compare semantics, and submit both through the same actual accelerator path after integration.

2. Require generation-reuse and stale-generation rejection checks; verify accepted launch counts and untouched sentinel output on failure.

3. Record genuine device capability and actual launch witness. Host-only or unavailable CUDA is not a successful GPU gate.

4. Specify per-run provenance and accepted evidence fields. These tests may be link-incomplete until I04; no stub backend may satisfy them.

## Completion evidence

1. The tests are ready to link against published interfaces without editing lane-owned production files.

2. Both orientations and both value generations are demanded from both origins.

3. No pre-generated pass receipt or expected-output echo is present.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
