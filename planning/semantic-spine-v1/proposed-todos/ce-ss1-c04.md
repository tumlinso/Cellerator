# CE-SS1-C04: Test the contract and publish the foundation artifact

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-CORE`. **Repository:** Cellerator.

## Objective

Prove the shared descriptor and provisional signatures are coherent before the parallel fan-out.

## Prerequisites

`CE-SS1-C03`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/core`
- `docs/semantic_spine_v1/core_validation.json`

## Implementation actions

1. Add a standalone, small host contract test build and declaration compilation checks for native/compiler consumers. Keep CPU code as an oracle/validation facility.

2. Test forward/transpose duality at the mathematical level, identity permutations, exact support, and recognized unsupported modes. Cover fieldwise copy/move and comparison.

3. Commit the core artifact and hand it to I02 with source hashes and real commands. Publish no installed/public ABI stability claim.

4. Report ambiguities now; do not let each downstream lane invent its own descriptor.

## Completion evidence

1. Contract tests run, not merely compile or search for names.

2. Core artifact contains both headers and validator implementation; all downstream signatures are visible.

3. I02 receives a concrete source revision; no consumer starts from an unpublished draft.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
