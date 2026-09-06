# CE-SS1-I02: Integrate the foundation and release parallel lanes

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Make the tested canonical contract available in one shared source base and publish its internal interface.

## Prerequisites

`CE-SS1-C04`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `include/Cellerator/compute/operation/relation_semantics.hh`
- `src/compute/operation/relation_semantics.cc`
- `include/Cellerator/compute/operation/prepared_relation.hh`

## Exclusive write scope

- `CMakeLists.txt`
- `cmake/SemanticSpineV1.cmake`
- `docs/semantic_spine_v1/foundation.json`

## Implementation actions

1. Integrate C04, add only the opt-in build wiring for the semantic contract, and verify its leaf tests.

2. Record the integrated foundation commit in the checkpoint. Materialize or reconcile downstream lane workspaces from this commit only after integration, never from stale pre-foundation work.

3. Publish CE-SS1-I-RELATION version 1 from the actual header files. This freezes an internal working agreement for this epic, not the future public API.

4. Release N01, F01, A01 and V01 together; I03 may proceed independently on library file disposition.

## Completion evidence

1. A clean checkout of the foundation compiles the real shared validator.

2. The four downstream lanes are simultaneously ready in the combined dependency/serial-queue graph.

3. Every consumer sees the same interface hash and foundation revision.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
