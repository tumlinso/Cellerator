# CE-SS1-C02: Implement shared semantic validation and fieldwise equivalence

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-CORE`. **Repository:** Cellerator.

## Objective

Both native and compiler callers must use the same relation semantics checks and equality rules.

## Prerequisites

`CE-SS1-C01`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `src/compute/operation/relation_semantics.cc`
- `tests/semantic_spine/core/descriptor_test.cc`

## Implementation actions

1. Implement explicit fieldwise comparison and validation; never compare padding or use pointer addresses as persistent keys.

2. Separate invalid semantics from valid-but-unimplemented physical capability. Recognize higher width or alternate arithmetic without silently executing the N1 contract.

3. Define reduction/FMA permissions independently from test tolerances. Do not convert tolerance-based testing into blanket permission to lower storage precision.

4. Check extents/count arithmetic and orientation consistency. Do not scan device values or support during hot validation.

## Completion evidence

1. Reject altered high identity bits, mismatched axes/order/epoch and illegal aliases; accept permitted equivalent descriptors from independent construction.

2. Distinguish zero dimensions/empty support from invalid pointers, with no unchecked 64-to-32-bit truncation.

3. Changing runtime value generation does not alter the topology preparation key.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
