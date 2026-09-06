# CE-SS1-A03: Update affected consumers and negative conformance tests

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-ALGEBRA`. **Repository:** Cellerator.

## Objective

Make the scoped classification and contraction corrections active in real consumers rather than descriptive comments.

## Prerequisites

`CE-SS1-A02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `src/compiler/ir/semantic/implement_bundle_chain_moments_hierarchy_and_exchange_op.cc`
- `src/compiler/ir/semantic/implement_contraction_segment_and_normalization_operatio.cc`
- `tests/semantic_spine/algebra`

## Implementation actions

1. Find bounded actual readers of the changed resolution and support-embedding contracts. Update only those consumers; report broader fallout for integration rather than rewriting unrelated subsystems.

2. Unknown/unimplemented composition lowering must retain composition meaning or return unsupported, never use a convenient substitute primitive.

3. Update old tests that asserted the wrong law with explicit migration rationale, retaining negative counterexamples.

4. Keep working sequence-predicate and other v1 implementations reachable; absence from a new enum is not deletion authorization.

## Completion evidence

1. A test which previously accepted dot split-K concatenation now fails for the right reason.

2. A chain/exchange/gradient cannot reach an apply/update launch by merely erasing its composition tag.

3. No opcode renumbering or schema replacement affects unrelated native implementations.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
