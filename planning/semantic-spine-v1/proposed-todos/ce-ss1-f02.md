# CE-SS1-F02: Connect the real parser/Sema relation slice

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-FRONTEND`. **Repository:** Cellerator.

## Objective

Demonstrate existing source parsing and semantic construction reach the shared descriptor for a bounded source fixture.

## Prerequisites

`CE-SS1-F01`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `src/compiler/frontend/parser/parse_non_relation_operation_families.cc`
- `src/compiler/frontend/parser/expose_parser_library_and_parse_tree_dump_apis.cc`

## Exclusive write scope

- `src/compiler/sema/relation_spine_bridge.cc`
- `include/Cellerator/compiler/sema/relation_spine_bridge.hh`
- `tests/semantic_spine/compiler/source_slice_test.cc`

## Implementation actions

1. Use existing parser/AST/Sema facilities for a small forward expression and transpose operation. Bind named domains, relations, state and exact identities through the existing environment or a narrowly defined binding adapter.

2. A fixture may supply runtime identities because source alone cannot know the loaded dataset; it may not precompute or substitute the expected operation descriptor.

3. Keep this an embedding/test route, not a replacement command-line compiler. Report unsupported source honestly.

4. If a narrow parse-to-IR connection is missing, add that connection without broad grammar completion.

## Completion evidence

1. Changing direction, symbol binding or numeric declaration in the source changes the descriptor or produces a relevant error.

2. The negative source fixture cannot pass through a hard-coded operation table or name-only assertion.

3. The full .cell executable route is still explicitly incomplete.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
