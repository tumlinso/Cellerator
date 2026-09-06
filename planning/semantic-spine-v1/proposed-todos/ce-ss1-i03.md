# CE-SS1-I03: Retire .ceh files without expanding the language milestone

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Apply the agreed file-format decision while preserving valuable source and recording deferred language infrastructure.

## Prerequisites

`CE-SS1-I02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `library`
- `stdlib`
- `docs/semantic_spine_v1/cell_units.md`
- `docs/semantic_spine_v1/ceh_disposition.json`

## Implementation actions

1. Inspect all tracked/untracked project-owned .ceh files locally; source-reader failure is not evidence they are empty. Preserve pre-existing user bytes and compare with the I01 inventory.

2. Remove only truly redundant files; convert meaningful definitions to .cell and update direct references. Do not duplicate definitions already in an umbrella.

3. Retain ordinary .hh headers. Do not implement a module loader, implicit-dialect frontend or cross-TU optimizer here.

4. Record .cell as the desired self-describing/importable source format, and mark full .cell execution/implicit mode/cross-TU infrastructure as future work. Only narrowly related references/manifests may change.

## Completion evidence

1. No active project-owned .ceh file remains; every previous file has a content-preservation or redundancy disposition.

2. Historical review/planning mentions need not be erased.

3. No library expansion, public SDK redesign or broad resource-layout migration occurred.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

If a .ceh contains substantial inseparable implementation, preserve it as .cell first and document missing language support. Never delete unread content. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
