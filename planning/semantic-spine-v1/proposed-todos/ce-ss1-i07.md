# CE-SS1-I07: Close the limited epic and preserve deferred obligations

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-INTEGRATE`. **Repository:** Cellerator.

## Objective

Publish an evidence-backed completion receipt and stop at the agreed boundary.

## Prerequisites

`CE-SS1-I06`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `docs/semantic_spine_v1/completion.json`
- `docs/semantic_spine_v1/deferred.md`
- `docs/semantic_spine_v1/source_disposition.json`

## Implementation actions

1. Re-run the integrated acceptance target suite and inspect actual pending artifacts/lane status before closing the epic.

2. List implemented semantics versus recognized-but-unimplemented operations and record every touched legacy implementation’s retained/adapted/replaced disposition.

3. Keep explicit obligations for full .cell-to-device execution, implicit .cell mode/imports/cross-TU optimization, broader dtype/width handling, full contraction/compositions, packaging, public API and performance work.

4. Commit/push completed work and integrate this epic only under later execution authorization; do not delete unrelated branches or auto-start another epic. Stop promptly on the user’s pause request.

## Completion evidence

1. All required real executable gates pass with exact provenance; no missing GPU evidence is papered over.

2. No active .ceh remains and useful code has not been discarded without a justified equivalent.

3. The completion statement explicitly denies full-language/SDK completion and performance supremacy.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
