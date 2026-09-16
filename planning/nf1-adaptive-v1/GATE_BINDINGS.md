# External acceptance bindings

These are execution configuration, not a second Todo authority. The root configures actual current tests/evidence and may replace this small adapter with an equivalent existing registered harness through an explicit amendment. It must preserve completion prerequisites, source/device identity, no-skip behavior and evidence honesty. Do not create a new testing platform merely to populate these fields.

The external JSON created by preview has `format`, `pair_id`, `project`, `evidence_dir` and `tasks`. All paths below are concrete host paths discovered by the controller, not strings to pass literally. No shell strings or guessed service flags are required.

## Shared runtime fields

`gpu_lease_verifier_argv` is the observed verifier's argv array, with `{lease_receipt}` as the receipt-path substitution. Include its actual project/device arguments. `shared_gpu_lock_file` is one host path shared by both projects; its parent must exist. `peer_bindings` points to the other project's binding file, whose lock path must match. `evidence_dir` is outside both repositories. These fields are needed for execution outcomes, not administrative adoption.

## Execution task object

```json
{
  "build_dir": "/actual/external-or-owned/build",
  "sources": {
    "cellerator": {"path": "/actual/qualified/cellerator/worktree", "commit": "actual-40-hex-head"},
    "glasshelix": {"path": "/actual/qualified/glasshelix/worktree", "commit": "actual-40-hex-head"}
  },
  "ctest_names": ["actual_registered_test_name"],
  "binary_paths": ["/actual/binary", "/actual/library-used-by-the-test"],
  "coverage": {
    "GH-ROLES": {"tests": ["actual_registered_test_name"], "evidence": []}
  },
  "timeout_seconds": 3600
}
```

Only include source projects actually used, but every GlassHelix execution outcome must bind Cellerator. The gate's working directory and CMake home must identify the owning tested worktree. `binary_paths` must cover actual executables and consumed libraries, not merely Python or CTest. Review actual linking/build dependency evidence; naming a commit in this object alone does not prove linkage.

Every requirement assigned to the outcome needs coverage with actual selected test names and/or hashed external evidence references `{ "path": "...", "sha256": "..." }`. A performance requirement can use the actual measurement report; a scientific convention can use reviewed analytic evidence alongside tests. Optional classifications alone may use `disposition: "not_selected"`, a reason, and actual supporting evidence. No required CUDA, exact-compaction, derivative, inference or Python capability may be excused that way.

CE QUALIFY and GH ACCEPT include independent oracle/counterexample, sanitizer and whole-program performance evidence. Test organization and names are delegated. Required tests must really run; exact-inventory matching rejects absent, duplicate, disabled, skipped or failed cases. Do not rerun all historical suites after every edit. Keep the actual retained/new cases affected by the final integrated source.

The runner seals the selected task and shared runtime configuration, not unrelated task-binding entries. Other agents updating their own entries need not invalidate an in-progress gate. Editing the current task, shared lock/verifier configuration, compiled source, libraries or build cache during its gate does invalidate it.

## ADOPT task object

Use `adoption.import_reviews` with a hashed CE and GH review reference; `adoption.preservation` with the hashed capture receipt; `adoption.retirement_receipts` with the two canonical retirement receipts; and `adoption.artifact_dispositions` with records containing project, workspace_id, successor, source_reference, reason and action.

Actions are `preserved_for_successor`, `already_integrated`, or `historical_no_source`. Every workspace in the capture receipt needs a disposition. The gate checks old-run cancellation, successful historical-row preservation, successor presence and archive/reference integrity. The root must inspect ancestry/actual source before declaring something already integrated. There is no magic Git migration performed by the gate.

## RELEASE and CONTROL task objects

`accepted_receipts` is an array of hashed passed NF1A outcome-gate receipts. CE RELEASE requires CE QUALIFY and GH ACCEPT; GH RELEASE requires GH ACCEPT and CE RELEASE; each CONTROL requires its own RELEASE. RELEASE additionally supplies the exact `sources` pair, a hashed `source_disposition_report`, and `remaining_limitations` (an explicit list, possibly empty after review).

The release check queries actual remote origin/main refs and requires the accepted source pair. A source change after acceptance requires a new qualified receipt under proper native authority; do not edit an old receipt or rewrite a historical completed task. Administrative closure reuses tests only when the relevant source/build/dependency identity remains supported.

## Limits of this adapter

It checks provenance and actual execution, not the intellectual truth of a test. It cannot prove that a newly written test catches every scientific mistake, that an operator-provided library list is exhaustive, or that an external performance report measures the right workload. Independent numerical/scientific review remains mandatory at the consequential integrated boundary. These helpers were tested with disposable local fixtures; real installed-kernel integration and GPU execution remain host qualification steps, not claims of this delivery.
