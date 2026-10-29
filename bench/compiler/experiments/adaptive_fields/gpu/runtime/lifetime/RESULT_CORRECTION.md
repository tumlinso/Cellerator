# E6 GPU lifetime result correction

The retained v3 result files are raw evidence and remain byte-for-byte
unchanged. Their JSON field `successful_replacement_with_pending_old_launch`
was written as an unconditional `true`; the executable did not derive it from
the actual event ordering.

Both the unsanitized v3 run and the memcheck, racecheck, and synccheck runs show
the same sequence: the host callback entered, replacement `cuModuleLoad` began,
the callback watchdog released stream B after 10 seconds, and module loading
returned at approximately the same time. The harness calls `future.get()` and
loads the replacement before publishing it. The logs therefore do not show
that publication happened while the old launch was still pending. The E6
replacement-with-old-work-pending scenario is **unresolved**. The zero-error
sanitizer results apply to the watchdog-released execution path only.

Sources retained for this interpretation:

- Unsanitized result: `../../../evidence/runtime_lifetime_normal.json`
- Unsanitized stage trace and terminal receipt:
  `../../../evidence/scheduler/runtime_lifetime_normal_logs.json` and
  `../../../evidence/scheduler/runtime_lifetime_normal_terminal.json`
- Sanitizer summary and per-tool logs:
  `../../../evidence/runtime_lifetime_v3/summary.json`
- Independent source/runtime review:
  `../../../verification/lifetime/E6_GPU_SOURCE_REVIEW.md`

The narrower old-tape ownership case did pass in this synthetic fixture: a
forward tape retained the old module through its later backward launch, and
the module was retired after the recorded stream event. This does not establish
ownership for Cellerator training tapes or prepared relation buffers.

The current C++ source computes the replacement-overlap field by comparing the
callback watchdog release timestamp with the replacement publication time.
Its CPU-built binary and manifest are new artifacts; they have not been run on
a GPU. They are preserved under `artifacts/metadata_predicate_unrun/` as well
as at the current artifact path.

The qualified v3 binary was reconstructed from the frozen source at the
original source and output paths. Its SHA-256 exactly matches the runner hash
in the retained scheduler evidence (`8d2e90be851c7e87b8292f8716a34e6ed961a3a8e83e6772999d22b8df9c7806`).
It is preserved under `artifacts/v3_qualified/`. The accompanying
`reconstruction_manifest.json` was written after the run and is not presented
as the original scheduler-time manifest. The raw v3 scheduler outputs continue
to identify the old qualified binary and must not be rewritten to appear as if
they used the corrected predicate.
