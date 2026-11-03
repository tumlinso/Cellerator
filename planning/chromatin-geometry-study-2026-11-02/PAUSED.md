# Chromatin execution study pause

**Latest: bounded probe paused after qualification packet failure on
2026-11-03.** The focused qualifier and four sanitizers passed, but the first
full-chain result was rejected for a native/adapter descriptor-label mismatch.
No timing ran. One of three GPU job slots and 25.589826107 of the same 600
GPU-seconds were consumed; no allowance reset. Cleanup is verified, all probe
lanes and the parent remain paused, and no reservation is open. See
[the bounded result](../../bench/chromatin_geometry/analysis/grouping-probe-result.md).

**Bounded probe explicitly resumed on 2026-11-03.** The user authorized fixing
the two packet issues and running qualification plus R1/R16 timing, retaining
the same 600 GPU-second/three-job allowance and pausing afterward. The broader
campaign and parent workspace remain paused. See
[the bounded resumption record](bounded-grouping-resume-2026-11-03.json);
the preparation checkpoints below remain historical.

**Latest checkpoint: bounded grouping continuation paused at its CPU-readiness
limit on 2026-11-03.** No GPU job was submitted. The study budget remains
unchanged at 2,332.120188475 GPU-seconds used, with all probe lanes inactive
and paused and no open reservation. See the
[bounded checkpoint](../../bench/chromatin_geometry/analysis/grouping-probe-checkpoint.md)
for retained source/binaries and the two unresolved pre-timing review findings.

**Paused again on 2026-11-03 for the user's strategy discussion.** All fresh
study lanes are inactive and paused, with no open GPU reservation. The study
has used 2,332.120188475 GPU-seconds (38.869 GPU-minutes). The
[current checkpoint](pause-resources-2026-11-03.json) supersedes the resource
figures below; the earlier checkpoints remain historical. No automatic
resumption is authorized during this discussion.

**Resumed by the user on 2026-11-03.** The historical pause checkpoint below
is retained. Resumption continues the original approved study and shared
240 GPU-minute limit; the parent workspace's separate pause remains in place.
Current measurements use fresh inactive study-only lanes and the same ledger.
See [resumption notes](resumption-2026-11-03.md).

The user first requested "Finish or pause asap" and then clarified
"After the next milestone" on 2026-11-02. The bounded milestone attempt is
finished and the study is **paused**. All four v3 sanitizer modes passed.
The paired timing attempt recorded all 48 base and 96 update records, but its
telemetry detected an unattributed foreign GPU process, so the timings are
unqualified. No retry was submitted. Resume only with explicit user direction;
preserve the parent workspace's independent Autobench pause.

The [interim report](../../bench/chromatin_geometry/analysis/report.md) states
the current result and limits. The standalone native implementation, public
replay, independent references, measurement tools and immutable captures are
retained. Nothing is promoted to an installed API or production default.

## Accepted bounded result

One v2 teacher/FP32 grouped-linear/public-replay/incremental scenario has six
qualified paired rounds. Neither cohort arm clears the strict greater-than-5%
latency improvement gate over bitmap, its strongest matched control. The
original one-tile cached and two-tile shared arms have different ownership;
this historical result does not isolate staging.

The v3 cached-two-tile arm matches shared staging's tile ownership. Its
component correctness, eight-arm complete public-replay correctness, and all
four representative sanitizer modes passed with verified Scheduler cleanup.
The sanitizer modes each cover all six gate/body cases on the exact retained
v3 source/binary. V3 paired performance remains unqualified. The earlier v2
sanitizer evidence retains its own identities.

## Resume frontier

1. Read this handoff, the [protocol](protocol.md), [workflow exception](workflow-exception.md),
   and the current study controller ledger. Reconcile current Scheduler state
   before any submission; do not infer authority from visible devices.
2. Preserve contract freeze v3, SHA-256
   `4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06`,
   and the predeclared measurement matrix v3. Numerical/workload bounds did not
   change from v2. Use the retained coherent source closure and verified binary
   rather than a mixture of current and historical files.
3. Retain the four passing v3 sanitizer receipts and independently joined
   source/binary identities. Broader tile/buffer coverage remains separate;
   do not launch directly on unleased devices.
4. Reattempt the predeclared balanced same-packet staging and rebuild/incremental
   comparisons only after checking interference and supported admission state.
   Keep the rejected packet and its contamination sample. Same-mode strongest-
   control decisions remain separate from staging and update ablations. STE
   predicate-input updates remain inapplicable.
5. Complete the remaining eight-scenario matrix and supported tile/buffer
   exploration under the shared aggregate budget. Retain unsupported tile128/
   two-buffer capacity rejections. Do not pool devices or change tolerances.
6. Collect representative direct/cached-two-tile/shared profiles with actual
   primal/derivative membership counts. The old product profile follows a
   100% input flip and is sparse, so it does not demonstrate dense HBM reuse.
   No padded-stride implementation is justified by that profile alone.
7. Regenerate selected-contract qualification and cost exports, complete the
   report, and obtain final independent review/verification. Do not label the
   study complete while required coverage or cleanup is unresolved.

The native Todo lifecycle remains unresolved under the user-approved exception.
No authority SQLite, generated task projection or foreign claim was edited to
simulate acceptance. Scientific evidence and native task state remain distinct.

[Preservation checks](pause-preservation.json) confirm the protected geometry
hash is unchanged. Five previously dirty file hashes differ from the initial
manifest; this study's owners worked in the private study/planning directories.
Those concurrent differences are retained without attributing or reverting them.
Five original directory entries lacked individual baseline hashes, so this
manifest does not claim byte equality for their contents.

The [final post-milestone receipt](pause-resources.json) records
2,052.714808464 GPU-seconds used (34.212 GPU-minutes), zero open/unknown GPU
reservations, verified cleanup, and all four study lanes explicitly paused
and inactive. The parent Autobench remains paused. The
[pre-milestone receipt](pause-resources-pre-milestone.json) is retained.

During bounded resumption, `autobench resume` unexpectedly auto-submitted two
staged jobs before one-shot mode was restored. Both exact jobs were cancelled
and retained: one consumed 22.412 GPU-seconds and is unqualified; the other
had no attempt or GPU allocation. The ledger records the incident and charges
the observed execution. Do not call `resume` expecting it to preserve inactive
one-shot behavior. Check supported admission state before any future submission.
The 240 aggregate GPU-minute limit remains in force across resumptions; a pause
does not reset consumed time.
