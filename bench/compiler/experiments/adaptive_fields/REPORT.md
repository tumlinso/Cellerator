# Adaptive execution-field experiment decisions

Date: 2026-10-28
Source base: `c63f591d04ef74b7566937d161c80936dcc4cf05` plus explicitly retained
experiment source snapshots and the pre-existing working-tree state
Scope: isolated prototypes; no production API, default, numerical-kernel,
or application-algorithm changes

## Decision boundaries

| Experiment | Decision | What the evidence establishes |
| --- | --- | --- |
| E1: language/native connected computation | Supported within the local adapter | Real field parsers drive native CPU relation and segmented-softmax callbacks; production vector semantic lowering remains absent. |
| E2: semantic registration | Supported within the local adapter | A callable C++ helper and selected ATen addition share a mathematical descriptor; a local rewrite selector respects native-write and opaque barriers. |
| E3: dependency-specific reuse | Unresolved as production integration | Real support mutation/restoration and a materialized local artifact replay behave correctly; production caches do not yet implement that dependency model. |
| E4: runtime CPU compilation | Supported within the dependency-bounded prototype | Existing object compilation can produce a callable shared object with dynamic values, in-process and cross-process reuse, and explicit cache invalidation. |
| E5: recurrent trainable field | Unresolved | The current shared native transition matches an independent oracle, but Python parameter storage describes per-mechanism matrices. Full qualification stops at that interface ambiguity. |
| E6: executable replacement | Supported for bounded leases; overlap unresolved | Host and synthetic GPU tapes retain code through consumers; replacement module loading does not complete while the deliberately blocked stream remains held in this fixture. |
| E7: complete execution economics | Unresolved; campaign stopped at the user's requested wrap-up | Compiler comparison and six-path correctness qualification completed; stationary timing has only one exploration block for small and event fixtures. Reuse horizons, the ranked timing block, confirmation rounds, and GPU adaptation remain unqualified. |

These decisions distinguish experimental feasibility from existing production
functionality. They do not adopt a production implementation plan.

The user requested an early wrap-up for repository cleanup. The remaining E7
campaign was cancelled through Device Scheduler, with verified cleanup. This is
an explicit scope stop, not completion of the requested performance protocol.
Large generated payloads are retained locally under `/mnt/block`, without remote
upload; [LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json) records the verified archive,
each member's identity, and restoration instructions. Source, reports, and raw
JSON timing records remain suitable for Git.

## Material architectural findings

### Trainable parameter shape must be part of mathematical semantics

The C++ operation explicitly declares a shared transition
`z' = tanh(Az + Bu + b)`, with mechanism lanes in the state. Its CUDA and host
implementations use one `[width,width]` A/B pair and one `[width]` bias. Python's
parameter layout and binding accept `[mechanisms,width,width]` matrices and
`[mechanisms,width]` bias. Later parameter blocks are accepted but not used by
that operation. This is an interface/storage contract mismatch, not evidence
that the native shared-matrix equation is implemented incorrectly.

Direct and field-staged execution initially agreed because both invoked the
same native primitive. Independent replay of the exported six-step event trace
was necessary to expose the distinction. Following the documented shared
contract, final output maximum absolute error was about `3.05e-7`, loss delta
`2.39e-7`, and flat-owner gradient maximum absolute error `3.13e-8` against
the native artifact. A reference interpreting the allocated matrices as
independent per-mechanism parameters disagreed substantially.

The broader E5 branch stops here without changing scientific or numerical
semantics. Shared versus indexed/per-mechanism parameters must be reconciled
through the Cellerator operation, binding, and parameter-layout owners before
qualifying the full ranked/event recurrent field. Subsequent history/union,
structural publication, and checkpoint-continuation coverage is not asserted.
The relation-only E7 experiment does not invoke this transition.

Evidence: [independent estimator report](verification/estimator/README.md),
[exported contract boundary](gpu/evidence/e5_contract_boundary.json),
[native smoke arrays](gpu/evidence/raw/e5_seed20261027_current_event_all.npz),
and its [actual scheduler receipt](evidence/scheduler/e5_smoke_wait.json).

### Asynchronous compilation does not imply asynchronous module loading

E6 uses actual loaded modules, two CUDA streams, a queued old forward/backward,
one compilation worker, and an event/callback gate. The first unbounded gate
attempt stalled; it was cancelled through Device Scheduler with verified cleanup.
The diagnostic adds exception-safe release and bounded watchdogs.

Under memcheck, racecheck, synccheck, and a separate unsanitized run, replacement
module loading spanned approximately 9.99 seconds and returned only after the
10-second callback watchdog released the blocked stream. Therefore the fixture
does not establish module publication while the old launch is definitely still
pending. This observation is specific to the tested binary, driver, context,
and blocked-stream setup; it is not a universal claim about CUDA loading.

The serialized paths were sanitizer-clean and retained the old module through
its later backward launch. CPU lease/publication checks also passed. A consumed
request-key bug in the experimental CPU worker was independently found and
fixed, with retries after success, cancellation, and failure rechecked.

The v3 GPU JSON contains an unconditional pending-work success flag; that flag
is overbroad and must not be used as acceptance. Raw records are preserved and
the correction is explicit. The qualified v3 binary was subsequently reproduced
with its exact original hash and retained separately; a later metadata-corrected
binary was CPU-built but not GPU-qualified.

Evidence: [independent GPU review](verification/lifetime/E6_GPU_SOURCE_REVIEW.md),
[result correction](gpu/runtime/lifetime/RESULT_CORRECTION.md),
[sanitizer stage logs](evidence/runtime_lifetime_v3/summary.json), and
[unsanitized trace](evidence/scheduler/runtime_lifetime_normal_logs.json).

This favors investigating quiescent module loading/publication or switching
among already loaded modules. It does not justify a general live-code swap
service that assumes loading can overlap arbitrary pending work.

### Existing code generation is useful, but the shared semantic bridge is new

E1's source-derived graph changes execution when an operand changes; real CPU
primitives match an independent FP64 oracle over three deterministic fixtures.
E2 invokes a compiled ordinary C++ addition helper and the selected CPU ATen
overload, performs a local epilogue rewrite, and refuses write/opaque crossings.
Those adapters do not turn the scalar demonstration IR into a vector compiler.

E3 materializes index/projection, workset, and value operands. A value rebind
changes native numerical results without code/index/projection reconstruction;
gate close/reopen refreshes primal/derivative worksets; stale epoch/order handles
are rejected. Its final local counts are compile 1, index 3, projection 3,
workset 3, value operands 4, rebind 1, and packing 0. Packing reuse was not tested.

E4 invokes a real generated shared object with runtime values and inputs. A
second process reuses it. Numerical contract, compiler identity, and topology
tag changes invalidate the cache. The executable itself uses generic topology
loops; topology-key changes are not evidence of beneficial structural code
specialization. Include-content invalidation beyond the generated-source and
standard-toolchain boundary remains uncovered.

Observed CPU cold compile/link/load were roughly 153/35.8/0.06 ms, with a
second-process warm lookup around 0.26 ms. These are bounded probe observations,
not a comparative compiler benchmark or application speedup.

Evidence: [CPU report](cpu/REPORT.md), [structured CPU decisions](cpu/results.json),
and [independent artifact/lifetime verification](verification/lifetime/REPORT.md).

## GPU measurement protocol and retained exclusions

GPU execution is serialized on UUID `GPU-d745da9c-7649-8334-41f1-483afa6f3206`
through finite exclusive Device Scheduler jobs. Actual JobSpecs, submissions,
attempts, logs, and cleanup receipts are retained. The installed scheduler
supplies UUID routing but does not inject job/attempt environment variables;
correlation keys join results to the external authority records. Markers alone
are not admission or isolation evidence.

The initial E7 fixture used soft, unpruned gates and its derivative-only criterion
correctly failed. A permissive process exit code obscured that failure in the
first wrapper summary; both wrapper and runner now fail closed on a rejected
JSON decision. Those earlier numerical/protocol and sanitizer records are not
acceptance of the requested hard-gated workload.

The corrected small validation passes all six paths with exact identities and
masks: 124 primal items and 32 derivative-only items per candidate. It also
tests duplicate/unsorted posting queries and exact fallback, forward/VJP/update
comparisons, and the real owner/Adam checkpoint-reset protocol. New bridge
memory, race, and synchronization checks pass for this corrected workload.

For clean timing, KDA orchestrates a trusted six-way collector over three frozen
fixtures and four rounds of 21 paired samples, with five warmups. Its single
collector preferred state is not a numerical winner; comparisons use actual
candidate/baseline pairs. Identical owner/Adam state is restored outside each
paired interval, with harness restore cost separately recorded. Cold costs
separate baseline, candidate, and joint comparison preparation. Peak allocation
in the paired process is explicitly joint, not isolated candidate memory.

Measured 1/16/256-use lifetimes and the adaptive support/density trace are distinct
from modeled amortization. No historical report, multiplication of a median,
precomputed workset, or supplied cost constant substitutes for their execution.

The first KDA attempt passed all three fixtures' six-candidate validations but
was cancelled before producing a complete timing block. Its sample-reset path
called full `load_checkpoint()`, which restores the numerical owner and then
reconstructs sparse support, rebinds the handle, and rebuilds endpoint indexes.
That is inappropriate granularity for fixed-structure paired samples. The
existing `NativeOwnerProtocol.snapshot/restore` is the numerical subpath already
used by the full loader; the experimental reset can reuse it while independently
checking unchanged topology, owner identity, and optimizer state. This is further
evidence for separating recovery, structural preparation, and value restoration.
The cancelled attempt is not pooled into later performance evidence.

The numerical-only adapter subsequently passed actual full-restore parity for
parameters, Adam state, generations, native forward/VJP/update, and stale-epoch
rejection. Its six-candidate fixture passed three deterministic seeds and
memcheck, racecheck, and synccheck before the second KDA timing attempt. Evidence:
[final reset qualification](evidence/e7_final_qualification_v2/summary.json).

### Runtime compiler sub-probe

The clean same-source GPU region compiled in about 449.84 ms through NVCC and
19.00 ms through NVRTC. Precompiled NVCC and NVRTC module loads were about
0.059 and 0.126 ms respectively. Both implementations exactly matched the
fixture's exported reference, including after numerical value rebinding.

Across 84 alternating paired executions, median synchronized wall times were
0.01326 ms (NVCC) and 0.01336 ms (NVRTC). A paired two-level bootstrap interval
for the mean NVRTC-minus-NVCC difference crossed zero
(`[-0.000146, 0.000343]` ms). This demonstrates a feasible runtime compilation
route, not an execution advantage for recompiling an otherwise equivalent
region. Structure-specific code would need a separately demonstrated benefit.
Compile times are application-cold observations; persistent driver/OS caches
were not cleared. This region is a small experimental kernel, not a training
throughput benchmark or a production code-generation integration.

Evidence: [comparison](evidence/runtime_clean/comparison.json),
[paired analysis](evidence/runtime_clean/analysis.json), and
[terminal admission/cleanup receipt](evidence/scheduler/runtime_clean_terminal.json).

### Stationary economics: retained exploratory evidence only

The second KDA attempt completed one six-candidate exploration block for the
small and large-event fixtures: five warmups and 21 paired samples per candidate.
All three fixtures passed six-path validation, but ranked timing was interrupted
before its aggregate block completed. Two exploration and two confirmation
rounds were not completed. The partial records cannot establish a qualified
winner, a break-even lifetime, or a production dispatch threshold.

The [partial summary](evidence/e7_partial_summary.json) explicitly marks every
row `complete_protocol: false`. It retains per-pair complete training wall costs,
separate cold setup, and exploratory uncertainty. A single block cannot estimate
variation between rounds; its bootstrap interval must not be interpreted as the
requested confirmation evidence. Partial ranked tensors/checkpoints survive in
the local archive, but are not substituted for a completed timing record.

Actual 1/16/256-use measurements were not run. The GPU low/high/low trace and its
cost calibration are source-only, unqualified prototypes; the existing CPU trace
establishes a support lifecycle, not GPU adaptation economics. The GPU trace
currently covers only two indexers, runs one fixed comparator, and conservatively
charges posting preparation again in its switch hurdle after preparing it.
Those limitations require resolution before claiming either a fixed-candidate
oracle or an adaptive policy result. No modeled costs stand in for measurements.

Evidence: [cancelled job and verified cleanup](evidence/scheduler/e7_kda_v2_terminal.json),
[GPU report](gpu/REPORT.md), and [trace source boundaries](gpu/trace/README.md).

## Smallest justified production follow-up

These are discussion recommendations, not changes made by this campaign:

1. Reconcile transition sharing, tensor shapes, and trainable parameter coverage
   before extending recurrent field qualification.
2. Establish one explicit semantic-operation and binding contract, including
   parameter sharing, effects, derivative/sampling policy, precision, identity,
   and dependency-specific generations; connect existing native owners to it.
3. Carry those dependencies through prepared artifacts and inspectable plans,
   rather than consolidating all indexes, projections, and caches at once.
4. Extend the existing object/code-generation route behind that contract, with
   bounded compilation and explicit executable lifetimes. Defer ORC, arbitrary
   Torch lowering, general C++ inference, and automatic gate-driven recompilation.
5. Separate background compilation from module loading and publication. Use
   measured safe boundaries and preloaded candidates where appropriate.

Biological applications retain their scientific meaning, eligibility and
approximation choices. These synthetic probes establish no biological validity.
