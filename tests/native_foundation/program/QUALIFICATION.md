# Prepared program qualification

Status: P01 host source qualification. Native acceptance is recorded externally.

P01 extends `execution/program/program_v2.h` by borrowing the canonical
`prepared_binding_contract` and `launch_bindings` owners. Callbacks retain their
signature and see typed input/output/value arrays through `binding.typed`.
All stages undergo structural and dynamic validation before any callback runs.
The runner never owns an allocator, stream, device context, scheduler, or operand
storage. This is host execution qualification; no CUDA performance is claimed.

Both new pointer members are appended and default null, preserving source use of
legacy aggregate initializers and callbacks. Their addition changes C++ object
layout: rebuild producers and consumers together. These pointer-bearing structs
are transient process objects, not serialized records. Program version 2 is
retained for source compatibility, not binary compatibility with older builds.

Typed mode requires both pointers. It rejects mixed legacy payload pointers,
divergent workspace descriptors, and a stream different from the caller's.
Existing axis, structure epoch, value generation, output effects, and alias
validation remain owned by `execution/launch_bindings.hh`. This initial owner
requires at least one real bound relation structure even for dense callbacks.
Prepared effect declarations express callback obligations: validation does not
implement numerical effects on behalf of the callback or prove its behavior.
Dense views do not carry allocation bounds; no capacity claim is made here.

`ce_nf1_p01` executes three dense inputs and one value plane into two distinct
outputs, with exact integer-valued float arithmetic and overwrite/accumulate
effects. It rejects stale generations, axis/count mismatch, forbidden alias,
invalid effect, ambiguous legacy/typed payloads, missing paired contracts and
stream mismatch. An invalid final stage leaves callback counters and earlier
outputs unchanged. A callback failure after dispatch cannot roll back prior
callbacks; callers must keep all descriptors immutable during execution.

`ce_nf1_program_legacy` compiles the unchanged existing
`tests/jbc/fragment/canonical_relation_apply_smoke.cc` and its real canonical
owners, exercising fragment compilation, external value binding, numerical
execution (23, 18), and output affordance publication.

Standalone qualification:

```sh
cmake -S tests/native_foundation/program -B /tmp/ce-nf1-program -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/ce-nf1-program -j 4
ctest --test-dir /tmp/ce-nf1-program --output-on-failure --no-tests=error
```

Four build jobs bound this small host test while other native lanes compile
concurrently on the shared controller host. No GPU reservation is needed.

## Host binding correction

Authority scope amendment 7155 and task brief v2 (7156) admitted the canonical
launch-binding owner correction. Host stream ordinal -1 requires null stream
and host workspace; device ordinals retain matching workspace-device checks.
All tests now compile against repository headers, without an include overlay.
The earlier partial checkpoint and prospective overlay are historical evidence.

## P02 dynamic preflight

`preflight_prepared_program_v2` validates the entire stage graph and all dynamic
bindings without launching work. `execute_prepared_program_v2` calls it every
time, then dispatches the same existing callbacks. `prepared_stage_v2::preflight`
is a borrowed, pure operation-owner callback using immutable prepared state plus
`launch_binding_v2::validation_state`. Owners use it for capacities, layout and
range-alias requirements which erased legacy pointers and typed axes cannot
represent. Legacy bindings retain their original callback validation obligations;
null preflight does not certify bounds. Neither dynamic validation nor its result
is cached. A preflight callback must never enqueue, allocate or mutate payloads.

`ce_nf1_p02` composes the existing `external_binding_v1` extent validator with a
concrete host operation's required count, order, address and range-alias checks.
Two dependent stages execute a real add-one calculation. Insufficient final
capacity, wrong order, partial overlap, absent validation metadata, missing
workspace and missing binding leave every output and enqueue counter untouched.
Changing accepted extents before a later execution is revalidated. The external
lease/readiness tokens in this host fixture test descriptor validity only; live
session ownership and CUDA readiness remain subsequent P03 qualification.

## P03 native session attachment

`program_session_v2` borrows one initialized, sealed `execution_session` and an
immutable `prepared_program_v2`; it neither owns streams nor allocates scratch.
Each native session stream is one program instance, with disjoint native scratch.
The existing `relation_value_readiness` owns event publication and external
reader completion for each instance. Host calls through the adapter use a
nonblocking atomic guard; reentrant/concurrent calls return busy. The borrowed
session/program and callback state must remain alive and immutable until checked
close succeeds. Direct session mutation or destruction while attached is forbidden.

`ce_nf1_p03` constructs a real two-stream CUDA session using existing runtime
initialization, persistent/transient allocation, library preparation and sealing.
Canonical program callbacks run a numerical kernel using independent scratch.
Controls reject foreign scratch, device mismatch, a sibling reader ticket,
mutation during an outstanding reader, and checked close during a live borrow.
A delayed consumer reads old scratch while a later owner execution changes it;
checked close observes completion before the caller clears the native session.
The callback reenters the adapter to prove host serialization rejection.

The task's host-labelled gate is strengthened by a native GPU lease plus the
shared cross-repository lock. Its supplemental CTest runs Compute Sanitizer.
These tests require CUDA 12.9 and the actual leased GPU. This section describes
required cases; authoritative passing evidence is external and produced only
by native finish after a clean committed rebuild.

P03 session ABI adds an exclusive borrower token to the existing runtime session.
Cold native session calls remain externally serialized. Checked close refuses a
live attachment; legacy void cleanup preserves it. Reinitialization refuses an
initialized session rather than overwriting owned handles. Attached session storage
must remain at a stable address; raw aggregate copies do not transfer ownership.
The adapter uses one session stream per instance and never switches streams for
an instance. Its nonblocking host guard rejects reentry. All runtime consumers
must rebuild for the transient session layout addition at M20. Tests cover duplicate
attachment, foreign detach, close refusal and initialized handle preservation.

P04 prepares whole-program stage scratch requirements and explicit inclusive
intermediate lifetimes in the existing program namespace. Deterministic first-fit
reuse is bounded but does not claim globally optimal packing. Preparation owns
only metadata; payload allocation remains in the native session or caller.
Stage-local slices reuse storage; retained intermediate slots overlap neither
live stage scratch nor each other. Invalid alignment, overflow and failed binds
preserve the prior plan/output. Two externally owned state buffers change roles
without copies. The caller commits roles only after completion and must return
all readers before the next write, enforced by the P03 native readiness boundary
when executing through the session adapter. Metadata must remain immutable during
execution. This is not automatic graph liveness inference.

Host qualification runs two canonical stages over twelve steps with an independent
recurrence oracle, unchanged operator-new count, and untouched exterior canaries.
It tests overlapping state, insufficient storage, overflow and inclusive last-use
boundaries. P04 requires additional qualification of real CUDA scratch subranges and delayed
reader safety with the P03 test and Compute Sanitizer through the native gate.
The session adapter accepts only contained slices of its fixed per-instance arena.

P04 corrects initialization retry after partial cleanup: actual stream, library,
scratch and persistent allocation fields prevent reset even before initialization
flags/counts are published. The CUDA regression creates a real owned stream with
unpublished flags, proves rejected retry retains it, then closes and resets it.
No CUDA failure is fabricated; this exercises the preserved failure-state shape.

P05 extends the canonical runner with a submission report: attempted and accepted
callbacks are distinct, a failing callback may have written/submitted partially,
and neither counter claims device completion. Existing execute delegates to the
same reported implementation. Whole-program preflight still precedes callbacks.
The existing native readiness owner now exposes explicit cold completion
observation; successful owner-stream synchronization records the observed
publication generation. Synchronization failure poisons the owner. Historical
published/observed generation numbers are diagnostic only after poisoning, and
all subsequent reads/writes remain forbidden. No rollback of in-place effects
is promised. Native teardown continues to use real CUDA synchronization.

Host tests inject a late callback failure after actual writes, preserve those
writes, and prove later stages do not run; late preflight failure writes nothing.
CUDA tests submit real kernels, observe generation 1, then inject a callback
failure after a second-generation kernel and verify changed payload after drain,
no new valid generation, and rejected reuse/borrowing. A separate synchronization
failure is injected via the existing runtime API test seam after real kernels;
removing the injection never unpoisons the result. This is deterministic failure
injection, not evidence of a naturally occurring GPU hardware fault. The
readiness layout/API additions require all consumers to rebuild at M20.

P06 retains numerical operation/source IDs using operation_core_v2::stable_id,
source structure epochs and value generations in immutable borrowed stage-origin
maps. Each row must match the actual prepared stage and selected candidate.
Fused stages may carry multiple origins; decomposed stages may repeat an origin.
Cold comparison checks origin sets including source generations, without a hot
registry. These IDs do not identify biological actors or establish molecular
identity. Provenance metadata must outlive execution and remains separate from
numeric payloads and candidate-specific implementation details.

Optional caller-owned per-stage attempted/accepted counters are collected by the
same canonical runner loop. They saturate explicitly at uint64 maximum rather
than wrapping. Missing capacity rejects before callbacks. Existing execute/report
entry points continue through this implementation without counter storage.
Executed host tests compare actual fused and unfused numeric outputs and origin
sets, changed source generations/epochs, stale candidate IDs, absent origin rows,
many-to-many decomposition, capacity rejection and saturation. No accelerator
fusion performance claim is made by this host qualification.
