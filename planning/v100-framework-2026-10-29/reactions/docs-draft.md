# CUDA reaction application: V100 correctness and performance

## Contract and baseline

`apply_reaction_events` applies caller-supplied reaction IDs to identity-local
int64 species counts. A reaction applies a sparse, sorted, unique row of signed
nonzero deltas. Each event validates the entire row and touched counts before
committing it. A cell stops at its first malformed event/count/log capacity
error; its preceding events stay committed, while the failing and following
events remain unapplied. Changed-source logging preserves one record per
changed species per event. This operation does not evaluate rates or sample
stochastic events.

The independent baseline is a host sequential interpreter that checks each
cell's event sequence against an immutable reaction bank, using wide integers
for precondition/overflow checks and committing only a valid event. Counts and
per-cell statuses must match exactly; FP tolerance is irrelevant because state
and deltas are integers.

## Main risks and unknowns

- Device input arrays are borrowed and device-content validation is asynchronous.
- Reaction events and CSR entries are replayed serially by one thread per cell.
- GPU throughput is unknown. The kernel may be launch-bound for few cells or
  sequential per cell for long histories / large reaction rows.
- Actual admission and device UUID must come from Device Scheduler.

## Candidate directions

1. First qualify the existing kernel, error prefixes, unchanged counts on a
   failing event, capacity accounting, local versus mapped change records, and
   empty/tail cases against the host interpreter.
2. Compare one-thread-per-cell replay against a simple host/device memcpy plus
   reference for tiny inputs only as a harness sanity check, not a contender.
3. If the measurements identify a regime, consider bounded alternative
   ownership (cell event segmentation or grouped reaction rows). Preserve event
   order because negative counts/overflow make reordering observable. Compare
   complete work and avoid an unmeasured path dispatcher.

## Validation / evaluation

Build only `sm_70`, with the focused reactions smoke executable. The validation
entry point is `cellerator_framework_cuda_reactions_smoke --device <scheduler-
assigned-ordinal>` after scheduler admission. Verify independently generated
cell counts, status flags, applied prefix, and changed-source logs. Then run
Compute Sanitizer memcheck and racecheck under the same exclusive grant.

For performance, after correctness, use a repeatable event-timed batch with a
resident bank and buffers. Sweep cell count, events per cell, nonzeros per
reaction, and changed-log enabled/disabled. Report kernel-only device time
separately from upload/initialization and total apply path. No performance
promotion occurs unless a candidate beats the current kernel on a repeatable,
semantically matched workload and preserves the independent oracle.

## Promotion boundary

A successful scheduler-admitted V100 device run establishes only the tested
shapes, integer boundary cases, source version, and sm_70 binary. It does not
establish GPU SSA, biological validity, a universal performance win, or
scientific model correctness.
