# Prepared relation update lifetime

The existing prepared_relation_pair remains the sole cold owner. Its physical
f16 values are authoritative; transpose maps and scalar gradients refer to the
same edge slots. Gradient/delta f32 buffers are caller-owned, tagged by exact
structure, epoch, physical order, count and device. A logical-to-physical map
is immutable host observation through pair lifetime. No second master values
or training object is introduced. Preparing gradient work reserves/reports
persistent maps/panels and reusable scratch under the caller's byte cap.

Input identity/version records content, independently of raw pointer rebinding.
A cached pack requires exact identity, version, layout and dtype plus lifetime
validity; pointer equality alone proves nothing. Gradient stamps bind a unique
pair lifetime, generation, input/cotangent versions, arithmetic profile and
producer serial. Pair incarnation must not be a reusable allocation address;
all serial/incarnation counters fail on exhaustion, never wrap. The runtime
retains the issued stamp and validates every field plus the actual gradient
buffer binding; caller-authored metadata cannot bless an unrelated gradient.
A new gradient, value replacement, or update invalidates the previous stamp.
Independent delta input does not require a gradient stamp.

| State | Permitted transition and rejection |
|---|---|
| Uninitialized values | Prepare succeeded; execution/read/update reject until publication. |
| Published G | Owner execution reads exact G; begin read waits on recorded ready(G). Replacement/update requires strictly greater nonzero generation. |
| Borrowed G | One external lease exists; reject another acquire and any writer before side effects. Owner read-only work may continue on its stream. |
| Returned G | Matching consumer records done; owner waits on that record; invalidate lease immediately. Sequential next lease may use another same-device stream. |
| Updating | Serialized submission validates everything before writing; record ready only after successful write submission. This is not a host-call concurrency window. |
| Poisoned | Partial CUDA submission/event/observed async failure forbids new usable generation or execution. No rollback is promised. Registered work must still drain safely. |
| Closed | Checked close succeeds only without an outstanding lease, drains owned dependencies, destroys resources and clears the handle. |

A lease is bound to pair incarnation, nonce, structure, epoch, device, physical
order, generation, pointer/count and the exact issuing consumer stream retained
by the runtime. Wrong, stale, cross-pair, copied or replayed returns reject
without invalidating a newer live lease. Returning once invalidates all copies.
Ready and done events are created cold, never allocated in steady execution.
Producer ready orders consumer reads; consumer done orders the next in-place
write. Neither alone supplies the complete lifetime guarantee.

Publication and updates use the same producer-ready mechanism. An enqueued G
means readiness is backed by its recorded event, not that all device work has
already succeeded. After overwriting with G+1, G cannot be reacquired as history.
Observation must surface asynchronous failure; metadata inspection alone is no
GPU completion proof. Generation maximum has no valid successor. Alpha zero
still advances generation with the specified arithmetic, including NaN handling.

All host calls are externally serialized. Relation execution remains on one
owner stream/device; a single read lease may use any same-device consumer
stream. New mutable and lease operations reject capture before side effects;
existing N1 read-only graph replay remains independently supported. Metadata,
capacity, finite nonnegative alpha, generation and conservative byte-overlap
checks precede submission. Writable gradient cannot overlap either dense input,
authoritative values, immutable maps or internal scratch; delta/gradient update
inputs cannot overlap the mutated plane or internal storage. Empty arrays have
no accessed range. Overflowed byte products/address ranges reject. Borrowed
inputs and outputs remain live until their queued operation completes.

Checked close rejects unreturned leases even in a poisoned pair. Legacy destroy
must not free such borrowed storage; callers using leases must use checked close
and handle rejection, then return the reader. Teardown may drain owned streams;
hot execution must never perform a device-wide fence.

C02 is the internal declaration/lifetime milestone. Native and readiness tasks
supply the corresponding implementations before any API installation or full
epic capability claim. The prospective demo names are preserved. Host contract
compilation does not substitute for the required delayed-producer/reader V100
and sanitizer tests in the integration gates.
