# Relation value readiness and read leases

The runtime component in `include/Cellerator/runtime/relation_value_readiness.hh`
provides reusable dependency edges for the existing prepared relation pair. It
owns two disable-timing CUDA events; it owns neither streams nor a second value
plane. See the [execution architecture](../core_execution_cp_math.qmd) and
[lifetime contract](../../planning/relation-update-spine-v1/04_READINESS_AND_OWNERSHIP.md)
for the wider ownership boundary.

The serialized protocol is:

```
owner:    write G -> record ready G
consumer: wait ready G -> read values -> record done ticket
owner:    wait done ticket -> write next G -> record ready next G
```

Initialization creates the ready and done events once, on the owning device.
The initial state has no ready generation. `validate_write` checks the exact
expected generation, a strictly greater nonzero next generation, owner stream,
current device, stream device and capture state before any producer submission.
`publish` records ready only after the producer submissions succeed. A generation
number reports an enqueue promise; it does not report observed GPU completion.
Acquiring a stale or future generation never waits on the event for another
generation. Structural identity and epoch must match exactly.

`begin_read` admits one outstanding host ticket and submits a producer-ready
wait on the consumer stream. The ticket binds structure, epoch, generation,
device, a process-unique component incarnation and a monotonically increasing
nonce. A second acquisition fails as busy without overwriting the first ticket.
This bound is a v1 runtime policy and does not constrain relation mathematics.
Host calls on the component are externally serialized.

`end_read` accepts only the exact active ticket and matching consumer stream.
It records the done event and submits the owner-stream wait before invalidating
the ticket. The next writer can then enqueue immediately. All waits referring
to an event record are submitted before that event is recycled; subsequent
records cannot redirect the already-submitted waits. Sequential tickets may use
different same-device streams. No event or buffer is allocated in this cycle,
and no device-global synchronization is introduced.

The bare `wait_current` method supplies only the producer-ready edge. The pair
must use the full begin/end protocol for external borrowing; a ready wait alone
cannot prevent a later overwrite. The component supplies tickets and ordering,
while the prepared pair validates the complete public view, including pointers,
capacities and physical order, before exposing values.

## Failure and retirement

Metadata rejection submits no event and does not disturb a newer ticket.
Unsupported capture is rejected before stream-device queries or event changes.
CUDA 12.9 rejects the stream-device query during active capture, so testing the
capture state first is required to preserve that rejection boundary.

Producer enqueue failure or failed ready/done record or wait poisons the
component. An earlier numeric generation remains available for diagnostics but
cannot authorize reads or writes. No rollback of an in-place value update is
claimed. Enqueue success cannot prove future asynchronous device execution will
succeed; explicit stream-local observation must check its CUDA result.

A failed reader return preserves the matching ticket. `end_read` may retry the
record and owner join solely for cleanup, even while poisoned. It never clears
poison. A forged or stale ticket still fails. This allows checked teardown after
a recoverable injected event failure without pretending that the numerical
operation succeeded.

`close` rejects an unreturned ticket. After all tickets return, it synchronizes
the owner stream, which already contains the registered reader joins, and then
destroys the events. A failed synchronization preserves the poisoned component
for explicit handling. The destructor attempts checked close; it cannot safely
reclaim events protected by a lost ticket. The owning pair must likewise retain
borrowed storage if checked close reports busy. A permanently broken CUDA
context cannot be repaired by this component.

A returned ticket does not make a saved raw pointer safe for later use. The API
cannot detect reads made directly through a raw pointer after return, nor can it
protect against destruction of a borrowed stream or allocation by its caller.
Those are caller lifetime violations, not additional supported concurrency.
No historical snapshots, multiple outstanding host tickets, concurrent prepared
execution, or full mutable-cycle graph capture are provided here. The existing
N1 read-only graph path has its own acceptance contract.

## Validation

Three standalone CUDA tests exercise the component:

- `readiness_component_test.cu`: actual nonblocking producer-ready D2D ordering,
  initial unready state, exact metadata, capture, overflow and failed producer.
- `read_lease_test.cu`: 128 delayed D2D readers on two alternating nonblocking
  streams; immediately submitted next writes, busy acquisition/mutation/close,
  stale and forged tickets, wrong-stream return and capture rejection.
- `readiness_adversarial_test.cu`: 4,096 generations with the same two event
  handles, delayed producer and reader schedules, enqueue-versus-completion
  observation, object-address reuse, and five deterministic enqueue/event
  failures, including cleanup retries after a failed return.

The delay kernels have finite device-clock bounds, contain no occupancy-dependent
handshake, and call no CUDA API from a host callback. Stream synchronization is
confined to warmup, final output observations and checked close. A checked
`cudaEventQuery` demonstrates an enqueued generation whose ready event is still
incomplete; a later stream observation checks numerical copies and completion.
Sanitizer complements these dependency tests and cannot alone prove ordering.

Tests build with CUDA 12.9, C++17 and `-arch=sm_70`. Execution receipts use the
CUDA controller GPU lease and benchmark mutex. These are correctness results;
no latency or throughput advantage is claimed. Integrated pair, compiler and
N1 graph acceptance remains separately owned by the epic's integration gates.

The lane acceptance on 6 September 2026 used Tesla V100-SXM2-16GB,
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`, driver 580.173.02. R01 passed under
controller evidence `3de7d3f4-4017-43a9-817e-e32edb056f02`; R02 passed under
`4039f561-5f0f-4639-8f87-c041887ea278`. R03 ran the actual test under memcheck
with evidence `092ce0ec-623a-474c-8cf7-e1307bdb2b2a`, exit zero, its explicit
4,096-generation PASS line and `ERROR SUMMARY: 0 errors` in the raw stdout.
The tested R03 snapshot was HEAD `586e796a0945f6262961da5dbd4ac363c59d0ab3`
plus the new adversarial test, fingerprint
`35b293122cdc59471635b228ecf498b5a1da749bd1ac785bd3712ea4ccfe82cc`.

Exact standalone build and instrumented execution argv:

```sh
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc -std=c++17 -arch=sm_70 -Iinclude src/runtime/relation_value_readiness.cu tests/relation_update_spine_v1/readiness_adversarial_test.cu -o /tmp/ru-readiness-adversarial
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/ru-readiness-adversarial
```

The second argv ran inside the controller's `baseline` recipe so it retained
its lease, quiescence proof and mutex. The sanitizer recipe wrapper's earlier
receipt `3461419e-8510-4aab-8cc4-44641bc272c4` is **not acceptance**: its raw log
selected a missing CUDA 13.1 executable and had no test PASS or error summary,
even though the outer wrapper returned zero. No controller machinery was
changed to work around that failure.

The tested binary SHA-256 was
`fa141a503d460dc147af7eeb15991c2a7c433783b2b54a139e6a2c1690445e5f`;
the adversarial test source SHA-256 was
`e67c1ae8c7dabf79c4c75a9bdef332e27833850e9d04a27032f0233656e73a9b`.
The test uses exact integer copies to test lifetimes, not a floating-point
numerical approximation profile.
