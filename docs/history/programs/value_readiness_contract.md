# CE-LIVE value readiness contract

Mutable biological values have persistent structure and generation identities,
but CUDA readiness is launch-time runtime state. `value_readiness_record` is
therefore owned beside an execution session, not by `value_plane`, a prepared
operation, CPE2, or any durable structure ABI. It owns one timing-disabled CUDA
event and only observes producer and consumer streams; it never owns a stream.

## Publication

The producer enqueues all writes for a new value generation, checks the enqueue
status, and then calls `publish_value_generation`. A failed producer status does
not record an event and does not alter the last published epoch or generation.
If event recording itself fails, publication likewise remains unchanged. A
published generation is thus never evidence for work that failed to enqueue.

Within one structure epoch, generations must increase strictly. A greater
structure epoch may start again at any nonzero generation; an older epoch or a
non-increasing generation is stale. Structure epoch and generation numbers—not
the value address—establish identity. Rebinding a value pointer does not change
this readiness protocol.

## Consumption

A consumer requests one exact structure epoch and value generation. If it uses
the producer stream, program order already provides visibility and the fast
path makes no CUDA call. A different stream receives `cudaStreamWaitEvent`,
which orders subsequent work without synchronizing the host or the whole
device. A device mismatch, unpublished record, or stale identity fails before a
wait is enqueued.

The stored stream handle is runtime launch state used only to recognize the
same-stream fast path. It is not a persistent identity and is never serialized.
The implementation contains no `cudaDeviceSynchronize`, `cudaEventSynchronize`,
or value-pointer identity comparison.

## Lifetime and failure boundary

Initialization creates the event on the execution session's current device.
Callers explicitly clear the record while that device and session are alive;
cleanup is idempotent and reports CUDA failure. The destructor is a final
fallback against event leakage, not the normal ownership path.

Host mutation of one readiness record is serialized by its owning execution
session. Concurrent host publication to the same record is unsupported. CUDA
stream concurrency after publication is ordered entirely by the recorded event
and stream waits.
