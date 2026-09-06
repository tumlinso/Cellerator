# Readiness, ownership and in-place mutation

## Two dependency directions

Publishing a producer event permits another stream to read the result only after the producer has written it. It does **not** stop the next producer update from overwriting the storage while that reader runs. This is a necessary consequence of in-place mutable values, not a general multi-stream execution expansion.

The bounded protocol is:

```
owner:    writes(gen G) -> record ready(G)
consumer: wait ready(G) -> read const physical values -> record done(ticket)
owner:    wait done(ticket) -> writes(gen G+1) -> record ready(G+1)
```

Only one owner stream executes relation operations and updates. Host API calls are externally serialized. The first version admits one external read lease at a time, reusable sequentially on any same-device nonblocking consumer stream. This keeps the API truthful without introducing a general multi-reader reclamation system. A later extension can raise this bound without changing relation mathematics.

## State transitions

| State/call | Result |
|---|---|
| Prepared, no published values | Execution/read acquisition rejected. |
| Published G, valid acquire on consumer | Enqueue wait on already-recorded ready event, return ticket and const physical view. |
| Lease unreturned, next update/replacement | Reject as busy before writing; do not wait for an event the caller has not recorded. |
| Valid return on matching consumer | Record done event and enqueue owner wait; invalidate ticket. |
| Returned lease, next update | Owner stream ordering delays writing until reader completes, without host-global sync. |
| Wrong/stale/replayed ticket | Reject without disturbing any newer lease. |
| Enqueue/event error after submission | Poison; no claim of transactional rollback or new usable generation. |
| Close with outstanding unreturned lease | Checked close rejects. Caller must return it; never silently free live borrowed storage. |
| Close after return | Drain registered owned work as documented and free bounded resources safely. |

Ready/done events are precreated at preparation and recycled only under this serialized protocol. CUDA event rerecording changes event state; waits already submitted capture the then-current record rather than all future records [R2]. Validate a token before submitting a wait. Never use an unrecorded event as proof of readiness. A returned ticket does not make old raw pointers valid for future use.

An enqueued generation is a promise backed by an event, not an immutable snapshot of all historical generations. Once G+1 overwrites the single value plane, acquiring G must fail. Explicit observation checks asynchronous errors; publication enqueue success alone is not proof that all future device work will complete successfully.

## Boundaries and tests

Use `cudaStreamNonBlocking` in concurrency tests so legacy default-stream ordering cannot conceal missing dependencies [R3]. Delay both producer and consumer in controlled tests and repeat over many event reuses. Test duplicate acquire, wrong stream, stale return, generation overflow, failure injection, external pointer lifetime and close behavior. Test-only delays are not production synchronization mechanisms.

No device-global synchronization in the hot cycle. Host waits are allowed at output observation, acceptance, explicit close and teardown, and are reported separately in performance evidence. New update/publication/lease operations must reject unsupported stream capture before the first side effect. Existing N1 graph replay remains a separate supported regression. Full mutable-cycle capture, concurrent relation execution, multiple in-flight external read leases, historical value snapshots and multi-device readers are explicitly deferred.
