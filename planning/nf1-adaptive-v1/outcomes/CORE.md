# CE-NF1A-CORE: Deliver the general-width linked numerical core

Reuse preserved B/P/V/N implementation and relevant D01/T evidence, coordinate donor ownership, and complete general-width FP32 primitives, prepared execution, independent values and a real separately linked CPU/CUDA consumer.

**Why:** The paused system already has the right execution/value owners. Finish and expose them rather than introduce another runner or tensor framework.

**Completion prerequisites:** CE-NF1A-ADOPT, GH-NF1A-ADOPT. These are enforced by the outcome gate; they are not all claim-time dependencies. Useful preliminary work is allowed before they are satisfied, but no capability may be declared delivered early.

**CE-CORE.** A separately linked CPU/CUDA numerical core over the existing prepared-program, session, relation, value-plane and binding owners; metadata-invalid requests preflight before any submission, while asynchronous partial failure remains explicitly non-transactional.

Evidence: Actual external consumer, late-invalid-binding no-effects test, completion/lifetime/poisoning controls; retained program and RU1 behavior.

**CE-WIDTH.** General-width FP32 forward/transpose, arithmetic, gather/scatter and needed reductions. Widths 1,3,15,16,17,33,65 plus meaningful empty/high-degree cases; declared destination effects and numerical policy.

Evidence: Independent logical-map FP64 oracle; tail, overwrite/accumulate/affine-effect and adjoint checks; no hidden padded biological coordinates.

**CE-VALUES.** One actual immutable structural owner supports independent value instances. Retain explicit FP32 authority, optional half projection, epoch replacement and proven update/lifetime semantics without pretending a generation is a saved snapshot.

Evidence: Source-close/sibling-survival, precision-below-half-ULP, independent update and stale-ticket tests; actual preparation/memory counters.

## Execution latitude

Choose the local design, implementation/testing sequence and sensible delegation. Follow existing owners; use an authoritative narrow scope transfer when a better location lies outside the initial ownership. Do not duplicate an implementation to satisfy a directory proposal. Do not create separate records for ordinary inspect/code/test/retry steps. Publish a compact result: source and actual tests/evidence, material decisions, limitations and consumer impact.

This outcome subsumes 7 observed unfinished legacy records; `machine/legacy-disposition.json` gives the exact mapping. Completed legacy inputs remain successful historical records and are reused, not redone. `machine/requirements.json` is the complete acceptance inventory; old procedural sequencing is superseded.
