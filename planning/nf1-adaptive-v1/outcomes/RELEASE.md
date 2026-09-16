# CE-NF1A-RELEASE: Verify real-consumer acceptance and close Cellerator

Verify GlassHelix ACCEPT and its exact Cellerator source, reconcile published main and preserved work, and issue the final Cellerator receipt without gratuitously rerunning unchanged qualification.

**Why:** Consumer-ready and consumer-accepted are different boundaries. Final verification must not create a mutual-final dependency.

**Completion prerequisites:** CE-NF1A-QUALIFY, GH-NF1A-ACCEPT. These are enforced by the outcome gate; they are not all claim-time dependencies. Useful preliminary work is allowed before they are satisfied, but no capability may be declared delivered early.

**CE-FINAL.** Verify the real GlassHelix acceptance against the exact qualified CE source (or an explicitly requalified descendant), publish final CE receipt and verify origin/main.

Evidence: Fresh producer/consumer task identity, source/interface/test receipt chain, inherited work disposition, no force rewriting or unrelated cleanup.

## Execution latitude

Choose the local design, implementation/testing sequence and sensible delegation. Follow existing owners; use an authoritative narrow scope transfer when a better location lies outside the initial ownership. Do not duplicate an implementation to satisfy a directory proposal. Do not create separate records for ordinary inspect/code/test/retry steps. Publish a compact result: source and actual tests/evidence, material decisions, limitations and consumer impact.

This outcome subsumes 2 observed unfinished legacy records; `machine/legacy-disposition.json` gives the exact mapping. Completed legacy inputs remain successful historical records and are reused, not redone. `machine/requirements.json` is the complete acceptance inventory; old procedural sequencing is superseded.
