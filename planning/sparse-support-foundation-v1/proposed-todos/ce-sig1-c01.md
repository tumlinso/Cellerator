# CE-SIG1-C01: Define native support and work-view contracts

Define general contracts for identity-bound directed relations and ordered multi-participant dependencies; shared topology/per-cell activity; structure epochs versus parameter/state generations; variable work sets and explicit overflow; logical indices, local masks, and prepared region descriptors. Specify preparation and hot execution ownership, caller-stream behavior, and no hidden allocation or host synchronization in steady execution. Keep logical contracts independent of CE-GEO physical packing.

Acceptance: contracts compile in a Torch-free native build and have an independent CPU reference for identity, ordering, capacity, and overflow behavior. Record existing packing mechanisms and their migration boundary; do not replace or silently discard useful packing code.
