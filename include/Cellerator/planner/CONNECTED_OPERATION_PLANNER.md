# Connected-operation planner v2

CE-ARCH-86 adds a bounded linear-chain planner above the existing
single-operation planner. It does not replace operation candidates, Objective
V2, prepared execution, or empirical autotuning.

Each stage supplies the same pointer-free planning keys and real operation
candidates used by `plan_end_to_end`. A boundary explicitly names its producer
and consumer candidate identities, order effect, optional conversion identity,
legality, and measured phase costs. Those costs use the existing CE-ARCH-73
accounting fields, including preparation, dynamic packing, order transforms,
communication, and reuse amortization. Missing or malformed boundaries are not
inferred and make that path unavailable.

The v1 algorithm is intentionally limited to at most eight operations and
eight candidates per stage. Dynamic programming retains the best prefix for
each terminal candidate; this is exact for the declared chain because future
cost depends only on the current candidate and its next boundary. A later DAG
planner can replace this layer without changing candidate or execution ABIs.

Analytical total cost only forms the bounded shortlist. Whole-path measurement
can change the winner and remains mandatory when requested or when an included
candidate declares empirical uncertainty. Cache entries contain no pointers:
they store graph identity, every stage's structure epochs, biological geometry
and order identities, device/build/policy keys, candidate identities, and
projection identities. Evidence revision, spread, and confidence are validated
before replay; a stale or unavailable winner falls back to current planning.

CE-ARCH-90 adds the persistent partition-hierarchy identity to the durable
connected key. The identity covers nested partition membership, ancestry, and
placement; changing any of those facts invalidates cached whole-path evidence.
Boundary communication and order work remain ordinary measured phase costs,
so the planner does not encode a preferred topology or transfer mechanism.
