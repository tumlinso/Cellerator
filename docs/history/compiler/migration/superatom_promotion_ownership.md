# Cellerator ownership of superatom promotion

Todo `CE-CCP1-A03-006` preserves the experimental JBC superatom implementation
from CellShard `b9749ad3e5146a04f847533d8c6f1a54146aed20` as an optional
Cellerator compiler composition. Candidate, membership, statistics, cost,
benchmark, lifecycle and evolution behavior remain evidence; a superatom is
not a CellShard storage shard.

Promotion requires a stable derivation identity, exact member set, independent
deconstruction digest, and complete measured comparison of the composed path
against its constituent-atom baseline. The comparison includes build, storage,
maintenance, invalidation and expected-use costs where applicable. Equality or
a slower composed result is a legitimate `evaluated_not_promoted` disposition.
Invalid or saturated evidence cannot promote.

Demotion and evolution remain versioned compiler decisions with exact lineage
and deconstruction. CellShard may materialize or cache the resulting immutable
ruleset, but residence does not define superatom identity or promotion.

Gate `CE-CCP1-A03-006-GATE` proves both promotion and valid non-promotion,
rejects missing exact deconstruction/independent evidence, and proves the
contract never classifies a superatom as a storage shard.
