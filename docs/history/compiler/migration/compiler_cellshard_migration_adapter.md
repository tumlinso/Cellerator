# Temporary compiler-to-CellShard migration adapters

Todo `CE-CCP1-A03-011` defines five versioned adapter families for preserved
JBC source and tests at CellShard
`b9749ad3e5146a04f847533d8c6f1a54146aed20`. They cover evidence/discovery,
atom/certification, grammar/basis/superatom, graph/schedule, and partials.

Each adapter is declared in the CellShard compatibility namespace, includes or
consumes a public Cellerator contract, maps fields exactly, preserves stable
identity and generation, and owns no compiler semantics. It may reject stale or
unrepresentable legacy input but may not infer missing identity, silently
certify a proposal, introduce a new enum value, or become a new planning API.

Every row names a retirement proof. Once the mapped consumers and preserved
tests compile directly against Cellerator, the adapter is removed in its own
scoped Todo. Adapters are one-way compatibility surfaces; no Cellerator header
depends on them. Gate `CE-CCP1-A03-011-GATE` requires a nonzero version, a
public `Cellerator::compiler` target, a concrete retirement proof, unique legacy
surface, and `owns_semantics == false` for every row.
