# Split persistent partial semantics from partial storage

Todo `CE-CCP1-A03-007` splits the seventeen JBC partial contracts reviewed at
CellShard `b9749ad3e5146a04f847533d8c6f1a54146aed20` across a one-way
interface. Cellerator owns additive, extrema, log-sum-exp, moments, gradient,
transform, structural, trajectory, gathered-panel and relation-contribution
algebras; dependency freshness/closure; numerical policy; and the legality and
promotion rules for persistence.

CellShard owns concrete partial-image bytes, object/file persistence,
replication, placement, leases, recovery and delivery. It consumes a stable
Cellerator ABI containing partial identity, algebra kind, exact coverage,
structure/value generation, dependency digest, numerical policy and legal
persistence modes. Cellerator imports no CellShard header or library; storage
receipts return through opaque adapter values, so the link graph is acyclic.

The interface table has four semantic/compiler rows and four concrete-storage
rows. It preserves current partial behavior while deliberately preventing
`partial_image_v1` or a resident payload from becoming the semantic authority.
Gate `CE-CCP1-A03-007-GATE` checks unique ownership and the one-way dependency.
