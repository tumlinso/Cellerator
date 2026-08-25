# Hierarchy and boundary planning v1

The hierarchy IR is an optional projection of existing semantic structure. A
persistent hierarchy identity covers nested partition membership, ancestry,
and device placement. It does not replace biological domain, geometry, order,
structure, projection, or value-generation identity.

`shared_value_hierarchy_view` is an immutable compact module-to-value index.
Repeated indices express reuse; numerical values remain in existing mutable
value planes. `module_activity_view` is generation-tagged launch state.
`build_active_module_plan` writes only active modules into caller storage and
never allocates.

Boundary planning consumes edges in execution order. It skips edges touching
inactive modules and omits same-device order-preserving edges. Local order
changes pay only the explicit transform phase. Cross-device edges record bytes
and a replaceable latency/bandwidth estimate as communication phase cost. The
result adapts to connected-planner transitions, where empirical measurement
remains authoritative.

Connected-plan cache keys include the persistent hierarchy identity. A change
to partition ancestry or placement therefore cannot reuse a stale path. The
single-device case stays a no-op unless an explicit order transform is needed.
