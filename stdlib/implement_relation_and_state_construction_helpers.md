# Relation and state construction helpers v1

`builders.cell` requires callers to supply every pointer, extent, domain/order
identity, value generation, support descriptor, and numeric type. Construction
is allocation-free and performs no reorder, copy, inference, or canonicalization.
