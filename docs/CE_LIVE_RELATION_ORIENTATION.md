# CE-LIVE logical relation orientation v1

`CELLERATOR_LOGICAL_EDGE_ORIENTATION_V1_READY` freezes one logical relation
for CP-BP and the current Cellerator sparse candidates:

```text
feature or gene source -> row, cell, or module destination
```

The relation's `structure_handle`, `structure_epoch`, and logical edge indices
remain stable across physical projections. A transpose/backward candidate does
not construct a row-to-feature logical relation. It consumes a distinct
transpose projection of the same structure and an explicit
`value_position_map_view` with `direction == transpose` to map each stable
logical edge to its projection-local value position.

## Adapter contract

`build_cp_bp_v1_compatibility_adapter_host` assigns the request's
`feature_axis` to `relation_structure::source_axis` and `row_axis` to
`destination_axis`. The geometry retains both execution-to-canonical and
canonical-to-execution maps independently for rows and features; changing
relation orientation does not canonicalize or rebuild CPK1.

`validate_cp_bp_v1_compatibility_adapter_host` rejects an adapter whose axes
are swapped, whose structure/value lifetimes disagree, whose logical edge
count differs from the CPK1 nonzero count, or whose direct views stop aliasing
the validated payload. Forward row-masked, feature-major, and CSR candidates
already require the same feature-source/row-destination relation at launch.

## Transpose and edge identity

Forward and transpose projections may have different projection identities
and byte orders. They must reference the same logical structure identity and
epoch, expose the same logical edge count, and provide exact inverse
logical-to-projection and projection-to-logical maps. Mutable value generations
do not enter persistent structure identity and do not change edge identity.

This contract does not select a physical layout, introduce another operation
framework, or make canonical row order an internal execution postcondition.
