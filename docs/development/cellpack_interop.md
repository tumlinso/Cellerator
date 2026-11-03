# CellPack payload interoperability

Existing CellPack plans, row records, warp tiles and CPK1/CPE2 images retain
their versioned encodings. Their packing objective, recovery maps and direct
weighted-row-reduction consumers keep their existing meanings.

`layout::CellPackingGeometryView` describes physical feature blocks, row
groups and recovery maps. Its statistics describe occupancy, reuse, imbalance
and metadata costs. These are physical packing facts; they do not establish
biological dependency restrictions. The canonical [cellular geometry
definition](../../cellular_geometry.md) describes that separate knowledge.

`execution::CellPackingBinding` associates a validated payload with logical
axes, runtime handles, generations and value planes. It aliases the existing
payload without thawing, decoding or rewriting it. The binding exposes CPK1's
embedded values as a projection-local value plane. Separate execution bindings
may share the same immutable physical maps.

The statistics manifest separates hot summaries from cold planning sidecars.
Statistics requiring external module, transpose, cross-partition, activation
or quantization information remain unavailable until that information is
supplied. The v1 `row_active_block_references` objective remains a versioned
surrogate; later objectives must not change its recorded meaning.

`interop::formats::borrow_validated_cellpack_records` and
`borrow_validated_cellpack_tiles` expose existing records through native
block-masked views. Their metadata and value pointers must have the placement
required by the consumer; a host pointer is not a device replica. These
adapters allocate no replacement values. Link the optional
`Cellerator::cellpack_interop` component when using this bridge.

The existing `make_persistent_feature_weighted_row_reduction_view` remains a
direct payload consumer. It needs no CSR or Blocked-ELL reconstruction.
Historical validation records remain bound to their original source and
binary identities; native API renames do not extend those measurements.
