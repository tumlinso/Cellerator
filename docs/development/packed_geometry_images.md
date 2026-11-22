# Packed geometry images

The geometry image readers live under `Cellerator/geometry/`. Existing packing
plans, row records, row-masked tiles, and execution images retain their versioned
encodings, including the `CPK1` and `CPE2` file identifiers. The packing objective,
recovery maps, and direct weighted row reduction preserve their meanings.

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

`geometry::bind_packed_row_records` and `geometry::bind_row_masked_tiles` in
`<Cellerator/geometry/persistence/packed_tensor_views.hh>` borrow the mask,
offset, and value arrays through the native block-masked tensor views. The
binding returns a validation status and allocates no storage. A supplied
feature-block offset replica must match the validated plan and reside in the
same memory placement as the source. The caller retains the source through the
last asynchronous consumer. The bindings are available with the installed
native geometry and tensor headers.

The existing `make_persistent_feature_weighted_row_reduction_view` remains a
direct payload consumer. It needs no CSR or Blocked-ELL reconstruction.
Historical validation records remain bound to their original source and
binary identities; native API renames do not extend those measurements.
