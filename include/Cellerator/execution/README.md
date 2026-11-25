# Execution contracts

`execution/planning` ranks operation and placement candidates. Preparation binds
the selected views and providers. Program execution enqueues prepared work on
the supplied streams. `execution/tuning` contains measurement and adaptation
policy. The device session and its resources live under `runtime`.

## Eager sparse fetch

Link `Cellerator::execution_streaming` to use `streaming::FetchQueue` with a
borrowed `FetchSource`. The optional RDS and AnnData components provide
`FetchReader` adapters. The caller supplies a source context, one worker
submission callback, compressed scratch, packet slots, and row-map storage.
`fetch` accepts ranges in logical occurrence order. The worker reads in source
order and packs within the requested packet. `poll` reports readiness and
requirements. Release a ticket after its final user and drain the queue before
closing the source.

Native fetch preserves the stored widths. SELL fetch requires an explicit
conversion policy for non-FP16 input. It packs into a single supplied span and
publishes the layout and row map. The default slices have 16 rows. Explicitly
permitted reordering uses stable length groups within 256-cell windows.

`Cellerator::execution_streaming_cuda` adds alternating placements in a caller
supplied `StripedBuffer`. Pass pinned packet storage and distinct ready and
last-reader events. `stage` takes the host ticket. The compute stream waits on
the returned ready event, and `release` records completion after the final
reader. Split copies can upload a disjoint span before that reader finishes.
Release the device tickets, drain CUDA staging, and then drain the fetch queue.

Fetch tuning is opt-in. Supply comparable timed windows of completed useful
work to trial future lookahead, row reordering, or split-copy policy within the
reserved capacity. The chosen settings apply to future requests and placements.
When an operation permits multiple packets per batch, `planning::choose_packet_cut`
compares caller-sized whole-slice cuts with their adjacent packets.

For indexed RDS with CUDA enabled, build `cellerator_rds_sell_fetch_bench`.
Pass the source, its index, packet count, cells per packet, and first cell.
It compares serial and fixed eager execution over the same warmed ranges and
checks column moments and row maps against a host reference.

## Operand and identity contracts

`biological_abi.hh` defines the Cellerator-owned CPU/CUDA operand and identity
contract for biological execution and Baseplane sequence bindings.

## Identity and size policy

Persistent identities are typed 128-bit values. Runtime launch records use
typed `(slot, generation)` handles resolved by a registry during preparation.
This keeps pointer addresses out of semantics while avoiding four persistent
hashes in every hot axis record. `axis_identity` is 32 bytes,
`sequence_domain` is 40 bytes, and the discriminated operand envelope is 224
bytes on the required 64-bit ABI. These are per-binding records; they must not
be copied into every edge, tile, or event.

Domain, order, semantic geometry, and partition remain separate. Structure and
projection identifiers are also separate. Device performance class and build
identity are planner/cache inputs and never biological identity. A handle is
an interned alias, not persistent identity, and is valid only in the registry
generation that interned its persistent identity.

## Persistence and versioning

Persistent records use an explicit schema version, record kind, and byte count.
Writers encode named fields in little-endian order; native struct bytes are not
a persistence format. Device pointers, padding, runtime handles, device
ordinals, and performance-class fields must not be serialized as semantic
identity. Version mismatch is rejected. A future change in field meaning,
identity semantics, or serialized width requires a new schema version.

## Migration map

| Existing surface | Biological ABI v1 disposition |
| --- | --- |
| CP-Math `feature_order_identity` | Becomes an adapter input for explicit domain, order, geometry, and partition handles; its canonical/packed distinction is preserved, not reinterpreted. |
| CP-Math `sparse_structure_identity` | Becomes an adapter input for typed persistent `structure_id` plus a generation-checked runtime handle. |
| Baseplane `dna2_planes32_stream_view` | Binds to `bit_plane_view` with base count, validity, coordinate axis, and residency supplied explicitly. |
| Baseplane motif hits | Bind to `event_stream_view`; total matches, stored records, dropped records, order, strand, rule, and bounded sequence coordinates remain distinct. |
| Baseplane segments | Bind to `segment_stream_view` with global/local/owned/halo sequence-domain semantics. |
| `parameter_descriptor` | Is not reused as the biological axis ABI; small launch parameters may bind `scalar_parameter_view`, while learned value-plane ownership belongs to CE-ARCH-11. |

## Ownership boundary

Cellerator owns identity semantics, operand discrimination, residency, relation
identity, and future planning interpretation. Baseplane continues to own packed
sequence bytes, validity-aware predicates, motif/event emission, and segment
construction. This target has no allocation, STL ownership, CUDA runtime,
vendor-library, biological ontology, or persistence-image dependency.

Prepared operations may require a bounded set of immutable relation structures.
Each mutable value plane names the exact relation handle and epoch whose logical
edges it values. Prepared state owns no ordinary launch-time pointers, and every
output has an explicit update/effect contract. These rules are implemented by
the adjacent lifetime, launch-binding, and execution-contract headers rather
than by growing every operand record.
