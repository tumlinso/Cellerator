# Cellerator biological ABI v1

`biological_abi.hh` is the minimal Cellerator-owned CPU/CUDA contract shared by
future execution code and Baseplane sequence consumers. It is deliberately
separate from `include/Cellerator/abi.h`: the latter remains the version-1
public sparse-layout C API, while this directory is the versioned internal
biological execution ABI. Neither surface silently changes the meaning of the
other.

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
valid only in the registry generation that interned its persistent identity.

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
| `cellerator_tensor_desc` | Remains public C API compatibility; future internal adapters bind a `dense_tensor_view` with explicit axes and residency. |
| `cellerator_sparse_layout_desc` | Remains public sparse-layout compatibility; it is not the common Baseplane ABI or a universal physical representation. |
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

The exact Baseplane conversion and public adapter surfaces remain gated on the
Baseplane correctness/prepared-plan checkpoints and CE-ARCH-40. CE-ARCH-11 owns
structure/value/binding lifetime and execution-order contracts; those concepts
must not be added here by growing every operand record.
