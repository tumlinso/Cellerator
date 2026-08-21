# Cellerator execution image v2

`CPE2` is Cellerator-owned, pointer-free execution IR. CellShard continues to
wrap these bytes in the unchanged `CPEXEC01` envelope and performs one opaque,
contiguous asynchronous upload. CellShard does not interpret sections,
projections, biological identities, or value mappings.

The image contains one immutable biological relation/semantic geometry and a
directory of zero or more concrete representations of that geometry. Required
domain, order/partition, relation, and semantic-geometry sections are distinct
from optional initial values. Structure epoch and optional initial value
generation are therefore independently visible. Mutable values normally bind
outside the image; the initial-values section is only a deployable default.

Projection entries may name row-masked CP-BP tiles, feature-major layouts, CTA
macrotiles, dense fragments, CSR, SELL, BSR, Blocked-ELL, vendor formats,
transpose/backward layouts, or architecture-specific layouts. This schema does
not implement those formats. An optional lazy entry advertises a constructible
projection without storing its bytes. Load never constructs CSR or BELL merely
because the catalog can name them.

Forward and transpose logical-edge maps and scheduling summaries are ordinary
optional sections referenced by a projection. An unknown section or projection
is accepted only when explicitly optional, allowing catalog extensions without
redefining existing schemas.

`CPK1` remains unchanged. A native row-masked projection may reference a
`cpk1_v1_compatibility` section; the compatibility loader delegates to the
frozen CPK1 validator and does not thaw, canonicalize, or convert it. CPK1's
combined v1 values remain a compatibility artifact rather than redefining the
v2 lifetime model.

Validation checks the complete image identity, per-section checksums, required
section cardinality, alignment and bounds, non-overlap, directory identities,
projection references, and the caller's structure/geometry/catalog identities.
Relocation rebases only directory pointers. A hot kernel receives
`prebound_projection_view_v1`, not the self-describing image.

Schema v2 assumes fixed-width little-endian records, recorded by the endian
marker. All alignment values are offsets relative to the image base; this keeps
the image compatible with CellShard's ordinary contiguous host allocation and
CUDA's naturally aligned device allocation.

No CPEXEC01 change is required. If a future Cellerator schema cannot fit inside
that opaque envelope, work must stop at an external CellShard decision
checkpoint rather than assigning CellShard implementation here.
