# Regulatory sequence execution

Cellerator binds native sequence bit planes to the biological execution ABI.
The binding borrows low, high, and validity arrays in caller-owned storage,
and attaches coordinate identity and placement. The tensor layout stays
separate from the compact execution operand. A nonempty execution operand
requires explicit validity, including when every biological base is valid.

The precompiled regulatory operation accepts one verified forward exact-motif
mask program. Preparation checks the native predicate plan and retains the
semantic hash, motif, coordinate-to-regulatory and regulatory-to-gene relation
identities and structure epochs, projection, and output contracts. Input,
output, and value pointers, value generation, and stream remain launch bindings.
Use the operation core's `run_prepared_operation` for checked uncached dispatch.

- `materialize_mask` writes one predicate bit per local coordinate, then joins
  the selected anchors against regulatory intervals.
- `fuse_predicate` tests the motif at owned anchors and joins them in one launch.
- `materialize_relation` writes one local regulatory-element ID per coordinate
  or `UINT32_MAX`. Repeated cell states can reuse this map without repeating
  the interval join.

Each path accumulates weights from the regulatory-to-gene relation into
caller-initialized gene state. A halo can complete a motif anchored in an owned
span. Invalid positions cannot contribute motif evidence. The caller supplies
device-resident inputs, materialized outputs where applicable, and a stream.
Execution enqueues work without allocating, synchronizing the host, or moving
data across the host/device boundary.

Automatic selection uses comparable measurements for the exact predicate,
coordinate order, projection, and runtime/device build. It accounts for first
use and cached uses over the supplied reuse horizon. Missing or incompatible
evidence leaves the strategy unresolved.

Mask and relation caches belong to the caller's execution session. The key
includes sequence generation, predicate hash, and coordinate identity and
order, along with the coordinate structure epoch. The direct relation key also
includes the projection identity. A new regulatory weight generation can reuse
either sequence result. The readiness event publishes the materialized payload
to subsequent streams. A captured use waits on an already published key through
an external event node. Publish cache misses outside capture. Keep the payload
and event bound to their key through all submitted uses, including graph replays.

The current interval projection is sorted and non-overlapping. The standalone
bounded predicate verifier describes additional programs. Regulatory execution
supports the single forward exact-motif mask operation described above.
