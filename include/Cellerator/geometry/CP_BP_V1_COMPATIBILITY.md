# CP-BP v1 compatibility and semantic geometry

The v1 plan, compact records, local row order, warp tiles, CPK1 image, exact
objective, referees, and direct weighted-row-reduction kernel remain valid v1
evidence. This bridge does not change their schemas or meanings.

`cp_bp_v1_compatibility_adapter` is a read-only aliasing view. It maps CPK1 to
the Cellerator structure/value lifetime contract while retaining the exact v1
feature and row recovery maps. The embedded CPK1 values are exposed as a
projection-local `value_plane`; later images may bind values separately, but
CPK1 bytes are not rewritten in place.

Semantic geometry consists of the biological row and feature axes, execution
orders, feature blocks, row groups, recovery maps, identities, and reusable
statistics. The row-masked warp tiles are one physical projection over that
geometry. Additional projections must not redefine the geometry.

The v1 `row_active_block_references` objective remains a versioned surrogate.
Its exact value stays with `frozen_packing_plan`; CPK1 persists its objective
kind and cost-policy identity. A later operation-aware objective must use a new
version rather than changing this meaning.

The statistics manifest separates compact hot summaries from cold planning
sidecars. Row-nnz, mask populations, occupancy, reuse, imbalance, metadata,
partial blocks, heavy rows, and dense-fragment candidates are derivable from
v1. Module, transpose, cross-partition, activation, and quantization summaries
require additional semantic input and remain explicitly unavailable rather
than being guessed.

Existing code can pass `adapter.payload` directly to
`make_persistent_feature_weighted_row_reduction_view` and then use the existing
CUDA evaluator. This performs no CSR or Blocked-ELL reconstruction and preserves
the current canonical-output compatibility behavior.
