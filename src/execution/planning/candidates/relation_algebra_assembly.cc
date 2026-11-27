#include <Cellerator/execution/planning/candidates/relation_algebra_catalog.hh>

namespace cellerator::execution::operation {

// Cold assembly seam used by the final central catalog. An invalid fragment
// fails closed as an empty view; no execution, device query, or preparation is
// performed while assembling the immutable relation-algebra declarations.
relation_algebra_catalog_view_v1 assembled_relation_algebra_catalog_v1()
    noexcept {
    const core::operation_status status =
        validate_relation_algebra_candidate_catalog_v1();
    if (!status) return {};
    return relation_algebra_candidate_catalog_v1();
}

} // namespace cellerator::execution::operation
