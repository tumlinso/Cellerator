#include <Cellerator/compiler/sema/implement_axis_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {
namespace {
bool same(semantic_identity a, semantic_identity b) noexcept {
    return a.low == b.low && a.high == b.high;
}
bool valid(semantic_identity value) noexcept {
    return value.low != 0 || value.high != 0;
}
}  // namespace

axis_compatibility compare_axes(const axis_type &left,
                                const axis_type &right) noexcept {
    if (!same_nominal_domain(left.domain, right.domain))
        return axis_compatibility::domain_mismatch;
    if (left.global_extent != right.global_extent)
        return axis_compatibility::extent_mismatch;
    if (!same(left.logical_order, right.logical_order))
        return axis_compatibility::order_mismatch;
    if (!same(left.geometry, right.geometry))
        return axis_compatibility::geometry_mismatch;
    if (!same(left.partition, right.partition))
        return axis_compatibility::partition_mismatch;
    if (left.local_extent != right.local_extent)
        return axis_compatibility::local_extent_mismatch;
    if (!same(left.recovery_identity, right.recovery_identity))
        return axis_compatibility::recovery_mismatch;
    return axis_compatibility::exact;
}

bool valid_explicit_axis_mapping(const explicit_axis_mapping &mapping) noexcept {
    return same_nominal_domain(mapping.source.domain, mapping.destination.domain)
        && mapping.source.global_extent != 0
        && mapping.destination.global_extent != 0
        && valid(mapping.source.logical_order)
        && valid(mapping.destination.logical_order)
        && mapping.total && mapping.one_to_one;
}

}  // namespace cellerator::compiler::sema::v1
