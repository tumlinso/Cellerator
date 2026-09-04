#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ir::semantic {
namespace {

constexpr bool same(semantic_identity_v1 left, semantic_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

template<typename Tag>
constexpr bool same(semantic_identity_v1 left,
                    cellerator::execution::persistent_identity<Tag> right) noexcept {
    return left.low == right.low && left.high == right.high;
}

}  // namespace

axis_ir_validation_code_v1
validate_domain_ir_type_v1(const domain_ir_type_v1& domain) noexcept {
    return domain.identity.valid() && !domain.nominal_tag.empty()
        ? axis_ir_validation_code_v1::success
        : axis_ir_validation_code_v1::invalid_domain;
}

axis_ir_validation_code_v1 validate_axis_ir_type_v1(const axis_ir_type_v1& axis) noexcept {
    if (!axis.identity.valid()) return axis_ir_validation_code_v1::invalid_axis_identity;
    if (validate_domain_ir_type_v1(axis.domain) != axis_ir_validation_code_v1::success)
        return axis_ir_validation_code_v1::invalid_domain;
    if (!axis.order.identity.valid() || !same(axis.order.domain, axis.domain.identity))
        return axis_ir_validation_code_v1::invalid_order;
    if (!axis.geometry.identity.valid() || !same(axis.geometry.domain, axis.domain.identity))
        return axis_ir_validation_code_v1::invalid_geometry;
    if (!axis.partition.identity.valid() || !same(axis.partition.domain, axis.domain.identity))
        return axis_ir_validation_code_v1::invalid_partition;

    switch (axis.extent.kind) {
    case extent_knowledge_kind_v1::unknown:
        if (axis.extent.lower_bound != 0 || axis.extent.upper_bound != 0)
            return axis_ir_validation_code_v1::invalid_extent;
        break;
    case extent_knowledge_kind_v1::bounded:
        if (axis.extent.upper_bound == 0 ||
            axis.extent.lower_bound > axis.extent.upper_bound)
            return axis_ir_validation_code_v1::invalid_extent;
        break;
    case extent_knowledge_kind_v1::exact:
        if (axis.extent.lower_bound != axis.extent.upper_bound)
            return axis_ir_validation_code_v1::invalid_extent;
        break;
    }

    const auto local_extent = axis.extent.kind == extent_knowledge_kind_v1::exact
        ? axis.extent.upper_bound : 0;
    switch (axis.recovery.kind) {
    case identity_recovery_kind_v1::identity:
        if (axis.recovery.stored_space != axis_identity_space_v1::global ||
            axis.recovery.affine_base != 0 || !axis.recovery.local_to_global.empty())
            return axis_ir_validation_code_v1::invalid_recovery;
        break;
    case identity_recovery_kind_v1::affine:
        if (axis.recovery.stored_space != axis_identity_space_v1::partition_local ||
            !axis.recovery.local_to_global.empty() ||
            (local_extent != 0 &&
             (axis.recovery.affine_base > axis.recovery.global_extent ||
              local_extent > axis.recovery.global_extent - axis.recovery.affine_base)))
            return axis_ir_validation_code_v1::invalid_recovery;
        break;
    case identity_recovery_kind_v1::explicit_map:
        if (axis.recovery.stored_space != axis_identity_space_v1::partition_local ||
            (local_extent != 0 && axis.recovery.local_to_global.size() != local_extent) ||
            std::any_of(axis.recovery.local_to_global.begin(),
                        axis.recovery.local_to_global.end(),
                        [&axis](std::uint64_t identity) {
                            return identity >= axis.recovery.global_extent;
                        }))
            return axis_ir_validation_code_v1::invalid_recovery;
        break;
    }
    return axis_ir_validation_code_v1::success;
}

axis_ir_validation_code_v1 validate_axis_ir_against_biological_abi_v1(
    const axis_ir_type_v1& axis,
    const cellerator::execution::persistent_axis_identity& abi_axis) noexcept {
    const auto status = validate_axis_ir_type_v1(axis);
    if (status != axis_ir_validation_code_v1::success) return status;
    if (cellerator::execution::validate_persistent_axis_identity(abi_axis) !=
            cellerator::execution::biological_validation_code::ok ||
        !same(axis.domain.identity, abi_axis.domain) ||
        !same(axis.order.identity, abi_axis.order) ||
        !same(axis.geometry.identity, abi_axis.geometry) ||
        !same(axis.partition.identity, abi_axis.partition)) {
        return axis_ir_validation_code_v1::biological_abi_mismatch;
    }
    return axis_ir_validation_code_v1::success;
}

}  // namespace Cellerator::compiler::ir::semantic
