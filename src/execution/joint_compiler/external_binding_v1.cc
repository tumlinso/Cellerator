#include <Cellerator/execution/joint_compiler/external_binding_v1.hh>

#include <cstdint>
#include <limits>

namespace cellerator::execution::joint_compiler {
namespace {

external_binding_validation_result_v1 failure(
    external_binding_validation_code_v1 code,
    std::uint64_t extent = 0u) noexcept {
    return {code, extent};
}

bool valid_token(opaque_runtime_token_v1 token) noexcept {
    return token.slot != 0u && token.generation != 0u;
}

}  // namespace

external_binding_validation_result_v1 validate_external_binding_v1(
    const external_binding_v1 &binding) noexcept {
    if (binding.schema_version != external_binding_schema_version_v1)
        return failure(external_binding_validation_code_v1::unsupported_schema);
    if (binding.record_bytes != sizeof(external_binding_v1))
        return failure(
            external_binding_validation_code_v1::invalid_record_bytes);
    if (!validate_persistent_identity_v1(binding.binding_identity))
        return failure(
            external_binding_validation_code_v1::invalid_binding_identity);
    if (!validate_persistent_identity_v1(binding.atom_identity))
        return failure(
            external_binding_validation_code_v1::invalid_atom_identity);
    if (!validate_persistent_identity_v1(binding.plane_identity))
        return failure(
            external_binding_validation_code_v1::invalid_plane_identity);
    if (binding.extent_count == 0u
        || binding.extent_count > maximum_external_extents_v1)
        return failure(
            external_binding_validation_code_v1::invalid_extent_count);
    if (binding.extents == nullptr)
        return failure(external_binding_validation_code_v1::missing_extents);

    std::uint64_t expected_offset = 0u;
    order_id expected_order{};
    value_generation expected_generation{};
    for (std::uint64_t index = 0u; index < binding.extent_count; ++index) {
        const external_extent_v1 &extent = binding.extents[index];
        if (extent.address == nullptr)
            return failure(
                external_binding_validation_code_v1::missing_address, index);
        if (!valid_location(extent.location))
            return failure(
                external_binding_validation_code_v1::invalid_location, index);
        if (extent.location.address_space == 0u)
            return failure(external_binding_validation_code_v1::
                invalid_address_space, index);
        if (extent.alignment == 0u
            || (extent.alignment & (extent.alignment - 1u)) != 0u)
            return failure(
                external_binding_validation_code_v1::invalid_alignment, index);
        if (reinterpret_cast<std::uintptr_t>(extent.address)
            % extent.alignment != 0u)
            return failure(
                external_binding_validation_code_v1::misaligned_address, index);
        if (extent.bytes == 0u)
            return failure(
                external_binding_validation_code_v1::empty_extent, index);
        if (extent.plane_byte_offset != expected_offset)
            return failure(external_binding_validation_code_v1::
                extent_offset_mismatch, index);
        if (expected_offset
            > std::numeric_limits<std::uint64_t>::max() - extent.bytes)
            return failure(
                external_binding_validation_code_v1::extent_overflow, index);
        if (!valid_identity(extent.order))
            return failure(
                external_binding_validation_code_v1::invalid_order, index);
        if (extent.generation.value == 0u)
            return failure(
                external_binding_validation_code_v1::invalid_generation, index);
        if (index == 0u) {
            expected_order = extent.order;
            expected_generation = extent.generation;
        } else {
            if (!same_identity(extent.order, expected_order))
                return failure(external_binding_validation_code_v1::
                    inconsistent_order, index);
            if (extent.generation.value != expected_generation.value)
                return failure(external_binding_validation_code_v1::
                    inconsistent_generation, index);
        }
        if (!valid_token(extent.readiness))
            return failure(external_binding_validation_code_v1::
                invalid_readiness_token, index);
        if (!valid_token(extent.lease))
            return failure(
                external_binding_validation_code_v1::invalid_lease_token, index);
        expected_offset += extent.bytes;
    }
    if (expected_offset != binding.total_bytes)
        return failure(
            external_binding_validation_code_v1::total_bytes_mismatch);
    return {};
}

}  // namespace cellerator::execution::joint_compiler
