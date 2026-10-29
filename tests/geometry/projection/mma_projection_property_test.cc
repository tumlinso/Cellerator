#define main ce_geo_mma_projection_roundtrip_regression
#include "mma_projection_roundtrip_test.cc"
#undef main

#include <Cellerator/execution/projection_activation_v2.hh>

namespace {

template<typename T>
T load_record(const std::vector<unsigned char> &image, std::uint64_t offset,
    std::uint64_t index = 0u) {
    T value{};
    std::memcpy(&value, image.data() + offset + index * sizeof(T), sizeof(T));
    return value;
}

template<typename T>
void store_record(std::vector<unsigned char> *image, std::uint64_t offset,
    const T &value, std::uint64_t index = 0u) {
    std::memcpy(image->data() + offset + index * sizeof(T), &value, sizeof(T));
}

void require_rejected(const std::vector<unsigned char> &image) {
    assert(!projection::validate_physical_mma_hybrid_image_v1(
        image.data(), image.size()));
}

} // namespace

int main() {
    assert(ce_geo_mma_projection_roundtrip_regression() == 0);
    const std::vector<unsigned char> valid = physical_image();
    assert(projection::validate_physical_mma_hybrid_image_v1(
        valid.data(), valid.size()));
    const auto header = load_record<projection::physical_mma_hybrid_header_v1>(
        valid, 0u);

    // A cover cannot omit an edge named by its logical cardinality.
    std::vector<unsigned char> missing = valid;
    auto missing_header = header;
    ++missing_header.logical_edge_count;
    store_record(&missing, 0u, missing_header);
    require_rejected(missing);

    // Logical ownership is exactly once across MMA and residual regions.
    std::vector<unsigned char> duplicate = valid;
    auto duplicate_map = load_record<projection::projection_value_map_v1>(
        duplicate, header.value_map_offset, 1u);
    duplicate_map.logical_edge_id.value = 0u;
    store_record(&duplicate, header.value_map_offset, duplicate_map, 1u);
    require_rejected(duplicate);

    // Padding cannot enlarge a physical group past the fixed 16x16 contract.
    std::vector<unsigned char> bad_padding = valid;
    auto group = load_record<projection::physical_group_v1>(
        bad_padding, header.source_group_offset);
    group.padded_count = 17u;
    store_record(&bad_padding, header.source_group_offset, group);
    require_rejected(bad_padding);

    // Residual row ownership must account for every residual contribution.
    std::vector<unsigned char> bad_residual = valid;
    std::uint32_t terminal_offset = 0u;
    store_record(&bad_residual, header.residual_row_offset_offset,
        terminal_offset, 1u);
    require_rejected(bad_residual);

    // Width tags and projection slots are independently recoverable facts.
    std::vector<unsigned char> bad_width = valid;
    auto width_map = load_record<projection::projection_value_map_v1>(
        bad_width, header.value_map_offset);
    width_map.logical_edge_id.width = projection::logical_edge_id_width_v1::u64;
    store_record(&bad_width, header.value_map_offset, width_map);
    require_rejected(bad_width);

    std::vector<unsigned char> bad_slot = valid;
    auto slot_map = load_record<projection::projection_value_map_v1>(
        bad_slot, header.value_map_offset, 1u);
    slot_map.projection_slot = 1u;
    store_record(&bad_slot, header.value_map_offset, slot_map, 1u);
    require_rejected(bad_slot);

    // Provider erasure retains exact projection, contract, and device identity.
    cellerator::compute::math::core::projection_key key{};
    key.persistent = {0x91u, 0x92u};
    key.runtime = {1u, 1u};
    key.kind = cellerator::compute::math::core::projection_kind::
        architecture_specific;
    key.schema_version = 1u;
    key.variant = 2u;
    cellerator::compute::math::core::candidate_projection_contract_v2 contract{};
    contract.view_type = {0xa1u, 0xa2u};
    contract.abi_major = 1u;
    contract.schema_version = 1u;
    contract.variant = 2u;
    cellerator::execution::projection_reference_binding_v2 binding{};
    binding.key = key;
    binding.provider_identity = {0xb1u, 0xb2u};
    binding.capability_identity = {0xc1u, 0xc2u};
    binding.contract = contract;
    binding.location = {cellerator::execution::residency_kind::device, {}, 0, 0u};
    binding.view = valid.data();
    binding.view_bytes = valid.size();
    cellerator::execution::activated_projection_reference_v2 activated{};
    assert(cellerator::execution::make_activated_projection_reference_v2(
        binding, &activated)
        == cellerator::execution::projection_reference_status_v2::success);
    assert(cellerator::execution::validate_activated_projection_reference_v2(
        activated)
        == cellerator::execution::projection_reference_status_v2::success);
    binding.view = nullptr;
    assert(cellerator::execution::make_activated_projection_reference_v2(
        binding, &activated)
        == cellerator::execution::projection_reference_status_v2::invalid_view);

    // Header corruption remains fail-closed before any section is consumed.
    std::vector<unsigned char> corrupt = valid;
    auto corrupt_header = header;
    corrupt_header.reserved[0] = 1u;
    store_record(&corrupt, 0u, corrupt_header);
    require_rejected(corrupt);
    return 0;
}
