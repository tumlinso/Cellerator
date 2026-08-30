#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/compute/projection/physical_mma_hybrid.hh>
#include <Cellerator/execution/projection_activation_v2.hh>
#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>
#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace architecture = cellerator::compute::architecture;
namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace persistence = cellpack::persistence;
namespace projection = cellerator::compute::projection;

namespace cellerator::compute::architecture::providers::nvidia {
bool realize_mma_provider_projection_v1(
    const architecture_provider_v1 &, const matrix_engine_capability_v1 &,
    const void *, std::size_t, core::stable_id, core::stable_id,
    core::stable_id, core::stable_id, core::stable_id, core::stable_id,
    std::uint32_t, std::uint32_t, core::projection_key,
    core::candidate_projection_contract_v2, execution::device_location,
    const void *, persistence::execution_section_source *,
    persistence::execution_projection_source *,
    persistence::execution_capability_manifest_v1 *,
    execution::activated_projection_reference_v2 *) noexcept;
}

namespace {

std::size_t align64(std::size_t value) {
    return (value + 63u) & ~std::size_t{63u};
}

template<typename T>
std::size_t append_records(std::vector<unsigned char> *image,
    const T *records, std::size_t count) {
    const std::size_t offset = align64(image->size());
    image->resize(offset + sizeof(T) * count);
    std::memcpy(image->data() + offset, records, sizeof(T) * count);
    return offset;
}

std::vector<unsigned char> physical_image(
    architecture::architecture_identity_v1 provider,
    architecture::architecture_identity_v1 capability) {
    std::vector<unsigned char> image(
        sizeof(projection::physical_mma_hybrid_header_v1));
    projection::physical_group_v1 group{};
    group.group_id = 0u;
    group.semantic_component_id = 1u;
    group.member_count = 1u;
    group.padded_count = 16u;
    projection::mma_tile_v1 tile{};
    tile.tile_id = 0u;
    tile.source_group_index = 0u;
    tile.destination_group_index = 0u;
    tile.semantic_component_id = 1u;
    tile.occupancy_mask[0] = 1u;
    tile.compact_slot_count = 1u;
    projection::mma_compact_slot_v1 slot{};
    slot.logical_edge_index = 0u;
    projection::residual_region_v1 residual{};
    residual.region_id = 0u;
    residual.semantic_component_id = 1u;
    residual.destination_group_index = 0u;
    residual.row_count = 1u;
    residual.edge_count = 1u;
    residual.value_map_offset = 1u;
    const std::uint32_t row_offsets[] = {0u, 1u};
    const std::uint32_t columns[] = {0u};
    projection::projection_schedule_entry_v1 schedules[2]{};
    schedules[0].kind = projection::schedule_work_kind_v1::mma_tile;
    schedules[0].work_index = 0u;
    schedules[0].destination_group_index = 0u;
    schedules[0].dense_column_count = 64u;
    schedules[1].kind = projection::schedule_work_kind_v1::residual_region;
    schedules[1].work_index = 0u;
    schedules[1].destination_group_index = 0u;
    schedules[1].dense_column_count = 64u;
    projection::projection_value_map_v1 maps[2]{};
    maps[0].logical_edge_id.value = 0u;
    maps[0].region_kind = projection::physical_region_kind_v1::mma;
    maps[0].region_index = 0u;
    maps[0].projection_slot = 0u;
    maps[1].logical_edge_id.value = 1u;
    maps[1].region_kind = projection::physical_region_kind_v1::residual;
    maps[1].region_index = 0u;
    maps[1].projection_slot = 0u;

    projection::physical_mma_hybrid_header_v1 header{};
    header.structure_identity.identity_version = 1u;
    header.structure_identity.value = 0x101u;
    header.source_order.feature_count = 1u;
    header.source_order.feature_axis_identity_version = 1u;
    header.source_order.feature_axis_identity = 0x201u;
    header.destination_order.feature_count = 1u;
    header.destination_order.feature_axis_identity_version = 1u;
    header.destination_order.feature_axis_identity = 0x301u;
    header.provider_identity_low = provider.low;
    header.provider_identity_high = provider.high;
    header.capability_identity_low = capability.low;
    header.capability_identity_high = capability.high;
    header.logical_edge_count = 2u;
    header.dense_width = 64u;
    header.source_group_count = 1u;
    header.destination_group_count = 1u;
    header.tile_count = 1u;
    header.compact_slot_count = 1u;
    header.residual_region_count = 1u;
    header.schedule_entry_count = 2u;
    header.value_map_count = 2u;
    header.source_group_offset = append_records(&image, &group, 1u);
    header.destination_group_offset = append_records(&image, &group, 1u);
    header.tile_offset = append_records(&image, &tile, 1u);
    header.compact_slot_offset = append_records(&image, &slot, 1u);
    header.residual_region_offset = append_records(&image, &residual, 1u);
    header.residual_row_offset_offset = append_records(&image, row_offsets, 2u);
    header.residual_column_index_offset = append_records(&image, columns, 1u);
    header.schedule_entry_offset = append_records(&image, schedules, 2u);
    header.value_map_offset = append_records(&image, maps, 2u);
    header.image_bytes = image.size();
    std::memcpy(image.data(), &header, sizeof(header));
    return image;
}

architecture::matrix_memory_operand_contract_v1 memory_contract() {
    architecture::matrix_memory_operand_contract_v1 result{};
    result.base_alignment_bytes = 16u;
    result.address_space_flags = architecture::memory_address_global;
    result.access_flags = architecture::memory_operand_read;
    return result;
}

} // namespace

int main() {
    const architecture::architecture_identity_v1 provider_id{0x11u, 0x12u};
    const architecture::architecture_identity_v1 capability_id{0x21u, 0x22u};
    const architecture::architecture_identity_v1 memory_id{0x31u, 0x32u};
    architecture::matrix_memory_interface_v1 memory{};
    memory.identity = memory_id;
    memory.flags = architecture::memory_interface_operand_a
        | architecture::memory_interface_operand_b;
    memory.operand_a = memory_contract();
    memory.operand_b = memory_contract();

    architecture::matrix_engine_capability_v1 capability{};
    capability.identity = capability_id;
    capability.provider_identity = provider_id;
    capability.memory_interface_identity = memory_id;
    capability.vendor = architecture::architecture_vendor_v1::nvidia;
    capability.architecture_class = 1u;
    capability.minimum_compute_major = 7u;
    capability.maximum_compute_major = 7u;
    capability.instruction_family =
        architecture::matrix_instruction_family_v1::nvidia_wmma;
    capability.collective_scope = architecture::collective_scope_v1::warp;
    capability.collective_threads = 32u;
    capability.instruction_m = 16u;
    capability.instruction_n = 16u;
    capability.instruction_k = 16u;
    capability.operand_a_type = execution::numeric_type::f16;
    capability.operand_b_type = execution::numeric_type::f16;
    capability.accumulation_type = execution::numeric_type::f32;
    capability.output_type = execution::numeric_type::f32;
    capability.operand_a_layout = architecture::matrix_layout_v1::row_major;
    capability.operand_b_layout = architecture::matrix_layout_v1::row_major;
    capability.accumulation_layout = architecture::matrix_layout_v1::opaque;
    capability.output_layout = architecture::matrix_layout_v1::row_major;
    capability.instruction_sparsity =
        architecture::instruction_sparsity_v1::dense;
    capability.flags = architecture::capability_source_linked_implementation
        | architecture::capability_fragment_layout_opaque
        | architecture::capability_requires_converged_collective
        | architecture::capability_memory_interface_present;
    capability.engine_requirements =
        architecture::matrix_engine_multiply_accumulate;
    assert(architecture::validate_matrix_engine_capability_v1(capability)
        == architecture::capability_status_v1::success);

    architecture::architecture_provider_v1 provider{};
    provider.identity = provider_id;
    provider.name = "test-sm70";
    provider.capabilities = &capability;
    provider.capability_count = 1u;
    provider.memory_interfaces = &memory;
    provider.memory_interface_count = 1u;
    std::vector<unsigned char> image = physical_image(provider_id, capability_id);
    void *device_view = nullptr;
    assert(cudaMalloc(&device_view, image.size()) == cudaSuccess);
    assert(cudaMemcpy(device_view, image.data(), image.size(),
        cudaMemcpyHostToDevice) == cudaSuccess);

    core::projection_key key{};
    key.persistent = {0x91u, 0x92u};
    key.runtime = {1u, 1u};
    key.kind = core::projection_kind::architecture_specific;
    key.schema_version = 1u;
    key.variant = 3u;
    core::candidate_projection_contract_v2 contract{};
    contract.view_type = {0xa1u, 0xa2u};
    contract.abi_major = 1u;
    contract.schema_version = 1u;
    contract.variant = 3u;
    const execution::device_location location{
        execution::residency_kind::device, {}, 0, 0u};

    for (std::uint64_t cover = 1u; cover <= 3u; ++cover) {
        persistence::execution_section_source sections[2]{};
        persistence::execution_projection_source source{};
        persistence::execution_capability_manifest_v1 manifest{};
        execution::activated_projection_reference_v2 activated{};
        assert(cellerator::compute::architecture::providers::nvidia::
            realize_mma_provider_projection_v1(provider, capability,
                image.data(), image.size(), {0x41u, 0x42u}, {0x51u, 0x52u},
                {0x61u, 0x62u}, {0x71u, 0x72u}, {0x80u + cover, 0x81u},
                {0x90u + cover, 0x91u}, 0u, 1u, key, contract, location,
                device_view, sections, &source, &manifest, &activated));
        assert(sections[0].kind ==
            persistence::execution_section_kind::projection_payload);
        assert(sections[1].kind ==
            persistence::execution_capability_manifest_v1_section_kind);
        assert(source.entry.capability_section == 1u);
        assert(persistence::validate_execution_capability_manifest_v1(manifest));
        assert(manifest.provider_identity_low == provider_id.low);
        assert(manifest.capability_identity_low == capability_id.low);
        assert(execution::validate_activated_projection_reference_v2(activated)
            == execution::projection_reference_status_v2::success);
        assert(activated.view == device_view);
    }

    architecture::matrix_engine_capability_v1 mismatched = capability;
    mismatched.provider_identity.low ^= 1u;
    persistence::execution_section_source sections[2]{};
    persistence::execution_projection_source source{};
    persistence::execution_capability_manifest_v1 manifest{};
    execution::activated_projection_reference_v2 activated{};
    assert(!cellerator::compute::architecture::providers::nvidia::
        realize_mma_provider_projection_v1(provider, mismatched,
            image.data(), image.size(), {1u, 1u}, {2u, 2u}, {3u, 3u},
            {4u, 4u}, {5u, 5u}, {6u, 6u}, 0u, 1u, key, contract, location,
            device_view, sections, &source, &manifest, &activated));
    assert(cudaFree(device_view) == cudaSuccess);
    return 0;
}
