#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/relation_apply_v1.cuh>

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cellerator::compute::architecture::nvidia::sm70::relation_apply;

int main() {
    const sm70_apply_inventory_v1 inventory = built_in_sm70_apply_inventory_v1();
    assert(validate_sm70_apply_inventory_v1(inventory)
        == apply_inventory_status_v1::success);
    assert(inventory.candidate_count == 15u);
    assert(inventory.candidates[10].identity.classification
        == catalog_v3::candidate_class::experimental);
    assert(inventory.candidates[10].identity.requires_measurement);

    std::vector<apply_candidate_registration_v1> registrations(
        inventory.candidate_count);
    std::vector<apply_resource_receipt_v1> receipts(inventory.candidate_count);
    assert(register_sm70_apply_candidates_v1(inventory, {}, false,
        {registrations.data(), registrations.size(), receipts.data(),
            receipts.size()}) == apply_registration_status_v1::success);
    for (std::uint64_t index = 0u; index < inventory.candidate_count; ++index) {
        assert(registrations[index].candidate == &inventory.candidates[index]);
        assert(registrations[index].kernel_symbol == nullptr);
        assert(receipts[index].state
            == apply_resource_receipt_state_v1::declared_only);
        assert(receipts[index].stage_id
            == inventory.candidates[index].stages[0].stage_id);
    }

    const auto fake_half = reinterpret_cast<const __half *>(0x1000u);
    const auto fake_u32 = reinterpret_cast<const std::uint32_t *>(0x2000u);
    const auto fake_float = reinterpret_cast<float *>(0x3000u);
    compact_apply_component_v1 component{fake_half, fake_u32, fake_u32,
        fake_half, fake_float, (std::uint64_t{1} << 32u) + 7u,
        1u, 5u, 16u, 16u};
    apply_n16_n32_request_v1 n16{component,
        apply_n16_n32_variant_v1::n16_feature_major, {}, 1u, nullptr};
    apply_launch_shape_v1 small_shape{};
    assert(validate_apply_n16_n32_v1(n16, &small_shape)
        == apply_launch_status_v1::success);
    assert(small_shape.grid_x == 5u && small_shape.block_x == 32u);
    n16.variant = apply_n16_n32_variant_v1::n32_dual_output_owner;
    n16.component.dense_width = 32u;
    assert(validate_apply_n16_n32_v1(n16, &small_shape)
        == apply_launch_status_v1::success);
    assert(small_shape.grid_x == 3u && small_shape.block_x == 128u);

    apply_n64_request_v1 n64{component, apply_n64_variant_v1::direct_global,
        {}, 2u, nullptr};
    n64.component.dense_width = 64u;
    apply_n64_launch_shape_v1 n64_shape{};
    assert(validate_apply_n64_v1(n64, &n64_shape)
        == apply_launch_status_v1::success);
    assert(n64_shape.dynamic_shared_bytes == 0u);
    n64.variant = apply_n64_variant_v1::shared_a;
    assert(validate_apply_n64_v1(n64, &n64_shape)
        == apply_launch_status_v1::success);
    assert(n64_shape.dynamic_shared_bytes == 512u);
    n64.variant = apply_n64_variant_v1::software_pipeline;
    assert(validate_apply_n64_v1(n64, &n64_shape)
        == apply_launch_status_v1::success);
    assert(n64_shape.dynamic_shared_bytes == 1024u);

    apply_wide_panels_request_v1 wide{component, 2u, 3u,
        (std::uint64_t{1} << 32u) + 9u, 3u, nullptr};
    wide.component.dense_width = 80u;
    apply_wide_panels_shape_v1 wide_shape{};
    assert(validate_apply_wide_panels_v1(wide, &wide_shape)
        == apply_launch_status_v1::success);
    assert(wide_shape.grid_y == 3u);
    wide.panel_count = 4u;
    assert(validate_apply_wide_panels_v1(wide, &wide_shape)
        == apply_launch_status_v1::invalid_argument);

    apply_wmma_shape_request_v1 wmma{fake_half, fake_u32, fake_u32,
        fake_half, fake_float, 0u, 1u, 1u, 16u,
        apply_wmma_shape_v1::m8n32k16, {}, 4u, nullptr};
    apply_wmma_shape_launch_v1 wmma_shape{};
    assert(validate_apply_wmma_shape_v1(wmma, &wmma_shape)
        == apply_launch_status_v1::success);
    assert(wmma_shape.output_rows == 8u && wmma_shape.output_columns == 32u);

    apply_ptx_m8n8k4_request_v1 ptx{fake_u32, fake_u32, fake_float,
        1u, 0u, 5u, nullptr};
    assert(validate_apply_ptx_m8n8k4_experiment_v1(ptx)
        == apply_launch_status_v1::success);
    assert(apply_ptx_m8n8k4_experimental_v1
        && apply_ptx_m8n8k4_requires_measurement_v1);

    constexpr std::uint32_t rows = 16u;
    constexpr std::uint32_t width = 80u;
    std::array<float, rows * 16u> relation{};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        relation[row * 16u + row] = 1.0f;
    }
    std::array<float, 16u * width> rhs{};
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        for (std::uint32_t column = 0u; column < width; ++column) {
            rhs[row * width + column] =
                static_cast<float>(row * width + column);
        }
    }
    std::array<float, rows * width> output{};
    const std::array<std::uint32_t, 2> offsets{{0u, 1u}};
    const std::array<std::uint32_t, 1> source_bases{{0u}};
    assert(apply_dense_tile_reference_v1({relation.data(), offsets.data(),
        source_bases.data(), rhs.data(), output.data(), output.size(), 1u, 1u,
        16u, rows, width}) == apply_reference_status_v1::success);
    assert(output == rhs);

    auto overflow_component = component;
    overflow_component.global_destination_group_base =
        std::numeric_limits<std::uint64_t>::max() - 2u;
    overflow_component.destination_group_count = 5u;
    n16.component = overflow_component;
    n16.component.dense_width = 32u;
    assert(validate_apply_n16_n32_v1(n16, &small_shape)
        == apply_launch_status_v1::arithmetic_overflow);
}
