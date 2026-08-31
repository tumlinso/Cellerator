#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_inventory_v1.hh"

#include <array>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

using catalog_v3::candidate_class;
using catalog_v3::candidate_descriptor_v3;
using catalog_v3::candidate_identity_v3;
using catalog_v3::candidate_resource_v3;
using catalog_v3::candidate_stage_v3;
using catalog_v3::numerical_mode;

constexpr std::uint64_t sm70_provider_id = 0x534d3730u;
constexpr std::uint64_t sm70_device_class_id = 0x7000u;
constexpr std::uint64_t relation_apply_operation_id = 1u;
constexpr std::uint64_t apply_capability_id = 0x4150504c59u;
constexpr std::uint64_t projection_sparse_id = 1u;
constexpr std::uint64_t projection_mma_id = 2u;
constexpr std::uint64_t projection_hybrid_id = 3u;

template<std::size_t Size>
constexpr candidate_stage_v3 make_stage(
    std::uint64_t stage_id,
    const char (&name)[Size]) noexcept {
    static_assert(Size <= sizeof(candidate_stage_v3::stable_name),
        "stable profiler stage name is too long");
    candidate_stage_v3 stage{};
    stage.stage_id = stage_id;
    stage.kernel_id = stage_id;
    stage.stage_kind = 4u;
    stage.launch_count = 1u;
    for (std::size_t index = 0u; index < Size; ++index) {
        stage.stable_name[index] = name[index];
    }
    return stage;
}

constexpr candidate_identity_v3 make_identity(
    std::uint64_t candidate_id,
    std::uint64_t projection_id,
    std::uint32_t width_min,
    std::uint32_t width_max,
    bool experimental = false) noexcept {
    candidate_identity_v3 identity{};
    identity.candidate_id = candidate_id;
    identity.provider_id = sm70_provider_id;
    identity.device_class_id = sm70_device_class_id;
    identity.projection_type_id = projection_id;
    identity.capability_id = apply_capability_id;
    identity.operation_id = relation_apply_operation_id;
    identity.width_min = width_min;
    identity.width_max = width_max;
    identity.numerics = numerical_mode::precise;
    identity.classification = experimental
        ? candidate_class::experimental : candidate_class::production;
    identity.requires_measurement = experimental;
    return identity;
}

constexpr std::array<candidate_stage_v3, 15> stages{{
    make_stage(0x15001u, "ce_sm70_apply_n16_feature_major"),
    make_stage(0x15002u, "ce_sm70_apply_n32_row_owner"),
    make_stage(0x15003u, "ce_sm70_apply_n32_dual_warp"),
    make_stage(0x15004u, "ce_sm70_apply_n64_direct_global"),
    make_stage(0x15005u, "ce_sm70_apply_n64_shared_a"),
    make_stage(0x15006u, "ce_sm70_apply_n64_software_pipeline"),
    make_stage(0x15007u, "ce_sm70_apply_wide_disjoint_panels"),
    make_stage(0x15008u, "ce_sm70_apply_wmma_m16n16k16"),
    make_stage(0x15009u, "ce_sm70_apply_wmma_m8n32k16"),
    make_stage(0x1500au, "ce_sm70_apply_wmma_m32n8k16"),
    make_stage(0x1500bu, "ce_sm70_apply_ptx_m8n8k4_experiment"),
    make_stage(0x1500cu, "ce_sm70_apply_pure_sparse"),
    make_stage(0x1500du, "ce_sm70_apply_hybrid_mma_residual"),
    make_stage(0x1500eu, "ce_sm70_apply_canonical_input"),
    make_stage(0x1500fu, "ce_sm70_apply_persistent_physical_input"),
}};

constexpr std::array<candidate_descriptor_v3, 15> candidates{{
    {make_identity(0x15001u, projection_sparse_id, 16u, 16u), &stages[0], 1u, 0u, {0u, 0u, 128u, 0u}},
    {make_identity(0x15002u, projection_sparse_id, 32u, 32u), &stages[1], 1u, 0u, {0u, 0u, 128u, 0u}},
    {make_identity(0x15003u, projection_sparse_id, 32u, 32u), &stages[2], 1u, 0u, {0u, 0u, 128u, 0u}},
    {make_identity(0x15004u, projection_sparse_id, 64u, 64u), &stages[3], 1u, 0u, {0u, 0u, 256u, 0u}},
    {make_identity(0x15005u, projection_sparse_id, 64u, 64u), &stages[4], 1u, 0u, {0u, 0u, 256u, 4096u}},
    {make_identity(0x15006u, projection_sparse_id, 64u, 64u), &stages[5], 1u, 0u, {0u, 0u, 256u, 8192u}},
    {make_identity(0x15007u, projection_sparse_id, 65u, std::numeric_limits<std::uint32_t>::max()), &stages[6], 1u, 0u, {0u, 0u, 256u, 0u}},
    {make_identity(0x15008u, projection_mma_id, 16u, 64u), &stages[7], 1u, 0u, {0u, 0u, 128u, 4096u}},
    {make_identity(0x15009u, projection_mma_id, 32u, 64u), &stages[8], 1u, 0u, {0u, 0u, 128u, 4096u}},
    {make_identity(0x1500au, projection_mma_id, 8u, 64u), &stages[9], 1u, 0u, {0u, 0u, 128u, 4096u}},
    {make_identity(0x1500bu, projection_mma_id, 8u, 32u, true), &stages[10], 1u, 0u, {0u, 0u, 128u, 0u}},
    {make_identity(0x1500cu, projection_sparse_id, 1u, std::numeric_limits<std::uint32_t>::max()), &stages[11], 1u, 0u, {0u, 0u, 128u, 0u}},
    {make_identity(0x1500du, projection_hybrid_id, 8u, std::numeric_limits<std::uint32_t>::max()), &stages[12], 1u, 0u, {0u, 0u, 256u, 8192u}},
    {make_identity(0x1500eu, projection_sparse_id, 1u, std::numeric_limits<std::uint32_t>::max()), &stages[13], 1u, 0u, {0u, 0u, 128u, 0u}},
    {make_identity(0x1500fu, projection_sparse_id, 1u, std::numeric_limits<std::uint32_t>::max()), &stages[14], 1u, 0u, {0u, 0u, 128u, 0u}},
}};

constexpr std::array<apply_candidate_capability_v1, 15> capabilities{{
    {apply_mechanism_v1::feature_major_n16, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 16u, 0u},
    {apply_mechanism_v1::n32_row_owner, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 32u, 0u},
    {apply_mechanism_v1::n32_dual_warp, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 32u, 0u},
    {apply_mechanism_v1::n64_direct_global, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 64u, 0u},
    {apply_mechanism_v1::n64_shared_a, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 64u, 0u},
    {apply_mechanism_v1::n64_software_pipeline, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1 | apply_requires_measurement_v1, 64u, 0u},
    {apply_mechanism_v1::wide_disjoint_panels, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1 | apply_disjoint_panels_v1, 64u, 0u},
    {apply_mechanism_v1::wmma_m16n16k16, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_mma_v1 | apply_requires_measurement_v1, 16u, 0u},
    {apply_mechanism_v1::wmma_m8n32k16, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_mma_v1 | apply_requires_measurement_v1, 32u, 0u},
    {apply_mechanism_v1::wmma_m32n8k16, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_mma_v1 | apply_requires_measurement_v1, 8u, 0u},
    {apply_mechanism_v1::ptx_mma_m8n8k4_experiment, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_mma_v1 | apply_experimental_v1 | apply_requires_measurement_v1, 8u, 0u},
    {apply_mechanism_v1::pure_sparse, apply_input_order_v1::either_explicit, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 0u, 0u},
    {apply_mechanism_v1::hybrid_mma_residual, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_mma_v1 | apply_residual_v1 | apply_requires_measurement_v1, 0u, 0u},
    {apply_mechanism_v1::canonical_input, apply_input_order_v1::canonical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 0u, 0u},
    {apply_mechanism_v1::persistent_physical_input, apply_input_order_v1::projection_physical, 0u, apply_profiler_visible_v1 | apply_pure_sparse_v1, 0u, 0u},
}};

bool valid_mechanism(apply_mechanism_v1 mechanism) noexcept {
    return mechanism >= apply_mechanism_v1::feature_major_n16
        && mechanism <= apply_mechanism_v1::persistent_physical_input;
}

}  // namespace

sm70_apply_inventory_v1 built_in_sm70_apply_inventory_v1() noexcept {
    return {sm70_apply_inventory_schema_v1, 0u, candidates.data(),
        capabilities.data(), candidates.size()};
}

apply_inventory_status_v1 validate_sm70_apply_inventory_v1(
    const sm70_apply_inventory_v1 &inventory) noexcept {
    if (inventory.schema_version != sm70_apply_inventory_schema_v1
        || inventory.candidates == nullptr || inventory.capabilities == nullptr
        || inventory.candidate_count == 0u) {
        return apply_inventory_status_v1::invalid_argument;
    }
    std::uint64_t previous_candidate_id = 0u;
    std::uint64_t previous_stage_id = 0u;
    for (std::uint64_t index = 0u; index < inventory.candidate_count; ++index) {
        const candidate_descriptor_v3 &candidate = inventory.candidates[index];
        const apply_candidate_capability_v1 &capability =
            inventory.capabilities[index];
        if (candidate.identity.candidate_id == 0u
            || candidate.identity.candidate_id <= previous_candidate_id
            || candidate.identity.provider_id != sm70_provider_id
            || candidate.identity.device_class_id != sm70_device_class_id
            || candidate.identity.operation_id != relation_apply_operation_id
            || candidate.identity.width_min == 0u
            || candidate.identity.width_min > candidate.identity.width_max
            || candidate.stage_count == 0u || candidate.stages == nullptr
            || candidate.resources.threads_per_cta == 0u
            || !valid_mechanism(capability.mechanism)
            || (capability.flags & apply_profiler_visible_v1) == 0u) {
            return apply_inventory_status_v1::invalid_candidate;
        }
        if (candidate.identity.classification == candidate_class::experimental
            && ((capability.flags & apply_experimental_v1) == 0u
                || !candidate.identity.requires_measurement)) {
            return apply_inventory_status_v1::invalid_candidate;
        }
        for (std::uint32_t stage_index = 0u;
             stage_index < candidate.stage_count; ++stage_index) {
            const candidate_stage_v3 &stage = candidate.stages[stage_index];
            if (stage.stage_id == 0u || stage.stage_id <= previous_stage_id
                || stage.kernel_id == 0u || stage.launch_count == 0u
                || stage.stable_name[0] == '\0') {
                return apply_inventory_status_v1::invalid_stage;
            }
            previous_stage_id = stage.stage_id;
        }
        previous_candidate_id = candidate.identity.candidate_id;
    }
    return apply_inventory_status_v1::success;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
