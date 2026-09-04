#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

namespace catalog_v3 = cellerator::compute::operation::catalog_v3;

enum candidate_provider_flags_v1 : std::uint32_t {
    candidate_provider_none_v1 = 0u,
    candidate_provider_experimental_v1 = 1u << 0u,
    candidate_provider_requires_measurement_v1 = 1u << 1u,
    candidate_provider_source_extension_v1 = 1u << 2u
};

struct candidate_provider_node_v1 {
    planning_identity_v1 node{};
    planning_identity_v1 provider{};
    planning_identity_v1 source_extension{};
    std::uint64_t candidate_id = 0u;
    std::uint64_t operation_id = 0u;
    std::uint64_t device_class_id = 0u;
    std::uint64_t projection_type_id = 0u;
    std::uint64_t capability_id = 0u;
    std::uint64_t preparation_entrypoint = 0u;
    std::uint32_t width_min = 0u;
    std::uint32_t width_max = 0u;
    catalog_v3::numerical_mode numerics = catalog_v3::numerical_mode::precise;
    std::uint32_t flags = candidate_provider_none_v1;
    catalog_v3::candidate_resource_v3 resources{};
    const catalog_v3::candidate_stage_v3 *stages = nullptr;
    std::uint32_t stage_count = 0u;
    std::uint32_t reserved = 0u;
};

enum class candidate_provider_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_identity, invalid_width,
    invalid_numerics, missing_stages, nonzero_reserved
};

candidate_provider_status_v1 import_candidate_catalog_v3(
    const catalog_v3::candidate_descriptor_v3 &candidate,
    planning_identity_v1 node, planning_identity_v1 source_extension,
    std::uint64_t preparation_entrypoint,
    candidate_provider_node_v1 *result) noexcept;
candidate_provider_status_v1 validate_candidate_provider_node_v1(
    const candidate_provider_node_v1 &node) noexcept;

static_assert(std::is_trivially_copyable_v<candidate_provider_node_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
