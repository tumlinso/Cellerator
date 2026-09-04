#pragma once

#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>
#include <Cellerator/compute/operation/candidate_catalog_v3/inventory.h>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::planning {

using candidate_preparation_function_v1 = bool (*)(
    std::uint64_t candidate_identity,
    const void* context) noexcept;

struct source_linked_preparation_hook_v1 {
    std::uint64_t source_candidate_identity = 0u;
    std::uint64_t source_catalog_index = 0u;
    candidate_preparation_function_v1 prepare = nullptr;
    const void* context = nullptr;
};

struct planning_candidate_stage_v1 {
    std::uint64_t stage_identity = 0u;
    std::uint64_t kernel_identity = 0u;
    std::uint32_t stage_kind = 0u;
    std::uint32_t launch_count = 0u;
    std::string stable_name;
};

struct planning_candidate_v1 {
    std::uint64_t candidate_identity = 0u;
    std::uint64_t provider_identity = 0u;
    std::uint64_t device_class_identity = 0u;
    std::uint64_t projection_type_identity = 0u;
    std::uint64_t capability_identity = 0u;
    std::uint64_t operation_identity = 0u;
    std::uint32_t width_min = 0u;
    std::uint32_t width_max = 0u;
    cellerator::compute::operation::catalog_v3::numerical_mode numerics{};
    cellerator::compute::operation::catalog_v3::candidate_class classification{};
    bool requires_measurement = false;
    cellerator::compute::operation::catalog_v3::candidate_resource_v3 resources{};
    source_linked_preparation_hook_v1 preparation{};
    std::vector<planning_candidate_stage_v1> stages;
};

struct planning_provider_v1 {
    std::uint64_t provider_identity = 0u;
    std::uint64_t capabilities = 0u;
    std::uint32_t minimum_compute_major = 0u;
    std::uint32_t minimum_compute_minor = 0u;
    bool compiled = false;
    std::string stable_name;
};

struct planning_operation_v1 {
    std::uint64_t operation_identity = 0u;
    std::uint64_t capabilities = 0u;
    cellerator::compute::operation::catalog_v3::relation_operation_v3 operation{};
    std::string stable_name;
};

struct candidate_catalog_planning_ir_v1 {
    std::vector<planning_provider_v1> providers;
    std::vector<planning_operation_v1> operations;
    std::vector<planning_candidate_v1> candidates;
};

enum class candidate_catalog_adapter_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_catalog,
    invalid_inventory,
    missing_provider,
    missing_operation,
    invalid_preparation_hook,
};

struct candidate_catalog_adapter_result_v1 {
    candidate_catalog_adapter_code_v1 code =
        candidate_catalog_adapter_code_v1::invalid_catalog;
    std::uint64_t source_index = 0u;
    candidate_catalog_planning_ir_v1 ir{};

    constexpr explicit operator bool() const noexcept {
        return code == candidate_catalog_adapter_code_v1::ok;
    }
};

[[nodiscard]] candidate_catalog_adapter_result_v1
adapt_candidate_catalog_v3_to_planning_ir_v1(
    const cellerator::compute::operation::catalog_v3::candidate_catalog_view_v3& catalog,
    const cellerator::compute::operation::catalog_v3::provider_operation_inventory_v3& inventory,
    const source_linked_preparation_hook_v1* hooks,
    std::uint64_t hook_count);

[[nodiscard]] bool cross_validate_candidate_catalog_planning_ir_v1(
    const cellerator::compute::operation::catalog_v3::candidate_catalog_view_v3& catalog,
    const cellerator::compute::operation::catalog_v3::provider_operation_inventory_v3& inventory,
    const candidate_catalog_planning_ir_v1& ir) noexcept;

}  // namespace Cellerator::compiler::planning
