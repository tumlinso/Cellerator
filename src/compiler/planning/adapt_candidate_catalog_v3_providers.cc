#include <Cellerator/compiler/planning/adapt_candidate_catalog_v3_providers_v1.hh>

#include <algorithm>
#include <cstring>

namespace Cellerator::compiler::planning {
namespace catalog_v3 = cellerator::compute::operation::catalog_v3;
namespace {

template <std::size_t N>
std::string stable_string(const char (&value)[N]) {
    return std::string(value, std::find(value, value + N, '\0'));
}

bool provider_exists(
    const catalog_v3::provider_operation_inventory_v3& inventory,
    std::uint64_t identity) noexcept {
    for (std::uint64_t i = 0u; i < inventory.provider_count; ++i) {
        if (inventory.providers[i].stable_provider_id == identity) return true;
    }
    return false;
}

bool operation_exists(
    const catalog_v3::provider_operation_inventory_v3& inventory,
    std::uint64_t identity) noexcept {
    for (std::uint64_t i = 0u; i < inventory.operation_count; ++i) {
        if (inventory.operations[i].stable_operation_id == identity) return true;
    }
    return false;
}

}  // namespace

candidate_catalog_adapter_result_v1 adapt_candidate_catalog_v3_to_planning_ir_v1(
    const catalog_v3::candidate_catalog_view_v3& catalog,
    const catalog_v3::provider_operation_inventory_v3& inventory,
    const source_linked_preparation_hook_v1* hooks,
    std::uint64_t hook_count) {
    candidate_catalog_adapter_result_v1 result{};
    if (catalog_v3::validate_candidate_catalog_v3(catalog) !=
        catalog_v3::catalog_status::success) {
        return result;
    }
    if (!catalog_v3::validate_provider_operation_inventory_v3(inventory)) {
        result.code = candidate_catalog_adapter_code_v1::invalid_inventory;
        return result;
    }
    if ((hook_count != 0u && hooks == nullptr) || hook_count != catalog.candidate_count) {
        result.code = candidate_catalog_adapter_code_v1::invalid_preparation_hook;
        return result;
    }

    result.ir.providers.reserve(inventory.provider_count);
    for (std::uint64_t i = 0u; i < inventory.provider_count; ++i) {
        const auto& source = inventory.providers[i];
        result.ir.providers.push_back({
            source.stable_provider_id, source.capabilities,
            source.minimum_compute_major, source.minimum_compute_minor,
            source.compiled, stable_string(source.stable_name)});
    }
    result.ir.operations.reserve(inventory.operation_count);
    for (std::uint64_t i = 0u; i < inventory.operation_count; ++i) {
        const auto& source = inventory.operations[i];
        result.ir.operations.push_back({source.stable_operation_id,
            source.capabilities, source.operation, stable_string(source.stable_name)});
    }

    result.ir.candidates.reserve(catalog.candidate_count);
    for (std::uint64_t i = 0u; i < catalog.candidate_count; ++i) {
        result.source_index = i;
        const auto& source = catalog.candidates[i];
        if (!provider_exists(inventory, source.identity.provider_id)) {
            result.code = candidate_catalog_adapter_code_v1::missing_provider;
            return result;
        }
        if (!operation_exists(inventory, source.identity.operation_id)) {
            result.code = candidate_catalog_adapter_code_v1::missing_operation;
            return result;
        }
        if (hooks[i].source_catalog_index != i ||
            hooks[i].source_candidate_identity != source.identity.candidate_id ||
            hooks[i].prepare == nullptr) {
            result.code = candidate_catalog_adapter_code_v1::invalid_preparation_hook;
            return result;
        }
        planning_candidate_v1 candidate{};
        candidate.candidate_identity = source.identity.candidate_id;
        candidate.provider_identity = source.identity.provider_id;
        candidate.device_class_identity = source.identity.device_class_id;
        candidate.projection_type_identity = source.identity.projection_type_id;
        candidate.capability_identity = source.identity.capability_id;
        candidate.operation_identity = source.identity.operation_id;
        candidate.width_min = source.identity.width_min;
        candidate.width_max = source.identity.width_max;
        candidate.numerics = source.identity.numerics;
        candidate.classification = source.identity.classification;
        candidate.requires_measurement = source.identity.requires_measurement;
        candidate.resources = source.resources;
        candidate.preparation = hooks[i];
        candidate.stages.reserve(source.stage_count);
        for (std::uint32_t stage = 0u; stage < source.stage_count; ++stage) {
            const auto& source_stage = source.stages[stage];
            candidate.stages.push_back({source_stage.stage_id, source_stage.kernel_id,
                source_stage.stage_kind, source_stage.launch_count,
                stable_string(source_stage.stable_name)});
        }
        result.ir.candidates.push_back(std::move(candidate));
    }
    result.code = candidate_catalog_adapter_code_v1::ok;
    return result;
}

bool cross_validate_candidate_catalog_planning_ir_v1(
    const catalog_v3::candidate_catalog_view_v3& catalog,
    const catalog_v3::provider_operation_inventory_v3& inventory,
    const candidate_catalog_planning_ir_v1& ir) noexcept {
    if (catalog.candidate_count != ir.candidates.size() ||
        inventory.provider_count != ir.providers.size() ||
        inventory.operation_count != ir.operations.size()) {
        return false;
    }
    for (std::uint64_t i = 0u; i < catalog.candidate_count; ++i) {
        const auto& source = catalog.candidates[i];
        const auto& candidate = ir.candidates[i];
        if (candidate.candidate_identity != source.identity.candidate_id ||
            candidate.provider_identity != source.identity.provider_id ||
            candidate.operation_identity != source.identity.operation_id ||
            candidate.projection_type_identity != source.identity.projection_type_id ||
            candidate.capability_identity != source.identity.capability_id ||
            candidate.stages.size() != source.stage_count ||
            candidate.preparation.source_catalog_index != i ||
            candidate.preparation.source_candidate_identity != source.identity.candidate_id) {
            return false;
        }
    }
    return true;
}

}  // namespace Cellerator::compiler::planning
