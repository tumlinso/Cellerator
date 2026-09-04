#include <Cellerator/compiler/ir/planning/implement_candidate_family_and_provider_nodes_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {
namespace {
bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}
}  // namespace

candidate_provider_status_v1 validate_candidate_provider_node_v1(
    const candidate_provider_node_v1 &node) noexcept {
    if (zero(node.node) || zero(node.provider) || node.candidate_id == 0u ||
        node.operation_id == 0u || node.capability_id == 0u) {
        return candidate_provider_status_v1::invalid_identity;
    }
    if (node.width_min == 0u || node.width_max < node.width_min) {
        return candidate_provider_status_v1::invalid_width;
    }
    if (node.numerics != catalog_v3::numerical_mode::precise &&
        node.numerics != catalog_v3::numerical_mode::relaxed) {
        return candidate_provider_status_v1::invalid_numerics;
    }
    if (node.stage_count != 0u && node.stages == nullptr) {
        return candidate_provider_status_v1::missing_stages;
    }
    if (node.reserved != 0u) {
        return candidate_provider_status_v1::nonzero_reserved;
    }
    return candidate_provider_status_v1::ok;
}

candidate_provider_status_v1 import_candidate_catalog_v3(
    const catalog_v3::candidate_descriptor_v3 &candidate,
    planning_identity_v1 node, planning_identity_v1 source_extension,
    std::uint64_t preparation_entrypoint,
    candidate_provider_node_v1 *result) noexcept {
    if (result == nullptr) {
        return candidate_provider_status_v1::invalid_argument;
    }
    candidate_provider_node_v1 imported{};
    imported.node = node;
    imported.provider = {candidate.identity.provider_id, candidate.identity.provider_id};
    imported.source_extension = source_extension;
    imported.candidate_id = candidate.identity.candidate_id;
    imported.operation_id = candidate.identity.operation_id;
    imported.device_class_id = candidate.identity.device_class_id;
    imported.projection_type_id = candidate.identity.projection_type_id;
    imported.capability_id = candidate.identity.capability_id;
    imported.preparation_entrypoint = preparation_entrypoint;
    imported.width_min = candidate.identity.width_min;
    imported.width_max = candidate.identity.width_max;
    imported.numerics = candidate.identity.numerics;
    imported.resources = candidate.resources;
    imported.stages = candidate.stages;
    imported.stage_count = candidate.stage_count;
    if (candidate.identity.classification == catalog_v3::candidate_class::experimental) {
        imported.flags |= candidate_provider_experimental_v1;
    }
    if (candidate.identity.requires_measurement) {
        imported.flags |= candidate_provider_requires_measurement_v1;
    }
    if (!zero(source_extension)) {
        imported.flags |= candidate_provider_source_extension_v1;
    }
    const auto status = validate_candidate_provider_node_v1(imported);
    if (status == candidate_provider_status_v1::ok) {
        *result = imported;
    }
    return status;
}

}  // namespace cellerator::compiler::ir::planning::v1
