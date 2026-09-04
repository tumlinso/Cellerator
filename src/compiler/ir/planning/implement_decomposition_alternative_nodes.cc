#include <Cellerator/compiler/ir/planning/implement_decomposition_alternative_nodes_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {
namespace {
bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}
bool zero(joint_compiler::persistent_identity_v1 value) noexcept {
    return value.producer_namespace == 0u || value.local_identity == 0u;
}
}  // namespace

decomposition_node_status_v1 validate_decomposition_alternative_node_v1(
    const decomposition_alternative_node_v1 &node) noexcept {
    const auto &alternative = node.alternative;
    if (zero(node.node) || zero(alternative.alternative_identity) ||
        zero(alternative.candidate_family)) {
        return decomposition_node_status_v1::invalid_identity;
    }
    if (node.reserved != 0u || node.reserved_count != 0u ||
        alternative.reserved0[0] != 0u || alternative.reserved0[1] != 0u ||
        alternative.reserved0[2] != 0u) {
        return decomposition_node_status_v1::nonzero_reserved;
    }
    if (static_cast<std::uint8_t>(alternative.split_axis) >
        static_cast<std::uint8_t>(decomposition::split_axis_v1::extents)) {
        return decomposition_node_status_v1::invalid_split;
    }
    if ((alternative.flags & ~decomposition::known_decomposition_flags_v1) != 0u ||
        (alternative.flags & decomposition::legal_alternative_v1) == 0u) {
        return decomposition_node_status_v1::invalid_flags;
    }
    const bool fallback = (alternative.flags & decomposition::complete_unsplit_fallback_v1) != 0u;
    if (fallback && alternative.split_axis != decomposition::split_axis_v1::none) {
        return decomposition_node_status_v1::invalid_fallback;
    }
    if (!fallback && (node.fragment_count == 0u || node.fragments == nullptr)) {
        return decomposition_node_status_v1::missing_fragments;
    }
    if (alternative.required_input_coverage_count == 0u ||
        alternative.required_input_coverages == nullptr || zero(alternative.output_coverage)) {
        return decomposition_node_status_v1::missing_coverage;
    }
    if ((alternative.flags & decomposition::produces_partial_result_v1) != 0u &&
        (zero(alternative.partial_algebra) || node.contribution_coverage_count == 0u ||
         node.contribution_coverages == nullptr)) {
        return decomposition_node_status_v1::missing_partial_algebra;
    }
    return decomposition_node_status_v1::ok;
}

decomposition_node_status_v1 import_decomposition_portfolio_v1(
    const decomposition::decomposition_portfolio_v1 &source,
    decomposition_alternative_node_v1 *nodes, std::uint32_t capacity,
    std::uint32_t *written) noexcept {
    if (written == nullptr || (source.alternative_count != 0u && source.alternatives == nullptr)) {
        return decomposition_node_status_v1::invalid_argument;
    }
    if (source.alternative_count > capacity || (source.alternative_count != 0u && nodes == nullptr)) {
        return decomposition_node_status_v1::invalid_argument;
    }
    *written = static_cast<std::uint32_t>(source.alternative_count);
    for (std::uint32_t index = 0u; index != *written; ++index) {
        nodes[index] = {};
        nodes[index].node = {source.alternatives[index].alternative_identity.local_identity,
                             source.alternatives[index].alternative_identity.producer_namespace};
        nodes[index].alternative = source.alternatives[index];
    }
    return decomposition_node_status_v1::ok;
}

}  // namespace cellerator::compiler::ir::planning::v1
