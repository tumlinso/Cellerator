#include <Cellerator/compiler/ir/planning/implement_decomposition_alternative_nodes_v1.hh>

#include <array>
#include <cassert>

int main() {
    namespace planning = cellerator::compiler::ir::planning::v1;
    namespace decomp = cellerator::compute::decomposition;
    namespace joint = cellerator::execution::joint_compiler;
    const joint::persistent_identity_v1 inputs[] = {{1u, 1u}};
    decomp::decomposition_alternative_v1 alternatives[2]{};
    alternatives[0].alternative_identity = {2u, 1u};
    alternatives[0].candidate_family = {3u, 1u};
    alternatives[0].flags = decomp::legal_alternative_v1 |
                            decomp::complete_unsplit_fallback_v1;
    alternatives[0].required_input_coverages = inputs;
    alternatives[0].required_input_coverage_count = 1u;
    alternatives[0].output_coverage = {4u, 1u};
    alternatives[1] = alternatives[0];
    alternatives[1].alternative_identity = {2u, 2u};
    alternatives[1].split_axis = decomp::split_axis_v1::source_axis;
    alternatives[1].flags = decomp::legal_alternative_v1 |
                            decomp::produces_partial_result_v1 |
                            decomp::requires_halo_v1;
    alternatives[1].partial_algebra = {5u, 1u};
    const decomp::decomposition_portfolio_v1 portfolio{
        decomp::decomposition_schema_version_v1, sizeof(portfolio), {6u, 1u}, alternatives, 2u};
    std::array<planning::decomposition_alternative_node_v1, 2> nodes{};
    std::uint32_t written = 0u;
    assert(planning::import_decomposition_portfolio_v1(portfolio, nodes.data(), nodes.size(),
                                                       &written) ==
           planning::decomposition_node_status_v1::ok);
    assert(written == 2u && nodes[0].alternative.flags == alternatives[0].flags);

    assert(planning::validate_decomposition_alternative_node_v1(nodes[0]) ==
           planning::decomposition_node_status_v1::ok);
    const planning::planning_identity_v1 fragments[] = {{7u, 1u}, {7u, 2u}};
    const joint::persistent_identity_v1 contributions[] = {{8u, 1u}};
    nodes[1].fragments = fragments;
    nodes[1].fragment_count = 2u;
    nodes[1].contribution_coverages = contributions;
    nodes[1].contribution_coverage_count = 1u;
    assert(planning::validate_decomposition_alternative_node_v1(nodes[1]) ==
           planning::decomposition_node_status_v1::ok);
    nodes[1].fragments = nullptr;
    assert(planning::validate_decomposition_alternative_node_v1(nodes[1]) ==
           planning::decomposition_node_status_v1::missing_fragments);
}
