#include <Cellerator/compute/decomposition/decomposition_v1.hh>

#include <cassert>

namespace decomposition = cellerator::compute::decomposition;
namespace execution = cellerator::execution;

void set_numeric(decomposition::decomposition_alternative_v1 *alternative) {
    alternative->numerical.relation_storage = execution::numeric_type::f16;
    alternative->numerical.state_storage = execution::numeric_type::f32;
    alternative->numerical.multiply = execution::numeric_type::f32;
    alternative->numerical.accumulation = execution::numeric_type::f32;
    alternative->numerical.output_storage = execution::numeric_type::f32;
    alternative->numerical.scalar = execution::numeric_type::f32;
}

int main() {
    const execution::joint_compiler::persistent_identity_v1 inputs[] = {
        {1u, 1u}, {1u, 2u}};
    const execution::joint_compiler::persistent_identity_v1 replicas[] = {
        {2u, 1u}};
    const execution::joint_compiler::persistent_identity_v1 halos[] = {
        {3u, 1u}};

    decomposition::decomposition_alternative_v1 alternatives[2]{};
    alternatives[0].alternative_identity = {4u, 1u};
    alternatives[0].candidate_family = {5u, 1u};
    alternatives[0].split_axis = decomposition::split_axis_v1::none;
    alternatives[0].flags = decomposition::legal_alternative_v1
        | decomposition::complete_unsplit_fallback_v1;
    alternatives[0].required_input_coverages = inputs;
    alternatives[0].required_input_coverage_count = 2u;
    alternatives[0].output_coverage = {6u, 1u};
    alternatives[0].input_order = {7u, 1u};
    alternatives[0].output_order = {7u, 2u};
    set_numeric(&alternatives[0]);

    alternatives[1] = alternatives[0];
    alternatives[1].alternative_identity = {4u, 2u};
    alternatives[1].split_axis = decomposition::split_axis_v1::relation_edges;
    alternatives[1].flags = decomposition::legal_alternative_v1
        | decomposition::produces_partial_result_v1
        | decomposition::requires_replication_v1
        | decomposition::requires_halo_v1;
    alternatives[1].replication_coverages = replicas;
    alternatives[1].replication_coverage_count = 1u;
    alternatives[1].halo_coverages = halos;
    alternatives[1].halo_coverage_count = 1u;
    alternatives[1].partial_algebra = {8u, 1u};

    decomposition::decomposition_portfolio_v1 portfolio{};
    portfolio.portfolio_identity = {9u, 1u};
    portfolio.alternatives = alternatives;
    portfolio.alternative_count = 2u;
    assert(decomposition::validate_decomposition_portfolio_v1(portfolio));

    // A baseline-only portfolio is the required valid negative promotion.
    portfolio.alternative_count = 1u;
    assert(decomposition::validate_decomposition_portfolio_v1(portfolio));
    portfolio.alternative_count = 2u;

    auto malformed = alternatives[1];
    malformed.partial_algebra = {};
    alternatives[1] = malformed;
    assert(decomposition::validate_decomposition_portfolio_v1(portfolio).code
        == decomposition::decomposition_validation_code_v1::
            invalid_partial_algebra);
    alternatives[1] = alternatives[0];
    alternatives[1].alternative_identity = {4u, 2u};
    alternatives[1].split_axis = decomposition::split_axis_v1::source_axis;
    alternatives[1].flags = decomposition::legal_alternative_v1;
    assert(decomposition::validate_decomposition_portfolio_v1(portfolio));

    alternatives[0].flags = decomposition::legal_alternative_v1;
    alternatives[0].split_axis = decomposition::split_axis_v1::source_axis;
    assert(decomposition::validate_decomposition_portfolio_v1(portfolio).code
        == decomposition::decomposition_validation_code_v1::missing_fallback);
    alternatives[0].flags = decomposition::legal_alternative_v1
        | decomposition::complete_unsplit_fallback_v1;
    alternatives[0].split_axis = decomposition::split_axis_v1::none;
    alternatives[1].replication_coverages = replicas;
    assert(decomposition::validate_decomposition_portfolio_v1(portfolio).code
        == decomposition::decomposition_validation_code_v1::
            invalid_replication_flag);
    return 0;
}
