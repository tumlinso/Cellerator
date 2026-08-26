#include <Cellerator/compute/math/operation_core/builtin_catalog.hh>
#include <Cellerator/compute/math/operation_core/cusparse_csr_candidate.hh>
#include <Cellerator/compute/math/operation_core/preparation_factory.hh>
#include <Cellerator/execution/projection_activation.hh>
#include <Cellerator/parameters.hh>
#include <Cellerator/runtime/value_readiness.cuh>

#include <bench/ce_live/planner/live_planner_inputs.hh>
#include <bench/ce_live/runtime_fixture/quantitative_fixture.hh>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace inputs = cellerator::ce_live::planner_inputs;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "executable_core_integration_test: " << message << '\n';
    std::exit(1);
}

void test_candidate_surface() {
    require(static_cast<bool>(core::validate_built_in_candidate_catalog()),
        "built-in catalog failed fan-in validation");
    core::candidate_registry registry{};
    require(static_cast<bool>(
                core::register_built_in_candidate_catalog(&registry))
            && registry.size == core::builtin_candidate_count,
        "built-in candidates did not register atomically");
    require(static_cast<bool>(core::register_cusparse_csr_candidates(&registry))
            && registry.size == core::builtin_candidate_count + 2u
            && core::find_candidate(
                registry, core::cusparse_csr_spmv_candidate_id) != nullptr
            && core::find_candidate(
                registry, core::cusparse_csr_spmm_candidate_id) != nullptr,
        "strong conventional CSR candidates did not join the registry");

    const auto activation = &execution::activate_csr_projection;
    const auto preparation = &core::prepare_catalog_csr;
    require(activation != nullptr && preparation != nullptr,
        "typed activation or preparation factory is not linkable");
}

void test_quantitative_planner_bridge() {
    const auto fixture_ids =
        cellerator::ce_live::pbmc3k_quantitative_v1_identities();
    require(execution::valid_identity(fixture_ids.feature_domain)
            && execution::valid_identity(fixture_ids.observation_domain)
            && execution::valid_identity(fixture_ids.structure),
        "checksum-pinned fixture identities are unavailable");

    constexpr std::uint64_t offsets[]{0u, 2u, 3u};
    constexpr std::uint32_t sources[]{0u, 2u, 1u};
    constexpr float values[]{2.0f, -1.0f, 3.0f};
    inputs::quantitative_relation_input relation{};
    relation.identities = {fixture_ids.feature_domain,
        fixture_ids.observation_domain, {31u, 32u}, {41u, 42u},
        fixture_ids.geometry, fixture_ids.partition, fixture_ids.structure,
        {1u}};
    relation.destination_offsets = offsets;
    relation.source_indices = sources;
    relation.values = values;
    relation.source_count = 3u;
    relation.destination_count = 2u;
    relation.logical_edge_count = 3u;
    relation.observed_generation = {1u};
    inputs::live_planner_input planner_input{};
    require(inputs::derive_live_planner_input(relation, {101u, 102u},
                {1u, 7u, 0u, 700u}, {201u, 202u, 203u, 204u},
                {64u, 16u, 4u}, 1u, 1u, 1u, 1u, &planner_input)
            == inputs::live_input_status::ok
            && planner_input.keys.policy.structure_reuse == 64u
            && planner_input.keys.policy.projection_reuse == 16u
            && planner_input.keys.policy.value_reuse == 4u
            && std::fabs(planner_input.structure.density - 0.5) < 1.0e-12,
        "quantitative relation did not produce planner-ready inputs");

    inputs::candidate_phase_input phases{};
    phases.phases.host_preparation_ns = 8.0;
    phases.phases.projection_construction_ns = 16.0;
    phases.phases.static_value_pack_ns = 4.0;
    phases.phases.kernel_ns = 2.0;
    phases.reuse = {64u, 16u, 4u};
    cellerator::planner::total_cost total{};
    require(inputs::account_candidate_phases(phases, &total)
            == inputs::live_input_status::ok
            && std::fabs(total.amortized_total_ns - 12.0) < 1.0e-12
            && !inputs::authoritative_for_promotion(phases),
        "complete-cost shortlist contract failed");
}

void test_runtime_only_state_boundary() {
    cellerator::runtime::value_readiness_record readiness;
    cellerator::native_parameter_descriptor parameter{};
    parameter.kind = cellerator::native_parameter_kind::relation_values;
    parameter.structure = {11u, 1u};
    parameter.structure_epoch = {2u};
    parameter.generation = {3u};
    require(!readiness.initialized() && !readiness.published()
            && parameter.generation.value == 3u
            && execution::valid_handle(parameter.structure),
        "runtime readiness leaked into persistent parameter identity");
}

} // namespace

int main() {
    test_candidate_surface();
    test_quantitative_planner_bridge();
    test_runtime_only_state_boundary();
    std::cout << "executable_core_integration_test passed candidates="
              << core::builtin_candidate_count + 2u << '\n';
    return 0;
}
