#include <Cellerator/execution/geometry_acquisition_diagnostics.hh>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace execution = cellerator::execution;
namespace planner = cellerator::planner;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "acquisition_cost_mapping_test: %s\n", message);
        std::exit(1);
    }
}

execution::geometry_acquisition_work_v1 work() {
    execution::geometry_acquisition_work_v1 result{};
    result.semantic_search_ns = 11.0;
    result.target_refinement_ns = 13.0;
    result.projection_construction_ns = 17.0;
    result.projection_upload_ns = 19.0;
    result.cpe2_prebind_ns = 23.0;
    result.candidate_preparation_ns = 29.0;
    result.value_pack_ns = 31.0;
    result.input_pack_ns = 37.0;
    result.kernel_ns = 41.0;
    result.epilogue_ns = 43.0;
    result.order_ns = 47.0;
    result.projection_upload_bytes = 4096u;
    result.persistent_bytes = 8192u;
    result.transient_bytes = 1024u;
    return result;
}

execution::geometry_acquisition_reuse_diagnostics_v1 reuse() {
    execution::geometry_acquisition_reuse_diagnostics_v1 result{};
    result.structure = {1u, 2u};
    result.epoch = {3u};
    result.semantic_geometry = {4u, 5u};
    result.projection = {6u, 7u};
    result.values = {8u};
    result.dense_layout = {9u, 10u};
    result.work_window = {11u, 12u};
    result.prepared_program = {13u, 14u};
    result.graph_replay = {15u, 16u};
    result.structure_observed_uses = 64u;
    result.semantic_geometry_observed_uses = 32u;
    result.projection_observed_uses = 16u;
    result.value_generation_observed_uses = 8u;
    result.dense_layout_observed_uses = 4u;
    result.work_window_observed_uses = 4u;
    result.prepared_program_observed_uses = 2u;
    result.graph_replay_observed_uses = 2u;
    return result;
}

} // namespace

int main() {
    using code = execution::geometry_acquisition_diagnostics_status_v1;
    const auto observed_work = work();
    const auto observed_reuse = reuse();
    execution::geometry_acquisition_diagnostics_v1 mapped{};
    require(execution::map_geometry_acquisition_diagnostics_v1(
                observed_work, observed_reuse, &mapped)
            == code::success,
        "valid acquisition diagnostics rejected");

    const auto &phases = mapped.planner_phases;
    require(phases.semantic_packing_ns == 11.0,
        "semantic search did not map to semantic packing");
    require(phases.projection_construction_ns == 72.0,
        "refinement/construction/upload/prebind mapping mismatch");
    require(phases.backend_prepare_ns == 29.0
            && phases.static_value_pack_ns == 31.0
            && phases.dynamic_input_pack_ns == 37.0
            && phases.kernel_ns == 41.0 && phases.epilogue_ns == 43.0
            && phases.order_transform_ns == 47.0,
        "named execution work mapped to the wrong planner phase");
    require(phases.h2d_ns == 0.0 && phases.h2d_bytes == 0u
            && mapped.persistent_projection_upload_bytes == 4096u,
        "persistent projection upload was charged again per execution");
    require(phases.persistent_bytes == 8192u
            && phases.transient_bytes == 1024u,
        "acquisition memory diagnostics were lost");
    require(mapped.reuse.structure_observed_uses == 64u
            && mapped.reuse.semantic_geometry_observed_uses == 32u
            && mapped.reuse.projection_observed_uses == 16u
            && mapped.reuse.value_generation_observed_uses == 8u
            && mapped.reuse.dense_layout_observed_uses == 4u
            && mapped.reuse.work_window_observed_uses == 4u
            && mapped.reuse.prepared_program_observed_uses == 2u
            && mapped.reuse.graph_replay_observed_uses == 2u,
        "reuse diagnostics lost an identity-keyed observation");

    planner::total_cost total{};
    require(planner::compute_total_cost(phases, 64u, 16u, 8u, &total),
        "mapped phases were rejected by planner-v2");
    const double expected = 11.0 / 64.0 + (72.0 + 29.0) / 16.0
        + 31.0 / 8.0 + 37.0 + 41.0 + 43.0 + 47.0;
    require(std::fabs(total.amortized_total_ns - expected) < 1.0e-12
            && total.structure_reuse == 64u
            && total.projection_reuse == 16u && total.value_reuse == 8u,
        "planner-v2 did not remain the reuse/amortization authority");

    auto no_graph = observed_reuse;
    no_graph.graph_replay = {};
    no_graph.graph_replay_observed_uses = 0u;
    require(execution::map_geometry_acquisition_diagnostics_v1(
                observed_work, no_graph, &mapped)
            == code::success,
        "valid non-graph route rejected");
    no_graph.graph_replay_observed_uses = 1u;
    require(execution::map_geometry_acquisition_diagnostics_v1(
                observed_work, no_graph, &mapped)
            == code::invalid_reuse,
        "graph reuse without graph identity accepted");

    auto invalid_work = observed_work;
    invalid_work.kernel_ns = std::numeric_limits<double>::quiet_NaN();
    require(execution::map_geometry_acquisition_diagnostics_v1(
                invalid_work, observed_reuse, &mapped)
            == code::invalid_cost,
        "non-finite work accepted");
    invalid_work = observed_work;
    invalid_work.target_refinement_ns =
        std::numeric_limits<double>::max();
    invalid_work.projection_construction_ns =
        std::numeric_limits<double>::max();
    require(execution::map_geometry_acquisition_diagnostics_v1(
                invalid_work, observed_reuse, &mapped)
            == code::invalid_cost,
        "overflowing projection construction total accepted");

    auto invalid_reuse = observed_reuse;
    invalid_reuse.projection_observed_uses = 0u;
    require(execution::map_geometry_acquisition_diagnostics_v1(
                observed_work, invalid_reuse, &mapped)
            == code::invalid_reuse,
        "zero observed reuse accepted");
    invalid_reuse = observed_reuse;
    invalid_reuse.projection = {};
    require(execution::map_geometry_acquisition_diagnostics_v1(
                observed_work, invalid_reuse, &mapped)
            == code::invalid_identity,
        "missing projection identity accepted");

    require(execution::map_geometry_acquisition_diagnostics_v1(
                observed_work, observed_reuse, nullptr)
            == code::invalid_argument,
        "null output accepted");
    std::puts("acquisition_cost_mapping_test: ok");
    return 0;
}
