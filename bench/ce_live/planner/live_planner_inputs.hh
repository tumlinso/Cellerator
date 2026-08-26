#pragma once

#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstdint>

namespace cellerator::ce_live::planner_inputs {

enum class live_input_status : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    invalid_identity = 2u,
    invalid_support = 3u,
    invalid_values = 4u,
    invalid_reuse = 5u,
    invalid_cost = 6u
};

struct relation_identity_input {
    execution::domain_id source_domain{};
    execution::domain_id destination_domain{};
    execution::order_id source_order{};
    execution::order_id destination_order{};
    execution::geometry_id geometry{};
    execution::partition_id partition{};
    execution::structure_id structure{};
    execution::structure_epoch structure_epoch{};
};

struct quantitative_relation_input {
    relation_identity_input identities{};
    const std::uint64_t *destination_offsets = nullptr;
    const std::uint32_t *source_indices = nullptr;
    const float *values = nullptr;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint64_t logical_edge_count = 0u;
    execution::value_generation observed_generation{};
};

struct reuse_horizons {
    std::uint64_t structure = 1u;
    std::uint64_t projection = 1u;
    std::uint64_t value = 1u;
};

struct structural_statistics {
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint64_t logical_edge_count = 0u;
    std::uint64_t minimum_destination_degree = 0u;
    std::uint64_t maximum_destination_degree = 0u;
    double mean_destination_degree = 0.0;
    double density = 0.0;
};

struct quantitative_statistics {
    execution::value_generation observed_generation{};
    std::uint64_t nonzero_count = 0u;
    double minimum = 0.0;
    double maximum = 0.0;
    double l1_norm = 0.0;
};

struct live_planner_input {
    cellerator::planner::planning_keys keys{};
    structural_statistics structure{};
    quantitative_statistics values{};
};

struct candidate_phase_input {
    cellerator::planner::phase_costs phases{};
    reuse_horizons reuse{};
    bool measured = false;
};

live_input_status derive_live_planner_input(
    const quantitative_relation_input &relation,
    compute::math::core::stable_id problem,
    const cellerator::planner::device_performance_key &device,
    const cellerator::planner::runtime_build_key &build,
    reuse_horizons reuse,
    std::uint32_t numeric_policy,
    std::uint32_t determinism_policy,
    std::uint32_t output_order_policy,
    std::uint32_t graph_policy,
    live_planner_input *output) noexcept;

live_input_status account_candidate_phases(
    const candidate_phase_input &input,
    cellerator::planner::total_cost *output) noexcept;

// Analytical costs are admissible shortlist inputs. A final promotion record
// must carry measured phases from an independent, correct candidate run.
constexpr bool authoritative_for_promotion(
    const candidate_phase_input &input) noexcept {
    return input.measured;
}

} // namespace cellerator::ce_live::planner_inputs
