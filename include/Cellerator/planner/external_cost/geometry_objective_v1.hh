#pragma once

#include <Cellerator/planner/external_cost/vector_v1.hh>

namespace cellerator::planner::external_cost {

struct geometry_objective_terms_v1 {
    double local_objective_ns = 0.0;
    double construction_ns = 0.0;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t input_movement_bytes = 0u;
    std::uint64_t output_movement_bytes = 0u;
    std::uint64_t communication_bytes = 0u;
    std::uint64_t global_expected_reuse = 1u;
};

struct priced_geometry_objective_v1 {
    double local_objective_ns = 0.0;
    double amortized_geometry_ns = 0.0;
    double movement_ns = 0.0;
    double communication_ns = 0.0;
    double applied_reuse_credit_ns = 0.0;
    double complete_objective_ns = 0.0;
};

enum class geometry_objective_status_v1 : std::uint8_t {
    success = 0u,
    null_output,
    invalid_external_cost,
    invalid_terms,
    arithmetic_overflow,
};

geometry_objective_status_v1 price_geometry_objective_v1(
    const geometry_objective_terms_v1 &terms,
    const external_cost_vector_v1 &external_cost,
    priced_geometry_objective_v1 *output) noexcept;

} // namespace cellerator::planner::external_cost
