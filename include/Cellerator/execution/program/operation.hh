#pragma once

#include <Cellerator/execution/preparation/preparation_factory.hh>
#include <Cellerator/execution/planning/operation_planning.hh>
#include <Cellerator/execution/preparation/readiness/value_readiness.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::execution {

inline constexpr std::uint32_t executable_program_schema_version_v2 = 2u;
inline constexpr std::uint32_t maximum_program_candidates =
    planning::maximum_planner_candidates;

// Complete costs are supplied by the existing measurement/model layer. The
// executable program never substitutes kernel-only ranking.
struct program_candidate_cost {
    operation_core::stable_id candidate{};
    structure::projection_id projection{};
    planning::phase_costs phases{};
    std::uint32_t planner_flags = 0u;
    std::uint32_t reserved = 0u;
};

struct program_axis {
    axis_identity live{};
    persistent_axis_identity persistent{};
};

// Program v2 consumes arbitrary source-linked preparation-bearing catalog
// entries and provider-erased activated views. Neither the request nor the
// engine enumerates physical projection types.
struct executable_program_request_v2 {
    std::uint32_t schema_version = executable_program_schema_version_v2;
    std::uint32_t reserved = 0u;
    operation_core::operation_problem problem{};
    operation_core::structure_set_key structures{};
    operation_core::numeric_policy numeric{};
    operation_core::prepare_policy preparation{};
    planning::planning_keys planning{};
    planning::planner_policy planner_policy{};
    planning::measurement_hook measurement{};
    planning::plan_cache_hooks cache{};
    std::uint64_t current_evidence_revision = 0u;
    operation_core::candidate_preparation_catalog_v2 catalog{};
    const activated_projection_reference_v2 *projections = nullptr;
    std::uint32_t projection_count = 0u;
    std::uint32_t reserved2 = 0u;
    const program_candidate_cost *costs = nullptr;
    std::uint32_t cost_count = 0u;
    std::uint32_t reserved3 = 0u;
    runtime::execution_session *session = nullptr;
    std::uint32_t dense_width = 0u;
    std::uint32_t reserved4 = 0u;
    program_axis source_axis{};
    program_axis destination_axis{};
    program_axis dense_column_axis{};
    operation_core::preparation_state_storage preparation_state{};
};

struct program_candidate_summary {
    operation_core::stable_id candidate{};
    const char *name = nullptr;
    operation_core::projection_key projection{};
    planning::total_cost analytical{};
    planning::candidate_rejection rejection = planning::candidate_rejection::none;
    bool shortlisted = false;
    bool measured = false;
    bool conventional = false;
    std::uint8_t reserved = 0u;
};

struct executable_program {
    std::uint32_t schema_version = executable_program_schema_version_v2;
    operation_core::prepared_operation prepared{};
    operation_core::stable_id selected_candidate{};
    operation_core::projection_key selected_projection{};
    planning::selection_source selection = planning::selection_source::analytical;
    planning::cache_state cache = planning::cache_state::not_configured;
    bool conventional_winner = false;
    std::uint8_t reserved[3]{};
    std::uint32_t candidate_count = 0u;
    std::uint32_t legal_count = 0u;
    std::uint32_t shortlist_count = 0u;
    std::uint32_t measurement_count = 0u;
    planning::total_cost expected_cost{};
    const char *selection_reason = nullptr;
    program_candidate_summary candidates[maximum_program_candidates]{};
    runtime::execution_session *session = nullptr;
    std::uint64_t preparation_count = 0u;
    std::uint64_t run_count = 0u;
};

enum class executable_program_status_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    identity_mismatch = 2u,
    no_compatible_candidate = 3u,
    planner_failed = 4u,
    preparation_failed = 5u,
    stale_structure = 6u,
    stale_or_unready_value = 7u,
    invalid_launch = 8u,
    execution_failed = 9u
};

struct executable_program_status {
    executable_program_status_code code = executable_program_status_code::ok;
    operation_core::operation_status operation{};
    planning::planner_status planning{};
    runtime::value_readiness_status readiness =
        runtime::value_readiness_status::success;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == executable_program_status_code::ok;
    }
};

executable_program_status prepare_operation_program(
    const executable_program_request_v2 &request,
    executable_program *program) noexcept;

struct executable_program_launch {
    launch_bindings bindings{};
    const runtime::value_readiness_record *value_readiness = nullptr;
    structure::structure_epoch expected_structure_epoch{};
    value_generation expected_value_generation{};
};

struct executable_program_result {
    operation_core::stable_id candidate{};
    operation_core::projection_key projection{};
    planning::selection_source selection = planning::selection_source::analytical;
    planning::total_cost expected_cost{};
    const output_axis_contract *output_orders = nullptr;
    std::uint32_t output_order_count = 0u;
    structure::structure_epoch structure_epoch_value{};
    value_generation consumed_generation{};
    stream_context completion_stream{};
    bool enqueued = false;
};

executable_program_status run_operation_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept;

} // namespace cellerator::execution
