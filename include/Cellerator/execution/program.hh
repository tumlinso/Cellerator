#pragma once

#include <Cellerator/compute/operation/preparation_factory.hh>
#include <Cellerator/execution/projection_activation.hh>
#include <Cellerator/planner/end_to_end_planner.hh>
#include <Cellerator/runtime/value_readiness.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::execution {

inline constexpr std::uint32_t executable_program_schema_version = 1u;
inline constexpr std::uint32_t maximum_program_candidates =
    planner::maximum_planner_candidates;

namespace operation_core = compute::math::core;

enum class activated_projection_type : std::uint8_t {
    row_masked = 1u,
    csr = 2u,
    feature_major = 3u,
    transpose = 4u
};

// One non-owning typed projection reference. The pointed-to view and all of
// its payload bytes remain owned by the caller/session. This is compile-time
// orchestration input, never persistent CPE2 state.
struct activated_projection_reference {
    operation_core::projection_key key{};
    activated_projection_type type = activated_projection_type::row_masked;
    const void *view = nullptr;
};

activated_projection_reference program_projection(
    operation_core::projection_key key,
    const cellpack::persistent_packing_payload_view &view) noexcept;
activated_projection_reference program_projection(
    operation_core::projection_key key,
    const compute::math::execution_csr_view &view) noexcept;
activated_projection_reference program_projection(
    operation_core::projection_key key,
    const compute::math::feature_major_projection_view &view) noexcept;
activated_projection_reference program_projection(
    operation_core::projection_key key,
    const compute::math::transpose_projection_view &view) noexcept;

// Complete costs are supplied by the existing measurement/model layer. The
// executable program never substitutes kernel-only ranking.
struct program_candidate_cost {
    operation_core::stable_id candidate{};
    projection_id projection{};
    planner::phase_costs phases{};
    std::uint32_t planner_flags = 0u;
    std::uint32_t reserved = 0u;
};

struct program_axis {
    axis_identity live{};
    persistent_axis_identity persistent{};
};

struct executable_program_request {
    std::uint32_t schema_version = executable_program_schema_version;
    operation_core::operation_problem problem{};
    operation_core::structure_set_key structures{};
    operation_core::numeric_policy numeric{};
    operation_core::prepare_policy preparation{};
    planner::planning_keys planning{};
    planner::planner_policy planner_policy{};
    planner::measurement_hook measurement{};
    planner::plan_cache_hooks cache{};
    std::uint64_t current_evidence_revision = 0u;
    operation_core::built_in_candidate_catalog_view catalog{};
    const activated_projection_reference *projections = nullptr;
    std::uint32_t projection_count = 0u;
    const program_candidate_cost *costs = nullptr;
    std::uint32_t cost_count = 0u;
    runtime::execution_session *session = nullptr;
    std::uint32_t dense_width = 0u;
    program_axis source_axis{};
    program_axis destination_axis{};
    program_axis dense_column_axis{};
    operation_core::preparation_state_storage preparation_state{};
};

struct program_candidate_summary {
    operation_core::stable_id candidate{};
    const char *name = nullptr;
    operation_core::projection_key projection{};
    planner::total_cost analytical{};
    planner::candidate_rejection rejection = planner::candidate_rejection::none;
    bool shortlisted = false;
    bool measured = false;
    bool conventional = false;
    std::uint8_t reserved = 0u;
};

struct executable_program {
    std::uint32_t schema_version = executable_program_schema_version;
    operation_core::prepared_operation prepared{};
    operation_core::stable_id selected_candidate{};
    operation_core::projection_key selected_projection{};
    planner::selection_source selection = planner::selection_source::analytical;
    planner::cache_state cache = planner::cache_state::not_configured;
    bool conventional_winner = false;
    std::uint8_t reserved[3]{};
    std::uint32_t candidate_count = 0u;
    std::uint32_t legal_count = 0u;
    std::uint32_t shortlist_count = 0u;
    std::uint32_t measurement_count = 0u;
    planner::total_cost expected_cost{};
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
    planner::planner_status planning{};
    runtime::value_readiness_status readiness =
        runtime::value_readiness_status::success;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == executable_program_status_code::ok;
    }
};

executable_program_status compile_executable_program(
    const executable_program_request &request,
    executable_program *program) noexcept;

struct executable_program_launch {
    launch_bindings bindings{};
    const runtime::value_readiness_record *value_readiness = nullptr;
    structure_epoch expected_structure_epoch{};
    value_generation expected_value_generation{};
};

struct executable_program_result {
    operation_core::stable_id candidate{};
    operation_core::projection_key projection{};
    planner::selection_source selection = planner::selection_source::analytical;
    planner::total_cost expected_cost{};
    const output_axis_contract *output_orders = nullptr;
    std::uint32_t output_order_count = 0u;
    structure_epoch structure_epoch_value{};
    value_generation consumed_generation{};
    stream_context completion_stream{};
    bool enqueued = false;
};

executable_program_status run_executable_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept;

} // namespace cellerator::execution
