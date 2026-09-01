#pragma once

#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t atom_fragment_result_schema_version_v1 = 1u;
inline constexpr std::uint64_t maximum_atom_fragment_candidates_v1 = 64u;

enum class empirical_status_v1 : std::uint8_t {
    analytical_only = 1u,
    measured_correct = 2u,
    measured_incorrect = 3u,
    unavailable = 4u
};

enum class no_candidate_reason_v1 : std::uint8_t {
    none = 0u,
    no_legal_decomposition = 1u,
    unmet_atom_requirement = 2u,
    unsupported_target = 3u,
    invalid_numerics = 4u,
    bounded_frontier_empty = 5u
};

enum fragment_candidate_flag_v1 : std::uint32_t {
    candidate_produces_partial_v1 = 1u << 0u,
    graph_capture_compatible_v1 = 1u << 1u,
    deterministic_candidate_v1 = 1u << 2u
};

inline constexpr std::uint32_t known_fragment_candidate_flags_v1 =
    candidate_produces_partial_v1 | graph_capture_compatible_v1
    | deterministic_candidate_v1;

struct fragment_resource_vector_v1 {
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    std::uint64_t transfer_bytes = 0u;
    std::uint64_t communication_bytes = 0u;
    std::uint32_t launch_count = 0u;
    std::uint32_t extent_count = 0u;
};

struct fragment_complete_cost_v1 {
    std::uint64_t preparation_ns = 0u;
    std::uint64_t transform_ns = 0u;
    std::uint64_t execution_ns = 0u;
    std::uint64_t epilogue_ns = 0u;
    std::uint64_t synchronization_ns = 0u;
    std::uint64_t communication_ns = 0u;
    std::uint64_t expected_reuse = 1u;
};

struct atom_fragment_candidate_v1 {
    persistent_identity_v1 candidate_identity{};
    persistent_identity_v1 exact_local_coverage{};
    const persistent_identity_v1 *required_atom_inputs = nullptr;
    std::uint64_t required_atom_input_count = 0u;
    persistent_identity_v1 program_recipe{};
    const persistent_identity_v1 *projection_requirements = nullptr;
    std::uint64_t projection_requirement_count = 0u;
    persistent_identity_v1 output_affordance{};
    persistent_identity_v1 partial_affordance{};
    order_id input_order{};
    order_id output_order{};
    fragment_resource_vector_v1 resources{};
    fragment_complete_cost_v1 complete_cost{};
    empirical_status_v1 empirical_status = empirical_status_v1::analytical_only;
    std::uint8_t reserved[3]{};
    std::uint32_t flags = 0u;
    persistent_identity_v1 validation_receipt{};
};

struct atom_fragment_result_v1 {
    std::uint32_t schema_version = atom_fragment_result_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_fragment_result_v1);
    persistent_identity_v1 result_identity{};
    persistent_identity_v1 request_identity{};
    const atom_fragment_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_count = 0u;
    std::uint64_t candidate_capacity = 0u;
    no_candidate_reason_v1 no_candidate_reason = no_candidate_reason_v1::none;
    bool frontier_truncated = false;
    std::uint8_t reserved[6]{};
};

enum class atom_fragment_result_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_result_identity = 4u,
    invalid_request_identity = 5u,
    invalid_candidate_bounds = 6u,
    invalid_no_candidate_reason = 7u,
    inconsistent_empty_frontier = 8u,
    inconsistent_truncation = 9u,
    invalid_candidate_identity = 10u,
    duplicate_or_unordered_candidate = 11u,
    invalid_local_coverage = 12u,
    missing_atom_inputs = 13u,
    invalid_atom_input = 14u,
    duplicate_or_unordered_atom_input = 15u,
    invalid_program_recipe = 16u,
    missing_projection_requirements = 17u,
    invalid_projection_requirement = 18u,
    duplicate_or_unordered_projection_requirement = 19u,
    invalid_output_affordance = 20u,
    invalid_partial_affordance = 21u,
    unexpected_partial_affordance = 22u,
    invalid_order = 23u,
    invalid_resource = 24u,
    invalid_complete_cost = 25u,
    invalid_empirical_status = 26u,
    unknown_flag = 27u,
    invalid_validation_receipt = 28u
};

struct atom_fragment_result_validation_result_v1 {
    atom_fragment_result_validation_code_v1 code =
        atom_fragment_result_validation_code_v1::ok;
    std::uint64_t candidate_index = 0u;
    std::uint64_t element_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_fragment_result_validation_code_v1::ok;
    }
};

atom_fragment_result_validation_result_v1 validate_atom_fragment_result_v1(
    const atom_fragment_result_v1 &result) noexcept;

static_assert(std::is_standard_layout_v<fragment_resource_vector_v1>);
static_assert(std::is_trivially_copyable_v<fragment_resource_vector_v1>);
static_assert(std::is_standard_layout_v<fragment_complete_cost_v1>);
static_assert(std::is_trivially_copyable_v<fragment_complete_cost_v1>);
static_assert(std::is_standard_layout_v<atom_fragment_candidate_v1>);
static_assert(std::is_trivially_copyable_v<atom_fragment_candidate_v1>);
static_assert(std::is_standard_layout_v<atom_fragment_result_v1>);
static_assert(std::is_trivially_copyable_v<atom_fragment_result_v1>);

}  // namespace cellerator::execution::joint_compiler
