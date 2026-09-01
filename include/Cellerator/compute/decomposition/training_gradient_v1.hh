#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t training_gradient_schema_version_v1 = 1u;

enum class gradient_target_v1 : std::uint8_t {
    source_state = 1u,
    destination_state = 2u,
    relation_values = 3u
};

enum class gradient_accumulation_v1 : std::uint8_t {
    independent = 1u,
    associative_partial = 2u,
    ordered_scatter = 3u
};

struct gradient_decomposition_v1 {
    gradient_target_v1 target = gradient_target_v1::source_state;
    std::uint64_t relation_index = 0u;
    execution::persistent_axis_identity biological_axis{};
    execution::order_id logical_edge_order{};
    split_axis_kind_v1 split_axis = split_axis_kind_v1::none;
    gradient_accumulation_v1 accumulation =
        gradient_accumulation_v1::independent;
    bool has_biological_axis = true;
    bool requires_transpose_projection = false;
    bool requires_partial_algebra = false;
    bool preserves_logical_edge_identity = false;
    bool deterministic_merge = true;
    std::uint8_t reserved[3]{};
};

struct training_gradient_contract_v1 {
    std::uint32_t schema_version = training_gradient_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id identity{};
    operation::v2::stable_id forward_decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    const gradient_decomposition_v1 *gradients = nullptr;
    std::uint64_t gradient_count = 0u;
};

enum class training_gradient_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    backward_not_requested,
    missing_gradients,
    invalid_gradient_count,
    invalid_target,
    target_order_mismatch,
    target_not_requested,
    invalid_relation_index,
    invalid_split_axis,
    invalid_accumulation,
    invalid_axis_contract,
    axis_identity_mismatch,
    edge_identity_mismatch,
    invalid_transpose_requirement,
    invalid_partial_algebra_requirement,
    nondeterministic_ordered_scatter,
    missing_requested_gradient
};

struct training_gradient_validation_result_v1 {
    training_gradient_validation_code_v1 code =
        training_gradient_validation_code_v1::ok;
    std::uint64_t gradient_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == training_gradient_validation_code_v1::ok;
    }
};

training_gradient_validation_result_v1 validate_training_gradient_contract_v1(
    const training_gradient_contract_v1 &contract) noexcept;

static_assert(std::is_trivially_copyable_v<gradient_decomposition_v1>);
static_assert(std::is_standard_layout_v<gradient_decomposition_v1>);
static_assert(std::is_trivially_copyable_v<training_gradient_contract_v1>);
static_assert(std::is_standard_layout_v<training_gradient_contract_v1>);
static_assert(std::is_trivially_copyable_v<training_gradient_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
