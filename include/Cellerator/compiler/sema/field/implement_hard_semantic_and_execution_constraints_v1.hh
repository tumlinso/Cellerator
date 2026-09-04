#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

struct hard_execution_constraints_v1 {
    bool require_determinism = false;
    bool require_exactness = false;
    double maximum_numerical_error = 0.0;
    std::uint64_t maximum_memory_bytes = 0;
    std::uint64_t required_target_capabilities = 0;
    std::uint64_t allowed_candidate_families = 0;
    std::uint64_t required_order_identity = 0;
    bool forbid_synchronization = false;
};

struct constrained_plan_candidate_v1 {
    std::uint64_t candidate_identity = 0;
    bool deterministic = false;
    bool exact = false;
    double numerical_error = 0.0;
    std::uint64_t memory_bytes = 0;
    std::uint64_t target_capabilities = 0;
    std::uint64_t candidate_family = 0;
    std::uint64_t output_order_identity = 0;
    bool synchronizes = false;
};

struct hard_constraint_rejection_v1 {
    std::uint64_t candidate_identity = 0;
    std::string reason;
};

struct hard_constraint_result_v1 {
    std::vector<std::uint64_t> legal_candidates;
    std::vector<hard_constraint_rejection_v1> rejections;
};

enum class hard_constraint_status_v1 : std::uint8_t {
    success = 0,
    invalid_output,
    invalid_constraint,
    invalid_candidate,
    no_legal_continuation,
};

[[nodiscard]] hard_constraint_status_v1 implement_hard_semantic_and_execution_constraints_v1(
    const hard_execution_constraints_v1& constraints,
    const std::vector<constrained_plan_candidate_v1>& candidates,
    hard_constraint_result_v1* result) noexcept;

}  // namespace Cellerator::compiler::sema::field
