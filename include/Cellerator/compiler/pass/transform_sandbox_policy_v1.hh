#pragma once

#include <cstdint>
#include <string>

namespace cellerator::compiler::pass::v1 {

enum class transform_execution_mode_v1 : std::uint8_t {
    trusted_in_process = 0,
    isolated_subprocess,
    isolated_verified,
};
enum class transform_observation_v1 : std::uint8_t {
    success = 0,
    rejected,
    crashed,
    timed_out,
    memory_limit,
};

struct transform_sandbox_policy_v1 {
    transform_execution_mode_v1 requested_mode =
        transform_execution_mode_v1::trusted_in_process;
    std::uint64_t time_limit_milliseconds = 0;
    std::uint64_t memory_limit_bytes = 0;
    bool unsafe_continue_after_failure = false;
};

using transform_execute_v1 = transform_observation_v1 (*)(void*) noexcept;
using transform_verify_v1 = bool (*)(void*) noexcept;

struct transform_sandbox_receipt_v1 {
    transform_execution_mode_v1 executed_mode =
        transform_execution_mode_v1::trusted_in_process;
    transform_observation_v1 observation = transform_observation_v1::success;
    bool isolated = false;
    bool verified = false;
    bool continuation_allowed = false;
    std::string diagnostic;
};

[[nodiscard]] transform_sandbox_receipt_v1 execute_transform_with_policy_v1(
    const transform_sandbox_policy_v1& policy, transform_execute_v1 execute,
    transform_verify_v1 verify, void* user_data) noexcept;

}  // namespace cellerator::compiler::pass::v1
