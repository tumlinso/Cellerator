#pragma once

#include <Cellerator/compute/operation/relation_algebra.hh>
#include <Cellerator/execution/launch_bindings.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation {

inline constexpr std::uint32_t relation_bundle_schema_version_v1 = 1u;
inline constexpr std::uint32_t invalid_relation_bundle_member_v1 = 0xffffffffu;

enum class relation_bundle_kind_v1 : std::uint8_t {
    destination_accumulate = 1u,
    incidence_pool = 2u,
    incidence_broadcast = 3u
};

enum class relation_bundle_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_schema = 2u,
    invalid_operation = 3u,
    invalid_identity = 4u,
    invalid_relation = 5u,
    incompatible_destination = 6u,
    invalid_numeric_policy = 7u,
    invalid_shape = 8u,
    execution_failed = 9u
};

struct relation_bundle_result_v1 {
    relation_bundle_status_v1 code = relation_bundle_status_v1::ok;
    std::uint32_t member_index = invalid_relation_bundle_member_v1;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == relation_bundle_status_v1::ok;
    }
};

// Member arrays remain caller-owned cold storage. For destination accumulation
// every relation is applied forward into one exact destination axis. Incidence
// pool/broadcast use one typed incidence relation in forward/transpose order.
struct relation_bundle_plan_v1 {
    std::uint32_t schema_version = relation_bundle_schema_version_v1;
    relation_bundle_kind_v1 kind =
        relation_bundle_kind_v1::destination_accumulate;
    std::uint8_t reserved[3]{};
    relation_bundle_view_v1 bundle{};
    const core::stable_id *operation_identities = nullptr;
    relation_numeric_semantics_v1 numeric{};
    std::uint32_t dense_width = 0u;
    std::uint32_t reserved1 = 0u;
};

// The executor owns prepared projection selection and launch bindings. The
// composition layer supplies only a validated typed relation operation and an
// explicit destination effect. Returning false stops the sequence immediately.
using relation_apply_step_function_v1 = bool (*)(
    const relation_algebra_problem_v1 &problem,
    execution::output_update_kind destination_update,
    void *context) noexcept;

relation_bundle_result_v1 validate_relation_bundle_plan_v1(
    const relation_bundle_plan_v1 &plan) noexcept;

relation_bundle_result_v1 run_relation_bundle_v1(
    const relation_bundle_plan_v1 &plan,
    relation_apply_step_function_v1 execute,
    void *context) noexcept;

static_assert(std::is_trivially_copyable<relation_bundle_plan_v1>::value,
    "relation bundle plans must remain caller-owned POD views");

} // namespace cellerator::compute::operation
