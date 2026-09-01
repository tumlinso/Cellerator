#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::planner::external_cost {

inline constexpr std::uint32_t external_cost_vector_schema_v1 = 1u;

struct external_cost_vector_v1 {
    std::uint32_t schema_version = external_cost_vector_schema_v1;
    std::uint32_t record_bytes = sizeof(external_cost_vector_v1);
    std::uint64_t contract_id = 0u;
    std::uint64_t pricing_epoch = 0u;
    double fixed_ns = 0.0;
    double persistent_byte_ns = 0.0;
    double transient_byte_ns = 0.0;
    double transfer_byte_ns = 0.0;
    double communication_byte_ns = 0.0;
    double launch_ns = 0.0;
    double synchronization_ns = 0.0;
    double reuse_credit_ns = 0.0;
    std::uint64_t expected_reuse = 1u;
};

enum class external_cost_vector_status_v1 : std::uint8_t {
    valid = 0u,
    unsupported_schema,
    invalid_record_bytes,
    invalid_contract,
    invalid_pricing_epoch,
    invalid_component,
    invalid_reuse,
};

external_cost_vector_status_v1 validate_external_cost_vector_v1(
    const external_cost_vector_v1 &cost) noexcept;

static_assert(std::is_standard_layout_v<external_cost_vector_v1>);
static_assert(std::is_trivially_copyable_v<external_cost_vector_v1>);

} // namespace cellerator::planner::external_cost
