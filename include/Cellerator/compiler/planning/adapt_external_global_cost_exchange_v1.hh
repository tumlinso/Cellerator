#pragma once

#include <Cellerator/planner/external_cost/vector_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::planning {

struct external_global_cost_query_v1 {
    std::uint64_t candidate_identity = 0u;
    std::uint64_t storage_bytes = 0u;
    std::uint64_t movement_bytes = 0u;
    std::uint64_t replication_bytes = 0u;
    std::uint64_t invalidation_events = 0u;
    std::uint64_t deadline_nanoseconds = 0u;
};

struct external_global_cost_evidence_v1 {
    std::uint64_t contract_identity = 0u;
    std::uint64_t pricing_epoch = 0u;
    double storage_byte_nanoseconds = 0.0;
    double movement_byte_nanoseconds = 0.0;
    double replication_byte_nanoseconds = 0.0;
    double invalidation_event_nanoseconds = 0.0;
    double latency_nanoseconds = 0.0;
    double throughput_bytes_per_nanosecond = 0.0;
    double application_fixed_nanoseconds = 0.0;
    double application_byte_nanoseconds = 0.0;
};

enum class external_global_cost_reply_code_v1 : std::uint8_t {
    success = 0u,
    timeout,
    failure,
};

struct external_global_cost_reply_v1 {
    external_global_cost_reply_code_v1 code =
        external_global_cost_reply_code_v1::failure;
    external_global_cost_evidence_v1 evidence{};
};

using external_global_cost_callback_v1 = external_global_cost_reply_v1 (*)(
    const external_global_cost_query_v1& query,
    const void* context) noexcept;

struct external_global_cost_source_v1 {
    external_global_cost_callback_v1 query = nullptr;
    const void* context = nullptr;
};

enum class external_global_cost_adapter_code_v1 : std::uint8_t {
    success = 0u,
    fallback_timeout,
    fallback_failure,
    invalid_query,
    invalid_evidence,
};

struct external_global_cost_adapter_result_v1 {
    external_global_cost_adapter_code_v1 code =
        external_global_cost_adapter_code_v1::invalid_query;
    external_global_cost_evidence_v1 evidence{};
    cellerator::planner::external_cost::external_cost_vector_v1 planner_cost{};
    bool used_fallback = false;

    constexpr explicit operator bool() const noexcept {
        return code == external_global_cost_adapter_code_v1::success ||
            code == external_global_cost_adapter_code_v1::fallback_timeout ||
            code == external_global_cost_adapter_code_v1::fallback_failure;
    }
};

[[nodiscard]] external_global_cost_adapter_result_v1
adapt_external_global_cost_exchange_v1(
    const external_global_cost_query_v1& query,
    const external_global_cost_source_v1& source,
    const external_global_cost_evidence_v1& fallback) noexcept;

}  // namespace Cellerator::compiler::planning
