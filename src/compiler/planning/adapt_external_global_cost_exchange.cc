#include <Cellerator/compiler/planning/adapt_external_global_cost_exchange_v1.hh>

#include <cmath>

namespace Cellerator::compiler::planning {
namespace external = cellerator::planner::external_cost;
namespace {

bool valid(const external_global_cost_evidence_v1& value) noexcept {
    const double fields[] = {value.storage_byte_nanoseconds,
        value.movement_byte_nanoseconds, value.replication_byte_nanoseconds,
        value.invalidation_event_nanoseconds, value.latency_nanoseconds,
        value.throughput_bytes_per_nanosecond,
        value.application_fixed_nanoseconds, value.application_byte_nanoseconds};
    if (value.contract_identity == 0u || value.pricing_epoch == 0u) return false;
    for (const auto field : fields)
        if (!std::isfinite(field) || field < 0.0) return false;
    return true;
}

external::external_cost_vector_v1 translate(
    const external_global_cost_evidence_v1& evidence) noexcept {
    external::external_cost_vector_v1 cost{};
    cost.contract_id = evidence.contract_identity;
    cost.pricing_epoch = evidence.pricing_epoch;
    cost.fixed_ns = evidence.latency_nanoseconds + evidence.application_fixed_nanoseconds;
    cost.persistent_byte_ns = evidence.storage_byte_nanoseconds;
    cost.transient_byte_ns = evidence.application_byte_nanoseconds;
    cost.transfer_byte_ns = evidence.movement_byte_nanoseconds +
        (evidence.throughput_bytes_per_nanosecond > 0.0
            ? 1.0 / evidence.throughput_bytes_per_nanosecond : 0.0);
    cost.communication_byte_ns = evidence.replication_byte_nanoseconds;
    cost.synchronization_ns = evidence.invalidation_event_nanoseconds;
    return cost;
}

}  // namespace

external_global_cost_adapter_result_v1 adapt_external_global_cost_exchange_v1(
    const external_global_cost_query_v1& query,
    const external_global_cost_source_v1& source,
    const external_global_cost_evidence_v1& fallback) noexcept {
    external_global_cost_adapter_result_v1 result{};
    if (query.candidate_identity == 0u || query.deadline_nanoseconds == 0u ||
        source.query == nullptr) return result;

    const auto reply = source.query(query, source.context);
    if (reply.code == external_global_cost_reply_code_v1::success) {
        result.code = external_global_cost_adapter_code_v1::success;
        result.evidence = reply.evidence;
    } else {
        result.used_fallback = true;
        result.code = reply.code == external_global_cost_reply_code_v1::timeout
            ? external_global_cost_adapter_code_v1::fallback_timeout
            : external_global_cost_adapter_code_v1::fallback_failure;
        result.evidence = fallback;
    }
    if (!valid(result.evidence)) {
        result.code = external_global_cost_adapter_code_v1::invalid_evidence;
        result.used_fallback = false;
        return result;
    }
    result.planner_cost = translate(result.evidence);
    if (external::validate_external_cost_vector_v1(result.planner_cost) !=
        external::external_cost_vector_status_v1::valid) {
        result.code = external_global_cost_adapter_code_v1::invalid_evidence;
        result.used_fallback = false;
    }
    return result;
}

}  // namespace Cellerator::compiler::planning
