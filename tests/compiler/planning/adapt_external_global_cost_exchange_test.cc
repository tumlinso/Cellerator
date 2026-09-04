#include <Cellerator/compiler/planning/adapt_external_global_cost_exchange_v1.hh>

#include <cassert>
#include <cmath>

namespace planning = Cellerator::compiler::planning;

namespace {

planning::external_global_cost_reply_v1 reply(
    const planning::external_global_cost_query_v1&,
    const void* context) noexcept {
    return *static_cast<const planning::external_global_cost_reply_v1*>(context);
}

planning::external_global_cost_evidence_v1 evidence(std::uint64_t contract) {
    return {contract, 2u, 0.1, 0.2, 0.3, 4.0, 5.0, 10.0, 6.0, 0.4};
}

}  // namespace

int main() {
    const planning::external_global_cost_query_v1 query{
        1u, 100u, 200u, 300u, 2u, 1000u};
    const auto fallback = evidence(10u);
    const planning::external_global_cost_reply_v1 success{
        planning::external_global_cost_reply_code_v1::success, evidence(20u)};
    auto result = planning::adapt_external_global_cost_exchange_v1(
        query, {reply, &success}, fallback);
    assert(result);
    assert(!result.used_fallback);
    assert(result.planner_cost.contract_id == 20u);
    assert(std::abs(result.planner_cost.fixed_ns - 11.0) < 1.0e-12);
    assert(std::abs(result.planner_cost.transfer_byte_ns - 0.3) < 1.0e-12);

    const planning::external_global_cost_reply_v1 timeout{
        planning::external_global_cost_reply_code_v1::timeout, {}};
    result = planning::adapt_external_global_cost_exchange_v1(
        query, {reply, &timeout}, fallback);
    assert(result.code == planning::external_global_cost_adapter_code_v1::fallback_timeout);
    assert(result.used_fallback && result.planner_cost.contract_id == 10u);

    const planning::external_global_cost_reply_v1 failure{
        planning::external_global_cost_reply_code_v1::failure, {}};
    result = planning::adapt_external_global_cost_exchange_v1(
        query, {reply, &failure}, fallback);
    assert(result.code == planning::external_global_cost_adapter_code_v1::fallback_failure);
    assert(result.used_fallback);
}
