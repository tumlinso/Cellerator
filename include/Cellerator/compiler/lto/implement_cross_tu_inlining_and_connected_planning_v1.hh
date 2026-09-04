#pragma once
#include <cstdint>
#include <string>
namespace cellerator::compiler::lto::v1 {
struct connected_relation_chain_v1{std::string producer_order,consumer_order,producer_decomposition,consumer_decomposition;std::uint64_t producer_ns=0,consumer_ns=0,materialize_ns=0,order_transition_ns=0;bool semantic_body_available=false,authorized=false,effects_permit_inline=false;};
struct connected_planning_result_v1{bool semantic_inlined=false,persistent_order=false,shared_decomposition=false;std::uint64_t selected_total_ns=0,materialized_total_ns=0;};
enum class connected_planning_status_v1:std::uint8_t{valid=0,body_unavailable,unauthorized,effect_boundary};
[[nodiscard]] connected_planning_status_v1 plan_connected_cross_tu_chain_v1(const connected_relation_chain_v1&,connected_planning_result_v1*)noexcept;
}
