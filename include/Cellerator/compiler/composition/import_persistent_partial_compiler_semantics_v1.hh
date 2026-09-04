#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct persistent_partial_semantics_v1{std::string id,coverage,merge_algebra,finalize_algebra,numerical_contract;std::vector<std::string> dependencies;std::uint64_t structure_epoch=0,value_generation=0,build_cost=0,reuse_savings=0,expected_reuse=0;};
struct partial_decision_v1{bool legal=false,persist=false;std::string reason;};
[[nodiscard]] partial_decision_v1 evaluate_persistent_partial_v1(const persistent_partial_semantics_v1&,std::uint64_t required_epoch,std::uint64_t required_generation);
} // namespace Cellerator::compiler::composition
