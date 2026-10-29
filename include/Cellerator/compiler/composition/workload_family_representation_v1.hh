#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
enum class mutation_horizon_v1:std::uint8_t{immutable,per_generation,per_step};
struct workload_objective_v1{std::string metric;double weight=0;};
struct workload_family_v1{std::string semantic_ir_family,profile_family,target_class;std::uint64_t recurrence=1;mutation_horizon_v1 mutation=mutation_horizon_v1::per_step;std::vector<workload_objective_v1> objectives;};
[[nodiscard]] bool validate_workload_family_v1(const workload_family_v1&,std::string*error=nullptr);
} // namespace Cellerator::compiler::composition
