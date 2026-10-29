#pragma once
#include <Cellerator/planner/end_to_end_planner.hh>
#include <array>
#include <cstdint>
namespace cellerator::compiler::ir::planning::v1 {
enum class cost_dimension_v1:std::uint8_t{preparation=0,conversion,transfer,pack,order,execution,residual,epilogue,synchronization,canonicalization,memory,compile_time,reuse_amortization,communication,count};
inline constexpr std::size_t cost_dimension_count_v1=static_cast<std::size_t>(cost_dimension_v1::count);
struct complete_cost_vector_v1{std::array<double,cost_dimension_count_v1> nanoseconds{};std::uint64_t persistent_bytes=0,transient_bytes=0,reuse_count=1;double total_nanoseconds=0;};
enum class complete_cost_status_v1:std::uint8_t{ok=0,invalid_cost,invalid_reuse,total_mismatch};
complete_cost_status_v1 import_phase_costs_v1(const cellerator::planner::phase_costs&,double conversion_ns,double residual_ns,double compile_ns,std::uint64_t reuse,complete_cost_vector_v1*) noexcept;
complete_cost_status_v1 validate_complete_cost_vector_v1(const complete_cost_vector_v1&) noexcept;
}
