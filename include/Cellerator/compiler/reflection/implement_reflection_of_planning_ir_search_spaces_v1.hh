#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class reflected_selection_v1:std::uint8_t{offered=1,rejected,selected,forced};
struct reflected_planning_alternative_v1{ir_handle_v1 handle{};std::vector<std::uint64_t>exact_coverage,atoms,decomposition,cost_ns;std::string candidate,rejection_reason;reflected_selection_v1 selection=reflected_selection_v1::offered;};
struct reflected_search_space_v1{availability_phase_v1 available_at=availability_phase_v1::planned;std::vector<reflected_planning_alternative_v1>alternatives;};
[[nodiscard]] bool can_reflect_search_space_v1(const reflected_search_space_v1&,availability_phase_v1)noexcept;
[[nodiscard]] const reflected_planning_alternative_v1* selected_alternative_v1(const reflected_search_space_v1&)noexcept;
}
