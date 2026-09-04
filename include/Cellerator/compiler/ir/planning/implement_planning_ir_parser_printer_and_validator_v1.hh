#pragma once
#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellerator::compiler::ir::planning::v1 {
struct planning_text_module_v1{planning_identity_v1 module{};std::uint32_t alternatives=0,coverages=0,atoms=0,costs=0,evidence=0,selections=0;const char*unknown_extensions=nullptr;std::size_t unknown_extension_bytes=0;};
enum class planning_text_status_v1:std::uint8_t{ok=0,invalid_argument,invalid_syntax,invalid_module,insufficient_capacity,trailing_input};
planning_text_status_v1 parse_planning_ir_v1(const char*,std::size_t,char*,std::size_t,planning_text_module_v1*) noexcept;
planning_text_status_v1 print_planning_ir_v1(const planning_text_module_v1&,char*,std::size_t,std::size_t*) noexcept;
planning_text_status_v1 validate_planning_text_module_v1(const planning_text_module_v1&) noexcept;
}
