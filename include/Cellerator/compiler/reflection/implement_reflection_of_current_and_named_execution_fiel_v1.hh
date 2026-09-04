#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
struct reflected_field_v1{std::string name;ir_handle_v1 handle{};bool exported=false;};
struct field_reflection_scope_v1{availability_phase_v1 phase=availability_phase_v1::source;std::optional<reflected_field_v1>current;std::vector<reflected_field_v1>named;};
[[nodiscard]] std::optional<ir_handle_v1> reflect_current_field_v1(const field_reflection_scope_v1&)noexcept;
[[nodiscard]] std::optional<ir_handle_v1> reflect_named_field_v1(const field_reflection_scope_v1&,const std::string&)noexcept;
template<class Tag>[[nodiscard]] std::optional<ir_handle_v1> reflect_field_v1(const field_reflection_scope_v1&s,const std::string&name){return reflect_named_field_v1(s,name);}
}
