#include <Cellerator/compiler/reflection/implement_reflection_of_current_and_named_execution_fiel_v1.hh>
namespace cellerator::compiler::reflection::v1 {
std::optional<ir_handle_v1> reflect_current_field_v1(const field_reflection_scope_v1&s)noexcept{if(s.phase<availability_phase_v1::semantic||!s.current)return{};return s.current->handle;}
std::optional<ir_handle_v1> reflect_named_field_v1(const field_reflection_scope_v1&s,const std::string&n)noexcept{if(s.phase<availability_phase_v1::semantic)return{};for(const auto&f:s.named)if(f.exported&&f.name==n)return f.handle;return{};}
}
