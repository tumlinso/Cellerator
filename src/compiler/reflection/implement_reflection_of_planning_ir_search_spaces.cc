#include <Cellerator/compiler/reflection/implement_reflection_of_planning_ir_search_spaces_v1.hh>
namespace cellerator::compiler::reflection::v1 {
bool can_reflect_search_space_v1(const reflected_search_space_v1&s,availability_phase_v1 p)noexcept{return p>=s.available_at;}
const reflected_planning_alternative_v1* selected_alternative_v1(const reflected_search_space_v1&s)noexcept{const reflected_planning_alternative_v1*r=nullptr;for(const auto&a:s.alternatives)if(a.selection==reflected_selection_v1::selected||a.selection==reflected_selection_v1::forced){if(r)return nullptr;r=&a;}return r;}
}
