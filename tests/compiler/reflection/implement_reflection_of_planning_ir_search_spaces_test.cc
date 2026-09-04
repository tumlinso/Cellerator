#include <Cellerator/compiler/reflection/implement_reflection_of_planning_ir_search_spaces_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){ir_handle_v1 h{1,1,1,1,handle_kind_v1::planning_alternative,availability_phase_v1::planned,handle_lifetime_v1::compilation};reflected_search_space_v1 s{availability_phase_v1::planned,{{h,{0,1},{1},{1},{4,5},"csr","",reflected_selection_v1::selected},{h,{0,1},{2},{2},{7},"native","cost",reflected_selection_v1::rejected}}};assert(!can_reflect_search_space_v1(s,availability_phase_v1::profiled));assert(can_reflect_search_space_v1(s,availability_phase_v1::planned));assert(selected_alternative_v1(s)->candidate=="csr");}
