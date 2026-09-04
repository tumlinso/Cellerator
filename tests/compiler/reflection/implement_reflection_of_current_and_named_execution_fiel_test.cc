#include <Cellerator/compiler/reflection/implement_reflection_of_current_and_named_execution_fiel_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1; struct tag{};
int main(){ir_handle_v1 h{1,1,1,1,handle_kind_v1::field,availability_phase_v1::semantic,handle_lifetime_v1::compilation};field_reflection_scope_v1 s{availability_phase_v1::semantic,reflected_field_v1{"current",h,false},{{"named",h,true},{"private",h,false}}};assert(reflect_current_field_v1(s));assert(reflect_field_v1<tag>(s,"named"));assert(!reflect_named_field_v1(s,"private"));s.phase=availability_phase_v1::source;assert(!reflect_current_field_v1(s));}
