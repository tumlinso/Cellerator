#include <Cellerator/compiler/reflection/implement_reflection_of_profile_environments_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){ir_handle_v1 h{1,1,1,1,handle_kind_v1::profile_state,availability_phase_v1::profiled,handle_lifetime_v1::compilation};reflected_profile_environment_v1 e{{{"base",h,{10,.9},{"values vary"},{"left"},{"cells"},{}},{"branch",h,{4,.8},{},{},{},"then"}},"base"};assert(validate_profile_environment_v1(e));assert(query_profile_state_v1(e,"base"));assert(query_profile_state_v1(e,"branch","then"));assert(!query_profile_state_v1(e,"branch"));}
