#include <Cellerator/compiler/reflection/implement_reflection_of_operations_and_relations_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
static ir_handle_v1 h(handle_kind_v1 k){return{1,1,1,1,k,availability_phase_v1::semantic,handle_lifetime_v1::compilation};}
int main(){reflected_value_v1 v{h(handle_kind_v1::field),"f32",1,2,3};reflected_operation_v1 o{h(handle_kind_v1::operation),"relation_apply",{v},{v},{h(handle_kind_v1::relation),1,2,3,4,5},reflected_effect_v1::reads_writes,h(handle_kind_v1::provenance)};assert(validate_reflected_operation_v1(o));assert(dump_reflected_operation_v1(o).find("relation_apply")!=std::string::npos);o.relation.order=0;assert(!validate_reflected_operation_v1(o));}
