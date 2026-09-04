#include <Cellerator/compiler/reflection/implement_reflection_of_realization_ir_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){ir_handle_v1 h{1,1,1,1,handle_kind_v1::selected_realization,availability_phase_v1::realized,handle_lifetime_v1::artifact};reflected_realization_v1 cpu{h,"cpu",{1},{2},{3},{4},{5},{6},{7},8,9},gpu{h,"cuda",{1},{2},{3},{4},{5},{6},{7},8,9};assert(validate_reflected_realization_v1(cpu)&&!realization_is_accelerated_v1(cpu));assert(validate_reflected_realization_v1(gpu)&&realization_is_accelerated_v1(gpu));}
