#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){ir_handle_v1 h{1,2,4,7,handle_kind_v1::field,availability_phase_v1::semantic,handle_lifetime_v1::compilation};handle_context_v1 c{4,availability_phase_v1::semantic};assert(validate_handle_v1(h,c,7)==handle_status_v1::valid);assert(preserve_handle_for_safe_transform_v1(h).object_generation==7);auto edited=invalidate_handle_for_edit_v1(h);assert(validate_handle_v1(h,c,edited.object_generation)==handle_status_v1::stale);c.phase=availability_phase_v1::source;assert(validate_handle_v1(h,c,7)==handle_status_v1::unavailable);}
