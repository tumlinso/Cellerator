#include <Cellerator/sdk/preserve_direct_runtime_execution_access_v1.hh>
#include <cassert>
int main(){
 cellerator::execution::persistent_axis_identity axis{};
 cellerator::compute::math::core::operation_problem operation{};
 cellerator::compute::operation::relation_algebra_problem_v1 relation{};
 cellerator::planner::planning_keys planning{};
 cellerator_execution_config runtime{CELLERATOR_ABI_VERSION,sizeof(cellerator_execution_config),-1,nullptr,nullptr,0};
 (void)axis;(void)operation;(void)relation;(void)planning;assert(runtime.version==CELLERATOR_ABI_VERSION);
 using namespace cellerator::compiler::api::v1;
 assert(is_direct_runtime_surface_v1("operation-core"));assert(!is_direct_runtime_surface_v1("compiler-required"));
}
