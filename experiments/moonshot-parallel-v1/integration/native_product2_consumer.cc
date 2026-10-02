#include <Cellerator/compute/operation/product2/product2.hh>
#include <iostream>
#include <stdexcept>
using namespace cellerator::compute::product2;
using namespace cellerator::execution::program;
void require(bool v){if(!v)throw std::runtime_error("installed product2 consumer check failed");}
int main(){
 int64_t a[]={0},b[]={0};float x[]={3},k[]={2},y[]={-77};
 prepared_owner owner({1,1,9,a,b,1,1});bound_view view{};view.numeric.x=x;view.numeric.x_count=1;view.numeric.k=k;view.numeric.k_count=1;view.numeric.y=y;view.numeric.y_count=1;view.numeric.expected_structure_generation=view.numeric.current_structure_generation=9;
 stage_state state{&owner,operation::forward};auto stage=make_stage(state,1,0);prepared_program_v2 program{2,0,&stage,1,nullptr,0};launch_binding_v2 binding{};binding.input=&view;
 require(execute_prepared_program_v2(program,&binding,1,nullptr)==program_status::success && y[0]==18);
 view.numeric.current_parameter_generation=1;y[0]=-88;require(execute_prepared_program_v2(program,&binding,1,nullptr)!=program_status::success && y[0]==-88);
 std::cout<<"installed Cellerator::product2 public C++ prepared-stage consumer PASS\n";
}
