#include <Cellerator/compiler/backend/nvcc/implement_relocatable_device_code_and_device_linking_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;device_link_request r{{{"a.o",{"kernel"},{}},{"b.o",{"helper"},{"kernel"}}},{"cudadevrt"},"link.o","register.o",{70}};auto p=plan_device_link(r);assert(p&&p->compile_actions.size()==2&&p->nvlink_argv[1]=="-dlink");r.objects[1].references={"missing"};assert(!plan_device_link(r));}
