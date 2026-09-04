#include <Cellerator/compiler/api/compiler.hpp>
#include <Cellerator/sdk/freeze_libcellerator_sdk_acceptance_v1.hh>
#include <Cellerator/sdk/runtime.hpp>
#include <cassert>
extern "C" int h02_c_consumer(void);
int main(){using namespace cellerator::compiler::api::v1;assert(h02_c_consumer());assert(accepted_sdk_umbrellas_v1.size()==3);assert(sdk_public_header_is_dependency_clean_v1("#include <vector>"));assert(!sdk_public_header_is_dependency_clean_v1("#include <llvm/IR/Module.h>"));runtime_axis_v1 axis{1,2,3};assert(axis.extent==3);}
