#include <Cellerator/sdk/create_installed_sdk_examples_v1.hh>
#include <cassert>
bool runtime_example();bool source_compiler_example();bool ceir_editing_example();bool custom_candidate_example();bool custom_pass_example();bool backend_example();
int main(){using namespace cellerator::compiler::api::v1;assert(installed_sdk_examples_v1.size()==6);assert(has_installed_sdk_example_v1("backend"));assert(runtime_example()&&source_compiler_example()&&ceir_editing_example()&&custom_candidate_example()&&custom_pass_example()&&backend_example());}
