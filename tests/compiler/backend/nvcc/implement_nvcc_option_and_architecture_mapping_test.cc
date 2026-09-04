#include <Cellerator/compiler/backend/nvcc/implement_nvcc_option_and_architecture_mapping_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;nvcc_options o{{80,70},{90},"gcc",17,3,false,true,true,{"cudart"},{"--use_fast_math"}};auto a=make_nvcc_argv(o);assert(a&&(*a)[5]=="-lineinfo"&&(*a)[7].find("sm_70")!=std::string::npos);option_status s{};o.user_options={"-arch=sm_80"};assert(!make_nvcc_argv(o,&s)&&s==option_status::unsafe_override);}
