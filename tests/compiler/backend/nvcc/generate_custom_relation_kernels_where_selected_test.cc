#include <Cellerator/compiler/backend/nvcc/generate_custom_relation_kernels_where_selected_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;custom_kernel_status s{};custom_relation_kernel_request r{"relation_field",32,48,2,7,16,true,true,true,false};auto k=generate_custom_relation_kernel(r,&s);assert(k&&k->declaration.find("i < 32ULL")!=std::string::npos&&k->declaration.find("+=")!=std::string::npos);r.prelinked_provider_selected=true;assert(!generate_custom_relation_kernel(r,&s));}
