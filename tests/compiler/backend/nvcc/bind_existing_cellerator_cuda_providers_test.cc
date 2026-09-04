#include <Cellerator/compiler/backend/nvcc/bind_existing_cellerator_cuda_providers_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;provider_binding_status s{};auto b=bind_existing_provider({9,70,3,70},&s);assert(b&&b->target=="Cellerator::provider_sm70"&&!b->generated_kernel);assert(!bind_existing_provider({9,99,3,70},&s));assert(s==provider_binding_status::unknown_provider);}
