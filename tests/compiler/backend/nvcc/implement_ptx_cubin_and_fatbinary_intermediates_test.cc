#include <Cellerator/compiler/backend/nvcc/implement_ptx_cubin_and_fatbinary_intermediates_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;artifact_bundle b{{{artifact_kind::ptx,"x.ptx","nvcc-12",70,1,false},{artifact_kind::cubin,"x.sm70.cubin","nvcc-12",70,2,true},{artifact_kind::fatbinary,"x.fatbin","nvcc-12",70,3,true}}};assert(validate_artifact_bundle(b)==artifact_status::ok);assert(select_artifact(b,70)->kind==artifact_kind::cubin);assert(select_artifact(b,80)->kind==artifact_kind::fatbinary);}
