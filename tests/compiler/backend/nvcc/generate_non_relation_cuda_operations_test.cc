#include <Cellerator/compiler/backend/nvcc/generate_non_relation_cuda_operations_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::backend::nvcc::v1;for(unsigned k=1;k<=11;++k){auto e=generate_non_relation_operation({static_cast<non_relation_operation>(k),"stage_"+std::to_string(k),4,5,true});assert(e&&e->kind==cuda_entity_kind::kernel);}non_relation_status s{};assert(!generate_non_relation_operation({non_relation_operation::publish,"publish",5,4,true},&s));}
