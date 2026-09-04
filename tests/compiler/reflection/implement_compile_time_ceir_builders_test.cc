#include <Cellerator/compiler/reflection/implement_compile_time_ceir_builders_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){ceir_builder_v1 b;auto op=append_ceir_node_v1(&b,ceir_builder_node_kind_v1::semantic_operation,"apply","f32");auto candidate=append_ceir_node_v1(&b,ceir_builder_node_kind_v1::planning_alternative,"csr","cost=7",{op});assert(op==0&&candidate==1&&validate_ceir_builder_v1(b));assert(append_ceir_node_v1(&b,ceir_builder_node_kind_v1::candidate,"bad","",{99})==UINT32_MAX&&!b.diagnostics.empty());}
