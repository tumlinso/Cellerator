#include <Cellerator/compiler/program/import_global_operation_graph_ir_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){planning_operation_graph_v1 g{{{"load","semantic.import","io","read","fragment.load",{}, {"a"},{"sm70","host"},{}},{"apply","semantic.relation_apply","forward","pure","fragment.apply",{"a"},{"b"},{"sm70"},{"fuse_epilogue"}}}};std::string e;assert(validate_planning_operation_graph_v1(g,&e));auto bad=g;bad.operations[1].semantic_ir_operation.clear();assert(!validate_planning_operation_graph_v1(bad,&e));auto missing=g;missing.operations[1].input_atoms={"storage_handle"};assert(!validate_planning_operation_graph_v1(missing,&e));}
