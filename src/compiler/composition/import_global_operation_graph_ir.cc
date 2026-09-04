#include <Cellerator/compiler/composition/import_global_operation_graph_ir_v1.hh>
#include <set>
namespace Cellerator::compiler::composition {
bool validate_planning_operation_graph_v1(const planning_operation_graph_v1&g,std::string*e){auto fail=[&](std::string m){if(e)*e=std::move(m);return false;};std::set<std::string> ids,produced;for(const auto&o:g.operations){if(o.id.empty()||o.semantic_ir_operation.empty()||o.graph_family.empty()||o.effect.empty()||o.local_fragment.empty()||o.profile_variants.empty())return fail("incomplete operation "+o.id);if(!ids.insert(o.id).second)return fail("duplicate operation "+o.id);for(const auto&i:o.input_atoms)if(!produced.count(i))return fail("unavailable atom "+i);for(const auto&x:o.output_atoms)if(!produced.insert(x).second)return fail("duplicate atom "+x);}return true;}
}
