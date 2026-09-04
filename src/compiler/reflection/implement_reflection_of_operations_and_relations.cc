#include <Cellerator/compiler/reflection/implement_reflection_of_operations_and_relations_v1.hh>
#include <sstream>
namespace cellerator::compiler::reflection::v1 {
bool validate_reflected_operation_v1(const reflected_operation_v1&o)noexcept{if(o.normalized_kind.empty()||o.handle.kind!=handle_kind_v1::operation||o.operands.empty()||o.results.empty()||o.relation.handle.kind!=handle_kind_v1::relation||!o.relation.source_domain||!o.relation.target_domain||!o.relation.order||!o.relation.structure_epoch)return false;for(const auto&v:o.operands)if(v.type.empty()||!v.domain||!v.order||!v.generation)return false;return o.provenance.kind==handle_kind_v1::provenance;}
std::string dump_reflected_operation_v1(const reflected_operation_v1&o){std::ostringstream s;s<<o.normalized_kind<<" operands="<<o.operands.size()<<" results="<<o.results.size()<<" relation="<<o.relation.source_domain<<":"<<o.relation.target_domain<<" order="<<o.relation.order<<" generation="<<o.relation.value_generation<<" effects="<<static_cast<unsigned>(o.effects);return s.str();}
}
