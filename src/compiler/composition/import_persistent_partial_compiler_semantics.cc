#include <Cellerator/compiler/composition/import_persistent_partial_compiler_semantics_v1.hh>
namespace Cellerator::compiler::composition {
partial_decision_v1 evaluate_persistent_partial_v1(const persistent_partial_semantics_v1&p,std::uint64_t epoch,std::uint64_t generation){if(p.id.empty()||p.coverage.empty()||p.merge_algebra.empty()||p.finalize_algebra.empty()||p.numerical_contract.empty()||p.dependencies.empty())return {false,false,"incomplete compiler semantics"};if(p.structure_epoch!=epoch||p.value_generation!=generation)return {false,false,"stale structure or values"};const bool wins=p.reuse_savings*p.expected_reuse>p.build_cost;return {true,wins,wins?"amortized":"legal but not amortized"};}
}
