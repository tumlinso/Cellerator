#include <Cellerator/compiler/composition/import_superatom_promotion_v1.hh>
namespace Cellerator::compiler::composition {
superatom_promotion_v1 evaluate_superatom_promotion_v1(const superatom_candidate_v1&c,std::string_view p){if(!c.experimental)return {false,"candidate must remain experimental"};if(c.profile!=p)return {false,"profile mismatch"};if(!c.exact||c.exact_derivation.empty()||c.deconstruction.empty())return {false,"exact reversible derivation required"};if(c.evidence.empty())return {false,"evidence required"};if(c.total_cost>=c.baseline_cost)return {false,"complete cost does not win"};return {true,"profile-specific complete-cost win"};}
}
