#include <Cellerator/compiler/diagnostics/implement_planning_barrier_explanations_v1.hh>
#include <array>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;constexpr std::array bs{planning_barrier::opaque_cxx_call,planning_barrier::field_boundary,planning_barrier::unknown_extension,planning_barrier::effect,planning_barrier::alias_uncertainty,planning_barrier::profile_widening,planning_barrier::hard_constraint};for(auto b:bs){auto r=explain_planning_barrier(b,{7,10,20});assert(r.valid&&!r.explanation.empty()&&r.range.begin==10);}assert(!explain_planning_barrier(planning_barrier::effect,{0,1,2}).valid);}
