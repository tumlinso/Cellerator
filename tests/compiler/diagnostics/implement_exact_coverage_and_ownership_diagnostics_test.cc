#include <Cellerator/compiler/diagnostics/implement_exact_coverage_and_ownership_diagnostics_v1.hh>
#include <array>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;constexpr std::array failures{coverage_failure::omission,coverage_failure::duplicate,coverage_failure::wrong_role,coverage_failure::incompatible_partial_algebra,coverage_failure::halo_as_contributor,coverage_failure::canonical_recovery_failure};for(auto f:failures){auto d=diagnose_exact_coverage({f,17,2,3});assert(!d.valid&&d.member==17&&!d.explanation.empty());}assert(diagnose_exact_coverage({}).valid);assert(diagnose_exact_coverage({coverage_failure::omission,0}).explanation=="failure lacks member-level evidence");}
