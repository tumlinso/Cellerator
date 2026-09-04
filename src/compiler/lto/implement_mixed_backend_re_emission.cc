#include <Cellerator/compiler/lto/implement_mixed_backend_re_emission_v1.hh>
namespace cellerator::compiler::lto::v1 {
std::vector<backend_emission_v1> plan_mixed_backend_re_emission_v1(const std::vector<backend_artifact_v1>&xs){std::vector<backend_emission_v1>r;r.reserve(xs.size());for(const auto&x:xs){const bool owned=x.backend!=emission_backend_v1::conventional;const bool emit=owned&&(x.program_region_changed||!x.artifact_valid);r.push_back({x.input,x.output,x.backend,emit?emission_action_v1::reemit:emission_action_v1::retain});}return r;}
}
