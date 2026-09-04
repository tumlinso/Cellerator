#include <Cellerator/compiler/tooling/merge_ordinary_c_and_cellerator_diagnostics_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::tooling;
int main(){
 tooling_diagnostic_v1 cxx{tooling_diagnostic_phase_v1::clangd,tooling_diagnostic_severity_v1::error,{10,14},"E1","bad field",{{{10,14},"good"}},{"clang note"}};
 tooling_diagnostic_v1 cell{tooling_diagnostic_phase_v1::semantic,tooling_diagnostic_severity_v1::error,{0,4},"E1","bad field",{}, {"semantic note"}};
 tooling_diagnostic_v1 plan{tooling_diagnostic_phase_v1::planning,tooling_diagnostic_severity_v1::warning,{20,21},"P1","slow",{}, {}};
 auto merged=merge_diagnostics_v1({cxx},{cell,plan},[](auto s)->std::optional<tooling_diagnostic_span_v1>{return tooling_diagnostic_span_v1{s.begin-10,s.end-10};});
 assert(merged.size()==2); assert(merged[0].phase==tooling_diagnostic_phase_v1::clangd);
 assert(merged[0].severity==tooling_diagnostic_severity_v1::error); assert(merged[0].fixes[0].range.begin==0);
 assert(merged[0].related.size()==2); assert(merged[1].phase==tooling_diagnostic_phase_v1::planning);
}
