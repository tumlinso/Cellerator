#include <Cellerator/compiler/diagnostics/implement_crash_and_timeout_diagnostics_v1.hh>
#include <array>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;for(auto k:std::array{failure_kind::crash,failure_kind::timeout})for(auto m:std::array{extension_mode::builtin,extension_mode::custom_in_process,extension_mode::custom_isolated}){auto d=diagnose_failure({k,failure_owner::pass,7,m,true});assert(d.valid&&d.preserve_temporaries&&d.owner==failure_owner::pass);assert(d.isolated==(m==extension_mode::custom_isolated));}assert(!diagnose_failure({failure_kind::crash,failure_owner::backend,0}).valid);}
