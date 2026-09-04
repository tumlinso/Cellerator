#include <Cellerator/sdk/define_abi_version_and_feature_queries_v1.hh>
#include <cassert>
namespace api=cellerator::compiler::api::v1;
int main(){auto f=api::query_compiler_features_v1();assert(f.ceir_levels.size()==4&&api::has_feature_v1(f,"reflection"));auto ok=api::check_provider_compatibility_v1({"ok",{1,4,0},{}});assert(ok.compatible);auto bad=api::check_provider_compatibility_v1({"future",{7,0,0},{}});assert(!bad.compatible&&bad.diagnostic.find("future")!=std::string::npos&&bad.diagnostic.find("7")!=std::string::npos);}
