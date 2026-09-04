#include <Cellerator/compiler/api/define_abi_version_and_feature_queries_v1.hh>
#include <algorithm>
#include <sstream>
namespace cellerator::compiler::api::v1 {
compiler_features_v1 query_compiler_features_v1(){return {{compiler_api_abi_v1,0,0},{1,2,3,4},{"profiles","reflection","external-backends"}};}
bool has_feature_v1(const compiler_features_v1& f,std::string_view n)noexcept{return std::find(f.optional_features.begin(),f.optional_features.end(),n)!=f.optional_features.end();}
compatibility_result_v1 check_provider_compatibility_v1(const provider_descriptor_v1& p){if(p.api.major==compiler_api_abi_v1)return {true,"compatible"};std::ostringstream s;s<<"provider '"<<p.name<<"' requires compiler API ABI "<<p.api.major<<", but libCellerator provides "<<compiler_api_abi_v1;return {false,s.str()};}
}
