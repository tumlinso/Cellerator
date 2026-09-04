#include <Cellerator/compiler/api/expose_profile_build_load_query_apis_v1.hh>
#include <sstream>
namespace cellerator::compiler::api::v1 {
profile_v1 build_profile_v1(const profile_entry_v1* e,std::size_t n){profile_v1 p;for(std::size_t i=0;e&&i<n;++i)if(e[i].name)p.states[e[i].name]=e[i].value;return p;}
profile_v1 load_profile_text_v1(std::string_view t){profile_v1 p;std::istringstream in{std::string(t)};for(std::string k;in>>k;){double v;if(!(in>>v))break;p.states[k]=v;}return p;}
profile_v1 load_profile_binary_v1(const std::uint8_t* b,std::size_t n){return b?load_profile_text_v1({reinterpret_cast<const char*>(b),n}):profile_v1{};}
const double* find_profile_state_v1(const profile_v1& p,std::string_view n) noexcept{auto i=p.states.find(std::string(n));return i==p.states.end()?nullptr:&i->second;}
std::vector<std::string> diff_profiles_v1(const profile_v1&a,const profile_v1&b){std::vector<std::string> d;for(auto&[k,v]:a.states){auto i=b.states.find(k);if(i==b.states.end()||i->second!=v)d.push_back(k);}for(auto&[k,v]:b.states)if(!a.states.count(k))d.push_back(k);return d;}
void transfer_profile_v1(profile_v1&p,profile_transfer_v1 f,void*d){if(f)for(auto&[k,v]:p.states)v=f(v,d);}
void bind_profile_environment_v1(profile_v1&p,std::string e){p.environment=std::move(e);}
}
