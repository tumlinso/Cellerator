#include <Cellerator/sdk/expose_profile_build_load_query_apis_v1.hh>
#include <cassert>
namespace ca=cellerator::compiler::api::v1; namespace{double scale(double v,void*p)noexcept{return v**static_cast<double*>(p);}}
int main(){ca::profile_entry_v1 e[]={{"active",1},{"stress",2}};auto p=ca::build_profile_v1(e,2);auto q=ca::load_profile_text_v1("active 1 stress 3");assert(*ca::find_profile_state_v1(p,"stress")==2);assert(ca::diff_profiles_v1(p,q).size()==1);double x=2;ca::transfer_profile_v1(p,scale,&x);ca::bind_profile_environment_v1(p,"pbmc");assert(*ca::find_profile_state_v1(p,"stress")==4&&p.environment=="pbmc");const char text[]="active 1";assert(*ca::find_profile_state_v1(ca::load_profile_binary_v1((const std::uint8_t*)text,8),"active")==1);}
