#include <cassert>
#include <set>
#include <string>
#include <vector>
enum class object_kind{embedded,sidecar,archive,plain};struct object{object_kind kind;std::string symbol,profile;bool hash=true,section=true;};
static std::string validate(const std::vector<object>&xs){std::set<std::string>s;std::string profile;for(const auto&x:xs){if(x.kind==object_kind::plain)continue;if(!x.hash)return "CE-LTO-HASH";if(!x.section&&x.kind!=object_kind::sidecar)return "CE-LTO-SECTION";if(!s.insert(x.symbol).second)return "CE-LTO-DUPLICATE";if(!profile.empty()&&x.profile!=profile)return "CE-LTO-PROFILE";profile=x.profile;}return "OK";}
int main(){std::vector<object>good={{object_kind::embedded,"a","p"},{object_kind::sidecar,"b","p"},{object_kind::archive,"c","p"},{object_kind::plain,"main",""}};assert(validate(good)=="OK");for(auto expected:{"CE-LTO-HASH","CE-LTO-SECTION","CE-LTO-DUPLICATE","CE-LTO-PROFILE"}){auto x=good;if(std::string(expected)=="CE-LTO-HASH")x[0].hash=false;else if(std::string(expected)=="CE-LTO-SECTION")x[0].section=false;else if(std::string(expected)=="CE-LTO-DUPLICATE")x[1].symbol="a";else x[1].profile="q";assert(validate(x)==expected);assert(validate(x)==expected);}assert(validate({good.back()})=="OK");}
