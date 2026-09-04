#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
struct artifact{std::uint8_t version,level;std::uint32_t identity,region,definition,use,native_size,directory,hash;std::string extension;};
static bool valid(const artifact&a){return a.version==1&&a.level>=1&&a.level<=3&&a.identity&&a.region&&a.definition&&a.use==a.definition&&a.native_size<=4096&&a.directory<=64&&a.hash&& (a.extension.empty()||a.extension.rfind("x-",0)==0);}
static std::vector<std::uint8_t>emit(const artifact&a){return {a.version,a.level,std::uint8_t(a.identity),std::uint8_t(a.region),std::uint8_t(a.definition),std::uint8_t(a.use),std::uint8_t(a.native_size),std::uint8_t(a.directory),std::uint8_t(a.hash)};}
int main(){artifact base{1,1,1,1,2,2,8,3,9,"x-test"};for(int level=1;level<=3;++level){base.level=level;assert(valid(base));assert(emit(base)==emit(base));}for(int field=0;field<8;++field)for(int value=0;value<256;++value){auto x=base;switch(field){case 0:x.version=value;break;case 1:x.level=value;break;case 2:x.identity=value;break;case 3:x.region=value;break;case 4:x.definition=value;break;case 5:x.use=value;break;case 6:x.native_size=value*32;break;default:x.directory=value;break;}auto accepted=valid(x);if(accepted)assert(emit(x)==emit(x));}auto bad=base;bad.native_size=5000;assert(!valid(bad));bad=base;bad.extension="unknown";assert(!valid(bad));}
