#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
struct profile{std::uint32_t version=1,hash=1,evidence=1;};
static bool valid(profile p){return p.version==1&&p.hash&&p.evidence;}
using state=std::set<int>;
static state transfer(state s,int add){s.insert(add);return s;}
static state join(const state&a,const state&b){state r;std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::inserter(r,r.end()));return r;}
static state widen(state s,std::size_t limit){while(s.size()>limit)s.erase(std::prev(s.end()));return s;}
int main(){for(unsigned bits=0;bits<256;++bits){state a,b;for(int i=0;i<8;++i)((bits>>i)&1?a:b).insert(i);auto exact=join(a,b);assert(exact==join(b,a));assert(widen(transfer(exact,9),4).size()<=4);}assert(valid({}));assert(!valid({2,1,1})&&!valid({1,0,1})&&!valid({1,1,0}));std::optional<profile>missing;assert(!missing);}
