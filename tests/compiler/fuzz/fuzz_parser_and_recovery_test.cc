#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
namespace {struct result{std::vector<std::size_t>errors;};result parse(const std::string&s){result r;std::vector<char>stack;for(std::size_t i=0;i<s.size();++i){char c=s[i];if(c=='('||c=='{'||c=='[')stack.push_back(c);else if(c==')'||c=='}'||c==']'){char want=c==')'?'(':c=='}'?'{':'[';if(stack.empty()||stack.back()!=want){if(r.errors.size()<64)r.errors.push_back(i);}else stack.pop_back();}}while(!stack.empty()&&r.errors.size()<64){r.errors.push_back(s.size());stack.pop_back();}return r;}std::uint64_t next(std::uint64_t&x){x=x*6364136223846793005ULL+1;return x;}}
int main(){std::vector<std::string>seeds={"domain gene;","<[ a -[r]-> b; ]>","require ce::deterministic;","ir_of<semantic>(f)","reflect(field)","pass transform {}","native { asm(); }"};std::uint64_t state=7;for(int n=0;n<20000;++n){std::string s=seeds[next(state)%seeds.size()];for(int m=0;m<8;++m){auto p=next(state)%(s.size()+1);s.insert(p,1,"(){}[]<>;"[next(state)%9]);}auto a=parse(s),b=parse(s);assert(a.errors==b.errors&&a.errors.size()<=64);}auto minimized=parse("]");assert(minimized.errors.size()==1&&minimized.errors[0]==0);}
