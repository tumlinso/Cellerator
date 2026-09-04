#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
struct scan_v1 { bool activated=false; std::size_t opens=0,closes=0; std::vector<std::size_t>map; };
scan_v1 scan(const std::string&s){scan_v1 r;bool block=false,line=false,string=false;for(std::size_t i=0;i<s.size();++i){if(line){if(s[i]=='\n')line=false;continue;}if(block){if(i+1<s.size()&&s[i]=='*'&&s[i+1]=='/'){block=false;++i;}continue;}if(string){if(s[i]=='\\')++i;else if(s[i]=='"')string=false;continue;}if(i+1<s.size()&&s[i]=='/'&&s[i+1]=='/'){line=true;++i;continue;}if(i+1<s.size()&&s[i]=='/'&&s[i+1]=='*'){block=true;++i;continue;}if(s[i]=='"'){string=true;continue;}const bool bol=i==0||s[i-1]=='\n';if(bol&&s.compare(i,25,"#pragma cellerator source")==0)r.activated=true;if(r.activated&&i+1<s.size()&&s[i]=='<'&&s[i+1]=='['){++r.opens;r.map.push_back(i);++i;}else if(r.activated&&i+1<s.size()&&s[i]==']'&&s[i+1]=='>'){++r.closes;r.map.push_back(i);++i;}}return r;}
std::uint64_t next(std::uint64_t&x){x^=x<<13;x^=x>>7;x^=x<<17;return x;}
}
int main(){std::uint64_t state=0xCE11E2A7u;const std::vector<std::string>parts={"template<class T> struct X {};\n","#include <vector>\n","#define FIELD <[ x ]>\n","// #pragma cellerator source\n","/* <[ ]> */\n","\"<[ not syntax ]>\"\n","#pragma cellerator source\n","<[ value ]>\n","// comment\n"};for(int trial=0;trial<10000;++trial){std::string source;for(int n=0;n<12;++n)source+=parts[next(state)%parts.size()];auto a=scan(source),b=scan(source);assert(a.activated==b.activated&&a.opens==b.opens&&a.closes==b.closes&&a.map==b.map);for(std::size_t i=1;i<a.map.size();++i)assert(a.map[i]>a.map[i-1]);if(source.find("#pragma cellerator source\n") == std::string::npos)assert(!a.activated);}auto leak=scan("// #pragma cellerator source\n<[x]>");assert(!leak.activated&&leak.opens==0);}
