#include <Cellerator/compiler/diagnostics/expose_structured_diagnostic_and_query_apis_v1.hh>
#include <atomic>
#include <cassert>
#include <thread>
using namespace cellerator::compiler::diagnostics::v1;bool collect(const structured_diagnostic&,void*p){++*static_cast<unsigned*>(p);return true;}bool cancelled(void*p){return static_cast<std::atomic<bool>*>(p)->load();}int main(){std::vector<structured_diagnostic> ds{{1,0,2,3,4,1,"CE001","bad \"axis\""},{2,1,2,4,1,2,"CE002","related"}};assert(to_lsp_json(ds[0]).find("bad \\\"axis\\\"")!=std::string::npos);unsigned n=0;std::atomic<bool> stop{false};std::thread q([&]{assert(stream_diagnostics(ds,collect,&n,cancelled,&stop)==2);});q.join();assert(n==2);stop=true;assert(stream_diagnostics(ds,collect,&n,cancelled,&stop)==0);}
