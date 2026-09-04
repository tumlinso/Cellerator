#include <Cellerator/compiler/tooling/benchmark_baseline_editor_latency_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::tooling;
int main(){std::vector<editor_measurement_v1> m;for(auto l:{"c++","cell"})for(auto t:{"cold","warm"})for(auto s:{"small","large"})for(auto o:{"startup","diagnostics","edit","completion","hover","navigation","memory","proxy"})m.push_back({o,l,t,s,o==std::string("startup")?90.0:9.0});assert(meets_editor_budgets_v1(m,{{"startup",100},{"diagnostics",10},{"edit",10},{"completion",10},{"hover",10},{"navigation",10},{"memory",10},{"proxy",10}}));assert(!meets_editor_budgets_v1(m,{{"startup",80}}));}
