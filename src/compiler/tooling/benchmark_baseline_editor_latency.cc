#include <Cellerator/compiler/tooling/benchmark_baseline_editor_latency_v1.hh>
#include <algorithm>
namespace Cellerator::compiler::tooling {
double percentile95_v1(std::vector<double>s){if(s.empty())return 0;std::sort(s.begin(),s.end());return s[(s.size()*95+99)/100-1];}
bool meets_editor_budgets_v1(const std::vector<editor_measurement_v1>&m,const std::vector<editor_budget_v1>&b){for(const auto&x:b){std::vector<double>s;for(const auto&v:m)if(v.operation==x.operation)s.push_back(v.milliseconds);if(s.empty()||percentile95_v1(s)>x.p95_ms)return false;}return true;}
}
