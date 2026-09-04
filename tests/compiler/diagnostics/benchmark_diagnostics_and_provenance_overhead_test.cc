#include <Cellerator/compiler/diagnostics/benchmark_diagnostics_and_provenance_overhead_v1.hh>
#include <cassert>
#include <array>
int main(){using namespace cellerator::compiler::diagnostics::v1;std::vector<provenance_measurement> m;for(auto s:std::array{translation_unit_size::small,translation_unit_size::large}){m.push_back({provenance_level::disabled,s,100,1000,1000,64});m.push_back({provenance_level::minimal,s,105,1050,1080,64});m.push_back({provenance_level::full,s,120,1150,1250,64});}assert(provenance_overhead_within_budget(m,{25,20,30}));m.back().hot_runtime_bytes=65;assert(!provenance_overhead_within_budget(m,{25,20,30}));}
