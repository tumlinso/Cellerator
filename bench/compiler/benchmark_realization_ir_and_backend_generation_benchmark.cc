#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-realization"); constexpr std::array<std::string_view,4> backends{"cpu","nvcc","clang_cuda","direct_ptx"}; constexpr std::array<std::string_view,9> metrics{"realization","projection_plan","packing_plan","stage_build","source_bytes","compiler_time","ptxas_resources","artifact_bytes","provenance"}; for(auto b:backends) for(auto m:metrics) std::cout<<b<<','<<m<<",available_or_explicitly_unavailable\n"; }
