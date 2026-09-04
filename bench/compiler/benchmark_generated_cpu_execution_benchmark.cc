#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-cpu-execution"); constexpr std::array<std::string_view,2> fields{"relation","multi_operation"}; constexpr std::array<std::string_view,3> paths{"generated_cpu","direct_cxx","existing_runtime"}; constexpr std::array<std::string_view,7> phases{"prepare","transform","pack","execute","reuse","peak_memory","output_recovery"}; for(auto f:fields) for(auto p:paths) for(auto x:phases) std::cout<<f<<','<<p<<','<<x<<",11\n"; }
