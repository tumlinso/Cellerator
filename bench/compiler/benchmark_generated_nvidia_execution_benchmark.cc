#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-nvidia-contract"); constexpr std::array<std::string_view,4> paths{"generated","prelinked_provider","vendor_fallback","direct_ptx"}; constexpr std::array<std::string_view,10> phases{"prepare","allocate","pack","transfer","synchronize","launches","kernel","canonicalize","graph_replay","amortized_reuse"}; for(auto p:paths) for(auto x:phases) std::cout<<p<<','<<x<<",sm70,requires_accelerator_lease\n"; }
