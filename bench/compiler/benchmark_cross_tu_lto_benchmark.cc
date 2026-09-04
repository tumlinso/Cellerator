#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-lto"); constexpr std::array<std::string_view,3> paths{"no_lto","conventional_materialization","ceir_lto"}; constexpr std::array<std::string_view,9> metrics{"object_ceir_bytes","extract","merge","incremental_cache","replan","reemit","link_time","binary_bytes","runtime"}; for(auto p:paths) for(auto m:metrics) std::cout<<p<<','<<m<<",authorized_field_chain,11\n"; }
