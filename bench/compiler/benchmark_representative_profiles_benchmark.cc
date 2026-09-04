#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-profiles"); constexpr std::array<std::string_view,9> phases{"build","exact_scan","sketch","load","mapped_query","state_transfer","branch_join","multi_state","peak_memory"}; for(auto p:phases) std::cout<<p<<",exact_small,estimated,absolute_error,11\n"; }
