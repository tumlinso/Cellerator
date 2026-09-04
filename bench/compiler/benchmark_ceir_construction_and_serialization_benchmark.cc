#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-ceir"); constexpr std::array<std::string_view,4> levels{"semantic","planning","realization","executable"}; constexpr std::array<std::string_view,9> phases{"construct","canonicalize","text_parse","text_print","binary_load","binary_store","map","unknown_extension","strip_provenance"}; for(auto l:levels) for(auto p:phases) std::cout<<l<<','<<p<<",nodes_per_second,bytes_per_node\n"; }
