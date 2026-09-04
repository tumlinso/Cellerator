#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-sdk-daemon"); constexpr std::array<std::string_view,10> operations{"session_start","concurrent_parse","cancel","editor_start","diagnostics","completion","hover","ir_query","candidate_query","peak_memory"}; for(auto source:{"ordinary_cxx","cellerator"}) for(auto op:operations) std::cout<<source<<','<<op<<",median,p95,11\n"; }
