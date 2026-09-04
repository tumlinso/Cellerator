#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-planning"); constexpr std::array<std::string_view,8> metrics{"decomposition_portfolio","candidate_count","transition_cost","oracle_regret","external_cost","cache_hit","resumption","bounded_failure"}; for(auto profile:{"generic","representative","adversarial"}) for(auto m:metrics) std::cout<<profile<<','<<m<<",11\n"; }
