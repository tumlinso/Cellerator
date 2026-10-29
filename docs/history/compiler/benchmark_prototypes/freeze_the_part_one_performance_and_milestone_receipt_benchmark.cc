#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-final-receipt"); constexpr std::array<std::string_view,12> campaigns{"plain_cxx","frontend","ceir","profiles","discovery","planning","realization","cpu_execution","nvidia_execution","lto","sdk_daemon","milestones"}; for(auto c:campaigns) std::cout<<c<<",complete_cost,identity,decision,regression_budget\n"; }
