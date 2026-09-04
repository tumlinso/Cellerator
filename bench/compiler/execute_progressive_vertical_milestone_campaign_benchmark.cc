#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-milestones"); constexpr std::array<std::string_view,11> names{"driver_passthrough","pragma_parse","semantic_ir","profile_compile","planning_candidate","cpu_object","nvcc_object","inline_rewrite","custom_pass","cross_tu_import","celleratord_hover"}; for(auto name:names) std::cout<<name<<",exact_command,content_hashed_artifact,reproducible_or_unavailable\n"; }
