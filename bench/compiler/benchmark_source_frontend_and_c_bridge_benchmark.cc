#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){
 cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-frontend");
 constexpr std::array<std::string_view,7> phases{"preprocess","activated_tokens","shadow_generation","clang_sema","ast_construction","incremental_reuse","source_map_memory"};
 for(auto mode:{"pure_cxx","cellerator_field"}) for(auto phase:phases) std::cout<<mode<<','<<phase<<",11\n";
}
