#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
int main(){ cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-discovery"); constexpr std::array<std::string_view,6> mechanisms{"support_sketch","signature","atom_plane","fragment","grammar_basis","exact_scan"}; for(auto m:mechanisms) for(auto baseline:{"matched_generic","matched_null","no_basis"}) std::cout<<m<<','<<baseline<<",candidates,exact_rescans,certified,peak_bytes,disposition\n"; }
