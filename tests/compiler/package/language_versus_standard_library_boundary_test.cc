#include <array>
#include <cassert>
#include <string_view>
int main(){constexpr std::array<std::string_view,14> language{"domain","axis","order","state","relation_transfer","operation","field","effect","exact_coverage","profile","planning_authority","identity_generation","ceir","native"};constexpr std::array<std::string_view,13> library{"owner","view","binder","builder","container","reorder","canonicalize","bundle","exchange","common_biology","algorithm","policy","helper"};for(auto x:language)for(auto y:library)assert(x!=y);}
