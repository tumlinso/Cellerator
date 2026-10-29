#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,7> phases{"preprocess","activated_tokens","shadow_generation","clang_sema","ast_construction","incremental_reuse","source_map_memory"}; static_assert(phases.size()==7); for(auto p:phases) assert(!p.empty()); }
