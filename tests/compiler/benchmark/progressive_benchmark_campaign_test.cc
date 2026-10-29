#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,11> names{"driver_passthrough","pragma_parse","semantic_ir","profile_compile","planning_candidate","cpu_object","nvcc_object","inline_rewrite","custom_pass","cross_tu_import","celleratord_hover"}; static_assert(names.size()==11); for(auto n:names) assert(!n.empty()); }
