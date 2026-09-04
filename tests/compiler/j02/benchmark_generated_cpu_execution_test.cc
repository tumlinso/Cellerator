#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,2> fields{"relation","multi_operation"}; constexpr std::array<std::string_view,3> paths{"generated_cpu","direct_cxx","existing_runtime"}; constexpr std::array<std::string_view,7> phases{"prepare","transform","pack","execute","reuse","peak_memory","output_recovery"}; static_assert(fields.size()*paths.size()*phases.size()==42); assert(phases.back()=="output_recovery"); }
