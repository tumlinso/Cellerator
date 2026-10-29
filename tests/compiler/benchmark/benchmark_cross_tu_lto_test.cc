#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,3> paths{"no_lto","conventional_materialization","ceir_lto"}; constexpr std::array<std::string_view,9> metrics{"object_ceir_bytes","extract","merge","incremental_cache","replan","reemit","link_time","binary_bytes","runtime"}; static_assert(paths.size()*metrics.size()==27); assert(metrics.front()=="object_ceir_bytes"); }
