#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,6> identities{"source","toolchain","hardware","profile","input","benchmark_binary"}; constexpr bool reject_kernel_only=true, reject_unexplained_regression=true, retain_negative=true; static_assert(identities.size()==6 && reject_kernel_only && reject_unexplained_regression && retain_negative); for(auto x:identities) assert(!x.empty()); }
