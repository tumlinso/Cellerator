#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,4> paths{"generated","prelinked_provider","vendor_fallback","direct_ptx"}; constexpr std::array<std::string_view,10> phases{"prepare","allocate","pack","transfer","synchronize","launches","kernel","canonicalize","graph_replay","amortized_reuse"}; constexpr bool lease_required=true, sm70=true; static_assert(paths.size()*phases.size()==40 && lease_required && sm70); assert(phases.back()=="amortized_reuse"); }
