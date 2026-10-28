#include <cstdint>

extern "C" void probe_add_f32(const float* lhs, const float* rhs,
                               float* out, std::uint64_t count) {
    for (std::uint64_t i=0;i<count;++i) out[i]=lhs[i]+rhs[i];
}
