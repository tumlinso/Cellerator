#include "learning/ste.hh"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
std::uint32_t bits(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}
}

// This executable deliberately uses only no-launch paths and therefore needs
// no GPU context. GPU numerical tests run separately in an admitted job.
int main() {
    cgs::GateRevisions r{};
    r.temperature_bits = bits(1.0f);
    r.threshold_bits = bits(0.5f);
    cgs::SteSnapshot s;
    auto success = cgs::ste_forward(0, 9, 1.0f, 0.5f, r, {}, &s, nullptr);
    bool ok = success == cudaSuccess && s.valid;
    ok = ok && cgs::ste_backward(s, r, nullptr, nullptr, nullptr, nullptr, nullptr) == cudaSuccess;
    auto stale = r;
    ++stale.accessibility_generation;
    ok = ok && cgs::ste_backward(s, stale, nullptr, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue;
    stale = r;
    stale.threshold_bits ^= 1;
    ok = ok && cgs::ste_backward(s, stale, nullptr, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue;
    ok = ok && cgs::ste_forward(1, 1, 0.0f, 0.5f, r, {}, &s, nullptr) == cudaErrorInvalidValue && !s.valid;
    ok = ok && cgs::ste_forward(1, 1, 1.0f, -0.1f, r, {}, &s, nullptr) == cudaErrorInvalidValue;
    ok = ok && cgs::ste_forward(1, 1, 1.0f, 0.5f, r, {}, &s, nullptr) == cudaErrorInvalidValue;
    ok = ok && cgs::ste_forward(std::numeric_limits<std::uint64_t>::max(), 9, 1.0f, 0.5f,
                                r, {}, &s, nullptr) == cudaErrorInvalidValue;
    ok = ok && cgs::ste_reduce_per_condition(0, 0, nullptr, nullptr, nullptr, nullptr) == cudaSuccess;
    ok = ok && cgs::ste_sgd_update(0, nullptr, nullptr, -1.0f, &s, nullptr) == cudaErrorInvalidValue;
    if (!ok) std::cerr << "STE host validation failed\n";
    else std::cout << "STE host validation passed\n";
    return ok ? 0 : 1;
}
