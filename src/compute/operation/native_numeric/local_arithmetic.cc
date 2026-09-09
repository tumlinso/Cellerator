#include <Cellerator/compute/operation/native_numeric/local_arithmetic.hh>
#include <cmath>
#include <cfenv>
#include <limits>

namespace cellerator::compute::native_numeric {
namespace {
template<class T> local_status value(local_operation op, T a, T b, T& out) noexcept {
    switch (op) {
    case local_operation::add: out = a + b; break;
    case local_operation::multiply: out = a * b; break;
    case local_operation::tanh: out = std::tanh(a); break;
    default: return local_status::unsupported_operation;
    }
    return local_status::success;
}
template<class T> bool overlaps(std::span<const T> a, std::span<T> b) noexcept {
    if (a.empty() || b.empty()) return false;
    auto x = reinterpret_cast<std::uintptr_t>(a.data());
    auto y = reinterpret_cast<std::uintptr_t>(b.data());
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    if (a.size() > (max - x) / sizeof(T) || b.size() > (max - y) / sizeof(T)) return true;
    return x < y + b.size_bytes() && y < x + a.size_bytes();
}
template<class T> local_status forward(local_operation op, std::span<const T> a,
        std::span<const T> b, std::span<T> out) noexcept {
    if (std::fegetround() != FE_TONEAREST) return local_status::unsupported_policy;
    const auto arity = local_arity(op);
    if (!arity) return local_status::unsupported_operation;
    if (a.size() != out.size() || (arity == 1 ? !b.empty() : b.size() != a.size()) ||
        (!a.empty() && (!a.data() || !out.data() || (arity == 2 && !b.data()))) ||
        overlaps(a, out) || overlaps(b, out)) return local_status::invalid_binding;
    for (std::size_t i = 0; i < out.size(); ++i) value(op, a[i], arity == 2 ? b[i] : T{}, out[i]);
    return local_status::success;
}
}
local_status detail::local_value_nearest(local_operation op, float a, float b, float& out) noexcept { return value(op,a,b,out); }
local_status detail::local_value_nearest(local_operation op, double a, double b, double& out) noexcept { return value(op,a,b,out); }
local_status local_value(local_operation op, float a, float b, float& out) noexcept {
    if (std::fegetround() != FE_TONEAREST) return local_status::unsupported_policy;
    return value(op,a,b,out);
}
local_status local_value(local_operation op, double a, double b, double& out) noexcept {
    if (std::fegetround() != FE_TONEAREST) return local_status::unsupported_policy;
    return value(op,a,b,out);
}
local_status local_forward(local_operation op, std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept { return forward(op,a,b,out); }
local_status local_forward(local_operation op, std::span<const double> a, std::span<const double> b, std::span<double> out) noexcept { return forward(op,a,b,out); }
} // namespace cellerator::compute::native_numeric
