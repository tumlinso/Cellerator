#pragma once
#include <cstdint>
#include <span>

namespace cellerator::compute::native_numeric {
// Initial smooth vocabulary. No broadcasting, reduction, scheduling or allocation.
enum class local_operation : std::uint8_t { add, multiply, tanh };
enum class local_status : std::uint8_t {
    success, unsupported_operation, invalid_binding, unsupported_policy
};
constexpr unsigned local_arity(local_operation op) noexcept {
    return op == local_operation::tanh ? 1 :
        (op == local_operation::add || op == local_operation::multiply ? 2 : 0);
}
namespace detail {
// Internal batch/derivative seam. Caller already checked FE_TONEAREST once.
// Keeps scalar primal arithmetic in one owner without a per-element mode query.
local_status local_value_nearest(local_operation, float, float, float&) noexcept;
local_status local_value_nearest(local_operation, double, double, double&) noexcept;
}
// Sole scalar primal implementation, reusable by derivative actions and N04.
local_status local_value(local_operation, float left, float right, float& output) noexcept;
local_status local_value(local_operation, double left, double right, double& output) noexcept;
// Caller owns synchronous host spans. Equal extents, no output/input overlap.
// Unary right must be empty; repeated binary inputs are legal. IEEE nonfinites
// propagate; no reassociation, storage quantization or implicit saturation.
local_status local_forward(local_operation, std::span<const float> left,
    std::span<const float> right, std::span<float> output) noexcept;
local_status local_forward(local_operation, std::span<const double> left,
    std::span<const double> right, std::span<double> output) noexcept;
} // namespace cellerator::compute::native_numeric
