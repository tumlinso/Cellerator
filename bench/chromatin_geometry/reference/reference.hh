#pragma once

#include "../contract.hh"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cgs::reference {

struct ProductForward final {
    std::vector<float> y;
};

struct ProductBackward final {
    std::vector<float> dx;
    std::vector<float> dtheta;
    std::vector<float> dq;
    std::vector<float> dscore;
};

struct LinearForward final {
    std::vector<float> y;
    // [batch, group_weight_outputs], after the declared local F rounding.
    std::vector<float> group_f;
};

struct LinearBackward final {
    std::vector<float> dx;
    std::vector<float> dweights;
    std::vector<float> dq;
    std::vector<float> dscore;
    // Decoded local storage values after the declared dtype rounding.
    std::vector<float> group_df;
    std::vector<float> group_dx;
    std::vector<float> group_dw;
};

struct LinearValidationBounds final {
    // Absolute bounds aligned with the corresponding FP32-decoded reference
    // values or local stored values. They combine per-dot error bounds from
    // the existing GroupedLinear smoke with FP32 overlap-reduction bounds.
    std::vector<float> y;
    std::vector<float> dx;
    std::vector<float> dweights;
    std::vector<float> dq;
    std::vector<float> dscore;
    std::vector<float> group_dx;
    std::vector<float> group_dw;
};

// Scalar host oracles. Every pointer in the views must refer to host memory.
// The product oracle follows each ordered argument slot independently, so
// repeated coordinates and zero-valued participants retain their VJPs.
// Supplied and hardened-teacher gates stop gate gradients: dq/dscore remain
// zero while numerical mechanism gradients are computed over primal members.
// STE alone computes dq/dscore over the complete declared derivative set.
ProductForward product_forward(const ProblemView&, const GateView&);
ProductBackward product_backward(const ProblemView&, const GateView&,
                                 const float* dy);

// The linear oracle rounds local F, dF, dX and dW to ProblemView::dtype at
// the same boundaries as GroupedLinear, then reduces overlapping results in
// FP32. group_f/group_df/group_dx/group_dw are decoded stored values.
LinearForward grouped_linear_forward(const ProblemView&, const GateView&);
LinearBackward grouped_linear_backward(const ProblemView&, const GateView&,
                                       const float* dy,
                                       const LinearForward& forward);
LinearValidationBounds grouped_linear_validation_bounds(
    const ProblemView&, const GateView&, const float* dy,
    const LinearForward&, const LinearBackward&);

bool within_absolute_bound(float actual, float expected,
                           float absolute_bound) noexcept;

float read_value(const void* values, DType dtype, std::uint64_t index);
float round_value(float value, DType dtype) noexcept;
std::uint16_t float_to_half_bits(float value) noexcept;
float half_bits_to_float(std::uint16_t bits) noexcept;

inline constexpr float product_abs_tolerance = 2.0e-4f;

// These constants mirror the compound bound in tests/framework/cuda/
// grouped_linear_smoke.cu. The bound is evaluated per dot output from its
// sum of absolute products and dot length, then includes one FP16 ULP when
// local storage is FP16.
float grouped_linear_allowed_error(DType dtype, float expected,
                                   std::size_t dot_length,
                                   double sum_abs_products) noexcept;

} // namespace cgs::reference
