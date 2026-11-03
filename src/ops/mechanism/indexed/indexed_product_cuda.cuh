#pragma once

#include <Cellerator/ops/mechanism/indexed/training.hh>

#include <cuda_runtime_api.h>

namespace cellerator::ops::mechanism::indexed::detail {

struct device_plan {
    std::uint64_t n, o, k, m, a;
    const std::uint64_t *arg_offsets, *args, *coefficient;
    const std::uint64_t *out_offsets, *out_mechanism;
    const float* out_scale;
    const std::uint64_t *input_offsets, *input_slots, *param_offsets, *param_mechanisms;
    const std::uint64_t *mechanism_out_offsets, *mechanism_outputs;
    const float* mechanism_scales;
};

struct execution_binding {
    device_plan plan;
    void* saved_input;
    float* products;
    float* argument_gradients;
    float* mechanism_gradients;
    const void* input;
    const void* coefficients;
    float* output;
    float* input_gradient;
    float* coefficient_gradient;
    std::uint64_t batch;
    bool input_half, mixed, backward;
};

execution::program::program_status launch_training(
    const void*, const execution::program::launch_binding_v2&, void* stream_ptr) noexcept;

} // namespace cellerator::ops::mechanism::indexed::detail
