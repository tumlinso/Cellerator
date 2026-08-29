#pragma once

#include "Cellerator/geometry/gating.hh"
#include "Cellerator/geometry/pack.hh"
#include "Cellerator/memory/view.hh"

#include <cuda_runtime_api.h>

#include <vector>

namespace cellpack {

struct alignas(16) region_coordinate_span {
    u32 region_id = invalid_id;
    u32 coordinate_begin = 0u;
    u32 coordinate_count = 0u;
    u32 reserved0 = 0u;
};

struct compiled_coordinate_plan {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    std::vector<region_coordinate_span> region_spans;
    std::vector<u32> row_ids;
    std::vector<u32> feature_ids;
    std::vector<float> values;
};

struct device_coordinate_plan_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 region_span_count = 0u;
    u32 coordinate_count = 0u;
    const region_coordinate_span *region_spans = nullptr;
    const u32 *row_ids = nullptr;
    const u32 *feature_ids = nullptr;
    const float *values = nullptr;
};

struct compiled_coordinate_plan_storage {
    ::cellerator::memory::array_view<region_coordinate_span> region_spans;
    ::cellerator::memory::array_view<u32> row_ids;
    ::cellerator::memory::array_view<u32> feature_ids;
    ::cellerator::memory::array_view<float> values;
};

validation_result build_compiled_coordinate_plan_into(
    const static_plan &plan,
    packed_coordinate_plan_view packed,
    compiled_coordinate_plan_storage storage,
    device_coordinate_plan_view *out);

validation_result build_compiled_coordinate_plan(
    const static_plan &plan,
    const packed_coordinate_plan &packed,
    compiled_coordinate_plan *out);

cudaError_t launch_route_forward(
    device_coordinate_plan_view plan,
    route_mask_view mask,
    const float *x,
    float *y,
    cudaStream_t stream = nullptr);

cudaError_t launch_route_backward_replay(
    device_coordinate_plan_view plan,
    route_tape_view tape,
    const float *grad_y,
    float *grad_x,
    cudaStream_t stream = nullptr);

} // namespace cellpack
