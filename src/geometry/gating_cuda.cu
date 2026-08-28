#include "Cellerator/geometry/gating_cuda.cuh"

#include <algorithm>
#include <utility>

namespace cellpack {
namespace {

const packed_region_desc *find_region(const static_plan &plan, u32 region_id) {
    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id == region_id) return &region;
    }
    return nullptr;
}

bool coordinate_less_by_region(const packed_coordinate &lhs, const packed_coordinate &rhs) {
    if (lhs.region_id != rhs.region_id) return lhs.region_id < rhs.region_id;
    if (lhs.permuted_row != rhs.permuted_row) return lhs.permuted_row < rhs.permuted_row;
    return lhs.permuted_feature < rhs.permuted_feature;
}

validation_result validate_coordinate_plan_source(
    const static_plan &plan,
    const packed_coordinate_plan &packed) {
    if (packed.row_count != plan.desc.row_count || packed.feature_count != plan.desc.feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "packed coordinate dimensions do not match plan");
    }
    validation_result desc_result = validate_plan_desc(plan.desc);
    if (!desc_result) return desc_result;
    validation_result region_result = validate_region_sequence(
        plan.regions.data(),
        static_cast<u32>(plan.regions.size()),
        plan.desc.row_count,
        plan.desc.feature_count);
    if (!region_result) return region_result;
    for (u32 i = 0; i < static_cast<u32>(packed.coordinates.size()); ++i) {
        const packed_coordinate &coordinate = packed.coordinates[i];
        const packed_region_desc *region = find_region(plan, coordinate.region_id);
        if (region == nullptr) {
            return validation_error(validation_code::missing_region, i, "packed coordinate references an unknown route region");
        }
        if (coordinate.original_row >= packed.row_count || coordinate.original_feature >= packed.feature_count) {
            return validation_error(validation_code::invalid_matrix_view, i, "packed coordinate original index is outside matrix bounds");
        }
        if (coordinate.permuted_row < region->row_begin
            || coordinate.permuted_row >= region->row_begin + region->row_count
            || coordinate.permuted_feature < region->feature_begin
            || coordinate.permuted_feature >= region->feature_begin + region->feature_count) {
            return validation_error(validation_code::invalid_region_bounds, i, "packed coordinate is outside its route region");
        }
    }
    return validation_ok();
}

__global__ void route_forward_kernel_(
    device_coordinate_plan_view plan,
    route_mask_view mask,
    const float *x,
    float *y) {
    const u32 active_index = static_cast<u32>(blockIdx.x);
    if (active_index >= mask.region_count) return;
    const u32 region_id = mask.region_ids[active_index];
    if (region_id >= plan.region_span_count) return;
    const region_coordinate_span span = plan.region_spans[region_id];
    if (span.region_id != region_id) return;
    for (u32 offset = static_cast<u32>(threadIdx.x);
         offset < span.coordinate_count;
         offset += static_cast<u32>(blockDim.x)) {
        const u32 coordinate = span.coordinate_begin + offset;
        atomicAdd(y + plan.row_ids[coordinate], plan.values[coordinate] * x[plan.feature_ids[coordinate]]);
    }
}

__global__ void route_backward_replay_kernel_(
    device_coordinate_plan_view plan,
    route_tape_view tape,
    const float *grad_y,
    float *grad_x) {
    const u32 active_index = static_cast<u32>(blockIdx.x);
    if (active_index >= tape.region_count) return;
    const u32 region_id = tape.region_ids[active_index];
    if (region_id >= plan.region_span_count) return;
    const region_coordinate_span span = plan.region_spans[region_id];
    if (span.region_id != region_id) return;
    for (u32 offset = static_cast<u32>(threadIdx.x);
         offset < span.coordinate_count;
         offset += static_cast<u32>(blockDim.x)) {
        const u32 coordinate = span.coordinate_begin + offset;
        atomicAdd(grad_x + plan.feature_ids[coordinate], plan.values[coordinate] * grad_y[plan.row_ids[coordinate]]);
    }
}

} // namespace

validation_result build_compiled_coordinate_plan(
    const static_plan &plan,
    const packed_coordinate_plan &packed,
    compiled_coordinate_plan *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "compiled coordinate output is null");
    }
    validation_result source_result = validate_coordinate_plan_source(plan, packed);
    if (!source_result) return source_result;

    std::vector<packed_coordinate> sorted = packed.coordinates;
    std::sort(sorted.begin(), sorted.end(), coordinate_less_by_region);

    compiled_coordinate_plan compiled;
    compiled.row_count = packed.row_count;
    compiled.feature_count = packed.feature_count;
    compiled.region_spans.resize(plan.regions.size());
    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id >= compiled.region_spans.size()) {
            return validation_error(validation_code::invalid_region_bounds, region.region_id, "region id is outside compiled span table");
        }
        compiled.region_spans[region.region_id].region_id = region.region_id;
    }
    compiled.row_ids.reserve(sorted.size());
    compiled.feature_ids.reserve(sorted.size());
    compiled.values.reserve(sorted.size());

    u32 coordinate_index = 0u;
    while (coordinate_index < static_cast<u32>(sorted.size())) {
        const u32 region_id = sorted[coordinate_index].region_id;
        if (region_id >= compiled.region_spans.size()) {
            return validation_error(validation_code::missing_region, coordinate_index, "coordinate region id is outside compiled span table");
        }
        region_coordinate_span &span = compiled.region_spans[region_id];
        span.coordinate_begin = static_cast<u32>(compiled.values.size());
        while (coordinate_index < static_cast<u32>(sorted.size())
               && sorted[coordinate_index].region_id == region_id) {
            const packed_coordinate &coordinate = sorted[coordinate_index];
            compiled.row_ids.push_back(coordinate.original_row);
            compiled.feature_ids.push_back(coordinate.original_feature);
            compiled.values.push_back(coordinate.value);
            ++coordinate_index;
        }
        span.coordinate_count = static_cast<u32>(compiled.values.size()) - span.coordinate_begin;
    }

    *out = std::move(compiled);
    return validation_ok();
}

cudaError_t launch_route_forward(
    device_coordinate_plan_view plan,
    route_mask_view mask,
    const float *x,
    float *y,
    cudaStream_t stream) {
    if (mask.region_count == 0u) return cudaSuccess;
    route_forward_kernel_<<<mask.region_count, 128, 0, stream>>>(plan, mask, x, y);
    return cudaGetLastError();
}

cudaError_t launch_route_backward_replay(
    device_coordinate_plan_view plan,
    route_tape_view tape,
    const float *grad_y,
    float *grad_x,
    cudaStream_t stream) {
    if (tape.region_count == 0u) return cudaSuccess;
    route_backward_replay_kernel_<<<tape.region_count, 128, 0, stream>>>(plan, tape, grad_y, grad_x);
    return cudaGetLastError();
}

} // namespace cellpack
