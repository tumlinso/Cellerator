#include "Cellerator/geometry/gating_cuda.cuh"

#include <utility>

namespace cellpack {
namespace {

const packed_region_desc *find_region(const static_plan &plan, u32 region_id) {
    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id == region_id) return &region;
    }
    return nullptr;
}

validation_result validate_coordinate_plan_source(
    const static_plan &plan,
    packed_coordinate_plan_view packed) {
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
    if (packed.coordinates.count != 0u && packed.coordinates.data == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "packed coordinate storage is null");
    }
    for (u32 i = 0; i < static_cast<u32>(packed.coordinates.count); ++i) {
        const packed_coordinate &coordinate = packed.coordinates.data[i];
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
    compiled_coordinate_plan compiled;
    compiled.row_count = packed.row_count;
    compiled.feature_count = packed.feature_count;
    compiled.region_spans.resize(plan.regions.size());
    compiled.row_ids.resize(packed.coordinates.size());
    compiled.feature_ids.resize(packed.coordinates.size());
    compiled.values.resize(packed.coordinates.size());
    device_coordinate_plan_view view;
    validation_result build_result = build_compiled_coordinate_plan_into(
        plan, view_packed_coordinates(packed),
        {{compiled.region_spans.data(), compiled.region_spans.size(), {}},
         {compiled.row_ids.data(), compiled.row_ids.size(), {}},
         {compiled.feature_ids.data(), compiled.feature_ids.size(), {}},
         {compiled.values.data(), compiled.values.size(), {}}},
        &view);
    if (!build_result) return build_result;

    *out = std::move(compiled);
    return validation_ok();
}

validation_result build_compiled_coordinate_plan_into(
    const static_plan &plan,
    packed_coordinate_plan_view packed,
    compiled_coordinate_plan_storage storage,
    device_coordinate_plan_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "compiled coordinate view output is null");
    }
    validation_result source_result = validate_coordinate_plan_source(plan, packed);
    if (!source_result) return source_result;
    const std::size_t region_count = plan.regions.size();
    const std::size_t coordinate_count = packed.coordinates.count;
    if ((region_count != 0u && storage.region_spans.data == nullptr)
        || (coordinate_count != 0u && (storage.row_ids.data == nullptr
            || storage.feature_ids.data == nullptr || storage.values.data == nullptr))) {
        return validation_error(validation_code::null_pointer, invalid_id, "compiled coordinate storage is null");
    }
    if (storage.region_spans.count < region_count
        || storage.row_ids.count < coordinate_count
        || storage.feature_ids.count < coordinate_count
        || storage.values.count < coordinate_count) {
        return validation_error(validation_code::invalid_offsets, invalid_id, "compiled coordinate storage capacity is insufficient");
    }

    for (std::size_t i = 0; i < region_count; ++i) storage.region_spans.data[i] = {};
    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id >= region_count) {
            return validation_error(validation_code::invalid_region_bounds, region.region_id, "region id is outside compiled span table");
        }
        storage.region_spans.data[region.region_id].region_id = region.region_id;
    }
    for (std::size_t i = 0; i < coordinate_count; ++i) {
        ++storage.region_spans.data[packed.coordinates.data[i].region_id].coordinate_count;
    }
    u32 coordinate_begin = 0u;
    for (std::size_t i = 0; i < region_count; ++i) {
        region_coordinate_span &span = storage.region_spans.data[i];
        span.coordinate_begin = coordinate_begin;
        coordinate_begin += span.coordinate_count;
    }
    for (std::size_t i = 0; i < coordinate_count; ++i) {
        const packed_coordinate &coordinate = packed.coordinates.data[i];
        region_coordinate_span &span = storage.region_spans.data[coordinate.region_id];
        const u32 destination = span.coordinate_begin + span.reserved0++;
        storage.row_ids.data[destination] = coordinate.original_row;
        storage.feature_ids.data[destination] = coordinate.original_feature;
        storage.values.data[destination] = coordinate.value;
    }
    for (std::size_t i = 0; i < region_count; ++i) storage.region_spans.data[i].reserved0 = 0u;

    out->row_count = packed.row_count;
    out->feature_count = packed.feature_count;
    out->region_span_count = static_cast<u32>(region_count);
    out->coordinate_count = static_cast<u32>(coordinate_count);
    out->region_spans = storage.region_spans.data;
    out->row_ids = storage.row_ids.data;
    out->feature_ids = storage.feature_ids.data;
    out->values = storage.values.data;
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
