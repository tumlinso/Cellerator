#include <Cellerator/execution/geometry_acquisition_v2/projections.hh>

#include <limits>

namespace cellerator::execution::acquisition_v2 {

status resolve_route(
    const acquisition_request &request, route_resolution *resolution) noexcept {
    if (resolution == nullptr) {
        return {status_code::invalid_argument, 0};
    }
    *resolution = {};
    const status request_status = validate_request(request);
    if (!request_status) {
        return request_status;
    }
    resolution->requested = request.preferred_route;
    resolution->selected = request.preferred_route;
    if (request.preferred_route != route::load_cpe2
        || request.cpe2 == cpe2_disposition::compatible) {
        return {};
    }
    if (request.fallback == fallback_policy::reject) {
        return {status_code::incompatible_cpe2_rejected, 0};
    }
    resolution->selected = route::load_csg1;
    resolution->rebuilt_from_embedded_csg1 = true;
    return {};
}

status validate_projection_set(
    const acquisition_request &request, const projection_set &set) noexcept {
    if (set.projections == nullptr || set.projection_count == 0
        || set.projection_count > request.projection_requirement_count
        || set.chunks == nullptr || set.chunk_count < set.projection_count
        || set.payload_bytes == 0) {
        return {status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < set.projection_count; ++index) {
        const projection_record &projection = set.projections[index];
        if (!valid_stable_identity(projection.candidate)
            || !valid_stable_identity(projection.physical_projection)
            || projection.logical_work_items == 0 || projection.physical_slots == 0
            || projection.chunk_count == 0
            || projection.first_chunk > set.chunk_count
            || projection.chunk_count > set.chunk_count - projection.first_chunk
            || !projection.preserves_permanent_holes
            || (projection.value_modes & request.required_value_modes) == 0) {
            return {status_code::invalid_result, index};
        }
        std::uint64_t expected_logical_begin = 0;
        for (std::uint64_t local = 0; local < projection.chunk_count; ++local) {
            const std::uint64_t chunk_index = projection.first_chunk + local;
            const projection_chunk &chunk = set.chunks[chunk_index];
            if (chunk.projection_index != index || chunk.chunk_index != local
                || chunk.logical_count == 0 || chunk.logical_begin != expected_logical_begin
                || (chunk.local_index_bits != 16 && chunk.local_index_bits != 32)
                || chunk.payload_bytes == 0 || chunk.payload_offset > set.payload_bytes
                || chunk.payload_bytes > set.payload_bytes - chunk.payload_offset) {
                return {status_code::invalid_result, chunk_index};
            }
            if (expected_logical_begin
                > std::numeric_limits<std::uint64_t>::max() - chunk.logical_count) {
                return {status_code::invalid_result, chunk_index};
            }
            expected_logical_begin += chunk.logical_count;
        }
        if (expected_logical_begin != projection.logical_work_items) {
            return {status_code::invalid_result, index};
        }
    }
    return {};
}

}  // namespace cellerator::execution::acquisition_v2
