#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

namespace cellerator::execution::acquisition_v2 {
namespace {

bool valid_route(route value) noexcept {
    return value >= route::compile_now && value <= route::adapt_cpk1;
}

bool valid_alignment(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

bool sufficient(byte_span span, buffer_requirement requirement) noexcept {
    if (requirement.bytes == 0) {
        return true;
    }
    return span.data != nullptr && span.bytes >= requirement.bytes
        && reinterpret_cast<std::uintptr_t>(span.data) % requirement.alignment == 0;
}

bool within(immutable_byte_span view, byte_span buffer) noexcept {
    if (view.data == nullptr || view.bytes == 0 || buffer.data == nullptr) {
        return false;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
    const auto end = begin + buffer.bytes;
    const auto view_begin = reinterpret_cast<std::uintptr_t>(view.data);
    return view_begin >= begin && view_begin <= end
        && view.bytes <= end - view_begin;
}

}  // namespace

status validate_request(const acquisition_request &request) noexcept {
    if (request.version != schema_version || request.record_bytes != sizeof(request)) {
        return {status_code::invalid_header, 0};
    }
    if (!valid_route(request.preferred_route)) {
        return {status_code::invalid_route, 0};
    }
    if (!valid_identity(request.structure) || request.epoch.value == 0) {
        return {status_code::invalid_identity, 0};
    }
    if (request.source.data == nullptr || request.source.bytes == 0
        || request.projection_requirements == nullptr
        || request.projection_requirement_count == 0
        || request.projection_requirement_count > maximum_projection_requirements) {
        return {status_code::invalid_argument, 0};
    }
    const std::uint8_t valid_modes = logical_primary_values | projection_primary_values;
    if (request.required_value_modes == 0
        || (request.required_value_modes & ~valid_modes) != 0) {
        return {status_code::invalid_argument, 0};
    }
    if (request.preferred_route != route::load_cpe2
        && (request.cpe2 != cpe2_disposition::not_applicable
            || request.fallback != fallback_policy::reject)) {
        return {status_code::invalid_route, 0};
    }
    if (request.preferred_route == route::load_cpe2
        && (request.cpe2 == cpe2_disposition::not_applicable
            || request.cpe2 == cpe2_disposition::invalid)) {
        return {status_code::invalid_route, 0};
    }
    for (std::uint64_t index = 0; index < request.projection_requirement_count; ++index) {
        const projection_requirement &projection = request.projection_requirements[index];
        if (!valid_stable_identity(projection.candidate)
            || projection.logical_work_items == 0
            || (projection.accepted_value_modes & request.required_value_modes) == 0) {
            return {status_code::invalid_argument, index};
        }
    }
    return {};
}

status validate_requirements(const acquisition_request &request,
    const acquisition_requirements &requirements) noexcept {
    if (requirements.version != schema_version
        || requirements.record_bytes != sizeof(requirements)
        || !valid_route(requirements.selected_route)
        || requirements.projection_count == 0
        || requirements.projection_count > request.projection_requirement_count
        || requirements.projection_chunk_count < requirements.projection_count) {
        return {status_code::invalid_requirements, 0};
    }
    if (requirements.rebuilt_from_embedded_csg1
        && (request.preferred_route != route::load_cpe2
            || request.cpe2 != cpe2_disposition::incompatible
            || request.fallback != fallback_policy::rebuild_from_embedded_csg1
            || requirements.selected_route != route::load_csg1)) {
        return {status_code::invalid_requirements, 0};
    }
    const buffer_requirement values[] = {requirements.semantic_geometry,
        requirements.projections, requirements.catalog, requirements.planner,
        requirements.program, requirements.transient, requirements.diagnostics};
    for (std::uint64_t index = 0; index < 7; ++index) {
        if (!valid_alignment(values[index].alignment)) {
            return {status_code::invalid_requirements, index};
        }
    }
    if (requirements.semantic_geometry.bytes == 0
        || requirements.projections.bytes == 0
        || requirements.program.bytes == 0
        || requirements.diagnostics.bytes == 0) {
        return {status_code::invalid_requirements, 0};
    }
    return {};
}

status query_requirements(const acquisition_facade &facade,
    const acquisition_request &request,
    acquisition_requirements *requirements) noexcept {
    if (requirements == nullptr) {
        return {status_code::invalid_argument, 0};
    }
    *requirements = {};
    const status request_status = validate_request(request);
    if (!request_status) {
        return request_status;
    }
    if (facade.query == nullptr) {
        return {status_code::callback_unavailable, 0};
    }
    const status callback_status = facade.query(request, requirements);
    if (!callback_status) {
        *requirements = {};
        return {status_code::callback_failed, callback_status.index};
    }
    return validate_requirements(request, *requirements);
}

status acquire(const acquisition_facade &facade,
    const acquisition_request &request,
    const acquisition_requirements &requirements,
    const acquisition_buffers &buffers,
    acquired_geometry *result) noexcept {
    if (result == nullptr) {
        return {status_code::invalid_argument, 0};
    }
    *result = {};
    const status request_status = validate_request(request);
    if (!request_status) {
        return request_status;
    }
    const status requirements_status = validate_requirements(request, requirements);
    if (!requirements_status) {
        return requirements_status;
    }
    const byte_span spans[] = {buffers.semantic_geometry, buffers.projections,
        buffers.catalog, buffers.planner, buffers.program, buffers.transient,
        buffers.diagnostics};
    const buffer_requirement required[] = {requirements.semantic_geometry,
        requirements.projections, requirements.catalog, requirements.planner,
        requirements.program, requirements.transient, requirements.diagnostics};
    for (std::uint64_t index = 0; index < 7; ++index) {
        if (!sufficient(spans[index], required[index])) {
            return {status_code::insufficient_capacity, index};
        }
    }
    if (facade.acquire == nullptr) {
        return {status_code::callback_unavailable, 0};
    }
    acquired_geometry candidate{};
    const status callback_status = facade.acquire(request, requirements, buffers, &candidate);
    if (!callback_status) {
        return {status_code::callback_failed, callback_status.index};
    }
    if (candidate.version != schema_version || candidate.record_bytes != sizeof(candidate)
        || !valid_stable_identity(candidate.semantic_geometry)
        || candidate.projection_count != requirements.projection_count
        || !within(candidate.semantic_geometry_image, buffers.semantic_geometry)
        || !within(candidate.projection_records, buffers.projections)
        || !within(candidate.prepared_program, buffers.program)
        || !within(candidate.diagnostics, buffers.diagnostics)) {
        return {status_code::invalid_result, 0};
    }
    *result = candidate;
    return {};
}

}  // namespace cellerator::execution::acquisition_v2
