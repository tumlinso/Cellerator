#include <Cellerator/memory/copy.cuh>

namespace cellerator::memory {

namespace {

bool is_host(domain value) noexcept {
    return value == domain::host || value == domain::host_numa
        || value == domain::host_pinned
        || value == domain::host_pinned_write_combined;
}

bool is_device(domain value) noexcept {
    return value == domain::device || value == domain::managed;
}

bool direction_matches(const copy_request &request) noexcept {
    switch (request.direction) {
    case copy_direction::host_to_device:
        return is_host(request.source_where.kind)
            && is_device(request.destination_where.kind);
    case copy_direction::device_to_host:
        return is_device(request.source_where.kind)
            && is_host(request.destination_where.kind);
    case copy_direction::device_to_device:
        return is_device(request.source_where.kind)
            && is_device(request.destination_where.kind);
    case copy_direction::host_to_host:
        return is_host(request.source_where.kind)
            && is_host(request.destination_where.kind);
    }
    return false;
}

cudaMemcpyKind cuda_kind(copy_direction value) noexcept {
    switch (value) {
    case copy_direction::host_to_device: return cudaMemcpyHostToDevice;
    case copy_direction::device_to_host: return cudaMemcpyDeviceToHost;
    case copy_direction::device_to_device: return cudaMemcpyDeviceToDevice;
    case copy_direction::host_to_host: return cudaMemcpyHostToHost;
    }
    return cudaMemcpyDefault;
}

} // namespace

status copy_async(const copy_request &request) noexcept {
    if (request.bytes == 0u) return status::success;
    if (request.destination == nullptr || request.source == nullptr)
        return status::invalid_argument;
    if (request.bytes > request.destination_capacity
        || request.bytes > request.source_capacity)
        return status::capacity_exceeded;
    if (!direction_matches(request)) return status::invalid_placement;
    return cudaMemcpyAsync(request.destination, request.source, request.bytes,
               cuda_kind(request.direction), request.stream) == cudaSuccess
        ? status::success : status::cuda_failure;
}

} // namespace cellerator::memory
