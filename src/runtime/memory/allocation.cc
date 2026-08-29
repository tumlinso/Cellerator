#include <Cellerator/memory/allocation.hh>

#include <cstdint>

namespace cellerator::memory {

namespace {

bool valid_placement(placement value) noexcept {
    switch (value.kind) {
    case domain::host:
    case domain::host_pinned:
    case domain::host_pinned_write_combined:
        return value.device_ordinal == -1 && value.numa_node == -1;
    case domain::host_numa:
        return value.device_ordinal == -1 && value.numa_node >= 0;
    case domain::device:
    case domain::managed:
        return value.device_ordinal >= 0 && value.numa_node == -1;
    case domain::external:
        return true;
    }
    return false;
}

} // namespace

status validate_allocation_request(
    const allocation_request &request) noexcept {
    if (!valid_alignment(request.alignment)) return status::invalid_alignment;
    if (!valid_placement(request.where)) return status::invalid_placement;
    return status::success;
}

status bind_external_allocation(
    void *base,
    std::size_t bytes,
    std::uint32_t alignment,
    std::uint32_t generation,
    allocation *out) noexcept {
    if (out == nullptr) return status::invalid_argument;
    *out = allocation{};
    if (!valid_alignment(alignment)) return status::invalid_alignment;
    if ((bytes != 0u && base == nullptr) || generation == 0u)
        return status::invalid_argument;
    if (base != nullptr
        && (reinterpret_cast<std::uintptr_t>(base) & (alignment - 1u)) != 0u)
        return status::invalid_alignment;
    *out = allocation{
        base, bytes, alignment, placement{domain::external, -1, -1, 0u}, generation};
    return status::success;
}

status reset_allocation_record(allocation *record) noexcept {
    if (record == nullptr) return status::invalid_argument;
    *record = allocation{};
    return status::success;
}

} // namespace cellerator::memory
