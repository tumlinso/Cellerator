#include <Cellerator/memory/image.hh>

#include <cstdint>
#include <limits>

namespace cellerator::memory {

status validate_image_span(
    std::size_t image_bytes,
    std::uint64_t byte_offset,
    std::size_t span_bytes,
    std::size_t alignment) noexcept {
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u)
        return status::invalid_alignment;
    if ((byte_offset & (alignment - 1u)) != 0u)
        return status::invalid_alignment;
    if (byte_offset > image_bytes) return status::capacity_exceeded;
    const std::size_t offset = static_cast<std::size_t>(byte_offset);
    if (span_bytes > image_bytes - offset) return status::capacity_exceeded;
    return status::success;
}

status resolve_image_span(
    const const_image_view &image,
    std::uint64_t byte_offset,
    std::size_t span_bytes,
    std::size_t alignment,
    const void **out) noexcept {
    if (out != nullptr) *out = nullptr;
    if (out == nullptr || (image.bytes != 0u && image.base == nullptr))
        return status::invalid_argument;
    const status checked = validate_image_span(
        image.bytes, byte_offset, span_bytes, alignment);
    if (checked != status::success) return checked;
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(image.base);
    if (byte_offset > std::numeric_limits<std::uintptr_t>::max() - base)
        return status::arithmetic_overflow;
    *out = reinterpret_cast<const void *>(base + byte_offset);
    return status::success;
}

status validate_image_header(
    const image_header &header,
    const const_image_view &image,
    std::uint32_t expected_magic,
    std::uint16_t expected_schema_version) noexcept {
    if (image.base == nullptr || header.magic != expected_magic
        || header.schema_version != expected_schema_version)
        return status::invalid_argument;
    if (header.total_bytes != image.bytes) return status::capacity_exceeded;
    if (header.required_alignment == 0u
        || (header.required_alignment & (header.required_alignment - 1u)) != 0u)
        return status::invalid_alignment;
    if ((reinterpret_cast<std::uintptr_t>(image.base)
            & (header.required_alignment - 1u)) != 0u)
        return status::invalid_alignment;
    return status::success;
}

} // namespace cellerator::memory
