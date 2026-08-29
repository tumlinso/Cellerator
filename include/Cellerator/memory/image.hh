#pragma once

#include "domain.hh"
#include "status.hh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::memory {

struct image_header {
    std::uint32_t magic = 0;
    std::uint16_t schema_version = 0;
    std::uint16_t flags = 0;
    std::uint64_t total_bytes = 0;
    std::uint32_t required_alignment = 0;
    std::uint32_t section_count = 0;
    std::uint64_t identity = 0;
};

struct rel32 {
    std::uint32_t byte_offset = 0;
};

struct rel64 {
    std::uint64_t byte_offset = 0;
};

struct image_buffer {
    void *base = nullptr;
    std::size_t bytes = 0;
    placement where{};
};

struct const_image_view {
    const void *base = nullptr;
    std::size_t bytes = 0;
    placement where{};
};

status validate_image_header(
    const image_header &header,
    const const_image_view &image,
    std::uint32_t expected_magic,
    std::uint16_t expected_schema_version) noexcept;

status validate_image_span(
    std::size_t image_bytes,
    std::uint64_t byte_offset,
    std::size_t span_bytes,
    std::size_t alignment) noexcept;

status resolve_image_span(
    const const_image_view &image,
    std::uint64_t byte_offset,
    std::size_t span_bytes,
    std::size_t alignment,
    const void **out) noexcept;

inline status validate_image_span(
    std::size_t image_bytes,
    rel32 offset,
    std::size_t span_bytes,
    std::size_t alignment) noexcept {
    return validate_image_span(
        image_bytes, offset.byte_offset, span_bytes, alignment);
}

inline status validate_image_span(
    std::size_t image_bytes,
    rel64 offset,
    std::size_t span_bytes,
    std::size_t alignment) noexcept {
    return validate_image_span(
        image_bytes, offset.byte_offset, span_bytes, alignment);
}

static_assert(sizeof(image_header) == 32u,
    "generic image header size is part of the substrate contract");
static_assert(std::is_trivially_copyable<image_header>::value,
    "image metadata must remain pointer-free");
static_assert(std::is_trivially_copyable<rel32>::value,
    "relative offsets must remain pointer-free");

} // namespace cellerator::memory
