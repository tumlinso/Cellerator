#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::layout {

// Width of an index into one independently bounded physical component. The
// numeric values are part of the persisted compact-index encoding.
enum class LocalIndexWidth : std::uint8_t {
    u16 = 2u,
    u32 = 4u,
    u64 = 8u,
};

// Non-owning mapping from component-local positions to global coordinates.
// local_to_global has local_extent entries; global_identity_sidecar is an
// optional parallel array carrying identity distinct from physical position.
struct LocalIndexSpaceViewV1 {
    std::uint64_t global_extent = 0u;
    std::uint64_t partition_identity = 0u;
    std::uint64_t local_extent = 0u;
    const std::uint64_t *local_to_global = nullptr;
    const std::uint64_t *global_identity_sidecar = nullptr;
    LocalIndexWidth local_width = LocalIndexWidth::u32;
    std::uint8_t reserved[7]{};
};

static_assert(std::is_trivially_copyable_v<LocalIndexSpaceViewV1>);
static_assert(std::is_standard_layout_v<LocalIndexSpaceViewV1>);
static_assert(sizeof(LocalIndexWidth) == 1u);
static_assert(static_cast<std::uint8_t>(LocalIndexWidth::u16) == 2u);
static_assert(static_cast<std::uint8_t>(LocalIndexWidth::u32) == 4u);
static_assert(static_cast<std::uint8_t>(LocalIndexWidth::u64) == 8u);

// The map is a native, pointer-bearing POD. Assert its host ABI layout so a
// namespace extraction cannot silently alter prepared views or record bytes;
// express pointer-dependent offsets so supported host pointer widths retain
// their existing layout.
static_assert(offsetof(LocalIndexSpaceViewV1, global_extent) == 0u);
static_assert(offsetof(LocalIndexSpaceViewV1, partition_identity) == 8u);
static_assert(offsetof(LocalIndexSpaceViewV1, local_extent) == 16u);
static_assert(offsetof(LocalIndexSpaceViewV1, local_to_global) == 24u);
static_assert(offsetof(LocalIndexSpaceViewV1, global_identity_sidecar)
              == 24u + sizeof(void *));
static_assert(offsetof(LocalIndexSpaceViewV1, local_width)
              == 24u + 2u * sizeof(void *));
static_assert(offsetof(LocalIndexSpaceViewV1, reserved)
              == 25u + 2u * sizeof(void *));
static_assert(sizeof(LocalIndexSpaceViewV1)
              == ((32u + 2u * sizeof(void *)
                   + alignof(LocalIndexSpaceViewV1) - 1u)
                  / alignof(LocalIndexSpaceViewV1))
                     * alignof(LocalIndexSpaceViewV1));

}  // namespace cellerator::layout
