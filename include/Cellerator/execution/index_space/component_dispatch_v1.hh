#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include <Cellerator/execution/index_space/hierarchical_index_space_v1.hh>

#if defined(__CUDACC__)
#define CELLERATOR_INDEX_HOST_DEVICE __host__ __device__
#else
#define CELLERATOR_INDEX_HOST_DEVICE
#endif

namespace cellerator::execution {

// Device-facing view for a single compact index stream.  A prepared dispatch
// binds one width, so kernels do not inspect aggregate relation size.
struct local_index_device_view_v1 {
    const void *data = nullptr;
    local_index_width_v1 width = local_index_width_v1::u32;
    std::uint8_t reserved[7]{};
};

CELLERATOR_INDEX_HOST_DEVICE inline std::uint64_t load_local_index_v1(
    const local_index_device_view_v1 &view, std::uint32_t position) noexcept {
    switch (view.width) {
        case local_index_width_v1::u16:
            return static_cast<const std::uint16_t *>(view.data)[position];
        case local_index_width_v1::u32:
            return static_cast<const std::uint32_t *>(view.data)[position];
        case local_index_width_v1::u64:
            return static_cast<const std::uint64_t *>(view.data)[position];
    }
    return 0u;
}

// One prepared launch consumes exactly one bounded component.  local_work_count
// is u32 by construction; aggregate_begin and global recovery stay u64.
struct component_dispatch_v1 {
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_begin = 0u;
    const std::uint64_t *local_to_global = nullptr;
    const std::uint64_t *global_identity_sidecar = nullptr;
    const void *payload = nullptr;
    std::uint32_t local_work_count = 0u;
    std::uint32_t block_threads = 0u;
    std::uint32_t grid_x = 0u;
    std::uint32_t grid_y = 0u;
    local_index_width_v1 local_width = local_index_width_v1::u32;
    std::uint8_t reserved[7]{};
};

struct chunked_dispatch_view_v1 {
    std::uint64_t dispatch_identity = 0u;
    std::uint64_t aggregate_work_count = 0u;
    const component_dispatch_v1 *components = nullptr;
    std::uint64_t component_count = 0u;
};

enum class component_dispatch_status_v1 : std::uint32_t {
    valid = 0u,
    invalid_block_size,
    invalid_grid_limit,
    local_extent_exceeds_u32,
    grid_y_exceeds_u32,
};

// Compute a portable two-dimensional grid without truncating intermediate
// sizes.  The returned descriptor still addresses only one u32-bounded local
// component; callers iterate the chunked dispatch outside sealed kernels.
inline component_dispatch_status_v1 make_component_grid_v1(
    std::uint64_t local_work_count, std::uint32_t block_threads,
    std::uint32_t maximum_grid_x, std::uint32_t *grid_x,
    std::uint32_t *grid_y) noexcept {
    if (block_threads == 0u || grid_x == nullptr || grid_y == nullptr) {
        return component_dispatch_status_v1::invalid_block_size;
    }
    if (maximum_grid_x == 0u) {
        return component_dispatch_status_v1::invalid_grid_limit;
    }
    if (local_work_count > std::numeric_limits<std::uint32_t>::max()) {
        return component_dispatch_status_v1::local_extent_exceeds_u32;
    }
    if (local_work_count == 0u) {
        *grid_x = 0u;
        *grid_y = 0u;
        return component_dispatch_status_v1::valid;
    }
    const std::uint64_t blocks =
        (local_work_count + block_threads - 1u) / block_threads;
    const std::uint64_t x = blocks < maximum_grid_x ? blocks : maximum_grid_x;
    const std::uint64_t y = (blocks + x - 1u) / x;
    if (y > std::numeric_limits<std::uint32_t>::max()) {
        return component_dispatch_status_v1::grid_y_exceeds_u32;
    }
    *grid_x = static_cast<std::uint32_t>(x);
    *grid_y = static_cast<std::uint32_t>(y);
    return component_dispatch_status_v1::valid;
}

CELLERATOR_INDEX_HOST_DEVICE inline std::uint64_t flattened_thread_index_v1(
    std::uint32_t block_x, std::uint32_t block_y,
    std::uint32_t grid_x, std::uint32_t block_threads,
    std::uint32_t thread_x) noexcept {
    return (static_cast<std::uint64_t>(block_y) * grid_x + block_x)
        * block_threads + thread_x;
}

CELLERATOR_INDEX_HOST_DEVICE inline std::uint64_t recover_global_index_v1(
    const component_dispatch_v1 &dispatch,
    std::uint32_t local_position) noexcept {
    return dispatch.local_to_global[local_position];
}

CELLERATOR_INDEX_HOST_DEVICE inline std::uint64_t recover_global_identity_v1(
    const component_dispatch_v1 &dispatch,
    std::uint32_t local_position) noexcept {
    return dispatch.global_identity_sidecar == nullptr
        ? dispatch.local_to_global[local_position]
        : dispatch.global_identity_sidecar[local_position];
}

static_assert(std::is_trivially_copyable_v<local_index_device_view_v1>);
static_assert(std::is_trivially_copyable_v<component_dispatch_v1>);
static_assert(std::is_trivially_copyable_v<chunked_dispatch_view_v1>);

}  // namespace cellerator::execution

#undef CELLERATOR_INDEX_HOST_DEVICE
