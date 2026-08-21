#pragma once

#include "libraries.cuh"
#include "scratch.cuh"
#include "stream.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::runtime {

inline constexpr std::uint32_t execution_session_max_streams = 8;
inline constexpr std::uint32_t execution_session_cache_capacity = 64;
inline constexpr std::uint32_t execution_session_max_persistent_allocations = 64;

enum class session_status : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_state,
    capacity_exceeded,
    workspace_exhausted,
    device_mismatch,
    cuda_failure
};

enum class persistent_lifetime : std::uint8_t {
    structure = 0,
    plan,
    graph_stable
};

enum class session_cache_kind : std::uint8_t {
    plan = 0,
    projection,
    order_transform
};

struct device_performance_class {
    int device = -1;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    int warp_size = 0;
    std::size_t global_memory_bytes = 0;
};

struct session_cache_key {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
};

struct session_cache_entry {
    session_cache_key key{};
    void *state = nullptr;
    std::uint64_t structure_epoch = 0;
    std::uint64_t generation = 0;
    bool occupied = false;
};

struct fixed_session_cache {
    session_cache_entry entries[execution_session_cache_capacity]{};
    std::uint32_t size = 0;
};

struct allocation_accounting {
    std::size_t current_bytes = 0;
    std::size_t high_water_bytes = 0;
    std::uint64_t allocation_count = 0;
};

struct persistent_allocation_record {
    void *data = nullptr;
    std::size_t bytes = 0;
    persistent_lifetime lifetime = persistent_lifetime::structure;
    bool occupied = false;
};

struct session_accounting {
    allocation_accounting structure{};
    allocation_accounting plan{};
    allocation_accounting graph_stable{};
    allocation_accounting transient{};
    std::uint64_t device_query_count = 0;
    std::uint64_t handle_prepare_count = 0;
    std::uint64_t launch_bind_count = 0;
    std::uint64_t synchronization_count = 0;
};

struct session_stream_slot {
    execution_context execution{};
    cublas_cache cublas{};
    cusparse_cache cusparse{};
    scratch_arena transient{};
    bool libraries_prepared = false;
};

struct execution_session_options {
    int device = -1;
    const cudaStream_t *external_streams = nullptr;
    std::uint32_t external_stream_count = 0;
    std::uint32_t owned_stream_count = 1;
};

struct execution_session {
    int device = -1;
    device_performance_class performance{};
    persistent_allocation_record
        persistent_allocations[execution_session_max_persistent_allocations]{};
    session_stream_slot streams[execution_session_max_streams]{};
    fixed_session_cache plans{};
    fixed_session_cache projections{};
    fixed_session_cache order_transforms{};
    session_accounting accounting{};
    std::uint32_t persistent_allocation_count = 0;
    std::uint32_t stream_count = 0;
    bool initialized = false;
    bool sealed = false;
};

struct launch_runtime_binding {
    session_status status = session_status::invalid_state;
    execution_context execution{};
    cublasHandle_t cublas = nullptr;
    cusparseHandle_t cusparse = nullptr;
    void *workspace = nullptr;
    std::size_t workspace_bytes = 0;
};

struct device_fleet_view {
    const device_performance_class *devices = nullptr;
    std::uint32_t device_count = 0;
};

session_status init_session(
    execution_session *session,
    const execution_session_options &options) noexcept;
void clear_session(execution_session *session) noexcept;

session_status reserve_persistent(
    execution_session *session,
    persistent_lifetime lifetime,
    std::size_t bytes,
    void **allocation) noexcept;
session_status reserve_transient(
    execution_session *session,
    std::uint32_t stream_index,
    std::size_t bytes,
    void **allocation) noexcept;
session_status prepare_stream_libraries(
    execution_session *session,
    std::uint32_t stream_index) noexcept;
session_status seal_session(execution_session *session) noexcept;

launch_runtime_binding bind_launch(
    execution_session *session,
    std::uint32_t stream_index,
    std::size_t required_workspace_bytes) noexcept;

session_status insert_session_cache(
    execution_session *session,
    session_cache_kind kind,
    session_cache_key key,
    void *state,
    std::uint64_t structure_epoch,
    std::uint64_t generation) noexcept;
const session_cache_entry *find_session_cache(
    const execution_session &session,
    session_cache_kind kind,
    session_cache_key key) noexcept;

bool graph_stable_address(
    const execution_session &session,
    const void *address,
    std::size_t bytes) noexcept;
device_fleet_view single_device_fleet(const execution_session &session) noexcept;

} // namespace cellerator::runtime
