#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation::sparse_axis_update {

enum class status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    unsupported = 2u,
    cuda_failure = 3u
};

enum class operation_v1 : std::uint8_t {
    assign = 0u,
    add = 1u,
    subtract = 2u,
    multiply = 3u,
    maximum = 4u
};

enum class index_kind_v1 : std::uint8_t {
    local_u32 = 0u,
    global_u64 = 1u
};

struct target_slice_v1 {
    std::uint64_t global_axis_begin = 0u;
    std::uint32_t local_axis_count = 0u;
    std::uint32_t component_count = 0u;
};

struct request_v1 {
    target_slice_v1 target_slice{};
    float *target = nullptr;
    const void *indices = nullptr;
    const float *updates = nullptr;
    std::uint32_t update_count = 0u;
    operation_v1 operation = operation_v1::assign;
    index_kind_v1 index_kind = index_kind_v1::local_u32;
    bool indices_are_unique = false;
    bool indices_are_in_persistent_order = false;
    std::uint8_t reserved[2]{};
    std::uint64_t target_axis_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t input_value_generation = 0u;
    std::uint64_t output_value_generation = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_request_v1(const request_v1 &request) noexcept;
status_v1 enqueue_v1(const request_v1 &request) noexcept;

static_assert(std::is_trivially_copyable<request_v1>::value,
    "sparse update launches are non-owning bindings");

} // namespace cellerator::compute::operation::sparse_axis_update
