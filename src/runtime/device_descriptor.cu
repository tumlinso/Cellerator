#include <Cellerator/runtime/device_descriptor.hh>

#include <Cellerator/execution/identity.hh>
#include <Cellerator/planner/end_to_end_planner.hh>
#include <Cellerator/runtime/session.cuh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

namespace cellerator::runtime {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void mix(std::uint64_t *hash, std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        *hash ^= value & 0xffu;
        *hash *= fnv_prime;
        value >>= 8u;
    }
}

device_architecture_class_v1 architecture_class(
    std::uint16_t major,
    std::uint16_t minor) noexcept {
    if (major == 7u && minor == 0u) return device_architecture_class_v1::nvidia_volta;
    if (major == 7u) return device_architecture_class_v1::nvidia_turing;
    if (major == 8u && minor < 9u) return device_architecture_class_v1::nvidia_ampere;
    if (major == 8u) return device_architecture_class_v1::nvidia_ada;
    if (major == 9u) return device_architecture_class_v1::nvidia_hopper;
    if (major >= 10u) return device_architecture_class_v1::nvidia_blackwell;
    return device_architecture_class_v1::nvidia_other;
}

std::uint64_t compatibility_identity(
    const device_descriptor_v1 &descriptor) noexcept {
    std::uint64_t hash = fnv_offset;
    mix(&hash, descriptor.schema_version);
    mix(&hash, descriptor.vendor);
    mix(&hash, descriptor.compute_major);
    mix(&hash, descriptor.compute_minor);
    mix(&hash, static_cast<std::uint32_t>(descriptor.architecture));
    mix(&hash, descriptor.warp_size);
    mix(&hash, descriptor.maximum_threads_per_block);
    mix(&hash, descriptor.maximum_thread_dimensions[0]);
    mix(&hash, descriptor.maximum_thread_dimensions[1]);
    mix(&hash, descriptor.maximum_thread_dimensions[2]);
    mix(&hash, descriptor.registers_per_block);
    mix(&hash, descriptor.shared_memory_per_block_bytes);
    mix(&hash, descriptor.optin_shared_memory_per_block_bytes);
    return hash == 0u ? 1u : hash;
}

std::uint64_t performance_identity(
    const device_descriptor_v1 &descriptor) noexcept {
    std::uint64_t hash = compatibility_identity(descriptor);
    mix(&hash, descriptor.multiprocessor_count);
    mix(&hash, descriptor.maximum_threads_per_multiprocessor);
    mix(&hash, descriptor.maximum_blocks_per_multiprocessor);
    mix(&hash, descriptor.registers_per_multiprocessor);
    mix(&hash, descriptor.shared_memory_per_multiprocessor_bytes);
    mix(&hash, descriptor.global_memory_bytes);
    mix(&hash, descriptor.l2_cache_bytes);
    return hash == 0u ? 1u : hash;
}

template<typename Value>
bool fits_unsigned(Value value) noexcept {
    return value >= 0
        && static_cast<unsigned long long>(value)
            <= std::numeric_limits<std::uint32_t>::max();
}

} // namespace

device_descriptor_status_v1 query_device_descriptor_v1(
    std::int32_t requested_ordinal,
    bool session_sealed,
    device_descriptor_v1 *descriptor,
    std::uint64_t *query_count) noexcept {
    if (descriptor == nullptr) return device_descriptor_status_v1::invalid_argument;
    if (session_sealed) return device_descriptor_status_v1::invalid_state;

    device_descriptor_v1 result{};
    int ordinal = requested_ordinal;
    if (ordinal < 0) {
        if (query_count != nullptr) ++*query_count;
        if (cudaGetDevice(&ordinal) != cudaSuccess) {
            return device_descriptor_status_v1::cuda_failure;
        }
    }

    cudaDeviceProp properties{};
    if (query_count != nullptr) ++*query_count;
    if (cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess) {
        return device_descriptor_status_v1::cuda_failure;
    }
    if (properties.major < 0 || properties.minor < 0
        || !fits_unsigned(properties.multiProcessorCount)
        || !fits_unsigned(properties.warpSize)
        || !fits_unsigned(properties.maxThreadsPerBlock)
        || !fits_unsigned(properties.maxThreadsPerMultiProcessor)
        || !fits_unsigned(properties.regsPerBlock)
        || !fits_unsigned(properties.regsPerMultiprocessor)) {
        return device_descriptor_status_v1::cuda_failure;
    }

    result.vendor = nvidia_pci_vendor_id;
    result.ordinal = ordinal;
    result.compute_major = static_cast<std::uint16_t>(properties.major);
    result.compute_minor = static_cast<std::uint16_t>(properties.minor);
    result.architecture = architecture_class(result.compute_major, result.compute_minor);
    result.multiprocessor_count =
        static_cast<std::uint32_t>(properties.multiProcessorCount);
    result.warp_size = static_cast<std::uint32_t>(properties.warpSize);
    result.maximum_threads_per_block =
        static_cast<std::uint32_t>(properties.maxThreadsPerBlock);
    result.maximum_threads_per_multiprocessor =
        static_cast<std::uint32_t>(properties.maxThreadsPerMultiProcessor);
    for (std::uint32_t dimension = 0u; dimension < 3u; ++dimension) {
        if (!fits_unsigned(properties.maxThreadsDim[dimension])
            || !fits_unsigned(properties.maxGridSize[dimension])) {
            return device_descriptor_status_v1::cuda_failure;
        }
        result.maximum_thread_dimensions[dimension] =
            static_cast<std::uint32_t>(properties.maxThreadsDim[dimension]);
        result.maximum_grid_dimensions[dimension] =
            static_cast<std::uint32_t>(properties.maxGridSize[dimension]);
    }
    result.registers_per_block = static_cast<std::uint32_t>(properties.regsPerBlock);
    result.registers_per_multiprocessor =
        static_cast<std::uint32_t>(properties.regsPerMultiprocessor);
    result.shared_memory_per_block_bytes = properties.sharedMemPerBlock;
#if CUDART_VERSION >= 9000
    result.optin_shared_memory_per_block_bytes = properties.sharedMemPerBlockOptin;
    result.shared_memory_per_multiprocessor_bytes = properties.sharedMemPerMultiprocessor;
#endif
    result.global_memory_bytes = properties.totalGlobalMem;
    if (properties.l2CacheSize < 0) return device_descriptor_status_v1::cuda_failure;
    result.l2_cache_bytes = static_cast<std::uint64_t>(properties.l2CacheSize);

#if CUDART_VERSION >= 11000
    int maximum_blocks = 0;
    if (query_count != nullptr) ++*query_count;
    if (cudaDeviceGetAttribute(
            &maximum_blocks,
            cudaDevAttrMaxBlocksPerMultiprocessor,
            ordinal) == cudaSuccess
        && maximum_blocks > 0) {
        result.maximum_blocks_per_multiprocessor =
            static_cast<std::uint32_t>(maximum_blocks);
    }
#endif

    result.hardware_compatibility_identity = compatibility_identity(result);
    result.performance_class_identity = performance_identity(result);
    if (!valid_device_descriptor_v1(result)) {
        return device_descriptor_status_v1::cuda_failure;
    }
    *descriptor = result;
    return device_descriptor_status_v1::success;
}

bool valid_device_descriptor_v1(
    const device_descriptor_v1 &descriptor) noexcept {
    return descriptor.schema_version == device_descriptor_schema_version_v1
        && descriptor.vendor != 0u && descriptor.ordinal >= 0
        && descriptor.compute_major != 0u
        && descriptor.architecture != device_architecture_class_v1::unknown
        && descriptor.multiprocessor_count != 0u && descriptor.warp_size != 0u
        && descriptor.maximum_threads_per_block != 0u
        && descriptor.maximum_threads_per_multiprocessor != 0u
        && descriptor.maximum_thread_dimensions[0] != 0u
        && descriptor.maximum_grid_dimensions[0] != 0u
        && descriptor.registers_per_block != 0u
        && descriptor.registers_per_multiprocessor != 0u
        && descriptor.shared_memory_per_block_bytes != 0u
        && descriptor.global_memory_bytes != 0u
        && descriptor.hardware_compatibility_identity != 0u
        && descriptor.performance_class_identity != 0u;
}

device_performance_class derive_runtime_device_performance_class(
    const device_descriptor_v1 &descriptor) noexcept {
    device_performance_class result{};
    if (!valid_device_descriptor_v1(descriptor)) return result;
    result.device = descriptor.ordinal;
    result.compute_major = descriptor.compute_major;
    result.compute_minor = descriptor.compute_minor;
    result.multiprocessor_count = descriptor.multiprocessor_count;
    result.warp_size = descriptor.warp_size;
    result.global_memory_bytes = descriptor.global_memory_bytes;
    return result;
}

execution::device_performance_class derive_execution_device_performance_class(
    const device_descriptor_v1 &descriptor) noexcept {
    execution::device_performance_class result{};
    if (!valid_device_descriptor_v1(descriptor)) return result;
    result.vendor = descriptor.vendor;
    result.architecture_major = descriptor.compute_major;
    result.architecture_minor = descriptor.compute_minor;
    // Historical field name: this compatibility view carries a hardware
    // performance identity, never a runtime or kernel build identity.
    result.build_identity = descriptor.performance_class_identity;
    return result;
}

planner::device_performance_key derive_planner_device_performance_key(
    const device_descriptor_v1 &descriptor) noexcept {
    planner::device_performance_key result{};
    if (!valid_device_descriptor_v1(descriptor)) return result;
    result.vendor = descriptor.vendor;
    result.architecture_major = descriptor.compute_major;
    result.architecture_minor = descriptor.compute_minor;
    result.performance_class = descriptor.performance_class_identity;
    return result;
}

} // namespace cellerator::runtime
