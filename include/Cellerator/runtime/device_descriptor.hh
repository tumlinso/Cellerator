#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::execution {
struct device_performance_class;
}

namespace cellerator::planner {
struct device_performance_key;
}

namespace cellerator::runtime {

struct device_performance_class;

inline constexpr std::uint32_t device_descriptor_schema_version_v1 = 1u;
inline constexpr std::uint32_t nvidia_pci_vendor_id = 0x10deu;

enum class device_architecture_class_v1 : std::uint32_t {
    unknown = 0u,
    nvidia_volta = 1u,
    nvidia_turing = 2u,
    nvidia_ampere = 3u,
    nvidia_ada = 4u,
    nvidia_hopper = 5u,
    nvidia_blackwell = 6u,
    nvidia_other = 0xffffu
};

enum class device_descriptor_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    invalid_state = 2u,
    cuda_failure = 3u
};

// Cold, process-local hardware truth. Runtime and planner compatibility
// records are derived from this descriptor; none independently query CUDA.
// Runtime and kernel build identities deliberately do not enter either
// hardware identity below.
struct device_descriptor_v1 {
    std::uint32_t schema_version = device_descriptor_schema_version_v1;
    std::uint32_t vendor = 0u;
    std::int32_t ordinal = -1;
    std::uint16_t compute_major = 0u;
    std::uint16_t compute_minor = 0u;
    device_architecture_class_v1 architecture =
        device_architecture_class_v1::unknown;
    std::uint32_t multiprocessor_count = 0u;
    std::uint32_t warp_size = 0u;
    std::uint32_t maximum_threads_per_block = 0u;
    std::uint32_t maximum_threads_per_multiprocessor = 0u;
    std::uint32_t maximum_blocks_per_multiprocessor = 0u;
    std::uint32_t maximum_thread_dimensions[3]{};
    std::uint32_t maximum_grid_dimensions[3]{};
    std::uint32_t registers_per_block = 0u;
    std::uint32_t registers_per_multiprocessor = 0u;
    std::uint64_t shared_memory_per_block_bytes = 0u;
    std::uint64_t optin_shared_memory_per_block_bytes = 0u;
    std::uint64_t shared_memory_per_multiprocessor_bytes = 0u;
    std::uint64_t global_memory_bytes = 0u;
    std::uint64_t l2_cache_bytes = 0u;
    std::uint64_t hardware_compatibility_identity = 0u;
    std::uint64_t performance_class_identity = 0u;
};

// requested_ordinal < 0 selects the current device. A sealed session is
// rejected before any CUDA query. query_count, when supplied, counts every
// attempted CUDA hardware-discovery call and is suitable for session
// accounting and post-seal tests.
device_descriptor_status_v1 query_device_descriptor_v1(
    std::int32_t requested_ordinal,
    bool session_sealed,
    device_descriptor_v1 *descriptor,
    std::uint64_t *query_count = nullptr) noexcept;

bool valid_device_descriptor_v1(
    const device_descriptor_v1 &descriptor) noexcept;

device_performance_class derive_runtime_device_performance_class(
    const device_descriptor_v1 &descriptor) noexcept;

execution::device_performance_class derive_execution_device_performance_class(
    const device_descriptor_v1 &descriptor) noexcept;

planner::device_performance_key derive_planner_device_performance_key(
    const device_descriptor_v1 &descriptor) noexcept;

static_assert(std::is_trivially_copyable<device_descriptor_v1>::value,
    "device descriptor must remain pointer-free and trivially copyable");

} // namespace cellerator::runtime
