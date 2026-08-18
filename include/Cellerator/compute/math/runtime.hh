#pragma once

#include <Cellerator/runtime/runtime.cuh>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::math {

inline constexpr std::uint32_t device_math_runtime_schema_version = 1u;

// Cached control-plane facts. They are captured once when the context is
// initialized and are not rediscovered by prepared-operation run calls.
struct DeviceCapabilities {
    std::uint32_t schema_version = device_math_runtime_schema_version;
    int device_ordinal = -1;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multiprocessor_count = 0;
    int warp_size = 0;
    int max_threads_per_block = 0;
    std::uint64_t total_global_memory_bytes = 0u;
    int driver_version = 0;
    int runtime_version = 0;
    int toolkit_version = CUDART_VERSION;
    bool tensor_core_capable = false;
    bool managed_memory = false;
    bool concurrent_managed_access = false;
    bool cooperative_launch = false;
};

// Stable physical-device and CUDA-environment identity. Live handles, streams,
// pointers, clock rates, and other mutable runtime state never participate.
struct DeviceFingerprint {
    std::uint32_t schema_version = device_math_runtime_schema_version;
    int device_ordinal = -1;
    int pci_domain_id = -1;
    int pci_bus_id = -1;
    int pci_device_id = -1;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::uint64_t total_global_memory_bytes = 0u;
    int driver_version = 0;
    int runtime_version = 0;
    int toolkit_version = CUDART_VERSION;
    unsigned char uuid[16]{};
};

// One device-affine reusable allocation. Backend plans may reserve bytes from
// it, but allocation shape and suballocation policy remain outside this owner.
struct WorkspacePool {
    runtime::scratch_arena storage{};
    int device_ordinal = -1;
    std::size_t allocation_count = 0u;
    std::size_t high_watermark_bytes = 0u;

    WorkspacePool() = default;
    ~WorkspacePool();
    WorkspacePool(const WorkspacePool &) = delete;
    WorkspacePool &operator=(const WorkspacePool &) = delete;
    WorkspacePool(WorkspacePool &&) = delete;
    WorkspacePool &operator=(WorkspacePool &&) = delete;
};

// DeviceMathContext owns CUDA stream, library-handle caches, and reusable
// workspace for one device. It is single-owner and not concurrently mutable.
struct DeviceMathContext {
    runtime::execution_context execution{};
    runtime::cublas_cache cublas{};
    runtime::cusparse_cache cusparse{};
    WorkspacePool workspace{};
    DeviceCapabilities capabilities{};
    DeviceFingerprint fingerprint{};
    bool initialized = false;

    DeviceMathContext() = default;
    ~DeviceMathContext();
    DeviceMathContext(const DeviceMathContext &) = delete;
    DeviceMathContext &operator=(const DeviceMathContext &) = delete;
    DeviceMathContext(DeviceMathContext &&) = delete;
    DeviceMathContext &operator=(DeviceMathContext &&) = delete;
};

DeviceCapabilities query_device_capabilities(int device_ordinal);
DeviceFingerprint query_device_fingerprint(
    int device_ordinal,
    const DeviceCapabilities &capabilities);
bool same_device_fingerprint(
    const DeviceFingerprint &lhs,
    const DeviceFingerprint &rhs) noexcept;

void init(WorkspacePool *pool, int device_ordinal);
void clear(WorkspacePool *pool) noexcept;
void *request_workspace(WorkspacePool *pool, std::size_t bytes);

void init(
    DeviceMathContext *context,
    int device_ordinal = -1,
    cudaStream_t stream = nullptr);
void clear(DeviceMathContext *context) noexcept;
void *request_workspace(DeviceMathContext *context, std::size_t bytes);
cublasHandle_t acquire_cublas(DeviceMathContext *context);
cusparseHandle_t acquire_cusparse(DeviceMathContext *context);

} // namespace cellerator::compute::math
