#include <Cellerator/compute/math/runtime.hh>

#include <cstring>
#include <stdexcept>

namespace cellerator::compute::math {

namespace {

void require_initialized(const DeviceMathContext *context, const char *operation) {
    if (context == nullptr) throw std::invalid_argument(operation);
    if (!context->initialized) throw std::logic_error("DeviceMathContext is not initialized");
}

} // namespace

DeviceCapabilities query_device_capabilities(int device_ordinal) {
    if (device_ordinal < 0) {
        runtime::cuda_require(
            cudaGetDevice(&device_ordinal),
            "cudaGetDevice(DeviceCapabilities)");
    }

    cudaDeviceProp properties{};
    runtime::cuda_require(
        cudaGetDeviceProperties(&properties, device_ordinal),
        "cudaGetDeviceProperties(DeviceCapabilities)");

    DeviceCapabilities out;
    out.device_ordinal = device_ordinal;
    out.compute_capability_major = properties.major;
    out.compute_capability_minor = properties.minor;
    out.multiprocessor_count = properties.multiProcessorCount;
    out.warp_size = properties.warpSize;
    out.max_threads_per_block = properties.maxThreadsPerBlock;
    out.total_global_memory_bytes =
        static_cast<std::uint64_t>(properties.totalGlobalMem);
    runtime::cuda_require(
        cudaDriverGetVersion(&out.driver_version),
        "cudaDriverGetVersion(DeviceCapabilities)");
    runtime::cuda_require(
        cudaRuntimeGetVersion(&out.runtime_version),
        "cudaRuntimeGetVersion(DeviceCapabilities)");
    out.tensor_core_capable = properties.major >= 7;
    out.managed_memory = properties.managedMemory != 0;
    out.concurrent_managed_access = properties.concurrentManagedAccess != 0;
    out.cooperative_launch = properties.cooperativeLaunch != 0;
    return out;
}

DeviceFingerprint query_device_fingerprint(
    int device_ordinal,
    const DeviceCapabilities &capabilities) {
    if (device_ordinal < 0) device_ordinal = capabilities.device_ordinal;
    if (device_ordinal < 0 || device_ordinal != capabilities.device_ordinal) {
        throw std::invalid_argument(
            "DeviceFingerprint requires capabilities for the selected device");
    }

    cudaDeviceProp properties{};
    runtime::cuda_require(
        cudaGetDeviceProperties(&properties, device_ordinal),
        "cudaGetDeviceProperties(DeviceFingerprint)");

    DeviceFingerprint out;
    out.device_ordinal = device_ordinal;
    out.pci_domain_id = properties.pciDomainID;
    out.pci_bus_id = properties.pciBusID;
    out.pci_device_id = properties.pciDeviceID;
    out.compute_capability_major = capabilities.compute_capability_major;
    out.compute_capability_minor = capabilities.compute_capability_minor;
    out.total_global_memory_bytes = capabilities.total_global_memory_bytes;
    out.driver_version = capabilities.driver_version;
    out.runtime_version = capabilities.runtime_version;
    out.toolkit_version = capabilities.toolkit_version;
    static_assert(sizeof(out.uuid) == sizeof(properties.uuid.bytes),
        "CUDA UUID width changed");
    std::memcpy(out.uuid, properties.uuid.bytes, sizeof(out.uuid));
    return out;
}

bool same_device_fingerprint(
    const DeviceFingerprint &lhs,
    const DeviceFingerprint &rhs) noexcept {
    return lhs.schema_version == rhs.schema_version
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.pci_domain_id == rhs.pci_domain_id
        && lhs.pci_bus_id == rhs.pci_bus_id
        && lhs.pci_device_id == rhs.pci_device_id
        && lhs.compute_capability_major == rhs.compute_capability_major
        && lhs.compute_capability_minor == rhs.compute_capability_minor
        && lhs.total_global_memory_bytes == rhs.total_global_memory_bytes
        && lhs.driver_version == rhs.driver_version
        && lhs.runtime_version == rhs.runtime_version
        && lhs.toolkit_version == rhs.toolkit_version
        && std::memcmp(lhs.uuid, rhs.uuid, sizeof(lhs.uuid)) == 0;
}

DeviceMathContext::~DeviceMathContext() {
    clear(this);
}

void init(DeviceMathContext *context, int device_ordinal, cudaStream_t stream) {
    if (context == nullptr) {
        throw std::invalid_argument("init(DeviceMathContext) requires a context");
    }
    clear(context);
    runtime::init(&context->cublas);
    runtime::init(&context->cusparse);

    try {
        runtime::init(&context->execution, device_ordinal, stream);
        context->capabilities =
            query_device_capabilities(context->execution.device);
        context->fingerprint = query_device_fingerprint(
            context->execution.device,
            context->capabilities);
        init(&context->workspace, context->execution.device);
        context->initialized = true;
    } catch (...) {
        clear(context);
        throw;
    }
}

void clear(DeviceMathContext *context) noexcept {
    if (context == nullptr) return;
    if (context->execution.device >= 0) {
        (void) cudaSetDevice(context->execution.device);
    }
    runtime::clear(&context->cusparse);
    runtime::clear(&context->cublas);
    clear(&context->workspace);
    runtime::clear(&context->execution);
    context->capabilities = DeviceCapabilities{};
    context->fingerprint = DeviceFingerprint{};
    context->initialized = false;
}

void *request_workspace(DeviceMathContext *context, std::size_t bytes) {
    require_initialized(
        context,
        "request_workspace(DeviceMathContext) requires a context");
    return request_workspace(&context->workspace, bytes);
}

cublasHandle_t acquire_cublas(DeviceMathContext *context) {
    require_initialized(
        context,
        "acquire_cublas(DeviceMathContext) requires a context");
    return runtime::acquire_cublas(&context->cublas, context->execution);
}

cusparseHandle_t acquire_cusparse(DeviceMathContext *context) {
    require_initialized(
        context,
        "acquire_cusparse(DeviceMathContext) requires a context");
    return runtime::acquire_cusparse(&context->cusparse, context->execution);
}

} // namespace cellerator::compute::math
