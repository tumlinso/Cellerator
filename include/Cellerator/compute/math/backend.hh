#pragma once

#include "execution_plan.hh"
#include "operation.hh"
#include "runtime.hh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>

namespace cellerator::compute::math {

inline constexpr u32 backend_abi_schema_version = 1u;

enum class capability_code : u32 {
    supported = 0u,
    invalid_request = 1u,
    unsupported_device = 2u,
    unsupported_type = 3u,
    unsupported_layout = 4u,
    unsupported_transpose = 5u,
    unsupported_epilogue = 6u,
    unsupported_determinism = 7u,
    workspace_policy_rejected = 8u,
    backend_unavailable = 9u
};

struct backend_capability {
    capability_code code = capability_code::supported;
    request_validation_code validation = request_validation_code::ok;
    const char *message = "supported";
    u32 physical_view_schema_version = 0u;
    u64 workspace_bytes = 0u;
    u64 algorithm_identity = 0u;
    u64 kernel_variant_identity = 0u;
    u64 tuning_identity = 0u;
    preprocessing_kind preprocessing = preprocessing_kind::none;
    epilogue_strategy_kind epilogue_strategy = epilogue_strategy_kind::none;

    constexpr explicit operator bool() const noexcept {
        return code == capability_code::supported;
    }
};

enum class backend_status_code : u32 {
    ok = 0u,
    invalid_argument = 1u,
    capability_rejected = 2u,
    not_prepared = 3u,
    backend_mismatch = 4u,
    allocation_during_run = 5u,
    runtime_failure = 6u,
    backend_failure = 7u
};

struct backend_status {
    backend_status_code code = backend_status_code::ok;
    capability_code capability = capability_code::supported;
    request_validation_code validation = request_validation_code::ok;
    cudaError_t cuda_error = cudaSuccess;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == backend_status_code::ok;
    }
};

struct PreparedExecution;

class SpMMBackend {
public:
    virtual ~SpMMBackend() = default;
    virtual u64 identity() const noexcept = 0;
    virtual const char *name() const noexcept = 0;
    virtual backend_capability query(
        const spmm_request &request,
        const DeviceCapabilities &device) const noexcept = 0;
    virtual backend_status prepare(PreparedExecution *prepared) noexcept = 0;
    virtual backend_status run(PreparedExecution *prepared) noexcept = 0;
    virtual void release(PreparedExecution *prepared) noexcept = 0;
};

// Owns all live state needed by a reusable execution. The backend is borrowed
// and must outlive this object; request bindings are copied and remain mutable
// only through a new prepare call.
struct PreparedExecution {
    DeviceMathContext device{};
    execution_plan plan{};
    math_request request{};
    const SpMMBackend *backend = nullptr;
    void *backend_state = nullptr;
    std::size_t prepared_workspace_allocation_count = 0u;
    std::size_t prepared_workspace_capacity_bytes = 0u;
    void *prepared_workspace_pointer = nullptr;
    u64 run_count = 0u;
    bool prepared = false;

    PreparedExecution() = default;
    ~PreparedExecution();
    PreparedExecution(const PreparedExecution &) = delete;
    PreparedExecution &operator=(const PreparedExecution &) = delete;
    PreparedExecution(PreparedExecution &&) = delete;
    PreparedExecution &operator=(PreparedExecution &&) = delete;
};

namespace detail {

inline u64 mix_fingerprint(u64 hash, const void *data, std::size_t bytes) noexcept {
    const auto *cursor = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0u; i < bytes; ++i) {
        hash ^= static_cast<u64>(cursor[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

inline u64 device_fingerprint_identity(const DeviceFingerprint &value) noexcept {
    u64 hash = 1469598103934665603ull;
    hash = mix_fingerprint(hash, &value.schema_version, sizeof(value.schema_version));
    hash = mix_fingerprint(hash, &value.device_ordinal, sizeof(value.device_ordinal));
    hash = mix_fingerprint(hash, &value.pci_domain_id, sizeof(value.pci_domain_id));
    hash = mix_fingerprint(hash, &value.pci_bus_id, sizeof(value.pci_bus_id));
    hash = mix_fingerprint(hash, &value.pci_device_id, sizeof(value.pci_device_id));
    hash = mix_fingerprint(hash, &value.compute_capability_major,
        sizeof(value.compute_capability_major));
    hash = mix_fingerprint(hash, &value.compute_capability_minor,
        sizeof(value.compute_capability_minor));
    hash = mix_fingerprint(hash, &value.total_global_memory_bytes,
        sizeof(value.total_global_memory_bytes));
    hash = mix_fingerprint(hash, value.uuid, sizeof(value.uuid));
    return hash;
}

inline u64 toolchain_fingerprint_identity(const DeviceFingerprint &value) noexcept {
    u64 hash = 1469598103934665603ull;
    hash = mix_fingerprint(hash, &value.driver_version, sizeof(value.driver_version));
    hash = mix_fingerprint(hash, &value.runtime_version, sizeof(value.runtime_version));
    hash = mix_fingerprint(hash, &value.toolkit_version, sizeof(value.toolkit_version));
    return hash;
}

inline backend_status capability_failure(const backend_capability &capability) noexcept {
    return {backend_status_code::capability_rejected,
        capability.code,
        capability.validation,
        cudaSuccess,
        capability.message};
}

} // namespace detail

inline void reset_prepared_execution(PreparedExecution *prepared) noexcept {
    if (prepared == nullptr) return;
    if (prepared->backend != nullptr) {
        const_cast<SpMMBackend *>(prepared->backend)->release(prepared);
    }
    clear(&prepared->device);
    prepared->plan = execution_plan{};
    prepared->request = math_request{};
    prepared->backend = nullptr;
    prepared->backend_state = nullptr;
    prepared->prepared_workspace_allocation_count = 0u;
    prepared->prepared_workspace_capacity_bytes = 0u;
    prepared->prepared_workspace_pointer = nullptr;
    prepared->run_count = 0u;
    prepared->prepared = false;
}

inline PreparedExecution::~PreparedExecution() {
    reset_prepared_execution(this);
}

inline backend_status prepare_execution(
    PreparedExecution *prepared,
    SpMMBackend *backend,
    const math_request &request,
    int device_ordinal = -1,
    cudaStream_t stream = nullptr) noexcept {
    if (prepared == nullptr || backend == nullptr) {
        return {backend_status_code::invalid_argument,
            capability_code::supported,
            request_validation_code::ok,
            cudaSuccess,
            "prepare_execution requires output and backend"};
    }
    reset_prepared_execution(prepared);

    const request_validation_result validation = validate_math_request(request);
    if (!validation) {
        return {backend_status_code::capability_rejected,
            capability_code::invalid_request,
            validation.code,
            cudaSuccess,
            validation.message};
    }

    DeviceCapabilities capabilities;
    try {
        capabilities = query_device_capabilities(device_ordinal);
    } catch (...) {
        return {backend_status_code::runtime_failure,
            capability_code::unsupported_device,
            request_validation_code::ok,
            cudaErrorUnknown,
            "CUDA device capability query failed"};
    }

    backend_capability capability = backend->query(request.operation, capabilities);
    if (!capability) return detail::capability_failure(capability);
    const workspace_policy &workspace = request.operation.workspace;
    if ((workspace.kind == workspace_policy_kind::no_additional_workspace
            && capability.workspace_bytes != 0u)
        || (workspace.kind == workspace_policy_kind::caller_limit
            && capability.workspace_bytes > workspace.byte_limit)) {
        capability.code = capability_code::workspace_policy_rejected;
        capability.message = "backend workspace exceeds request policy";
        return detail::capability_failure(capability);
    }

    try {
        init(&prepared->device, capabilities.device_ordinal, stream);
        void *workspace_pointer = request_workspace(
            &prepared->device,
            static_cast<std::size_t>(capability.workspace_bytes));
        prepared->request = request;
        prepared->backend = backend;
        prepared->plan.operation = make_operation_signature(request.operation);
        prepared->plan.physical_view_schema_version =
            capability.physical_view_schema_version;
        prepared->plan.backend_identity = backend->identity();
        prepared->plan.algorithm_identity = capability.algorithm_identity;
        prepared->plan.kernel_variant_identity = capability.kernel_variant_identity;
        prepared->plan.workspace_bytes = capability.workspace_bytes;
        prepared->plan.preprocessing = capability.preprocessing;
        prepared->plan.epilogue_strategy = capability.epilogue_strategy;
        prepared->plan.device_fingerprint =
            detail::device_fingerprint_identity(prepared->device.fingerprint);
        prepared->plan.toolchain_fingerprint =
            detail::toolchain_fingerprint_identity(prepared->device.fingerprint);
        prepared->plan.tuning_identity = capability.tuning_identity;
        prepared->backend_state = workspace_pointer;

        backend_status status = backend->prepare(prepared);
        if (!status) {
            reset_prepared_execution(prepared);
            return status;
        }
        prepared->prepared_workspace_allocation_count =
            prepared->device.workspace.allocation_count;
        prepared->prepared_workspace_capacity_bytes =
            prepared->device.workspace.storage.bytes;
        prepared->prepared_workspace_pointer = prepared->device.workspace.storage.data;
        prepared->prepared = true;
        return {};
    } catch (...) {
        reset_prepared_execution(prepared);
        return {backend_status_code::runtime_failure,
            capability_code::supported,
            request_validation_code::ok,
            cudaErrorUnknown,
            "CUDA runtime initialization or workspace reservation failed"};
    }
}

inline backend_status run_prepared_execution(PreparedExecution *prepared) noexcept {
    if (prepared == nullptr || !prepared->prepared || prepared->backend == nullptr) {
        return {backend_status_code::not_prepared,
            capability_code::supported,
            request_validation_code::ok,
            cudaSuccess,
            "execution is not prepared"};
    }
    SpMMBackend *const backend = const_cast<SpMMBackend *>(prepared->backend);
    backend_status status = backend->run(prepared);
    if (!status) return status;
    if (prepared->device.workspace.allocation_count
            != prepared->prepared_workspace_allocation_count
        || prepared->device.workspace.storage.bytes
            != prepared->prepared_workspace_capacity_bytes
        || prepared->device.workspace.storage.data
            != prepared->prepared_workspace_pointer) {
        return {backend_status_code::allocation_during_run,
            capability_code::supported,
            request_validation_code::ok,
            cudaSuccess,
            "backend changed reusable workspace during run"};
    }
    ++prepared->run_count;
    return {};
}

backend_capability query_generic_unfused_epilogue_capability(
    const spmm_request &request) noexcept;
backend_status launch_generic_unfused_epilogue(
    DeviceMathContext *context,
    const spmm_request &request,
    const spmm_bindings &bindings) noexcept;

} // namespace cellerator::compute::math
