#include <Cellerator/compute/math/runtime.hh>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <type_traits>

namespace cm = cellerator::compute::math;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpMathDeviceRuntimeTest: " << message << '\n';
        std::exit(1);
    }
}

void test_cached_device_facts(const cm::DeviceMathContext &context) {
    const cm::DeviceCapabilities &capabilities = context.capabilities;
    require(capabilities.device_ordinal == context.execution.device,
        "capabilities device does not match the execution context");
    require(capabilities.compute_capability_major >= 7,
        "native device is not Tensor Core capable");
    require(capabilities.tensor_core_capable,
        "Tensor Core capability was not cached");
    require(capabilities.warp_size == 32,
        "unexpected CUDA warp width");
    require(capabilities.total_global_memory_bytes != 0u,
        "global-memory capacity was not cached");
    require(capabilities.runtime_version != 0
            && capabilities.driver_version != 0
            && capabilities.toolkit_version != 0,
        "CUDA version facts were not cached");

    const cm::DeviceFingerprint queried = cm::query_device_fingerprint(
        context.execution.device,
        capabilities);
    require(cm::same_device_fingerprint(context.fingerprint, queried),
        "device fingerprint changed across repeated queries");
}

void test_library_handle_reuse(cm::DeviceMathContext *context) {
    const cublasHandle_t cublas_first = cm::acquire_cublas(context);
    const cublasHandle_t cublas_second = cm::acquire_cublas(context);
    require(cublas_first != nullptr && cublas_first == cublas_second,
        "cuBLAS handle was recreated");

    const cusparseHandle_t cusparse_first = cm::acquire_cusparse(context);
    const cusparseHandle_t cusparse_second = cm::acquire_cusparse(context);
    require(cusparse_first != nullptr && cusparse_first == cusparse_second,
        "cuSPARSE handle was recreated");
}

void test_workspace_reuse(cm::DeviceMathContext *context) {
    void *const first = cm::request_workspace(context, 4096u);
    require(first != nullptr, "workspace allocation returned null");
    require(context->workspace.allocation_count == 1u,
        "first workspace request was not counted");

    void *const same = cm::request_workspace(context, 4096u);
    void *const smaller = cm::request_workspace(context, 1024u);
    require(same == first && smaller == first,
        "equal or smaller requests did not reuse workspace");
    require(context->workspace.allocation_count == 1u,
        "reused workspace performed another allocation");

    void *const grown = cm::request_workspace(context, 8192u);
    require(grown != nullptr, "grown workspace allocation returned null");
    require(context->workspace.allocation_count == 2u,
        "workspace growth was not counted exactly once");
    require(context->workspace.high_watermark_bytes >= 8192u,
        "workspace high watermark was not updated");
    require(cm::request_workspace(context, 8192u) == grown,
        "grown workspace was not reused");
    require(context->workspace.allocation_count == 2u,
        "steady-state workspace performed a repeated allocation");
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible<cm::DeviceMathContext>::value,
        "DeviceMathContext must have unique ownership");
    static_assert(!std::is_copy_constructible<cm::WorkspacePool>::value,
        "WorkspacePool must have unique ownership");

    cm::DeviceMathContext context;
    cm::init(&context, 0);
    require(context.initialized, "device math context did not initialize");
    test_cached_device_facts(context);
    test_library_handle_reuse(&context);
    test_workspace_reuse(&context);
    cm::clear(&context);
    require(!context.initialized, "device math context did not clear");

    std::cout << "cpMathDeviceRuntimeTest passed\n";
    return 0;
}
