#include <Cellerator/compute/math/backend.hh>
#include <Cellerator/types.cuh>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

namespace cm = cellerator::compute::math;
namespace cr = cellerator::real;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpMathRuntimeTest: " << message << '\n';
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "cpMathRuntimeTest: " << message << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

cm::feature_order_identity canonical_order(cm::u32 features) {
    cm::feature_order_identity order;
    order.feature_count = features;
    order.feature_axis_identity_version = 1u;
    order.feature_axis_identity = 0x63706d6174683032ull;
    return order;
}

cm::math_request request_fixture(
    float *output,
    const float *bias,
    cm::dense_layout_kind layout,
    cm::epilogue_kind epilogue) {
    static int sparse = 0;
    static int dense = 0;
    cm::math_request request;
    request.operation.m = 2u;
    request.operation.k = 4u;
    request.operation.n = 3u;
    request.operation.sparse_nnz = 4u;
    request.operation.sparse_structure.identity_version = 1u;
    request.operation.sparse_structure.value = 0x02u;
    request.operation.dense_rhs_leading_dimension = 3u;
    request.operation.output_leading_dimension =
        layout == cm::dense_layout_kind::row_major ? 3u : 2u;
    request.operation.output_layout = layout;
    request.operation.sparse_storage_type_code = cr::value_f32;
    request.operation.dense_storage_type_code = cr::value_f32;
    request.operation.output_storage_type_code = cr::value_f32;
    request.operation.compute_type_code = cr::value_f32;
    request.operation.accumulation_type_code = cr::value_f32;
    request.operation.alpha = cm::make_scalar(1.0f);
    request.operation.beta = cm::make_scalar(0.0f);
    request.operation.reuse.kind = cm::expected_reuse_kind::persistent;
    request.operation.reuse.expected_run_count = 0u;
    request.operation.epilogue.kind = epilogue;
    if (bias != nullptr) {
        request.operation.epilogue.bias_type_code = cr::value_f32;
        request.operation.epilogue.bias_element_count = request.operation.n;
    }
    request.operation.sparse_feature_order = canonical_order(4u);
    request.operation.dense_feature_order = request.operation.sparse_feature_order;
    request.bindings.sparse_matrix = &sparse;
    request.bindings.dense_rhs = &dense;
    request.bindings.output = output;
    request.bindings.bias = bias;
    return request;
}

class EpilogueBackend final : public cm::SpMMBackend {
public:
    cm::u64 identity() const noexcept override {
        return 0x43504d4154483032ull;
    }

    const char *name() const noexcept override {
        return "cp-math-test-epilogue";
    }

    cm::backend_capability query(
        const cm::spmm_request &request,
        const cm::DeviceCapabilities &) const noexcept override {
        cm::backend_capability result =
            cm::query_generic_unfused_epilogue_capability(request);
        if (result) result.workspace_bytes = 4096u;
        return result;
    }

    cm::backend_status prepare(cm::PreparedExecution *prepared) noexcept override {
        if (prepared == nullptr || prepared->backend_state == nullptr) {
            return {cm::backend_status_code::backend_failure,
                cm::capability_code::supported,
                cm::request_validation_code::ok,
                cudaSuccess,
                "test backend did not receive prepared workspace"};
        }
        return {};
    }

    cm::backend_status run(cm::PreparedExecution *prepared) noexcept override {
        return cm::launch_generic_unfused_epilogue(
            &prepared->device,
            prepared->request.operation,
            prepared->request.bindings);
    }

    void release(cm::PreparedExecution *prepared) noexcept override {
        if (prepared != nullptr) prepared->backend_state = nullptr;
    }
};

float exact_gelu(float value) {
    return 0.5f * value
        * (1.0f + std::erf(value * 0.7071067811865475244f));
}

float tanh_gelu(float value) {
    return 0.5f * value
        * (1.0f + std::tanh(0.7978845608028653559f
            * (value + 0.044715f * value * value * value)));
}

void require_close(float actual, float expected, float tolerance, const char *message) {
    require(std::fabs(actual - expected) <= tolerance, message);
}

void test_structured_rejection(EpilogueBackend &backend) {
    float *device_output = nullptr;
    cuda_require(cudaMalloc(&device_output, 6u * sizeof(float)), "allocate rejection output");
    cm::math_request request = request_fixture(
        device_output, nullptr, cm::dense_layout_kind::row_major,
        cm::epilogue_kind::relu);
    request.operation.output_storage_type_code = cr::value_f64;
    cm::PreparedExecution prepared;
    const cm::backend_status unsupported =
        cm::prepare_execution(&prepared, &backend, request);
    require(unsupported.code == cm::backend_status_code::capability_rejected
            && unsupported.capability == cm::capability_code::unsupported_type,
        "unsupported output type was not rejected structurally");

    request.operation.output_storage_type_code = cr::value_f32;
    request.operation.workspace.kind =
        cm::workspace_policy_kind::no_additional_workspace;
    const cm::backend_status workspace =
        cm::prepare_execution(&prepared, &backend, request);
    require(workspace.code == cm::backend_status_code::capability_rejected
            && workspace.capability
                == cm::capability_code::workspace_policy_rejected,
        "workspace policy rejection was not structured");
    cuda_require(cudaFree(device_output), "free rejection output");
}

void test_repeated_run_without_allocation(EpilogueBackend &backend) {
    const std::vector<float> input{-2.0f, -0.5f, 1.0f, 2.0f, -3.0f, 4.0f};
    const std::vector<float> bias{1.0f, -1.0f, 0.5f};
    float *device_output = nullptr;
    float *device_bias = nullptr;
    cuda_require(cudaMalloc(&device_output, input.size() * sizeof(float)),
        "allocate output");
    cuda_require(cudaMalloc(&device_bias, bias.size() * sizeof(float)),
        "allocate bias");
    cuda_require(cudaMemcpy(device_output, input.data(), input.size() * sizeof(float),
        cudaMemcpyHostToDevice), "copy output input");
    cuda_require(cudaMemcpy(device_bias, bias.data(), bias.size() * sizeof(float),
        cudaMemcpyHostToDevice), "copy bias");

    cm::math_request request = request_fixture(
        device_output, device_bias, cm::dense_layout_kind::row_major,
        cm::epilogue_kind::bias_relu);
    cm::PreparedExecution prepared;
    require(static_cast<bool>(cm::prepare_execution(&prepared, &backend, request)),
        "prepare failed");
    const std::size_t allocations = prepared.device.workspace.allocation_count;
    void *const workspace = prepared.device.workspace.storage.data;
    require(static_cast<bool>(cm::run_prepared_execution(&prepared)),
        "first prepared run failed");
    require(prepared.device.workspace.allocation_count == allocations
            && prepared.device.workspace.storage.data == workspace,
        "first run changed workspace");
    require(static_cast<bool>(cm::run_prepared_execution(&prepared)),
        "second prepared run failed");
    require(prepared.run_count == 2u
            && prepared.device.workspace.allocation_count == allocations
            && prepared.device.workspace.storage.data == workspace,
        "repeated run allocated or did not update run count");

    std::vector<float> output(input.size());
    cuda_require(cudaMemcpy(output.data(), device_output,
        output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy row output");
    for (std::size_t i = 0u; i < output.size(); ++i) {
        const float once = std::fmax(input[i] + bias[i % 3u], 0.0f);
        const float twice = std::fmax(once + bias[i % 3u], 0.0f);
        require_close(output[i], twice, 2e-6f, "row-major bias ReLU mismatch");
    }
    cuda_require(cudaFree(device_bias), "free bias");
    cuda_require(cudaFree(device_output), "free output");
}

void test_column_major_gelu(EpilogueBackend &backend, cm::epilogue_kind kind) {
    const std::vector<float> input{-2.0f, 2.0f, -0.5f, -3.0f, 1.0f, 4.0f};
    float *device_output = nullptr;
    cuda_require(cudaMalloc(&device_output, input.size() * sizeof(float)),
        "allocate column output");
    cuda_require(cudaMemcpy(device_output, input.data(), input.size() * sizeof(float),
        cudaMemcpyHostToDevice), "copy column input");
    cm::math_request request = request_fixture(
        device_output, nullptr, cm::dense_layout_kind::column_major, kind);
    cm::PreparedExecution prepared;
    require(static_cast<bool>(cm::prepare_execution(&prepared, &backend, request)),
        "column GELU prepare failed");
    require(static_cast<bool>(cm::run_prepared_execution(&prepared)),
        "column GELU run failed");
    std::vector<float> output(input.size());
    cuda_require(cudaMemcpy(output.data(), device_output,
        output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy column output");
    for (std::size_t i = 0u; i < output.size(); ++i) {
        const float expected = kind == cm::epilogue_kind::gelu_exact_erf
            ? exact_gelu(input[i]) : tanh_gelu(input[i]);
        require_close(output[i], expected,
            kind == cm::epilogue_kind::gelu_exact_erf ? 2e-6f : 2e-5f,
            "column-major GELU mismatch");
    }
    cuda_require(cudaFree(device_output), "free column output");
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible<cm::PreparedExecution>::value,
        "PreparedExecution must be unique-owner");
    static_assert(!std::is_move_constructible<cm::PreparedExecution>::value,
        "PreparedExecution address must remain stable for backend state");
    EpilogueBackend backend;
    test_structured_rejection(backend);
    test_repeated_run_without_allocation(backend);
    test_column_major_gelu(backend, cm::epilogue_kind::gelu_exact_erf);
    test_column_major_gelu(backend, cm::epilogue_kind::gelu_tanh_approximate);
    std::cout << "cpMathRuntimeTest passed\n";
    return 0;
}
