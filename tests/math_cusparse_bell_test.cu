#include "../src/compute/math/backends/cusparse_bell.hh"

#include <Cellerator/types.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace cm = cellerator::compute::math;
namespace cr = cellerator::real;

namespace {

constexpr cm::u32 rows = 64u;
constexpr cm::u32 columns = 64u;
constexpr cm::u32 rhs_columns = 32u;
constexpr cm::u64 geometry_identity = 0x43504d4154483037ull;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpMathCusparseBellTest: " << message << '\n';
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "cpMathCusparseBellTest: " << message << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

cm::feature_order_identity packed_order(
    cm::u32 feature_count = columns,
    cm::u64 packing_geometry = geometry_identity) {
    cm::feature_order_identity result;
    result.kind = cm::feature_order_kind::packed;
    result.feature_count = feature_count;
    result.feature_axis_identity_version = 1u;
    result.feature_axis_identity = 0x4645415455524537ull;
    result.packing_geometry_identity = packing_geometry;
    return result;
}

void test_padded_candidate_rejection() {
    struct padded_case {
        cm::u32 block_size;
        cm::u32 logical_rows;
        cm::u32 logical_features;
        cm::u32 padded_rows;
        cm::u32 padded_features;
    };
    constexpr std::array<padded_case, 3u> cases{{
        {8u, 9u, 13u, 16u, 16u},
        {16u, 17u, 31u, 32u, 32u},
        {32u, 33u, 65u, 64u, 96u}
    }};
    constexpr cm::u32 guard_elements = 16u;
    constexpr cm::u32 test_columns = 3u;
    const cm::DeviceCapabilities device = cm::query_device_capabilities(-1);

    for (const padded_case &test : cases) {
        const std::size_t dense_elements = guard_elements
            + static_cast<std::size_t>(test.logical_features) * test_columns
            + guard_elements;
        const std::size_t output_elements = guard_elements
            + static_cast<std::size_t>(test.logical_rows) * test_columns
            + guard_elements;
        const __half dense_canary = __float2half(-37.0f);
        const float output_canary = -73.0f;
        std::vector<__half> host_dense(dense_elements, dense_canary);
        std::vector<float> host_output(output_elements, output_canary);
        __half *device_dense_base = nullptr;
        float *device_output_base = nullptr;
        cuda_require(cudaMalloc(&device_dense_base,
            dense_elements * sizeof(*device_dense_base)),
            "allocate padded-rejection dense guards");
        cuda_require(cudaMalloc(&device_output_base,
            output_elements * sizeof(*device_output_base)),
            "allocate padded-rejection output guards");
        cuda_require(cudaMemcpy(device_dense_base, host_dense.data(),
            dense_elements * sizeof(*device_dense_base), cudaMemcpyHostToDevice),
            "initialize padded-rejection dense guards");
        cuda_require(cudaMemcpy(device_output_base, host_output.data(),
            output_elements * sizeof(*device_output_base), cudaMemcpyHostToDevice),
            "initialize padded-rejection output guards");

        cm::physical_bell_view view;
        view.block_size = test.block_size;
        view.row_count = test.logical_rows;
        view.feature_count = test.logical_features;
        view.padded_row_count = test.padded_rows;
        view.padded_feature_count = test.padded_features;
        view.ell_columns = test.block_size;
        view.value_size_bytes = sizeof(__half);
        view.feature_block_geometry_identity = geometry_identity + test.block_size;
        view.ordering_identity = 0x5041444445444f52ull + test.block_size;
        view.row_domain_identity = 0x504144444544524full + test.block_size;
        view.candidate_identity = 0x5041444445440000ull + test.block_size;
        view.column_indices = reinterpret_cast<const std::int32_t *>(
            device_output_base);
        view.values = device_dense_base;

        cm::math_request request;
        request.operation.m = test.logical_rows;
        request.operation.k = test.logical_features;
        request.operation.n = test_columns;
        request.operation.sparse_nnz = 1u;
        request.operation.sparse_structure.identity_version = 1u;
        request.operation.sparse_structure.value = view.candidate_identity;
        request.operation.dense_rhs_leading_dimension = test_columns;
        request.operation.output_leading_dimension = test_columns;
        request.operation.sparse_storage_type_code = cr::value_f16;
        request.operation.dense_storage_type_code = cr::value_f16;
        request.operation.output_storage_type_code = cr::value_f32;
        request.operation.compute_type_code = cr::value_f32;
        request.operation.accumulation_type_code = cr::value_f32;
        request.operation.alpha = cm::make_scalar(1.0f);
        request.operation.beta = cm::make_scalar(0.0f);
        request.operation.sparse_feature_order = packed_order(
            test.logical_features, view.feature_block_geometry_identity);
        request.operation.dense_feature_order =
            request.operation.sparse_feature_order;
        request.bindings.sparse_matrix = &view;
        request.bindings.dense_rhs = device_dense_base + guard_elements;
        request.bindings.output = device_output_base + guard_elements;
        request.bindings.sparse_matrix_identity = view.candidate_identity;
        request.bindings.dense_rhs_identity = 0x50414444454e5345ull;
        request.bindings.output_identity = 0x5041444f55545055ull;

        cm::CusparseBellBackend backend(view);
        const cm::backend_capability capability =
            backend.query(request.operation, device);
        require(capability.code == cm::capability_code::unsupported_layout,
            "padded BELL capability was not rejected");
        cm::PreparedExecution prepared;
        const cm::backend_status status =
            cm::prepare_execution(&prepared, &backend, request);
        require(status.code == cm::backend_status_code::capability_rejected
                && status.capability == cm::capability_code::unsupported_layout
                && !prepared.prepared && prepared.backend_state == nullptr
                && !prepared.device.initialized,
            "padded BELL reached preparation state");

        std::vector<__half> observed_dense(dense_elements);
        std::vector<float> observed_output(output_elements);
        cuda_require(cudaMemcpy(observed_dense.data(), device_dense_base,
            dense_elements * sizeof(*device_dense_base), cudaMemcpyDeviceToHost),
            "read padded-rejection dense guards");
        cuda_require(cudaMemcpy(observed_output.data(), device_output_base,
            output_elements * sizeof(*device_output_base), cudaMemcpyDeviceToHost),
            "read padded-rejection output guards");
        for (const __half value : observed_dense) {
            require(__half2float(value) == __half2float(dense_canary),
                "padded BELL rejection changed dense guard storage");
        }
        for (const float value : observed_output) {
            require(value == output_canary,
                "padded BELL rejection changed output guard storage");
        }
        cuda_require(cudaFree(device_output_base),
            "free padded-rejection output guards");
        cuda_require(cudaFree(device_dense_base),
            "free padded-rejection dense guards");
    }
}

cm::math_request request_for(
    const cm::physical_bell_view *view,
    const __half *dense_rhs,
    float *output,
    cm::dense_layout_kind layout) {
    cm::math_request result;
    result.operation.m = rows;
    result.operation.k = columns;
    result.operation.n = rhs_columns;
    result.operation.sparse_nnz = rows;
    result.operation.sparse_structure.identity_version = 1u;
    result.operation.sparse_structure.value = view->candidate_identity;
    result.operation.dense_rhs_layout = layout;
    result.operation.output_layout = layout;
    result.operation.dense_rhs_leading_dimension =
        layout == cm::dense_layout_kind::row_major ? rhs_columns : columns;
    result.operation.output_leading_dimension =
        layout == cm::dense_layout_kind::row_major ? rhs_columns : rows;
    result.operation.sparse_storage_type_code = cr::value_f16;
    result.operation.dense_storage_type_code = cr::value_f16;
    result.operation.output_storage_type_code = cr::value_f32;
    result.operation.compute_type_code = cr::value_f32;
    result.operation.accumulation_type_code = cr::value_f32;
    result.operation.alpha = cm::make_scalar(1.25f);
    result.operation.beta = cm::make_scalar(0.0f);
    result.operation.reuse.kind = cm::expected_reuse_kind::persistent;
    result.operation.reuse.expected_run_count = 0u;
    result.operation.sparse_feature_order = packed_order();
    result.operation.dense_feature_order = result.operation.sparse_feature_order;
    result.bindings.sparse_matrix = view;
    result.bindings.dense_rhs = dense_rhs;
    result.bindings.output = output;
    result.bindings.sparse_matrix_identity = view->candidate_identity;
    result.bindings.dense_rhs_identity = 0x44454e5345524853ull;
    result.bindings.output_identity = 0x4f55545055543037ull;
    return result;
}

cm::u64 test_candidate(
    cm::u32 block_size,
    cm::dense_layout_kind layout = cm::dense_layout_kind::column_major,
    cm::u32 ell_block_count = 2u) {
    const std::size_t block_rows = rows / block_size;
    const cm::u32 ell_columns = ell_block_count * block_size;
    const std::size_t blocks_per_row = ell_columns / block_size;
    std::vector<std::int32_t> host_columns(
        block_rows * blocks_per_row, -1);
    std::vector<__half> host_values(
        static_cast<std::size_t>(rows) * ell_columns, __float2half(0.0f));
    for (std::size_t block_row = 0u; block_row < block_rows; ++block_row) {
        host_columns[block_row * blocks_per_row]
            = static_cast<std::int32_t>(block_row);
        for (cm::u32 lane = 0u; lane < block_size; ++lane) {
            const std::size_t value = block_row * blocks_per_row
                * block_size * block_size
                + static_cast<std::size_t>(lane) * block_size + lane;
            host_values[value] = __float2half(1.0f);
        }
    }

    std::vector<__half> host_rhs(rows * rhs_columns);
    std::vector<float> host_output(rows * rhs_columns, 2.0f);
    for (cm::u32 row = 0u; row < rows; ++row) {
        for (cm::u32 column = 0u; column < rhs_columns; ++column) {
            const std::size_t index = layout == cm::dense_layout_kind::row_major
                ? static_cast<std::size_t>(row) * rhs_columns + column
                : static_cast<std::size_t>(column) * rows + row;
            host_rhs[index] = __float2half(
                static_cast<float>((row % 5u) + column + 1u));
        }
    }

    std::int32_t *device_columns = nullptr;
    __half *device_values = nullptr, *device_rhs = nullptr;
    float *device_output = nullptr;
    cuda_require(cudaMalloc(&device_columns,
        host_columns.size() * sizeof(*device_columns)), "allocate BELL columns");
    cuda_require(cudaMalloc(&device_values,
        host_values.size() * sizeof(*device_values)), "allocate BELL values");
    cuda_require(cudaMalloc(&device_rhs,
        host_rhs.size() * sizeof(*device_rhs)), "allocate BELL RHS");
    cuda_require(cudaMalloc(&device_output,
        host_output.size() * sizeof(*device_output)), "allocate BELL output");
    cuda_require(cudaMemcpy(device_columns, host_columns.data(),
        host_columns.size() * sizeof(*device_columns), cudaMemcpyHostToDevice),
        "copy BELL columns");
    cuda_require(cudaMemcpy(device_values, host_values.data(),
        host_values.size() * sizeof(*device_values), cudaMemcpyHostToDevice),
        "copy BELL values");
    cuda_require(cudaMemcpy(device_rhs, host_rhs.data(),
        host_rhs.size() * sizeof(*device_rhs), cudaMemcpyHostToDevice),
        "copy BELL RHS");
    cuda_require(cudaMemcpy(device_output, host_output.data(),
        host_output.size() * sizeof(*device_output), cudaMemcpyHostToDevice),
        "copy BELL output");

    cm::physical_bell_view view;
    view.block_size = block_size;
    view.row_count = rows;
    view.feature_count = columns;
    view.padded_row_count = rows;
    view.padded_feature_count = columns;
    view.ell_columns = ell_columns;
    view.value_size_bytes = sizeof(__half);
    view.feature_block_geometry_identity = geometry_identity;
    view.ordering_identity = 0x4f52444552494e47ull;
    view.row_domain_identity = 0x524f57444f4d4149ull;
    view.candidate_identity = 0x42454c4c00000000ull
        | (static_cast<cm::u64>(ell_block_count) << 8u) | block_size;
    view.column_indices = device_columns;
    view.values = device_values;

    cm::CusparseBellBackend backend(view);
    cm::math_request request = request_for(
        &view, device_rhs, device_output, layout);
    const cm::DeviceCapabilities device = cm::query_device_capabilities(-1);
    const cm::backend_capability capability = backend.query(request.operation, device);
    require(capability
            && capability.physical_view_schema_version
                == cm::physical_bell_schema_version
            && capability.kernel_variant_identity != 0u
            && capability.tuning_identity == view.candidate_identity,
        "legal BELL candidate was not advertised");

    cm::PreparedExecution prepared;
    require(static_cast<bool>(cm::prepare_execution(
        &prepared, &backend, request)), "BELL prepare failed");
    const std::size_t allocations = prepared.device.workspace.allocation_count;
    void *const workspace = prepared.device.workspace.storage.data;
    require(static_cast<bool>(cm::run_prepared_execution(&prepared)),
        "first BELL run failed");
    require(static_cast<bool>(cm::run_prepared_execution(&prepared)),
        "second BELL run failed");
    require(prepared.run_count == 2u
            && prepared.device.workspace.allocation_count == allocations
            && prepared.device.workspace.storage.data == workspace,
        "repeated BELL run changed prepared workspace");
    require(prepared.plan.workspace_bytes
            == prepared.device.workspace.storage.bytes,
        "BELL plan did not record exact prepared workspace");

    cuda_require(cudaMemcpy(host_output.data(), device_output,
        host_output.size() * sizeof(*device_output), cudaMemcpyDeviceToHost),
        "copy BELL result");
    std::size_t mismatch_count = 0u;
    for (std::size_t index = 0u; index < host_output.size(); ++index) {
        const float rhs = __half2float(host_rhs[index]);
        const float expected = 1.25f * rhs;
        if (std::fabs(host_output[index] - expected) >= 2.0e-5f) {
            if (mismatch_count < 12u) {
                std::cerr << "cpMathCusparseBellTest: block=" << block_size
                          << " layout=" << static_cast<cm::u32>(layout)
                          << " index=" << index << " actual=" << host_output[index]
                          << " expected=" << expected << '\n';
            }
            ++mismatch_count;
        }
    }
    require(mismatch_count == 0u,
        "BELL result disagrees with f32 accumulation reference");

    const cm::u64 completed_runs = prepared.run_count;
    const cm::u64 original_structure =
        prepared.request.operation.sparse_structure.value;
    prepared.request.operation.sparse_structure.value += 1u;
    cm::backend_status stale = cm::run_prepared_execution(&prepared);
    require(stale.code == cm::backend_status_code::backend_mismatch
            && prepared.run_count == completed_runs,
        "stale sparse structure identity reached BELL launch");
    prepared.request.operation.sparse_structure.value = original_structure;

    const cm::u64 original_axis =
        prepared.request.operation.sparse_feature_order.feature_axis_identity;
    prepared.request.operation.sparse_feature_order.feature_axis_identity += 1u;
    prepared.request.operation.dense_feature_order.feature_axis_identity += 1u;
    stale = cm::run_prepared_execution(&prepared);
    require(stale.code == cm::backend_status_code::backend_mismatch
            && prepared.run_count == completed_runs,
        "stale execution order reached BELL launch");
    prepared.request.operation.sparse_feature_order.feature_axis_identity =
        original_axis;
    prepared.request.operation.dense_feature_order.feature_axis_identity =
        original_axis;

    const void *const original_dense = prepared.request.bindings.dense_rhs;
    prepared.request.bindings.dense_rhs = device_values;
    stale = cm::run_prepared_execution(&prepared);
    require(stale.code == cm::backend_status_code::backend_mismatch
            && prepared.run_count == completed_runs,
        "changed dense binding reached BELL launch");
    prepared.request.bindings.dense_rhs = original_dense;

    view.feature_block_geometry_identity += 1u;
    stale = cm::run_prepared_execution(&prepared);
    require(stale.code == cm::backend_status_code::backend_mismatch
            && prepared.run_count == completed_runs,
        "mutated BELL geometry reached vendor launch");
    view.feature_block_geometry_identity -= 1u;

    request.operation.determinism = cm::determinism_requirement::deterministic;
    const cm::backend_capability deterministic =
        backend.query(request.operation, device);
    require(deterministic.code == cm::capability_code::unsupported_determinism,
        "BELL deterministic request was not rejected structurally");
    request.operation.determinism =
        cm::determinism_requirement::allow_nondeterministic;
    request.operation.beta = cm::make_scalar(0.5f);
    const cm::backend_capability beta = backend.query(request.operation, device);
    require(beta.code == cm::capability_code::backend_unavailable,
        "BELL nonzero beta request was not rejected structurally");
    request.operation.beta = cm::make_scalar(0.0f);
    request.operation.dense_storage_type_code = cr::value_f32;
    const cm::backend_capability unfair = backend.query(request.operation, device);
    require(unfair.code == cm::capability_code::unsupported_type,
        "unfair BELL dtype comparison was not rejected structurally");
    request.operation.dense_storage_type_code = cr::value_f16;
    request.operation.sparse_storage_type_code = cr::value_bf16;
    require(backend.query(request.operation, device).code
            == cm::capability_code::unsupported_type,
        "unsupported BELL sparse storage type was advertised");
    request.operation.sparse_storage_type_code = cr::value_f16;
    request.operation.output_storage_type_code = cr::value_f16;
    require(backend.query(request.operation, device).code
            == cm::capability_code::unsupported_type,
        "unsupported BELL output storage type was advertised");
    request.operation.output_storage_type_code = cr::value_f32;
    request.operation.accumulation_type_code = cr::value_f16;
    require(backend.query(request.operation, device).code
            == cm::capability_code::unsupported_type,
        "unsupported BELL accumulation type was advertised");

    const cm::u64 identity = backend.identity();
    cm::reset_prepared_execution(&prepared);
    cuda_require(cudaFree(device_output), "free BELL output");
    cuda_require(cudaFree(device_rhs), "free BELL RHS");
    cuda_require(cudaFree(device_values), "free BELL values");
    cuda_require(cudaFree(device_columns), "free BELL columns");
    return identity;
}

} // namespace

int main(int argc, char **argv) {
    test_padded_candidate_rejection();
    if (argc == 2 && std::strcmp(argv[1], "--padded-rejection-only") == 0) {
        std::cout << "cpMathCusparseBellTest padded rejection passed\n";
        return 0;
    }
    const cm::u64 bell8 = test_candidate(8u);
    const cm::u64 bell16 = test_candidate(16u);
    const cm::u64 bell32 = test_candidate(32u);
    (void) test_candidate(8u, cm::dense_layout_kind::column_major, 1u);
    (void) test_candidate(16u, cm::dense_layout_kind::row_major);
    require(bell8 != bell16 && bell8 != bell32 && bell16 != bell32,
        "BELL candidates do not have distinct backend identities");
    std::cout << "cpMathCusparseBellTest passed\n";
    return 0;
}
