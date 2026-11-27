#include <Cellerator/execution/preparation/preparation_factory.hh>
#include "../fixtures/model_execution/activated_projection.hh"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace core = cellerator::execution::operation_core;
namespace cm = cellerator::layout::projection;
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "preparation_factory_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "preparation_factory_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "preparation_factory_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::abort();
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

core::numeric_policy csr_numeric() {
    core::numeric_policy result{};
    result.sparse_storage = execution::numeric_type::f16;
    result.dense_storage = execution::numeric_type::f32;
    result.output_storage = execution::numeric_type::f32;
    result.multiply = execution::numeric_type::f32;
    result.accumulation = execution::numeric_type::f32;
    result.scalar = execution::numeric_type::u32;
    result.bias = execution::numeric_type::invalid;
    return result;
}

struct device_fixture {
    std::uint32_t *row_offsets = nullptr;
    std::uint32_t *features = nullptr;
    __half *values = nullptr;

    device_fixture() {
        require_cuda(cudaMalloc(&row_offsets, 3u * sizeof(std::uint32_t)),
            "allocate row offsets");
        require_cuda(cudaMalloc(&features, 3u * sizeof(std::uint32_t)),
            "allocate features");
        require_cuda(cudaMalloc(&values, 3u * sizeof(__half)),
            "allocate values");
    }
    ~device_fixture() {
        if (values != nullptr) (void) cudaFree(values);
        if (features != nullptr) (void) cudaFree(features);
        if (row_offsets != nullptr) (void) cudaFree(row_offsets);
    }
};

core::candidate_preparation_request_v2 request(
    runtime::execution_session *session,
    core::csr_fallback_prepared_state *state) {
    core::candidate_preparation_request_v2 result{};
    result.problem.kind = core::operation_kind::weighted_relation_reduce;
    result.problem.operation = {101u, 102u};
    result.problem.input_count = 1u;
    result.problem.output_count = 1u;
    result.problem.logical_work_items = 3u;
    result.structures.count = 1u;
    result.structures.structures[0] = {{11u, 12u}, {21u, 1u}, {7u}};
    result.numeric = csr_numeric();
    result.policy = {true, false, true, true, 8u, 0u, 0u};
    result.session = session;
    result.dense_width = 1u;
    result.feature_axis = axis(10u);
    result.row_axis = axis(20u);
    result.dense_column_axis = axis(30u);
    result.state = {state, sizeof(*state)};
    return result;
}

::cellerator::layout::projection::CsrViewV1 csr(const device_fixture &device) {
    ::cellerator::layout::projection::CsrViewV1 result{};
    result.row_count = 2u;
    result.full_row_count = 2u;
    result.feature_count = 3u;
    result.nnz_count = 3u;
    result.value_size_bytes = sizeof(__half);
    result.row_domain_identity = 0x3003u;
    result.structure.identity_version = ::cellerator::layout::projection::execution_csr_structure_identity_version;
    result.structure.value = 0x7072u;
    result.feature_order.kind = ::cellerator::layout::projection::feature_order_kind::packed;
    result.feature_order.feature_count = 3u;
    result.feature_order.feature_axis_identity_version = 1u;
    result.feature_order.feature_axis_identity = 0x5005u;
    result.feature_order.packing_geometry_identity = 0x1001u;
    result.row_offsets = device.row_offsets;
    result.execution_feature_ids = device.features;
    result.values = device.values;
    return result;
}

void test_csr_preparation_retains_caller_state() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "query device");
    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "initialize session");
    require(runtime::prepare_stream_libraries(&session, 0u)
            == runtime::session_status::success,
        "prepare session libraries");
    device_fixture device_projection;
    alignas(core::csr_fallback_prepared_state)
        core::csr_fallback_prepared_state state{};
    const auto catalog = core::built_in_candidate_preparation_catalog_v2();
    const auto *adapter = core::find_candidate_preparation_adapter_v2(catalog,
        core::csr_fallback_candidate_id);
    require(adapter != nullptr, "find CSR preparation adapter");
    const core::projection_key key{{31u, 32u}, {41u, 1u},
        core::projection_kind::csr,
        ::cellerator::layout::projection::execution_csr_schema_version, 1u};
    const auto view = csr(device_projection);
    execution::activated_projection_reference_v2 projection{};
    require(cellerator::fixtures::activate_builtin_projection(catalog,
        core::csr_fallback_candidate_id, key, device, view, &projection),
        "activate CSR projection");
    auto factory_request = request(&session, &state);
    core::prepared_operation prepared{};
    require(core::prepare_catalog_candidate_v2(*adapter, factory_request,
        projection, &prepared), "prepare catalog CSR");
    require(core::same_stable_id(
            prepared.kernel, core::csr_fallback_candidate_id)
        && prepared.persistent.data == &state
        && prepared.structures.structures[0].epoch.value == 7u,
        "prepared operation retains caller-owned state and structure epoch");

    auto bad_adapter = *adapter;
    bad_adapter.prepare = nullptr;
    require(core::prepare_catalog_candidate_v2(bad_adapter, factory_request,
        projection, &prepared).code
            == core::operation_status_code::invalid_argument,
        "invalid preparation adapter rejection");

    core::csr_fallback_prepared_state rejected_state{};
    factory_request.state = {&rejected_state, sizeof(rejected_state)};
    factory_request.state.bytes = sizeof(rejected_state) - 1u;
    require(core::prepare_catalog_candidate_v2(*adapter, factory_request,
        projection, &prepared).code
            == core::operation_status_code::preparation_failed,
        "candidate state capacity rejection");

    factory_request.state = {&rejected_state, sizeof(rejected_state)};
    auto wrong_projection = projection;
    wrong_projection.key.kind = core::projection_kind::native_feature_major;
    require(core::prepare_catalog_candidate_v2(*adapter, factory_request,
        wrong_projection, &prepared).code
            == core::operation_status_code::unsupported_projection,
        "projection family mismatch rejection");

    require(runtime::seal_session(&session) == runtime::session_status::success,
        "seal session");
    require(core::prepare_catalog_candidate_v2(*adapter, factory_request,
        projection, &prepared).code
            == core::operation_status_code::preparation_failed,
        "sealed session preparation rejection");
    runtime::clear_session(&session);
}

} // namespace

int main() {
    test_csr_preparation_retains_caller_state();
    return 0;
}
