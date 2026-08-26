#include <Cellerator/compute/math/operation_core/preparation_factory.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
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

core::preparation_factory_request request(
    runtime::execution_session *session,
    core::csr_fallback_prepared_state *state) {
    core::preparation_factory_request result{};
    result.catalog_entry = core::find_built_in_candidate(
        core::csr_fallback_candidate_id);
    result.problem.kind = core::operation_kind::weighted_relation_reduce;
    result.problem.operation = {101u, 102u};
    result.problem.input_count = 1u;
    result.problem.output_count = 1u;
    result.problem.logical_work_items = 3u;
    result.structures.count = 1u;
    result.structures.structures[0] = {{11u, 12u}, {21u, 1u}, {7u}};
    result.projection = {{31u, 32u}, {41u, 1u},
        core::projection_kind::csr, cm::execution_csr_schema_version, 1u};
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

cm::execution_csr_view csr(const device_fixture &device) {
    cm::execution_csr_view result{};
    result.row_count = 2u;
    result.full_row_count = 2u;
    result.feature_count = 3u;
    result.nnz_count = 3u;
    result.value_size_bytes = sizeof(__half);
    result.row_domain_identity = 0x3003u;
    result.structure.identity_version = cm::execution_csr_structure_identity_version;
    result.structure.value = 0x7072u;
    result.feature_order.kind = cm::feature_order_kind::packed;
    result.feature_order.feature_count = 3u;
    result.feature_order.feature_axis_identity_version = 1u;
    result.feature_order.feature_axis_identity = 0x5005u;
    result.feature_order.packing_geometry_identity = 0x1001u;
    result.row_offsets = device.row_offsets;
    result.execution_feature_ids = device.features;
    result.values = device.values;
    return result;
}

void test_typed_csr_factory_and_session_cache() {
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
    auto factory_request = request(&session, &state);
    core::prepared_operation prepared{};
    require(core::prepare_catalog_csr(factory_request,
        csr(device_projection), &prepared), "prepare catalog CSR");
    require(core::same_stable_id(
            prepared.kernel, core::csr_fallback_candidate_id)
        && prepared.persistent.data == &state
        && session.plans.size == 1u
        && session.plans.entries[0].state == &state
        && session.plans.entries[0].structure_epoch == 7u,
        "prepared operation and session cache ownership");

    core::built_in_candidate_descriptor copied =
        *factory_request.catalog_entry;
    factory_request.catalog_entry = &copied;
    core::csr_fallback_prepared_state rejected_state{};
    factory_request.state = {&rejected_state, sizeof(rejected_state)};
    require(core::prepare_catalog_csr(factory_request,
        csr(device_projection), &prepared).code
            == core::operation_status_code::invalid_argument,
        "noncanonical catalog entry rejection");

    factory_request.catalog_entry = core::find_built_in_candidate(
        core::csr_fallback_candidate_id);
    factory_request.state.bytes = sizeof(rejected_state) - 1u;
    require(core::prepare_catalog_csr(factory_request,
        csr(device_projection), &prepared).code
            == core::operation_status_code::preparation_failed,
        "candidate state capacity rejection");

    factory_request.state = {&rejected_state, sizeof(rejected_state)};
    cm::feature_major_projection_view wrong_typed_projection{};
    require(core::prepare_catalog_feature_major(factory_request,
        wrong_typed_projection, &prepared).code
            == core::operation_status_code::unsupported_projection,
        "typed projection family mismatch rejection");

    require(runtime::seal_session(&session) == runtime::session_status::success,
        "seal session");
    require(core::prepare_catalog_csr(factory_request,
        csr(device_projection), &prepared).code
            == core::operation_status_code::preparation_failed,
        "sealed session preparation rejection");
    runtime::clear_session(&session);
}

} // namespace

int main() {
    test_typed_csr_factory_and_session_cache();
    return 0;
}
