#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>
#include <Cellerator/execution/execution_contract.hh>
#include <Cellerator/planner/candidate_measurement.hh>

#include <Cellerator/geometry/persistent_packing_payload.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace cp = cellpack;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "candidate_measurement_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "candidate_measurement_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "candidate_measurement_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::abort();
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;

    explicit device_array(std::size_t count) : size(count) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
    ~device_array() { if (data != nullptr) cudaFree(data); }
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "device upload capacity");
    if (!host.empty())
        require_cuda(cudaMemcpy(device.data, host.data(),
            host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location device_location(int ordinal) {
    return {execution::residency_kind::device, {}, ordinal, 0u};
}

execution::dense_tensor_view dense(void *pointer,
    execution::numeric_type type, execution::axis_identity value_axis,
    std::uint64_t count, int ordinal) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(ordinal);
    view.value_type = type;
    view.rank = 1u;
    view.axes[0] = value_axis;
    view.shape[0] = count;
    view.stride[0] = 1;
    return view;
}

core::numeric_policy numeric() {
    core::numeric_policy value{};
    value.sparse_storage = execution::numeric_type::f16;
    value.dense_storage = execution::numeric_type::f32;
    value.output_storage = execution::numeric_type::f32;
    value.multiply = execution::numeric_type::f32;
    value.accumulation = execution::numeric_type::f32;
    value.scalar = execution::numeric_type::u32;
    value.bias = execution::numeric_type::invalid;
    return value;
}

struct projection_fixture {
    device_array<unsigned char> image{1u};
    device_array<cp::u32> feature_offsets{2u};
    device_array<cp::u32> feature_permutation{3u};
    device_array<cp::u32> row_permutation{2u};
    device_array<cp::u32> tile_offsets{2u};
    device_array<cp::u32> tile_blocks{1u};
    device_array<cp::u32> cell_masks{1u};
    device_array<cp::u32> entry_offsets{2u};
    device_array<cp::u32> gene_masks{2u};
    device_array<cp::u32> value_offsets{3u};
    device_array<std::uint32_t> csr_rows{3u};
    device_array<std::uint32_t> csr_features{3u};
    device_array<__half> values{3u};
    cp::persistent_packing_payload_view cpk1{};
    cm::execution_csr_view csr{};

    projection_fixture() {
        upload(feature_offsets, std::vector<cp::u32>{0u, 3u});
        upload(feature_permutation, std::vector<cp::u32>{0u, 1u, 2u});
        upload(row_permutation, std::vector<cp::u32>{0u, 1u});
        upload(tile_offsets, std::vector<cp::u32>{0u, 1u});
        upload(tile_blocks, std::vector<cp::u32>{0u});
        upload(cell_masks, std::vector<cp::u32>{0x3u});
        upload(entry_offsets, std::vector<cp::u32>{0u, 2u});
        upload(gene_masks, std::vector<cp::u32>{0x5u, 0x2u});
        upload(value_offsets, std::vector<cp::u32>{0u, 2u, 3u});
        upload(csr_rows, std::vector<std::uint32_t>{0u, 2u, 3u});
        upload(csr_features, std::vector<std::uint32_t>{0u, 2u, 1u});
        upload(values, std::vector<__half>{
            __float2half(1.0f), __float2half(2.0f), __float2half(3.0f)});

        cpk1.payload_schema_version = cp::persistent_packing_payload_schema_version;
        cpk1.payload_kind = cp::persistent_packing_payload_kind;
        cpk1.payload_identity = 0x43504b31u;
        cpk1.image_base = image.data;
        cpk1.image_bytes = image.size;
        cpk1.plan.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
        cpk1.plan.geometry_identity_version = cp::feature_block_geometry_identity_version;
        cpk1.plan.feature_count = 3u;
        cpk1.plan.feature_block_count = 1u;
        cpk1.plan.feature_block_geometry_identity = 0x1001u;
        cpk1.plan.feature_block_offsets = feature_offsets.data;
        cpk1.plan.feature_permutation = feature_permutation.data;
        cpk1.order.order_schema_version = cp::local_cell_order_schema_version;
        cpk1.order.signature_algorithm_version = cp::local_cell_signature_algorithm_version;
        cpk1.order.kind = cp::local_cell_order_kind::inferred_minhash;
        cpk1.order.window_size = 2u;
        cpk1.order.group_width = 2u;
        cpk1.order.ordering_identity = 0x2002u;
        cpk1.order.full_row_count = 2u;
        cpk1.order.row_count = 2u;
        cpk1.order.feature_block_count = 1u;
        cpk1.order.feature_block_geometry_identity = 0x1001u;
        cpk1.order.row_domain_identity = 0x3003u;
        cpk1.order.row_permutation = row_permutation.data;
        cpk1.tiles.tile_schema_version = cp::warp_tile_schema_version;
        cpk1.tiles.record_schema_version = cp::cell_block_record_schema_version;
        cpk1.tiles.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
        cpk1.tiles.geometry_identity_version = cp::feature_block_geometry_identity_version;
        cpk1.tiles.order_schema_version = cp::local_cell_order_schema_version;
        cpk1.tiles.tile_identity = 0x4004u;
        cpk1.tiles.feature_block_geometry_identity = 0x1001u;
        cpk1.tiles.ordering_identity = 0x2002u;
        cpk1.tiles.full_row_count = 2u;
        cpk1.tiles.row_count = 2u;
        cpk1.tiles.feature_count = 3u;
        cpk1.tiles.feature_block_count = 1u;
        cpk1.tiles.tile_row_width = 2u;
        cpk1.tiles.tile_count = 1u;
        cpk1.tiles.nnz_count = 3u;
        cpk1.tiles.tile_block_count = 1u;
        cpk1.tiles.row_block_entry_count = 2u;
        cpk1.tiles.value_size_bytes = sizeof(__half);
        cpk1.tiles.feature_axis_fingerprint = 0x5005u;
        cpk1.tiles.feature_axis_fingerprint_version = 1u;
        cpk1.tiles.row_domain_identity = 0x3003u;
        cpk1.tiles.tile_block_offsets = tile_offsets.data;
        cpk1.tiles.tile_block_ids = tile_blocks.data;
        cpk1.tiles.tile_block_cell_masks = cell_masks.data;
        cpk1.tiles.block_row_entry_offsets = entry_offsets.data;
        cpk1.tiles.row_block_gene_masks = gene_masks.data;
        cpk1.tiles.row_block_value_offsets = value_offsets.data;
        cpk1.tiles.values = values.data;

        csr.row_count = 2u;
        csr.full_row_count = 2u;
        csr.feature_count = 3u;
        csr.nnz_count = 3u;
        csr.value_size_bytes = sizeof(__half);
        csr.row_domain_identity = 0x3003u;
        csr.structure.identity_version = cm::execution_csr_structure_identity_version;
        csr.structure.value = 0x7073u;
        csr.feature_order.kind = cm::feature_order_kind::packed;
        csr.feature_order.feature_count = 3u;
        csr.feature_order.feature_axis_identity_version = 1u;
        csr.feature_order.feature_axis_identity = 0x5005u;
        csr.feature_order.packing_geometry_identity = 0x1001u;
        csr.row_offsets = csr_rows.data;
        csr.execution_feature_ids = csr_features.data;
        csr.values = values.data;
    }
};

struct referee_context {
    const float *device_output = nullptr;
};

bool referee(void *opaque, const execution::launch_bindings &) noexcept {
    const auto &context = *static_cast<const referee_context *>(opaque);
    float host[2]{};
    return cudaMemcpy(host, context.device_output, sizeof(host),
               cudaMemcpyDeviceToHost) == cudaSuccess
        && std::fabs(host[0] - 16.0f) < 1.0e-5f
        && std::fabs(host[1] - 15.0f) < 1.0e-5f;
}

struct cache_context {
    bool found = false;
    planner::plan_cache_entry entry{};
};

bool cache_lookup(void *opaque, const planner::planning_keys &,
    planner::plan_cache_entry *entry) noexcept {
    const auto &cache = *static_cast<const cache_context *>(opaque);
    if (!cache.found) return false;
    *entry = cache.entry;
    return true;
}

bool cache_store(void *opaque, const planner::plan_cache_entry &entry) noexcept {
    auto &cache = *static_cast<cache_context *>(opaque);
    cache.entry = entry;
    cache.found = true;
    return true;
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    projection_fixture projection_data;
    device_array<float> weights{3u};
    device_array<float> row_output{2u};
    device_array<float> csr_output{2u};
    device_array<float> caller_output{2u};
    upload(weights, std::vector<float>{2.0f, 5.0f, 7.0f});
    upload(caller_output, std::vector<float>{123.0f, 456.0f});

    const execution::axis_identity feature_axis = axis(10u);
    const execution::axis_identity row_axis = axis(20u);
    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u, {73u, 1u},
        1u, 1u, 3u};
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {{11u, 12u}, {21u, 1u}, {1u}};
    const core::projection_key row_projection{{31u, 32u}, {41u, 1u},
        core::projection_kind::native_row_masked,
        cp::persistent_packing_payload_schema_version, 1u};
    const core::projection_key csr_projection{{33u, 34u}, {42u, 1u},
        core::projection_kind::csr, cm::execution_csr_schema_version, 1u};
    const core::prepare_policy prepare_policy{true, false, true, true,
        8u, 0u, 0u};
    core::row_masked_n1_prepared_state row_state{};
    core::csr_fallback_prepared_state csr_state{};
    core::prepared_operation row_prepared{}, csr_prepared{};
    require(core::prepare_row_masked_n1_operation(problem, structures,
        row_projection, numeric(), prepare_policy, projection_data.cpk1,
        feature_axis, row_axis, &row_state, &row_prepared),
        "row-masked preparation");
    require(core::prepare_csr_fallback_operation(problem, structures,
        csr_projection, numeric(), prepare_policy, projection_data.csr,
        device, feature_axis, row_axis, &csr_state, &csr_prepared),
        "CSR preparation");

    execution::relation_structure relation{};
    relation.identity = structures.structures[0].runtime;
    relation.epoch = structures.structures[0].epoch;
    relation.source_axis = feature_axis;
    relation.destination_axis = row_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = 3u;
    execution::value_plane plane{};
    plane.structure = relation.identity;
    plane.structure_epoch_value = relation.epoch;
    plane.values = projection_data.values.data;
    plane.location = device_location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 3u;
    plane.value_bytes = 3u * sizeof(__half);
    execution::value_binding value_binding{&plane, {1u}};
    execution::biological_operand_view input{};
    input.kind = execution::operand_kind::dense_tensor;
    input.storage.dense = dense(weights.data, execution::numeric_type::f32,
        feature_axis, 3u, device);
    execution::biological_operand_view outputs[2]{};
    outputs[0].kind = execution::operand_kind::dense_tensor;
    outputs[0].storage.dense = dense(row_output.data,
        execution::numeric_type::f32, row_axis, 2u, device);
    outputs[1].kind = execution::operand_kind::dense_tensor;
    outputs[1].storage.dense = dense(csr_output.data,
        execution::numeric_type::f32, row_axis, 2u, device);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(
        &stream, cudaStreamNonBlocking), "create stream");
    execution::launch_bindings launches[2]{};
    for (std::uint32_t index = 0u; index < 2u; ++index) {
        launches[index].structures = &relation;
        launches[index].inputs = &input;
        launches[index].outputs = &outputs[index];
        launches[index].values = &value_binding;
        launches[index].input_count = 1u;
        launches[index].output_count = 1u;
        launches[index].value_count = 1u;
        launches[index].structure_count = 1u;
        launches[index].scalars.values[0] = {
            core::row_masked_n1_feature_weight_generation_binding,
            execution::numeric_type::u32, {}, 7u};
        launches[index].scalars.count = 1u;
        launches[index].stream = {stream, device, 0u};
        launches[index].workspace = {nullptr, 0u, device_location(device)};
    }
    referee_context referees[2]{{row_output.data}, {csr_output.data}};

    core::operation_candidate operations[2]{
        core::row_masked_n1_candidate(), core::csr_fallback_candidate()};
    planner::planner_candidate candidates[2]{};
    candidates[0].identity = operations[0].identity;
    candidates[0].name = operations[0].name;
    candidates[0].operation = &operations[0];
    candidates[0].projection = row_projection;
    candidates[0].analytical.kernel_ns = 1000.0;
    candidates[0].analytical.persistent_bytes = operations[0].persistent_bytes;
    candidates[0].flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic
        | planner::planner_candidate_graph_capture;
    candidates[1].identity = operations[1].identity;
    candidates[1].name = operations[1].name;
    candidates[1].operation = &operations[1];
    candidates[1].projection = csr_projection;
    candidates[1].analytical.kernel_ns = 1000.0;
    candidates[1].analytical.persistent_bytes = operations[1].persistent_bytes;
    candidates[1].flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic
        | planner::planner_candidate_conventional;

    planner::candidate_measurement_entry entries[2]{};
    entries[0].candidate = candidates[0].identity;
    entries[0].projection = row_projection;
    entries[0].prepared = &row_prepared;
    entries[0].private_launch = launches[0];
    entries[0].caller_visible_output = caller_output.data;
    entries[0].premeasured.semantic_packing_ns = 2000.0;
    entries[0].premeasured.projection_construction_ns = 1000.0;
    entries[0].referee_context = &referees[0];
    entries[0].referee = referee;
    entries[1].candidate = candidates[1].identity;
    entries[1].projection = csr_projection;
    entries[1].prepared = &csr_prepared;
    entries[1].private_launch = launches[1];
    entries[1].caller_visible_output = caller_output.data;
    entries[1].premeasured.projection_construction_ns = 8000.0;
    entries[1].premeasured.backend_prepare_ns = 2000.0;
    entries[1].premeasured.static_value_pack_ns = 1000.0;
    entries[1].referee_context = &referees[1];
    entries[1].referee = referee;
    planner::candidate_measurement_session measurement_session{entries, 2u};

    planner::planner_request request{};
    request.problem = problem;
    request.keys.problem.identity = problem.operation;
    require(planner::make_persistent_structure_set_key(
        structures, &request.keys.structures), "persistent structure key");
    request.keys.geometry = {{1u, 1u}, {2u, 1u}, {3u, 1u},
        {4u, 1u}, {5u, 1u}, {6u, 1u}};
    request.keys.device = {1u, 7u, 0u, 700u};
    request.keys.build = {10u, 20u, 30u, 40u};
    request.keys.policy = {8u, 8u, 8u, 1u, 1u, 1u, 0u};
    request.candidates = candidates;
    request.candidate_count = 2u;
    request.policy.shortlist_size = 2u;
    request.policy.maximum_measurements = 2u;
    request.policy.minimum_tuning_work_items = 1u;
    request.policy.maximum_spread_percent = 100.0;
    request.policy.minimum_cache_confidence = 0.0;
    request.measurement = {&measurement_session,
        planner::measure_prepared_candidate};
    cache_context cache{};
    request.cache = {&cache, cache_lookup, cache_store};
    request.current_evidence_revision = 1u;
    planner::planner_result result{};
    require(static_cast<bool>(planner::plan_end_to_end(request, &result)),
        "measured planner selection");
    require(result.source == planner::selection_source::empirical
        && result.measurement_count == 2u
        && result.legal_count == 2u
        && result.selected != nullptr
        && result.diagnostics[0].sample_count == 5u
        && result.diagnostics[1].sample_count == 5u
        && result.diagnostics[0].empirical.phases.kernel_ns > 0.0
        && result.diagnostics[1].empirical.phases.kernel_ns > 0.0
        && result.diagnostics[1].empirical.phases.static_value_pack_ns
            == 1000.0,
        "real candidate timing and phase accounting");
    const std::uint32_t winner = result.selected == &candidates[0] ? 0u : 1u;
    const std::uint32_t loser = 1u - winner;
    require(result.diagnostics[winner].empirical.amortized_total_ns
            <= result.diagnostics[loser].empirical.amortized_total_ns,
        "measured winner is not the lower end-to-end cost");
    float caller_host[2]{};
    require_cuda(cudaMemcpy(caller_host, caller_output.data,
        sizeof(caller_host), cudaMemcpyDeviceToHost), "copy caller output");
    require(caller_host[0] == 123.0f && caller_host[1] == 456.0f,
        "autotuning touched caller-visible output");

    request.measurement = {};
    require(planner::plan_end_to_end(request, &result)
        && result.source == planner::selection_source::cache,
        "fresh measured cache evidence");
    ++request.current_evidence_revision;
    require(planner::plan_end_to_end(request, &result)
        && result.cache == planner::cache_state::stale
        && result.source == planner::selection_source::analytical,
        "stale evidence rejection");

    planner::candidate_measurement_entry reordered_entries[2]{entries[0], entries[1]};
    reordered_entries[winner].premeasured.order_transform_ns += 1.0e9;
    planner::candidate_measurement_session reordered_session{
        reordered_entries, 2u};
    request.cache = {};
    request.measurement = {&reordered_session,
        planner::measure_prepared_candidate};
    request.current_evidence_revision = 3u;
    require(planner::plan_end_to_end(request, &result)
        && result.source == planner::selection_source::empirical
        && result.selected == &candidates[loser],
        "candidate-dependent order cost did not change the winner");

    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
