#include <Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace cp = cellpack;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "row_masked_n1_candidate_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "row_masked_n1_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", binding=" << static_cast<unsigned>(status.binding)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "row_masked_n1_candidate_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::abort();
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;

    device_array() = default;
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

struct fixture {
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
    device_array<cellerator::real::storage_t> values{3u};
    device_array<cellerator::real::compute_t> weights{3u};
    device_array<cellerator::real::accum_t> output{2u};
    cp::persistent_packing_payload_view payload{};

    fixture() {
        upload(feature_offsets, std::vector<cp::u32>{0u, 3u});
        upload(feature_permutation, std::vector<cp::u32>{0u, 1u, 2u});
        upload(row_permutation, std::vector<cp::u32>{1u, 0u});
        upload(tile_offsets, std::vector<cp::u32>{0u, 1u});
        upload(tile_blocks, std::vector<cp::u32>{0u});
        upload(cell_masks, std::vector<cp::u32>{0x3u});
        upload(entry_offsets, std::vector<cp::u32>{0u, 2u});
        upload(gene_masks, std::vector<cp::u32>{0x5u, 0x2u});
        upload(value_offsets, std::vector<cp::u32>{0u, 2u, 3u});
        upload(values, std::vector<cellerator::real::storage_t>{
            __float2half(1.0f), __float2half(2.0f), __float2half(3.0f)});
        upload(weights, std::vector<cellerator::real::compute_t>{
            2.0f, 5.0f, 7.0f});

        payload.payload_schema_version =
            cp::persistent_packing_payload_schema_version;
        payload.payload_kind = cp::persistent_packing_payload_kind;
        payload.payload_identity = 0x43504b31u;
        payload.image_base = image.data;
        payload.image_bytes = image.size;
        payload.plan.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        payload.plan.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        payload.plan.feature_count = 3u;
        payload.plan.feature_block_count = 1u;
        payload.plan.feature_block_geometry_identity = 0x1001u;
        payload.plan.feature_block_offsets = feature_offsets.data;
        payload.plan.feature_permutation = feature_permutation.data;
        payload.order.order_schema_version = cp::local_cell_order_schema_version;
        payload.order.signature_algorithm_version =
            cp::local_cell_signature_algorithm_version;
        payload.order.kind = cp::local_cell_order_kind::inferred_minhash;
        payload.order.window_size = 2u;
        payload.order.group_width = 2u;
        payload.order.ordering_identity = 0x2002u;
        payload.order.full_row_count = 2u;
        payload.order.row_count = 2u;
        payload.order.feature_block_count = 1u;
        payload.order.feature_block_geometry_identity = 0x1001u;
        payload.order.row_domain_identity = 0x3003u;
        payload.order.row_permutation = row_permutation.data;
        payload.tiles.tile_schema_version = cp::warp_tile_schema_version;
        payload.tiles.record_schema_version = cp::cell_block_record_schema_version;
        payload.tiles.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        payload.tiles.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        payload.tiles.order_schema_version = cp::local_cell_order_schema_version;
        payload.tiles.tile_identity = 0x4004u;
        payload.tiles.feature_block_geometry_identity = 0x1001u;
        payload.tiles.ordering_identity = 0x2002u;
        payload.tiles.full_row_count = 2u;
        payload.tiles.row_count = 2u;
        payload.tiles.feature_count = 3u;
        payload.tiles.feature_block_count = 1u;
        payload.tiles.tile_row_width = 2u;
        payload.tiles.tile_count = 1u;
        payload.tiles.nnz_count = 3u;
        payload.tiles.tile_block_count = 1u;
        payload.tiles.row_block_entry_count = 2u;
        payload.tiles.value_size_bytes = sizeof(cellerator::real::storage_t);
        payload.tiles.feature_axis_fingerprint = 0x5005u;
        payload.tiles.feature_axis_fingerprint_version = 1u;
        payload.tiles.row_domain_identity = 0x3003u;
        payload.tiles.tile_block_offsets = tile_offsets.data;
        payload.tiles.tile_block_ids = tile_blocks.data;
        payload.tiles.tile_block_cell_masks = cell_masks.data;
        payload.tiles.block_row_entry_offsets = entry_offsets.data;
        payload.tiles.row_block_gene_masks = gene_masks.data;
        payload.tiles.row_block_value_offsets = value_offsets.data;
        payload.tiles.values = values.data;
    }
};

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

void test_registration_and_planner_enumeration(
    const core::operation_problem &problem,
    const core::structure_set_key &structures,
    const core::projection_key &projection) {
    core::candidate_registry registry{};
    require(core::register_row_masked_n1_candidate(&registry),
        "candidate registration");
    require(registry.size == 1u
        && registry.candidates[0].operation
            == core::operation_kind::weighted_relation_reduce
        && registry.candidates[0].projection
            == core::projection_kind::native_row_masked
        && registry.candidates[0].backend == core::backend_kind::native_direct
        && registry.candidates[0].transient_bytes == 0u,
        "candidate capability record");
    require(core::register_row_masked_n1_candidate(&registry).code
            == core::operation_status_code::duplicate_candidate,
        "duplicate candidate rejection");

    planner::planner_candidate candidate{};
    candidate.identity = registry.candidates[0].identity;
    candidate.name = registry.candidates[0].name;
    candidate.operation = &registry.candidates[0];
    candidate.projection = projection;
    candidate.analytical.kernel_ns = 1.0;
    candidate.analytical.persistent_bytes =
        registry.candidates[0].persistent_bytes;
    candidate.flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic
        | planner::planner_candidate_graph_capture;
    planner::planner_request request{};
    request.problem = problem;
    request.keys.problem.identity = problem.operation;
    require(planner::make_persistent_structure_set_key(
        structures, &request.keys.structures), "persistent structure key");
    request.keys.geometry = {{1u, 1u}, {2u, 1u}, {3u, 1u},
        {4u, 1u}, {5u, 1u}, {6u, 1u}};
    request.keys.device = {1u, 7u, 0u, 700u};
    request.keys.build = {1u, 2u, 3u, 4u};
    request.keys.policy = {8u, 8u, 8u, 1u, 1u, 1u, 1u};
    request.candidates = &candidate;
    request.candidate_count = 1u;
    request.policy.shortlist_size = 1u;
    request.policy.maximum_measurements = 1u;
    request.policy.minimum_tuning_work_items = 4096u;
    request.current_evidence_revision = 1u;
    planner::planner_result result{};
    require(planner::plan_end_to_end(request, &result)
        && result.selected == &candidate
        && result.winner.low == candidate.identity.low
        && result.tuning_skipped, "planner candidate enumeration");
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    fixture data;
    const execution::axis_identity feature_axis = axis(10u);
    const execution::axis_identity row_axis = axis(20u);
    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u, {71u, 1u},
        1u, 1u, 3u};
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {{11u, 12u}, {21u, 1u}, {1u}};
    const core::projection_key projection{{31u, 32u}, {41u, 1u},
        core::projection_kind::native_row_masked,
        cp::persistent_packing_payload_schema_version, 1u};
    test_registration_and_planner_enumeration(problem, structures, projection);

    core::row_masked_n1_prepared_state state{};
    core::prepared_operation prepared{};
    const core::prepare_policy policy{true, true, true, true, 8u, 0u, 0u};
    require(core::prepare_row_masked_n1_operation(problem, structures,
        projection, numeric(), policy, data.payload, feature_axis, row_axis,
        &state, &prepared), "candidate preparation");
    require(prepared.binding_contract.workspace.minimum_bytes == 0u
        && prepared.binding_contract.output_effects[0].update
            == execution::output_update_kind::overwrite
        && prepared.binding_contract.output_orders[0].transition
            == execution::order_transition_kind::preserve,
        "prepared workspace, effect, and order contract");

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
    plane.values = data.values.data;
    plane.location = device_location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 3u;
    plane.value_bytes = 3u * sizeof(cellerator::real::storage_t);
    execution::value_binding value_binding{&plane, {1u}};
    execution::biological_operand_view input{}, output{};
    input.kind = execution::operand_kind::dense_tensor;
    input.storage.dense = dense(data.weights.data, execution::numeric_type::f32,
        feature_axis, 3u, device);
    output.kind = execution::operand_kind::dense_tensor;
    output.storage.dense = dense(data.output.data, execution::numeric_type::f32,
        row_axis, 2u, device);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(
        &stream, cudaStreamNonBlocking), "create stream");
    execution::launch_bindings launch{};
    launch.structures = &relation;
    launch.inputs = &input;
    launch.outputs = &output;
    launch.values = &value_binding;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.value_count = 1u;
    launch.structure_count = 1u;
    launch.scalars.values[0] = {
        core::row_masked_n1_feature_weight_generation_binding,
        execution::numeric_type::u32, {}, 7u};
    launch.scalars.count = 1u;
    launch.stream = {stream, device, 0u};
    launch.workspace = {nullptr, 0u, device_location(device)};
    require(core::run_prepared_operation(prepared, launch),
        "direct prepared execution");
    require_cuda(cudaStreamSynchronize(stream), "synchronize result");
    std::vector<float> result(2u);
    require_cuda(cudaMemcpy(result.data(), data.output.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost), "download result");
    require(std::fabs(result[0] - 15.0f) < 1.0e-5f
        && std::fabs(result[1] - 16.0f) < 1.0e-5f,
        "canonical numerical parity");

    core::numeric_policy rejected_numeric = numeric();
    rejected_numeric.dense_storage = execution::numeric_type::f64;
    core::row_masked_n1_prepared_state rejected_state{};
    core::prepared_operation rejected{};
    require(core::prepare_row_masked_n1_operation(problem, structures,
        projection, rejected_numeric, policy, data.payload,
        feature_axis, row_axis, &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_numeric_policy,
        "numeric capability rejection");
    core::projection_key wrong_projection = projection;
    wrong_projection.kind = core::projection_kind::csr;
    require(core::prepare_row_masked_n1_operation(problem, structures,
        wrong_projection, numeric(), policy, data.payload,
        feature_axis, row_axis, &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_projection,
        "projection capability rejection");
    core::prepare_policy no_preprocessing = policy;
    no_preprocessing.allow_persistent_preprocessing = false;
    require(core::prepare_row_masked_n1_operation(problem, structures,
        projection, numeric(), no_preprocessing, data.payload,
        feature_axis, row_axis, &rejected_state, &rejected).code
            == core::operation_status_code::capability_rejected,
        "persistent preparation policy rejection");
    value_binding.expected_generation.value = 2u;
    require(core::run_prepared_operation(prepared, launch).binding
            == execution::binding_validation_code::stale_value,
        "stale value generation rejection");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
