#include <Cellerator/compute/candidate/feature_major_small_n_candidate.hh>
#include <Cellerator/execution/opaque_artifact.hh>
#include <Cellerator/execution/program.hh>

#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

namespace cm = cellerator::compute::math;
namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace persistence = cellpack::persistence;
namespace planner = cellerator::planner;
namespace runtime = cellerator::runtime;
namespace cp = cellpack;
namespace cs = cellshard;

namespace {

template<typename T>
void require(T condition, const char *message) {
    if (static_cast<bool>(condition)) return;
    std::fprintf(stderr, "ce_live_program_replay_test: %s\n", message);
    std::exit(1);
}

void require(execution::executable_program_status status,
    const char *message) {
    if (status) return;
    std::fprintf(stderr,
        "ce_live_program_replay_test: %s (code=%u detail=%s operation=%s)\n",
        message, static_cast<unsigned>(status.code), status.message,
        status.operation.message);
    std::exit(1);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "ce_live_program_replay_test: %s: %s\n",
        message, cudaGetErrorString(status));
    std::exit(1);
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t size = 0u;

    explicit device_buffer(std::size_t count) : size(count) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data != nullptr) (void)cudaFree(data); }
};

execution::persistent_axis_identity persistent_axis(std::uint64_t base) {
    execution::persistent_axis_identity value{};
    value.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(execution::persistent_axis_identity)};
    value.domain = {base, 1u};
    value.order = {base + 1u, 1u};
    value.geometry = {base + 2u, 1u};
    value.partition = {base + 3u, 1u};
    return value;
}

execution::program_axis program_axis(std::uint32_t live,
    std::uint64_t persistent) {
    return {{{live, 1u}, {live + 1u, 1u},
                {live + 2u, 1u}, {live + 3u, 1u}},
        persistent_axis(persistent)};
}

execution::device_location location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense(void *pointer,
    execution::axis_identity major, execution::axis_identity minor,
    std::uint64_t rows, int device) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = location(device);
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = major;
    view.axes[1] = minor;
    view.shape[0] = rows;
    view.shape[1] = 1u;
    view.stride[0] = 1;
    view.stride[1] = 1;
    return view;
}

core::numeric_policy numeric() {
    core::numeric_policy value{};
    value.sparse_storage = execution::numeric_type::f16;
    value.dense_storage = execution::numeric_type::f32;
    value.output_storage = execution::numeric_type::f32;
    value.multiply = execution::numeric_type::f32;
    value.accumulation = execution::numeric_type::f32;
    value.scalar = execution::numeric_type::f32;
    value.bias = execution::numeric_type::invalid;
    return value;
}

struct source_fixture {
    std::vector<cp::u32> feature_offsets{0u, 4u};
    std::vector<cp::u32> feature_permutation{0u, 1u, 2u, 3u};
    std::vector<cp::u32> row_permutation{0u, 1u, 2u, 3u, 4u};
    std::vector<cp::u32> tile_offsets{0u, 1u, 2u};
    std::vector<cp::u32> tile_blocks{0u, 0u};
    std::vector<cp::u32> cell_masks{0xdu, 0x1u};
    std::vector<cp::u32> entry_offsets{0u, 3u, 4u};
    std::vector<cp::u32> gene_masks{0x5u, 0xeu, 0x1u, 0xcu};
    std::vector<cp::u32> value_offsets{0u, 2u, 5u, 6u, 8u};
    std::vector<__half> values;
    unsigned char image_byte = 0u;
    cp::persistent_packing_payload_view payload{};

    source_fixture() {
        for (float value : {1.0f, 2.0f, 3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f})
            values.push_back(__float2half(value));
        payload.payload_schema_version =
            cp::persistent_packing_payload_schema_version;
        payload.payload_kind = cp::persistent_packing_payload_kind;
        payload.payload_identity = 0x43504b3134u;
        payload.image_base = &image_byte;
        payload.image_bytes = 1u;
        payload.plan.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        payload.plan.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        payload.plan.feature_count = 4u;
        payload.plan.feature_block_count = 1u;
        payload.plan.feature_block_geometry_identity = 0x100134u;
        payload.plan.feature_block_offsets = feature_offsets.data();
        payload.plan.feature_permutation = feature_permutation.data();
        payload.order.order_schema_version = cp::local_cell_order_schema_version;
        payload.order.signature_algorithm_version =
            cp::local_cell_signature_algorithm_version;
        payload.order.kind = cp::local_cell_order_kind::inferred_minhash;
        payload.order.window_size = 4u;
        payload.order.group_width = 4u;
        payload.order.ordering_identity = 0x200234u;
        payload.order.full_row_count = 5u;
        payload.order.row_count = 5u;
        payload.order.feature_block_count = 1u;
        payload.order.feature_block_geometry_identity = 0x100134u;
        payload.order.row_domain_identity = 0x300334u;
        payload.order.row_permutation = row_permutation.data();
        payload.tiles.tile_schema_version = cp::warp_tile_schema_version;
        payload.tiles.record_schema_version = cp::cell_block_record_schema_version;
        payload.tiles.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        payload.tiles.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        payload.tiles.order_schema_version = cp::local_cell_order_schema_version;
        payload.tiles.tile_identity = 0x400434u;
        payload.tiles.feature_block_geometry_identity = 0x100134u;
        payload.tiles.ordering_identity = 0x200234u;
        payload.tiles.full_row_count = 5u;
        payload.tiles.row_count = 5u;
        payload.tiles.feature_count = 4u;
        payload.tiles.feature_block_count = 1u;
        payload.tiles.tile_row_width = 4u;
        payload.tiles.tile_count = 2u;
        payload.tiles.nnz_count = 8u;
        payload.tiles.tile_block_count = 2u;
        payload.tiles.row_block_entry_count = 4u;
        payload.tiles.value_size_bytes = sizeof(__half);
        payload.tiles.feature_axis_fingerprint = 0x500534u;
        payload.tiles.feature_axis_fingerprint_version = 1u;
        payload.tiles.row_domain_identity = 0x300334u;
        payload.tiles.tile_block_offsets = tile_offsets.data();
        payload.tiles.tile_block_ids = tile_blocks.data();
        payload.tiles.tile_block_cell_masks = cell_masks.data();
        payload.tiles.block_row_entry_offsets = entry_offsets.data();
        payload.tiles.row_block_gene_masks = gene_masks.data();
        payload.tiles.row_block_value_offsets = value_offsets.data();
        payload.tiles.values = values.data();
    }
};

std::string temporary_path() {
    std::string path = "/tmp/cellerator_ce_live_replayXXXXXX";
    const int descriptor = ::mkstemp(path.data());
    require(descriptor >= 0, "create temporary replay path");
    ::close(descriptor);
    ::unlink(path.c_str());
    return path + ".cspack";
}

} // namespace

int main() {
    const execution::structure_id structure{0x1134u, 0x1234u};
    const execution::structure_handle structure_handle{21u, 1u};
    const execution::structure_epoch epoch{7u};
    const execution::projection_id projection_id{0x3134u, 0x3234u};
    const execution::projection_handle projection_handle{41u, 1u};
    source_fixture source;
    cm::feature_major_projection_build_request projection_request{};
    projection_request.structure_identity = structure;
    projection_request.runtime_structure = structure_handle;
    projection_request.structure_epoch_value = epoch;
    projection_request.projection_identity = projection_id;
    projection_request.runtime_projection = projection_handle;
    projection_request.source = source.payload;
    cm::feature_major_projection_requirements projection_requirements{};
    require(cm::query_feature_major_projection_requirements_host(
        projection_request, &projection_requirements), "query FMP1 projection");
    std::vector<unsigned char> projection_payload(
        projection_requirements.payload_bytes);
    cm::feature_major_projection_view built_projection{};
    require(cm::build_feature_major_projection_host(projection_request,
        {projection_payload.data(), projection_payload.size()}, &built_projection),
        "build pointer-free FMP1 projection");

    const std::array<unsigned char, 8> domain{{1,2,3,4,5,6,7,8}};
    const std::array<unsigned char, 8> order{{2,3,4,5,6,7,8,9}};
    const std::array<unsigned char, 8> relation{{3,4,5,6,7,8,9,10}};
    const std::array<unsigned char, 8> geometry{{4,5,6,7,8,9,10,11}};
    persistence::execution_section_source sections[5]{};
    sections[0] = {persistence::execution_section_kind::domain_table,
        1u, 0u, 8u, 1u, 1u, domain.data(), domain.size(), 1u, 8u};
    sections[1] = {persistence::execution_section_kind::order_partition_table,
        1u, 0u, 8u, 2u, 1u, order.data(), order.size(), 1u, 8u};
    sections[2] = {persistence::execution_section_kind::relation_structure,
        1u, 0u, 8u, 3u, 1u, relation.data(), relation.size(), 1u, 8u};
    sections[3] = {persistence::execution_section_kind::semantic_geometry,
        1u, 0u, 8u, 4u, 1u, geometry.data(), geometry.size(), 1u, 8u};
    sections[4] = {persistence::execution_section_kind::projection_payload,
        cm::feature_major_projection_schema_version,
        persistence::directory_device_readable, 64u,
        projection_id.low, projection_id.high, projection_payload.data(),
        projection_payload.size(), 0u, 0u};
    persistence::execution_projection_source projection_source{};
    auto &entry = projection_source.entry;
    entry.identity_low = projection_id.low;
    entry.identity_high = projection_id.high;
    entry.kind = persistence::execution_projection_kind::native_feature_major;
    entry.schema_version = cm::feature_major_projection_schema_version;
    entry.flags = persistence::directory_device_readable
        | persistence::projection_forward_capable;
    entry.operation_family = static_cast<std::uint32_t>(
        core::operation_kind::sparse_dense_multiply);
    entry.storage_type = static_cast<std::uint16_t>(execution::numeric_type::f16);
    entry.compute_type = static_cast<std::uint16_t>(execution::numeric_type::f32);
    entry.accumulation_type = static_cast<std::uint16_t>(execution::numeric_type::f32);
    entry.orientation = static_cast<std::uint16_t>(
        execution::relation_orientation::forward);
    entry.architecture_class = 70u;
    entry.payload_section = 4u;
    entry.forward_map_section = persistence::invalid_directory_index;
    entry.transpose_map_section = persistence::invalid_directory_index;
    entry.scheduling_summary_section = persistence::invalid_directory_index;
    entry.capability_section = persistence::invalid_directory_index;

    const auto source_axis = program_axis(100u, 0x1000u);
    auto destination_axis = program_axis(200u, 0x2000u);
    destination_axis.persistent.geometry = source_axis.persistent.geometry;
    destination_axis.persistent.partition = source_axis.persistent.partition;
    const auto dense_axis = program_axis(300u, 0x3000u);
    persistence::execution_image_v2_build_request image_request{};
    image_request.structure_identity = structure;
    image_request.structure_epoch = epoch.value;
    image_request.semantic_geometry_identity = {0x5134u, 0x5234u};
    image_request.projection_catalog_identity = {0x6134u, 0x6234u};
    image_request.source_axis = source_axis.persistent;
    image_request.destination_axis = destination_axis.persistent;
    image_request.sections = sections;
    image_request.section_count = 5u;
    image_request.projections = &projection_source;
    image_request.projection_count = 1u;
    persistence::execution_image_v2_requirements image_requirements{};
    require(persistence::query_execution_image_v2_requirements_host(
        image_request, &image_requirements), "query CPE2 image");
    std::vector<unsigned char> image(image_requirements.image_bytes);
    persistence::execution_image_v2_view built_image{};
    require(persistence::build_execution_image_v2_host(image_request,
        {image.data(), image.size()}, &built_image), "build CPE2 image");

    cs::execution_payload_identity transport{};
    transport.dataset_identity = 0xce341001u;
    transport.generation = {1u, 2u, 3u, 4u};
    transport.partition_identity = 0xce341002u;
    transport.row_count = 5u;
    transport.feature_count = 4u;
    transport.feature_axis_fingerprint = 0x500534u;
    transport.feature_axis_fingerprint_version = 1u;
    transport.payload_kind = persistence::execution_image_v2_payload_kind;
    transport.payload_schema_version = persistence::execution_image_v2_schema_version;
    transport.row_domain_identity = 0x300334u;
    transport.payload_identity = built_image.header.image_identity;
    const cs::execution_payload_source transport_source{
        transport, image.data(), image.size()};
    const std::string path = temporary_path();
    require(cs::store_execution_cspack(path.c_str(), 34u,
        &transport_source, 1u), "store opaque CPEXEC01 payload");
    cs::execution_payload_host loaded{};
    require(cs::load_execution_cspack_partition(path.c_str(), 34u, 0u,
        transport, &loaded), "reload opaque CPEXEC01 payload");

    execution::opaque_execution_artifact_expected expected{};
    expected.transport = transport;
    expected.image = {structure, epoch.value,
        image_request.semantic_geometry_identity,
        image_request.projection_catalog_identity,
        built_image.header.image_identity};
    execution::validated_opaque_execution_artifact validated{};
    require(execution::validate_opaque_execution_artifact_host(
        loaded, expected, &validated), "validate reloaded CPE2 semantics");
    persistence::prebound_projection_view_v1 host_prebound{};
    require(persistence::prebind_execution_projection_host(
        validated.host_image, 0u, &host_prebound), "prebind reloaded FMP1");
    cm::feature_major_projection_view validated_projection{};
    require(cm::validate_feature_major_projection_payload_host(
        host_prebound.payload, host_prebound.payload_bytes, structure, epoch,
        projection_id, structure_handle, projection_handle,
        &validated_projection), "validate reloaded typed FMP1");

    int device = -1;
    require_cuda(cudaGetDevice(&device), "query CUDA device");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");
    cs::execution_payload_device residency{};
    require_cuda(cs::upload_execution_payload_async(
        loaded, device, stream, &residency), "upload opaque CPE2 once");
    execution::bound_opaque_execution_artifact bound{};
    require(execution::bind_opaque_execution_artifact_device(
        validated, residency, &bound), "bind caller-owned CPE2 residency");
    execution::projection_activation_context activation{};
    activation.structure = structure;
    activation.runtime_structure = structure_handle;
    activation.epoch = epoch;
    activation.projection = projection_id;
    activation.runtime_projection = projection_handle;
    activation.location = location(device);
    cm::feature_major_projection_view device_projection{};
    require(execution::activate_feature_major_projection(
        bound.projection, activation, validated_projection, &device_projection),
        "activate non-owning typed FMP1 device view");
    require(device_projection.payload_base == bound.projection.payload,
        "typed activation does not alias the uploaded CPE2 image");

    std::vector<__half> packed_values(source.values.size());
    require(cm::pack_feature_major_values_host(validated_projection,
        source.values.data(), source.values.size() * sizeof(__half),
        {packed_values.data(), packed_values.size() * sizeof(__half)}),
        "pack mutable values outside CPE2");
    device_buffer<__half> values(packed_values.size());
    device_buffer<float> input(4u);
    device_buffer<float> output(5u);
    require_cuda(cudaMemcpyAsync(values.data, packed_values.data(),
        packed_values.size() * sizeof(__half), cudaMemcpyHostToDevice, stream),
        "upload values");
    const float host_input[]{1.0f, 2.0f, 3.0f, 4.0f};
    require_cuda(cudaMemcpyAsync(input.data, host_input, sizeof(host_input),
        cudaMemcpyHostToDevice, stream), "upload dense operand");

    runtime::execution_session session{};
    runtime::execution_session_options session_options{};
    session_options.device = device;
    require(runtime::init_session(&session, session_options)
            == runtime::session_status::success,
        "initialize sole execution session");
    const core::projection_key projection_key{projection_id, projection_handle,
        core::projection_kind::native_feature_major,
        cm::feature_major_projection_schema_version,
        cm::feature_major_projection_variant};
    const auto projection = execution::program_projection(
        projection_key, device_projection);
    execution::program_candidate_cost cost{};
    cost.candidate = core::feature_major_small_n_candidate_id;
    cost.projection = projection_id;
    cost.phases.host_preparation_ns = 1000.0;
    cost.phases.projection_construction_ns = 1000.0;
    cost.phases.backend_prepare_ns = 1000.0;
    cost.phases.kernel_ns = 1000.0;
    cost.planner_flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic
        | planner::planner_candidate_graph_capture;
    alignas(64) unsigned char preparation_state[4096]{};
    execution::executable_program_request request{};
    request.problem = {core::operation_core_schema_version,
        core::operation_kind::sparse_dense_multiply, 0u,
        {0xce340001u, 0xce340002u}, 1u, 1u, 8u};
    request.structures.count = 1u;
    request.structures.structures[0] = {structure, structure_handle, epoch};
    request.numeric = numeric();
    request.preparation = {true, false, true, true, 8u, 0u, 0u};
    request.planning.problem.identity = request.problem.operation;
    require(planner::make_persistent_structure_set_key(
        request.structures, &request.planning.structures),
        "make persistent structure key");
    request.planning.geometry = {source_axis.persistent.domain,
        destination_axis.persistent.domain,
        source_axis.persistent.geometry, source_axis.persistent.order,
        destination_axis.persistent.order, source_axis.persistent.partition};
    request.planning.device = {1u, 7u, 0u, 700u};
    request.planning.build = {12090u, 700u, 1u, 1u};
    request.planning.policy = {8u, 8u, 8u, 1u, 1u, 1u, 1u};
    request.planner_policy.minimum_tuning_work_items =
        std::numeric_limits<std::uint64_t>::max();
    request.current_evidence_revision = 1u;
    request.catalog = core::built_in_candidate_catalog();
    request.projections = &projection;
    request.projection_count = 1u;
    request.costs = &cost;
    request.cost_count = 1u;
    request.session = &session;
    request.dense_width = 1u;
    request.source_axis = source_axis;
    request.destination_axis = destination_axis;
    request.dense_column_axis = dense_axis;
    request.preparation_state = {preparation_state, sizeof(preparation_state)};
    execution::executable_program program{};
    require(execution::compile_executable_program(request, &program),
        "compile planner-selected replay program");
    require(core::same_stable_id(program.selected_candidate,
            core::feature_major_small_n_candidate_id)
        && program.selected_projection.persistent.low == projection_id.low
        && program.selected_projection.persistent.high == projection_id.high,
        "planner did not select the reloaded FMP1 candidate");

    runtime::value_readiness_record readiness{};
    require(runtime::initialize_value_readiness(&readiness, device)
            == runtime::value_readiness_status::success,
        "initialize value readiness");
    require(runtime::publish_value_generation(
        &readiness, epoch.value, 1u, stream, cudaSuccess)
            == runtime::value_readiness_status::success,
        "publish replay value generation");
    execution::value_plane plane{};
    plane.structure = structure_handle;
    plane.structure_epoch_value = epoch;
    plane.location = location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.values = values.data;
    plane.element_count = packed_values.size();
    plane.value_bytes = packed_values.size() * sizeof(__half);
    plane.generation = {1u};
    execution::value_binding value{&plane, {1u}};
    execution::biological_operand_view input_operand{}, output_operand{};
    input_operand.kind = output_operand.kind = execution::operand_kind::dense_tensor;
    input_operand.storage.dense = dense(input.data,
        source_axis.live, dense_axis.live, 4u, device);
    output_operand.storage.dense = dense(output.data,
        destination_axis.live, dense_axis.live, 5u, device);
    execution::launch_bindings bindings{};
    execution::relation_structure live_structure{};
    live_structure.identity = structure_handle;
    live_structure.epoch = epoch;
    live_structure.source_axis = source_axis.live;
    live_structure.destination_axis = destination_axis.live;
    live_structure.projections = {1u, 1u};
    live_structure.logical_edge_count = 8u;
    bindings.structures = &live_structure;
    bindings.inputs = &input_operand;
    bindings.outputs = &output_operand;
    bindings.values = &value;
    bindings.structure_count = bindings.input_count =
        bindings.output_count = bindings.value_count = 1u;
    bindings.stream = {stream, device, 0u};
    bindings.workspace = {nullptr, 0u, location(device)};
    execution::executable_program_launch launch{
        bindings, &readiness, epoch, {1u}};
    execution::executable_program_result result{};
    require(execution::run_executable_program(&program, launch, &result),
        "execute reloaded planner-backed program");
    require(result.enqueued && result.consumed_generation.value == 1u
        && result.output_order_count == 2u,
        "replay result metadata is incomplete");
    std::array<float, 5> actual{};
    require_cuda(cudaMemcpyAsync(actual.data(), output.data, sizeof(actual),
        cudaMemcpyDeviceToHost, stream), "download replay output");
    require_cuda(cudaStreamSynchronize(stream), "wait caller stream");
    const std::array<float, 5> expected_output{{7.0f, 0.0f, 38.0f, 6.0f, 53.0f}};
    for (std::size_t index = 0u; index < actual.size(); ++index)
        require(std::fabs(actual[index] - expected_output[index]) < 1.0e-5f,
            "replayed CUDA output disagrees with independent referee");

    require(runtime::clear_value_readiness(&readiness)
            == runtime::value_readiness_status::success,
        "clear value readiness");
    runtime::clear_session(&session);
    require_cuda(cs::clear_execution_payload_device(&residency),
        "release caller-owned CPE2 residency");
    require_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    cs::clear_execution_payload_host(&loaded);
    ::unlink(path.c_str());
    std::puts("ce_live_program_replay_test passed cpe2=1 cpexec01=1 "
        "typed_activation=1 planner=1 quantitative_cuda=1");
    return 0;
}
