#include <Cellerator/compute/candidate/transpose_backward_candidate.hh>
#include <Cellerator/compat/cp_math_v1/referee.hh>

#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace cp = cellpack;
namespace persistence = cellpack::persistence;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "transpose_backward_candidate_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "transpose_backward_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", binding=" << static_cast<unsigned>(status.binding)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cm::physical_view_status status, const char *message) {
    if (status) return;
    std::cerr << "transpose_backward_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cm::referee_status status, const char *message) {
    if (status) return;
    std::cerr << "transpose_backward_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cp::validation_result status, const char *message) {
    if (status) return;
    std::cerr << "transpose_backward_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "transpose_backward_candidate_test: " << message << ": "
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
    ~device_array() { if (data != nullptr) cudaFree(data); }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "upload capacity");
    if (!host.empty())
        require_cuda(cudaMemcpy(device.data, host.data(),
            host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

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

execution::device_location device_location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense_matrix(void *pointer,
    execution::axis_identity major, execution::axis_identity minor,
    std::uint64_t rows, int device) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(device);
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

struct fixture {
    std::vector<cp::u32> feature_offsets{0u, 5u};
    std::vector<cp::u32> feature_permutation{0u, 1u, 2u, 3u, 4u};
    std::vector<cp::u32> row_permutation{0u, 1u, 2u, 3u, 4u};
    std::vector<cp::u32> tile_offsets{0u, 1u, 2u};
    std::vector<cp::u32> tile_blocks{0u, 0u};
    std::vector<cp::u32> cell_masks{0xdu, 0x1u};
    std::vector<cp::u32> entry_offsets{0u, 3u, 4u};
    std::vector<cp::u32> gene_masks{0x5u, 0xeu, 0x1u, 0xcu};
    std::vector<cp::u32> value_offsets{0u, 2u, 5u, 6u, 8u};
    std::vector<__half> values_a;
    std::vector<__half> values_b;
    unsigned char image_byte = 0u;
    cp::persistent_packing_payload_view source{};
    execution::structure_id structure_id{0x1185u, 0x1285u};
    execution::structure_handle structure_handle{22u, 1u};
    execution::structure_epoch epoch{8u};
    execution::projection_id forward_id{0x3185u, 0x3285u};
    execution::projection_handle forward_handle{42u, 1u};
    execution::projection_id transpose_id{0x5185u, 0x5285u};
    execution::projection_handle transpose_handle{62u, 1u};
    std::vector<unsigned char> forward_payload;
    std::vector<unsigned char> transpose_payload;
    cm::feature_major_projection_view forward_view{};
    cm::transpose_projection_view transpose_view{};

    fixture() {
        for (float value : {1.0f, 2.0f, 3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f})
            values_a.push_back(__float2half(value));
        for (float value : {2.0f, -1.0f, 0.5f, 3.0f,
                            -2.0f, 4.0f, 1.5f, -0.5f})
            values_b.push_back(__float2half(value));
        source.payload_schema_version = cp::persistent_packing_payload_schema_version;
        source.payload_kind = cp::persistent_packing_payload_kind;
        source.payload_identity = 0x43504b3185u;
        source.image_base = &image_byte;
        source.image_bytes = 1u;
        source.plan.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        source.plan.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        source.plan.feature_count = 5u;
        source.plan.feature_block_count = 1u;
        source.plan.feature_block_geometry_identity = 0x100185u;
        source.plan.feature_block_offsets = feature_offsets.data();
        source.plan.feature_permutation = feature_permutation.data();
        source.order.order_schema_version = cp::local_cell_order_schema_version;
        source.order.signature_algorithm_version =
            cp::local_cell_signature_algorithm_version;
        source.order.kind = cp::local_cell_order_kind::inferred_minhash;
        source.order.window_size = 4u;
        source.order.group_width = 4u;
        source.order.ordering_identity = 0x200285u;
        source.order.full_row_count = 5u;
        source.order.row_count = 5u;
        source.order.feature_block_count = 1u;
        source.order.feature_block_geometry_identity = 0x100185u;
        source.order.row_domain_identity = 0x300385u;
        source.order.row_permutation = row_permutation.data();
        source.tiles.tile_schema_version = cp::warp_tile_schema_version;
        source.tiles.record_schema_version = cp::cell_block_record_schema_version;
        source.tiles.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        source.tiles.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        source.tiles.order_schema_version = cp::local_cell_order_schema_version;
        source.tiles.tile_identity = 0x400485u;
        source.tiles.feature_block_geometry_identity = 0x100185u;
        source.tiles.ordering_identity = 0x200285u;
        source.tiles.full_row_count = 5u;
        source.tiles.row_count = 5u;
        source.tiles.feature_count = 5u;
        source.tiles.feature_block_count = 1u;
        source.tiles.tile_row_width = 4u;
        source.tiles.tile_count = 2u;
        source.tiles.nnz_count = 8u;
        source.tiles.tile_block_count = 2u;
        source.tiles.row_block_entry_count = 4u;
        source.tiles.value_size_bytes = sizeof(__half);
        source.tiles.feature_axis_fingerprint = 0x500585u;
        source.tiles.feature_axis_fingerprint_version = 1u;
        source.tiles.row_domain_identity = 0x300385u;
        source.tiles.tile_block_offsets = tile_offsets.data();
        source.tiles.tile_block_ids = tile_blocks.data();
        source.tiles.tile_block_cell_masks = cell_masks.data();
        source.tiles.block_row_entry_offsets = entry_offsets.data();
        source.tiles.row_block_gene_masks = gene_masks.data();
        source.tiles.row_block_value_offsets = value_offsets.data();
        source.tiles.values = values_a.data();

        cm::feature_major_projection_build_request forward_request{};
        forward_request.structure_identity = structure_id;
        forward_request.runtime_structure = structure_handle;
        forward_request.structure_epoch_value = epoch;
        forward_request.projection_identity = forward_id;
        forward_request.runtime_projection = forward_handle;
        forward_request.source = source;
        cm::feature_major_projection_requirements forward_required{};
        require(cm::query_feature_major_projection_requirements_host(
            forward_request, &forward_required), "query FMP1");
        forward_payload.resize(forward_required.payload_bytes);
        require(cm::build_feature_major_projection_host(forward_request,
            {forward_payload.data(), forward_payload.size()}, &forward_view),
            "build FMP1");

        cm::transpose_projection_build_request transpose_request{};
        transpose_request.projection_identity = transpose_id;
        transpose_request.runtime_projection = transpose_handle;
        transpose_request.forward = forward_view;
        cm::transpose_projection_requirements transpose_required{};
        require(cm::query_transpose_projection_requirements_host(
            transpose_request, &transpose_required), "query CTP1");
        transpose_payload.resize(transpose_required.payload_bytes);
        require(cm::build_transpose_projection_host(transpose_request,
            {transpose_payload.data(), transpose_payload.size()},
            &transpose_view), "build CTP1");
    }
};

void test_projection(const fixture &f) {
    const auto &view = f.transpose_view;
    const std::array<cp::u32, 6> offsets{{0u, 2u, 3u, 6u, 8u, 8u}};
    const std::array<cp::u32, 8> rows{{0u, 3u, 2u, 0u, 2u, 4u, 2u, 4u}};
    const std::array<cp::u32, 8> forward{{0u, 1u, 2u, 3u, 4u, 6u, 5u, 7u}};
    const std::array<cp::u32, 8> logical{{0u, 5u, 2u, 1u, 3u, 6u, 4u, 7u}};
    const std::array<cp::u32, 8> inverse{{0u, 3u, 2u, 4u, 6u, 1u, 5u, 7u}};
    require(std::equal(offsets.begin(), offsets.end(), view.feature_offsets)
        && std::equal(rows.begin(), rows.end(), view.execution_row_ids)
        && std::equal(forward.begin(), forward.end(),
            view.forward_value_positions)
        && std::equal(logical.begin(), logical.end(),
            view.transpose_to_logical)
        && std::equal(inverse.begin(), inverse.end(),
            view.logical_to_transpose),
        "exact transpose projection and bidirectional edge identity");
    require(execution::same_identity(view.header.structure_identity,
            f.forward_view.header.structure_identity)
        && view.header.structure_epoch == f.forward_view.header.structure_epoch
        && execution::same_identity(view.header.forward_projection_identity,
            f.forward_id)
        && execution::same_identity(view.header.projection_identity,
            f.transpose_id)
        && !execution::same_identity(view.header.projection_identity,
            view.header.forward_projection_identity),
        "shared structure and distinct forward/transpose projection identities");
    execution::relation_structure relation{f.structure_handle, f.epoch,
        axis(10u), axis(20u), {1u, 1u}, 8u};
    const auto map = cm::transpose_value_position_map(view,
        {execution::residency_kind::host, {}, -1, 0u});
    require(execution::validate_value_position_map(relation, map)
            == execution::order_validation_code::ok,
        "transpose value map contract");
    cm::transpose_projection_view rejected{};
    execution::projection_id mismatch = f.transpose_id;
    mismatch.low += 1u;
    require(cm::validate_transpose_projection_payload_host(
            f.transpose_payload.data(), f.transpose_payload.size(),
            f.structure_id, f.epoch, mismatch, f.forward_id,
            f.structure_handle, f.transpose_handle, f.forward_handle,
            &rejected).code
            == cm::physical_view_status_code::incompatible_identity,
        "mismatched transpose ProjectionId rejection");
    execution::structure_epoch stale = f.epoch;
    stale.value += 1u;
    require(cm::validate_transpose_projection_payload_host(
            f.transpose_payload.data(), f.transpose_payload.size(),
            f.structure_id, stale, f.transpose_id, f.forward_id,
            f.structure_handle, f.transpose_handle, f.forward_handle,
            &rejected).code
            == cm::physical_view_status_code::incompatible_identity,
        "stale transpose structure epoch rejection");
}

void test_execution_image(const fixture &f) {
    const std::array<unsigned char, 8> domain{{1,2,3,4,5,6,7,8}};
    const std::array<unsigned char, 8> order{{2,3,4,5,6,7,8,9}};
    const std::array<unsigned char, 8> relation{{3,4,5,6,7,8,9,10}};
    const std::array<unsigned char, 8> geometry{{4,5,6,7,8,9,10,11}};
    std::array<cp::u32, 16> transpose_map{};
    std::copy(f.transpose_view.logical_to_transpose,
        f.transpose_view.logical_to_transpose + 8u, transpose_map.begin());
    std::copy(f.transpose_view.transpose_to_logical,
        f.transpose_view.transpose_to_logical + 8u,
        transpose_map.begin() + 8u);
    persistence::execution_section_source sections[6]{};
    sections[0] = {persistence::execution_section_kind::domain_table,
        1u, 0u, 8u, 1u, 1u, domain.data(), domain.size(), 1u, 8u};
    sections[1] = {persistence::execution_section_kind::order_partition_table,
        1u, 0u, 8u, 2u, 1u, order.data(), order.size(), 1u, 8u};
    sections[2] = {persistence::execution_section_kind::relation_structure,
        1u, 0u, 8u, 3u, 1u, relation.data(), relation.size(), 1u, 8u};
    sections[3] = {persistence::execution_section_kind::semantic_geometry,
        1u, 0u, 8u, 4u, 1u, geometry.data(), geometry.size(), 1u, 8u};
    sections[4] = {persistence::execution_section_kind::projection_payload,
        cm::transpose_projection_schema_version,
        persistence::directory_device_readable, 64u,
        f.transpose_id.low, f.transpose_id.high,
        f.transpose_payload.data(), f.transpose_payload.size(), 0u, 0u};
    sections[5] = {persistence::execution_section_kind::transpose_value_map,
        1u, persistence::directory_device_readable, 64u,
        0x7185u, 0x7285u, transpose_map.data(),
        transpose_map.size() * sizeof(cp::u32),
        static_cast<cp::u32>(transpose_map.size()), sizeof(cp::u32)};
    persistence::execution_projection_source projection{};
    auto &entry = projection.entry;
    entry.identity_low = f.transpose_id.low;
    entry.identity_high = f.transpose_id.high;
    entry.kind = persistence::execution_projection_kind::transpose_backward;
    entry.schema_version = cm::transpose_projection_schema_version;
    entry.flags = persistence::directory_device_readable
        | persistence::projection_transpose_capable;
    entry.operation_family = static_cast<std::uint32_t>(
        core::operation_kind::sparse_dense_multiply);
    entry.storage_type = static_cast<std::uint16_t>(execution::numeric_type::f16);
    entry.compute_type = static_cast<std::uint16_t>(execution::numeric_type::f32);
    entry.accumulation_type = static_cast<std::uint16_t>(
        execution::numeric_type::f32);
    entry.orientation = 2u;
    entry.architecture_class = 70u;
    entry.payload_section = 4u;
    entry.forward_map_section = persistence::invalid_directory_index;
    entry.transpose_map_section = 5u;
    entry.scheduling_summary_section = persistence::invalid_directory_index;
    entry.capability_section = persistence::invalid_directory_index;
    persistence::execution_image_v2_build_request request{};
    request.structure_identity = f.structure_id;
    request.structure_epoch = f.epoch.value;
    request.semantic_geometry_identity = {0x8185u, 0x8285u};
    request.projection_catalog_identity = {0x9185u, 0x9285u};
    request.source_axis = persistent_axis(100u);
    request.destination_axis = persistent_axis(200u);
    request.sections = sections;
    request.section_count = 6u;
    request.projections = &projection;
    request.projection_count = 1u;
    persistence::execution_image_v2_requirements required{};
    require(persistence::query_execution_image_v2_requirements_host(
        request, &required), "query CPE2 transpose image");
    std::vector<unsigned char> image(required.image_bytes);
    persistence::execution_image_v2_view view{};
    require(persistence::build_execution_image_v2_host(request,
        {image.data(), image.size()}, &view), "build CPE2 transpose image");
    persistence::prebound_projection_view_v1 prebound{};
    require(persistence::prebind_execution_projection_host(view, 0u,
        &prebound), "prebind CPE2 transpose projection");
    require(prebound.payload_bytes == f.transpose_payload.size()
        && prebound.transpose_map_bytes
            == transpose_map.size() * sizeof(cp::u32)
        && prebound.forward_map == nullptr
        && prebound.descriptor.kind
            == persistence::execution_projection_kind::transpose_backward,
        "CPE2 transpose payload/map and capability binding");
    cm::transpose_projection_view typed{};
    require(cm::validate_transpose_projection_payload_host(prebound.payload,
        prebound.payload_bytes, f.structure_id, f.epoch, f.transpose_id,
        f.forward_id, f.structure_handle, f.transpose_handle,
        f.forward_handle, &typed), "typed CPE2 transpose validation");
}

std::vector<double> reference(const fixture &f,
    const std::vector<__half> &values,
    const std::vector<float> &rows) {
    const std::array<std::uint64_t, 6> row_offsets{{0u, 2u, 2u, 5u, 6u, 8u}};
    const std::array<std::uint32_t, 8> feature_ids{{0u, 2u, 1u, 2u,
        3u, 0u, 2u, 3u}};
    cm::spmm_request request{};
    request.m = 5u;
    request.k = 5u;
    request.n = 1u;
    request.sparse_nnz = 8u;
    request.transpose_sparse = cm::transpose_kind::transpose;
    request.sparse_structure.identity_version = 1u;
    request.sparse_structure.value = 0x8585u;
    request.dense_rhs_leading_dimension = 1u;
    request.output_leading_dimension = 1u;
    request.sparse_storage_type_code = cellerator::real::value_f16;
    request.dense_storage_type_code = cellerator::real::value_f32;
    request.output_storage_type_code = cellerator::real::value_f32;
    request.compute_type_code = cellerator::real::value_f32;
    request.accumulation_type_code = cellerator::real::value_f32;
    request.alpha = cm::make_scalar(1.0f);
    request.beta = cm::make_scalar(0.0f);
    request.workspace.kind = cm::workspace_policy_kind::no_additional_workspace;
    request.reuse.kind = cm::expected_reuse_kind::persistent;
    request.reuse.expected_run_count = 0u;
    request.sparse_feature_order.kind = cm::feature_order_kind::packed;
    request.sparse_feature_order.feature_count = 5u;
    request.sparse_feature_order.feature_axis_identity_version = 1u;
    request.sparse_feature_order.feature_axis_identity = 0x500585u;
    request.sparse_feature_order.packing_geometry_identity = 0x100185u;
    request.dense_feature_order = request.sparse_feature_order;
    const cm::logical_csr_view sparse{5u, 5u, 8u, row_offsets.data(),
        feature_ids.data(), values.data(), cellerator::real::value_f16};
    const cm::logical_dense_view dense{rows.data(), 5u, 1u, 1u,
        cm::dense_layout_kind::row_major, cellerator::real::value_f32};
    const cm::logical_dense_view initial{};
    std::vector<double> result(5u);
    require(cm::build_spmm_reference(request, sparse, dense, initial,
        nullptr, result.data(), result.size()),
        "independent transposed SpMM referee");
    (void)f;
    return result;
}

void run_candidate(const fixture &f, int device) {
    device_array<unsigned char> device_payload(f.transpose_payload.size());
    require_cuda(cudaMemcpy(device_payload.data, f.transpose_payload.data(),
        f.transpose_payload.size(), cudaMemcpyHostToDevice), "upload CTP1");
    cm::transpose_projection_view device_view{};
    require(cm::rebind_transpose_projection(f.transpose_view,
        device_payload.data, f.transpose_payload.size(), &device_view),
        "rebind CTP1");
    std::vector<__half> packed_a(8u), packed_b(8u);
    require(cm::pack_feature_major_values_host(f.forward_view,
        f.values_a.data(), f.values_a.size() * sizeof(__half),
        {packed_a.data(), packed_a.size() * sizeof(__half)}),
        "pack forward generation A");
    require(cm::pack_feature_major_values_host(f.forward_view,
        f.values_b.data(), f.values_b.size() * sizeof(__half),
        {packed_b.data(), packed_b.size() * sizeof(__half)}),
        "pack forward generation B");
    const std::vector<float> row_input{1.0f, 2.0f, -1.0f, 0.5f, 3.0f};
    device_array<__half> device_values_a(8u), device_values_b(8u);
    device_array<float> device_input(5u), device_output(5u);
    upload(device_values_a, packed_a);
    upload(device_values_b, packed_b);
    upload(device_input, row_input);

    const auto feature_axis = axis(10u);
    const auto row_axis = axis(20u);
    const auto dense_axis = axis(30u);
    core::candidate_registry registry{};
    require(core::register_transpose_backward_n1_candidate(&registry),
        "register transpose candidate");
    require(registry.size == 1u
        && registry.candidates[0].projection
            == core::projection_kind::transpose_or_backward
        && registry.candidates[0].transient_bytes == 0u,
        "truthful transpose candidate capabilities");
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {f.structure_id, f.structure_handle, f.epoch};
    core::projection_key projection{f.transpose_id, f.transpose_handle,
        core::projection_kind::transpose_or_backward,
        cm::transpose_projection_schema_version,
        cm::transpose_projection_variant};
    core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::sparse_dense_multiply, 0u,
        core::transpose_backward_n1_candidate_id, 1u, 1u, 8u};
    core::transpose_backward_prepared_state state{};
    core::prepared_operation prepared{};
    require(core::prepare_transpose_backward_n1_operation(problem, structures,
        projection, numeric(), {true, true, true, true, 8u, 0u, 0u},
        device_view, device, feature_axis, row_axis, dense_axis,
        &state, &prepared), "prepare transpose backward");
    require(prepared.binding_contract.workspace.minimum_bytes == 0u
        && prepared.binding_contract.output_orders[0].transition
            == execution::order_transition_kind::preserve
        && prepared.binding_contract.output_effects[0].update
            == execution::output_update_kind::overwrite,
        "transpose output order/effect and zero workspace");

    execution::relation_structure relation{f.structure_handle, f.epoch,
        feature_axis, row_axis, {1u, 1u}, 8u};
    execution::value_plane plane{};
    plane.structure = f.structure_handle;
    plane.structure_epoch_value = f.epoch;
    plane.values = device_values_a.data;
    plane.location = device_location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 8u;
    plane.value_bytes = 8u * sizeof(__half);
    execution::value_binding binding{&plane, plane.generation};
    execution::biological_operand_view input{}, output{};
    input.kind = execution::operand_kind::dense_tensor;
    input.storage.dense = dense_matrix(device_input.data, row_axis,
        dense_axis, 5u, device);
    output.kind = execution::operand_kind::dense_tensor;
    output.storage.dense = dense_matrix(device_output.data, feature_axis,
        dense_axis, 5u, device);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");
    execution::launch_bindings launch{};
    launch.structures = &relation;
    launch.inputs = &input;
    launch.outputs = &output;
    launch.values = &binding;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.value_count = 1u;
    launch.structure_count = 1u;
    launch.stream = {stream, device, 0u};
    launch.workspace = {nullptr, 0u, device_location(device)};

    auto run_generation = [&](const std::vector<__half> &logical_values,
                              const char *message) {
        require(core::run_prepared_operation(prepared, launch), message);
        require_cuda(cudaStreamSynchronize(stream), "synchronize backward");
        std::vector<float> actual(5u);
        require_cuda(cudaMemcpy(actual.data(), device_output.data,
            actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "download backward output");
        const auto expected = reference(f, logical_values, row_input);
        cm::logical_dense_view actual_view{actual.data(), 5u, 1u, 1u,
            cm::dense_layout_kind::row_major, cellerator::real::value_f32};
        cm::numerical_comparison comparison{};
        require(cm::compare_spmm_reference(expected.data(), expected.size(),
            actual_view, {1.0e-5, 1.0e-5, 1.0e-30}, &comparison),
            "compare transposed referee");
        require(comparison.within_tolerance && comparison.mismatch_count == 0u
            && actual[4] == 0.0f,
            "transpose numerical parity including empty feature");
    };
    run_generation(f.values_a, "run backward generation A");
    plane.values = device_values_b.data;
    plane.generation = {2u};
    binding.expected_generation = plane.generation;
    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "memory before generation B");
    run_generation(f.values_b, "run backward generation B");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "memory after generation B");
    require(free_before == free_after && total_before == total_after,
        "steady backward allocated or converted storage");
    binding.expected_generation = {3u};
    require(core::run_prepared_operation(prepared, launch).binding
            == execution::binding_validation_code::stale_value,
        "stale backward value generation rejection");
    binding.expected_generation = plane.generation;
    relation.epoch.value += 1u;
    require(core::run_prepared_operation(prepared, launch).code
            == core::operation_status_code::stale_structure,
        "stale backward structure rejection");

    core::projection_key mismatched = projection;
    mismatched.persistent.low += 1u;
    core::transpose_backward_prepared_state rejected_state{};
    core::prepared_operation rejected{};
    require(core::prepare_transpose_backward_n1_operation(problem, structures,
            mismatched, numeric(), {}, device_view, device, feature_axis,
            row_axis, dense_axis, &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_problem,
        "mismatched prepared transpose projection rejection");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    fixture f;
    test_projection(f);
    test_execution_image(f);
    run_candidate(f, device);
    return 0;
}
