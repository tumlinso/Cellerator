#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>
#include <Cellerator/compute/candidate/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>
#include <Cellerator/compat/cp_math_v1/referee.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace cp = cellpack;
namespace persistence = cellpack::persistence;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "feature_major_small_n_candidate_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "feature_major_small_n_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", binding=" << static_cast<unsigned>(status.binding)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cm::physical_view_status status, const char *message) {
    if (status) return;
    std::cerr << "feature_major_small_n_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cp::validation_result status, const char *message) {
    if (status) return;
    std::cerr << "feature_major_small_n_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cm::referee_status status, const char *message) {
    if (status) return;
    std::cerr << "feature_major_small_n_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "feature_major_small_n_candidate_test: " << message << ": "
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

execution::device_location device_location(int ordinal) {
    return {execution::residency_kind::device, {}, ordinal, 0u};
}

execution::dense_tensor_view dense_matrix(void *pointer,
    execution::numeric_type type, execution::axis_identity major_axis,
    execution::axis_identity minor_axis, std::uint64_t rows,
    std::uint64_t columns, int ordinal) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(ordinal);
    view.value_type = type;
    view.rank = 2u;
    view.axes[0] = major_axis;
    view.axes[1] = minor_axis;
    view.shape[0] = rows;
    view.shape[1] = columns;
    view.stride[0] = static_cast<std::int64_t>(columns);
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

struct host_source_fixture {
    std::vector<cp::u32> feature_offsets{0u, 4u};
    std::vector<cp::u32> feature_permutation{0u, 1u, 2u, 3u};
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
    cp::persistent_packing_payload_view payload{};

    host_source_fixture() {
        for (float value : {1.0f, 2.0f, 3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f})
            values_a.push_back(__float2half(value));
        for (float value : {2.0f, -1.0f, 0.5f, 3.0f,
                            -2.0f, 4.0f, 1.5f, -0.5f})
            values_b.push_back(__float2half(value));
        payload.payload_schema_version =
            cp::persistent_packing_payload_schema_version;
        payload.payload_kind = cp::persistent_packing_payload_kind;
        payload.payload_identity = 0x43504b3175u;
        payload.image_base = &image_byte;
        payload.image_bytes = 1u;
        payload.plan.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        payload.plan.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        payload.plan.feature_count = 4u;
        payload.plan.feature_block_count = 1u;
        payload.plan.feature_block_geometry_identity = 0x100175u;
        payload.plan.feature_block_offsets = feature_offsets.data();
        payload.plan.feature_permutation = feature_permutation.data();
        payload.order.order_schema_version = cp::local_cell_order_schema_version;
        payload.order.signature_algorithm_version =
            cp::local_cell_signature_algorithm_version;
        payload.order.kind = cp::local_cell_order_kind::inferred_minhash;
        payload.order.window_size = 4u;
        payload.order.group_width = 4u;
        payload.order.ordering_identity = 0x200275u;
        payload.order.full_row_count = 5u;
        payload.order.row_count = 5u;
        payload.order.feature_block_count = 1u;
        payload.order.feature_block_geometry_identity = 0x100175u;
        payload.order.row_domain_identity = 0x300375u;
        payload.order.row_permutation = row_permutation.data();
        payload.tiles.tile_schema_version = cp::warp_tile_schema_version;
        payload.tiles.record_schema_version = cp::cell_block_record_schema_version;
        payload.tiles.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        payload.tiles.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        payload.tiles.order_schema_version = cp::local_cell_order_schema_version;
        payload.tiles.tile_identity = 0x400475u;
        payload.tiles.feature_block_geometry_identity = 0x100175u;
        payload.tiles.ordering_identity = 0x200275u;
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
        payload.tiles.feature_axis_fingerprint = 0x500575u;
        payload.tiles.feature_axis_fingerprint_version = 1u;
        payload.tiles.row_domain_identity = 0x300375u;
        payload.tiles.tile_block_offsets = tile_offsets.data();
        payload.tiles.tile_block_ids = tile_blocks.data();
        payload.tiles.tile_block_cell_masks = cell_masks.data();
        payload.tiles.block_row_entry_offsets = entry_offsets.data();
        payload.tiles.row_block_gene_masks = gene_masks.data();
        payload.tiles.row_block_value_offsets = value_offsets.data();
        payload.tiles.values = values_a.data();
    }
};

struct projection_fixture {
    execution::structure_id structure_id{0x1175u, 0x1275u};
    execution::structure_handle structure_handle{21u, 1u};
    execution::structure_epoch epoch{7u};
    execution::projection_id projection_id{0x3175u, 0x3275u};
    execution::projection_handle projection_handle{41u, 1u};
    cm::feature_major_projection_build_request request{};
    cm::feature_major_projection_requirements requirements{};
    std::vector<unsigned char> payload;
    cm::feature_major_projection_view host_view{};

    explicit projection_fixture(host_source_fixture &source) {
        request.structure_identity = structure_id;
        request.runtime_structure = structure_handle;
        request.structure_epoch_value = epoch;
        request.projection_identity = projection_id;
        request.runtime_projection = projection_handle;
        request.source = source.payload;
        require(cm::query_feature_major_projection_requirements_host(
            request, &requirements), "query feature-major projection");
        payload.resize(requirements.payload_bytes);
        require(cm::build_feature_major_projection_host(request,
            {payload.data(), payload.size()}, &host_view),
            "build feature-major projection");
    }
};

void test_exact_projection_and_value_reconstruction(
    const host_source_fixture &source,
    const projection_fixture &projection) {
    const auto &view = projection.host_view;
    require(view.header.feature_record_count == 6u
        && view.header.nnz_count == 8u
        && view.tile_feature_offsets[0] == 0u
        && view.tile_feature_offsets[1] == 4u
        && view.tile_feature_offsets[2] == 6u,
        "feature-major record counts and tile offsets");
    const std::vector<cp::u32> expected_features{0u, 1u, 2u, 3u, 2u, 3u};
    const std::vector<cp::u32> expected_masks{0x9u, 0x4u, 0x5u, 0x4u, 0x1u, 0x1u};
    const std::vector<cp::u32> expected_value_offsets{0u, 2u, 3u, 5u, 6u, 7u, 8u};
    const std::vector<cp::u32> expected_source_positions{0u, 5u, 2u, 1u, 3u, 4u, 6u, 7u};
    require(std::equal(expected_features.begin(), expected_features.end(),
            view.execution_feature_ids)
        && std::equal(expected_masks.begin(), expected_masks.end(),
            view.participating_row_masks)
        && std::equal(expected_value_offsets.begin(), expected_value_offsets.end(),
            view.feature_value_offsets)
        && std::equal(expected_source_positions.begin(),
            expected_source_positions.end(), view.source_value_positions),
        "exact feature-major reconstruction");

    std::vector<__half> packed(source.values_a.size());
    require(cm::pack_feature_major_values_host(view, source.values_a.data(),
        source.values_a.size() * sizeof(__half),
        {packed.data(), packed.size() * sizeof(__half)}),
        "pack first feature-major value generation");
    for (std::size_t index = 0u; index < packed.size(); ++index)
        require(__half2float(packed[index]) == __half2float(
            source.values_a[expected_source_positions[index]]),
            "feature-major value map parity");
}

void test_execution_image_integration(const projection_fixture &projection) {
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
        projection.projection_id.low, projection.projection_id.high,
        projection.payload.data(), projection.payload.size(), 0u, 0u};
    persistence::execution_projection_source projection_source{};
    auto &entry = projection_source.entry;
    entry.identity_low = projection.projection_id.low;
    entry.identity_high = projection.projection_id.high;
    entry.kind = persistence::execution_projection_kind::native_feature_major;
    entry.schema_version = cm::feature_major_projection_schema_version;
    entry.flags = persistence::directory_device_readable
        | persistence::projection_forward_capable;
    entry.operation_family = static_cast<std::uint32_t>(
        core::operation_kind::sparse_dense_multiply);
    entry.storage_type = static_cast<std::uint16_t>(execution::numeric_type::f16);
    entry.compute_type = static_cast<std::uint16_t>(execution::numeric_type::f32);
    entry.accumulation_type = static_cast<std::uint16_t>(execution::numeric_type::f32);
    entry.orientation = 1u;
    entry.architecture_class = 70u;
    entry.payload_section = 4u;
    entry.forward_map_section = persistence::invalid_directory_index;
    entry.transpose_map_section = persistence::invalid_directory_index;
    entry.scheduling_summary_section = persistence::invalid_directory_index;
    entry.capability_section = persistence::invalid_directory_index;

    persistence::execution_image_v2_build_request request{};
    request.structure_identity = projection.structure_id;
    request.structure_epoch = projection.epoch.value;
    request.semantic_geometry_identity = {0x5175u, 0x5275u};
    request.projection_catalog_identity = {0x6175u, 0x6275u};
    request.source_axis = persistent_axis(100u);
    request.destination_axis = persistent_axis(200u);
    request.sections = sections;
    request.section_count = 5u;
    request.projections = &projection_source;
    request.projection_count = 1u;
    persistence::execution_image_v2_requirements required{};
    require(persistence::query_execution_image_v2_requirements_host(
        request, &required), "query CPE2 feature-major image");
    std::vector<unsigned char> image(required.image_bytes);
    persistence::execution_image_v2_view image_view{};
    require(persistence::build_execution_image_v2_host(request,
        {image.data(), image.size()}, &image_view),
        "build CPE2 feature-major image");
    persistence::prebound_projection_view_v1 prebound{};
    require(persistence::prebind_execution_projection_host(
        image_view, 0u, &prebound), "prebind CPE2 feature-major projection");
    cm::feature_major_projection_view typed{};
    require(cm::validate_feature_major_projection_payload_host(
        prebound.payload, prebound.payload_bytes, projection.structure_id,
        projection.epoch, projection.projection_id,
        projection.structure_handle, projection.projection_handle, &typed),
        "typed CPE2 feature-major payload validation");
    require(typed.header.feature_record_count
            == projection.host_view.header.feature_record_count
        && typed.header.projection_identity.low == projection.projection_id.low,
        "CPE2 projection identity and typed payload parity");
}

void test_registry_and_planner(
    const core::operation_problem &problem,
    const core::structure_set_key &structures,
    const core::projection_key &projection_key) {
    core::candidate_registry registry{};
    require(core::register_row_masked_n1_candidate(&registry),
        "row-masked coexistence registration");
    require(core::register_csr_fallback_candidate(&registry),
        "CSR coexistence registration");
    require(core::register_feature_major_small_n_candidate(&registry),
        "feature-major registration");
    require(core::register_feature_major_cta_medium_n_candidate(&registry),
        "feature-major CTA registration");
    require(registry.size == 4u
        && registry.candidates[0].projection
            == core::projection_kind::native_row_masked
        && registry.candidates[1].projection == core::projection_kind::csr
        && registry.candidates[2].operation
            == core::operation_kind::sparse_dense_multiply
        && registry.candidates[2].projection
            == core::projection_kind::native_feature_major
        && registry.candidates[2].transient_bytes == 0u
        && (registry.candidates[2].capability_flags
            & core::candidate_graph_capture) != 0u,
        "truthful feature-major capability and candidate coexistence");
    require(registry.candidates[3].projection
            == core::projection_kind::native_feature_major
        && registry.candidates[3].operation
            == core::operation_kind::sparse_dense_multiply
        && registry.candidates[3].transient_bytes == 0u
        && !core::same_stable_id(registry.candidates[2].identity,
            registry.candidates[3].identity),
        "CTA schedule has a distinct candidate identity over FMP1");

    planner::planner_candidate candidate{};
    candidate.identity = registry.candidates[2].identity;
    candidate.name = registry.candidates[2].name;
    candidate.operation = &registry.candidates[2];
    candidate.projection = projection_key;
    candidate.analytical.projection_construction_ns = 1.0;
    candidate.analytical.kernel_ns = 1.0;
    candidate.analytical.persistent_bytes =
        registry.candidates[2].persistent_bytes;
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
        && result.selected == &candidate && result.tuning_skipped,
        "feature-major real planner candidate");
}

std::vector<double> build_reference(const host_source_fixture &source,
    const std::vector<__half> &values,
    const std::vector<float> &rhs, std::uint32_t dense_width) {
    const std::array<std::uint64_t, 6> row_offsets{{0u, 2u, 2u, 5u, 6u, 8u}};
    const std::array<std::uint32_t, 8> feature_ids{{0u, 2u, 1u, 2u,
        3u, 0u, 2u, 3u}};
    cm::spmm_request request{};
    request.m = 5u;
    request.k = 4u;
    request.n = dense_width;
    request.sparse_nnz = 8u;
    request.sparse_structure.identity_version = 1u;
    request.sparse_structure.value = 0x7575u;
    request.dense_rhs_leading_dimension = dense_width;
    request.output_leading_dimension = dense_width;
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
    request.sparse_feature_order.feature_count = 4u;
    request.sparse_feature_order.feature_axis_identity_version = 1u;
    request.sparse_feature_order.feature_axis_identity = 0x500575u;
    request.sparse_feature_order.packing_geometry_identity = 0x100175u;
    request.dense_feature_order = request.sparse_feature_order;
    const cm::logical_csr_view sparse{5u, 4u, 8u, row_offsets.data(),
        feature_ids.data(), values.data(), cellerator::real::value_f16};
    const cm::logical_dense_view dense{rhs.data(), 4u, dense_width,
        dense_width, cm::dense_layout_kind::row_major,
        cellerator::real::value_f32};
    const cm::logical_dense_view initial{};
    std::vector<double> reference(5u * dense_width);
    require(cm::build_spmm_reference(request, sparse, dense, initial,
        nullptr, reference.data(), reference.size()),
        "independent SpMM referee");
    (void)source;
    return reference;
}

void run_supported_boundary(std::uint32_t dense_width,
    const host_source_fixture &source,
    const projection_fixture &projection,
    int device) {
    device_array<unsigned char> device_payload(projection.payload.size());
    require_cuda(cudaMemcpy(device_payload.data, projection.payload.data(),
        projection.payload.size(), cudaMemcpyHostToDevice),
        "upload feature-major payload");
    cm::feature_major_projection_view device_view{};
    require(cm::rebind_feature_major_projection(projection.host_view,
        device_payload.data, projection.payload.size(), &device_view),
        "rebind feature-major device projection");

    std::vector<float> rhs(4u * dense_width);
    for (std::uint32_t feature = 0u; feature < 4u; ++feature)
        for (std::uint32_t column = 0u; column < dense_width; ++column)
            rhs[feature * dense_width + column] =
                static_cast<float>((feature + 1u) * 3u + column) * 0.125f;
    std::vector<__half> packed_a(8u), packed_b(8u);
    require(cm::pack_feature_major_values_host(projection.host_view,
        source.values_a.data(), source.values_a.size() * sizeof(__half),
        {packed_a.data(), packed_a.size() * sizeof(__half)}),
        "pack generation A");
    require(cm::pack_feature_major_values_host(projection.host_view,
        source.values_b.data(), source.values_b.size() * sizeof(__half),
        {packed_b.data(), packed_b.size() * sizeof(__half)}),
        "pack generation B");
    device_array<float> device_rhs(rhs.size());
    device_array<__half> device_values_a(packed_a.size());
    device_array<__half> device_values_b(packed_b.size());
    device_array<float> device_output(5u * dense_width);
    upload(device_rhs, rhs);
    upload(device_values_a, packed_a);
    upload(device_values_b, packed_b);

    const execution::axis_identity feature_axis = axis(10u);
    const execution::axis_identity row_axis = axis(20u);
    const execution::axis_identity dense_axis = axis(30u);
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {projection.structure_id,
        projection.structure_handle, projection.epoch};
    const core::projection_key projection_key{projection.projection_id,
        projection.projection_handle,
        core::projection_kind::native_feature_major,
        cm::feature_major_projection_schema_version,
        cm::feature_major_projection_variant};
    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::sparse_dense_multiply, 0u, {75u, dense_width},
        1u, 1u, static_cast<std::uint64_t>(8u) * dense_width};
    if (dense_width == cm::feature_major_small_n_minimum)
        test_registry_and_planner(problem, structures, projection_key);

    const bool medium_n = dense_width
        >= core::feature_major_cta_medium_n_minimum;
    core::feature_major_small_n_prepared_state state{};
    core::prepared_operation prepared{};
    const core::prepare_policy policy{true, true, true, true, 8u, 0u, 0u};
    const core::operation_status prepare_status = medium_n
        ? core::prepare_feature_major_cta_medium_n_operation(problem, structures,
            projection_key, numeric(), policy, device_view, device, dense_width,
            feature_axis, row_axis, dense_axis, &state, &prepared)
        : core::prepare_feature_major_small_n_operation(problem, structures,
            projection_key, numeric(), policy, device_view, device, dense_width,
            feature_axis, row_axis, dense_axis, &state, &prepared);
    require(prepare_status, medium_n
        ? "prepare feature-major CTA medium-N operation"
        : "prepare feature-major small-N operation");
    require(prepared.binding_contract.workspace.minimum_bytes == 0u
        && prepared.binding_contract.output_order_count == 2u
        && prepared.binding_contract.output_orders[0].transition
            == execution::order_transition_kind::preserve
        && prepared.binding_contract.output_orders[1].transition
            == execution::order_transition_kind::preserve
        && prepared.binding_contract.output_effects[0].update
            == execution::output_update_kind::overwrite,
        "feature-major workspace, order, and output effect");
    const void *const prepared_state = prepared.persistent.data;
    const void *const prepared_records = state.projection.execution_feature_ids;

    execution::relation_structure relation{};
    relation.identity = projection.structure_handle;
    relation.epoch = projection.epoch;
    relation.source_axis = feature_axis;
    relation.destination_axis = row_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = 8u;
    execution::value_plane plane{};
    plane.structure = relation.identity;
    plane.structure_epoch_value = relation.epoch;
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
    input.storage.dense = dense_matrix(device_rhs.data,
        execution::numeric_type::f32, feature_axis, dense_axis,
        4u, dense_width, device);
    output.kind = execution::operand_kind::dense_tensor;
    output.storage.dense = dense_matrix(device_output.data,
        execution::numeric_type::f32, row_axis, dense_axis,
        5u, dense_width, device);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(
        &stream, cudaStreamNonBlocking), "create stream");
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

    require(core::run_prepared_operation(prepared, launch),
        "feature-major generation A execution");
    require_cuda(cudaStreamSynchronize(stream), "synchronize generation A");
    std::vector<float> result(5u * dense_width);
    require_cuda(cudaMemcpy(result.data(), device_output.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download generation A");
    const std::vector<double> reference_a =
        build_reference(source, source.values_a, rhs, dense_width);
    cm::logical_dense_view result_view{result.data(), 5u, dense_width,
        dense_width, cm::dense_layout_kind::row_major,
        cellerator::real::value_f32};
    cm::numerical_comparison comparison{};
    require(cm::compare_spmm_reference(reference_a.data(), reference_a.size(),
        result_view, {1.0e-5, 1.0e-5, 1.0e-30}, &comparison),
        "compare generation A referee");
    require(comparison.within_tolerance && comparison.mismatch_count == 0u,
        "generation A numerical parity including empty row");

    plane.values = device_values_b.data;
    plane.generation = {2u};
    binding.expected_generation = plane.generation;
    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "memory before steady generation");
    require(core::run_prepared_operation(prepared, launch),
        "feature-major generation B execution");
    require_cuda(cudaStreamSynchronize(stream), "synchronize generation B");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "memory after steady generation");
    require(free_before == free_after && total_before == total_after,
        "steady execution allocated or converted storage");
    require_cuda(cudaMemcpy(result.data(), device_output.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download generation B");
    const std::vector<double> reference_b =
        build_reference(source, source.values_b, rhs, dense_width);
    require(cm::compare_spmm_reference(reference_b.data(), reference_b.size(),
        result_view, {1.0e-5, 1.0e-5, 1.0e-30}, &comparison),
        "compare generation B referee");
    require(comparison.within_tolerance && comparison.mismatch_count == 0u
        && prepared.persistent.data == prepared_state
        && state.projection.execution_feature_ids == prepared_records,
        "generation B parity and immutable prepared structure reuse");

    binding.expected_generation.value = 3u;
    require(core::run_prepared_operation(prepared, launch).binding
            == execution::binding_validation_code::stale_value,
        "stale feature-major value generation rejection");
    binding.expected_generation = plane.generation;
    relation.epoch.value += 1u;
    require(core::run_prepared_operation(prepared, launch).code
            == core::operation_status_code::stale_structure,
        "stale feature-major structure rejection");
    relation.epoch = projection.epoch;

    core::feature_major_small_n_prepared_state rejected_state{};
    core::prepared_operation rejected{};
    core::projection_key mismatched_projection = projection_key;
    mismatched_projection.persistent.low += 1u;
    const core::operation_status mismatched_status = medium_n
        ? core::prepare_feature_major_cta_medium_n_operation(problem, structures,
            mismatched_projection, numeric(), policy, device_view, device,
            dense_width, feature_axis, row_axis, dense_axis,
            &rejected_state, &rejected)
        : core::prepare_feature_major_small_n_operation(problem, structures,
            mismatched_projection, numeric(), policy, device_view, device,
            dense_width, feature_axis, row_axis, dense_axis,
            &rejected_state, &rejected);
    require(mismatched_status.code
            == core::operation_status_code::unsupported_problem,
        "mismatched feature-major ProjectionId rejection");
    core::operation_problem unsupported_problem = problem;
    const std::uint32_t below = medium_n ? 16u : 0u;
    const std::uint32_t above = medium_n ? 65u : 17u;
    unsupported_problem.logical_work_items = below == 0u ? 1u : 8u * below;
    const core::operation_status below_status = medium_n
        ? core::prepare_feature_major_cta_medium_n_operation(unsupported_problem,
            structures, projection_key, numeric(), policy, device_view, device,
            below, feature_axis, row_axis, dense_axis,
            &rejected_state, &rejected)
        : core::prepare_feature_major_small_n_operation(unsupported_problem,
            structures, projection_key, numeric(), policy, device_view, device,
            below, feature_axis, row_axis, dense_axis,
            &rejected_state, &rejected);
    require(below_status.code == core::operation_status_code::unsupported_problem,
        "N below feature-major regime rejection");
    unsupported_problem.logical_work_items = 8u * above;
    const core::operation_status above_status = medium_n
        ? core::prepare_feature_major_cta_medium_n_operation(unsupported_problem,
            structures, projection_key, numeric(), policy, device_view, device,
            above, feature_axis, row_axis, dense_axis,
            &rejected_state, &rejected)
        : core::prepare_feature_major_small_n_operation(unsupported_problem,
            structures, projection_key, numeric(), policy, device_view, device,
            above, feature_axis, row_axis, dense_axis,
            &rejected_state, &rejected);
    require(above_status.code == core::operation_status_code::unsupported_problem,
        "N above feature-major regime rejection");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    host_source_fixture source;
    projection_fixture projection(source);
    test_exact_projection_and_value_reconstruction(source, projection);
    test_execution_image_integration(projection);
    run_supported_boundary(cm::feature_major_small_n_minimum,
        source, projection, device);
    run_supported_boundary(cm::feature_major_small_n_maximum,
        source, projection, device);
    run_supported_boundary(core::feature_major_cta_medium_n_minimum,
        source, projection, device);
    run_supported_boundary(core::feature_major_cta_medium_n_maximum,
        source, projection, device);
    return 0;
}
