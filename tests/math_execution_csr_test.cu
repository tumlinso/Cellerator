#include <Cellerator/compat/cp_math_v1/packed_dense_operand.hh>
#include <Cellerator/compat/cp_math_v1/referee.hh>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace cm = cellerator::compute::math;
namespace cr = cellerator::real;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpMathExecutionCsrTest: " << message << '\n';
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "cpMathExecutionCsrTest: " << message << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

struct plan_fixture {
    cm::u32 offsets[3]{0u, 2u, 4u};
    cm::u32 permutation[4]{2u, 0u, 3u, 1u};

    cellpack::feature_weighted_row_reduction_plan_view view() const {
        cellpack::feature_weighted_row_reduction_plan_view result;
        result.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
        result.geometry_identity_version = cellpack::feature_block_geometry_identity_version;
        result.feature_count = 4u;
        result.feature_block_count = 2u;
        result.feature_block_geometry_identity = 0x03030303ull;
        result.feature_block_offsets = offsets;
        result.feature_permutation = permutation;
        return result;
    }
};

cm::feature_order_identity canonical_order() {
    cm::feature_order_identity order;
    order.feature_count = 4u;
    order.feature_axis_identity_version = 1u;
    order.feature_axis_identity = 0x0303a11ull;
    return order;
}

cm::execution_csr_view test_ordered_plan_adaptation(const plan_fixture &fixture) {
    static const cm::u32 rows[]{0u, 2u, 4u};
    const cm::u32 blocks[]{0u, 0u, 1u, 1u};
    const cm::u32 locals[]{0u, 1u, 0u, 1u};
    const cm::u32 canonical[]{2u, 0u, 3u, 1u};
    static const float values[]{2.0f, 1.0f, 4.0f, 3.0f};
    cellpack::ordered_plan_partition_view ordered;
    ordered.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
    ordered.full_row_count = 2u;
    ordered.row_count = 2u;
    ordered.feature_count = 4u;
    ordered.nnz_count = 4u;
    ordered.value_size_bytes = sizeof(float);
    ordered.feature_axis_fingerprint = canonical_order().feature_axis_identity;
    ordered.feature_axis_fingerprint_version = 1u;
    ordered.row_domain_identity = 0x303ull;
    ordered.row_offsets = rows;
    ordered.block_ids = blocks;
    ordered.local_feature_ids = locals;
    ordered.canonical_feature_ids = canonical;
    ordered.values = values;
    static cm::u32 execution_features[4]{};
    cm::execution_csr_view result;
    require(static_cast<bool>(cm::build_execution_csr_view_host(
        fixture.view(), ordered, {4u, execution_features}, &result)),
        "ordered-plan adaptation failed");
    for (cm::u32 i = 0u; i < 4u; ++i) {
        require(result.execution_feature_ids[i] == i,
            "ordered feature did not map to execution order");
    }
    require(result.values == values && result.row_offsets == rows
            && result.feature_order.kind == cm::feature_order_kind::packed
            && result.structure.value != 0u,
        "execution CSR did not preserve aliases or identities");
    return result;
}

cm::packed_dense_operand_view test_dense_pack(const plan_fixture &fixture) {
    static const float canonical_values[]{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    cm::canonical_dense_operand_view source;
    source.values = canonical_values;
    source.feature_count = 4u;
    source.column_count = 2u;
    source.leading_dimension = 2u;
    source.value_size_bytes = sizeof(float);
    source.feature_order = canonical_order();
    source.operand_identity = 0x30303ull;
    static float host_packed[8]{};
    cm::packed_dense_operand_view host;
    require(static_cast<bool>(cm::pack_dense_operand_host(
        fixture.view(), source, {sizeof(host_packed), host_packed}, &host)),
        "host dense pack failed");
    const float expected[]{5.0f, 6.0f, 1.0f, 2.0f, 7.0f, 8.0f, 3.0f, 4.0f};
    require(std::memcmp(host_packed, expected, sizeof(expected)) == 0,
        "host dense pack changed mathematical feature identity");

    cm::u32 *device_permutation = nullptr;
    float *device_source = nullptr, *device_packed = nullptr;
    cuda_require(cudaMalloc(&device_permutation, sizeof(fixture.permutation)),
        "allocate device permutation");
    cuda_require(cudaMalloc(&device_source, sizeof(canonical_values)),
        "allocate device source");
    cuda_require(cudaMalloc(&device_packed, sizeof(host_packed)),
        "allocate device packed output");
    cuda_require(cudaMemcpy(device_permutation, fixture.permutation,
        sizeof(fixture.permutation), cudaMemcpyHostToDevice), "copy permutation");
    cuda_require(cudaMemcpy(device_source, canonical_values,
        sizeof(canonical_values), cudaMemcpyHostToDevice), "copy dense source");
    auto device_plan = fixture.view();
    device_plan.feature_permutation = device_permutation;
    source.values = device_source;
    cm::packed_dense_operand_view device;
    require(static_cast<bool>(cm::pack_dense_operand_cuda(device_plan, source,
        {sizeof(host_packed), device_packed}, nullptr, &device)),
        "device dense pack failed");
    float device_result[8]{};
    cuda_require(cudaMemcpy(device_result, device_packed, sizeof(device_result),
        cudaMemcpyDeviceToHost), "copy device packed output");
    require(std::memcmp(device_result, expected, sizeof(expected)) == 0
            && device.operand_identity == host.operand_identity,
        "device dense pack disagrees with host or identity");
    cuda_require(cudaFree(device_packed), "free device packed output");
    cuda_require(cudaFree(device_source), "free device source");
    cuda_require(cudaFree(device_permutation), "free device permutation");
    return host;
}

void test_packed_math(
    const cm::execution_csr_view &sparse,
    const cm::packed_dense_operand_view &dense) {
    const cm::u64 rows[]{0u, 2u, 4u};
    cm::logical_csr_view logical_sparse{2u, 4u, 4u, rows,
        sparse.execution_feature_ids, sparse.values, cr::value_f32};
    cm::logical_dense_view logical_dense{dense.values, 4u, 2u, 2u,
        cm::dense_layout_kind::row_major, cr::value_f32};
    cm::spmm_request request;
    request.m = 2u;
    request.k = 4u;
    request.n = 2u;
    request.sparse_nnz = 4u;
    request.sparse_structure = sparse.structure;
    request.dense_rhs_leading_dimension = 2u;
    request.output_leading_dimension = 2u;
    request.sparse_storage_type_code = cr::value_f32;
    request.dense_storage_type_code = cr::value_f32;
    request.output_storage_type_code = cr::value_f32;
    request.compute_type_code = cr::value_f32;
    request.accumulation_type_code = cr::value_f32;
    request.alpha = cm::make_scalar(1.0f);
    request.beta = cm::make_scalar(0.0f);
    request.sparse_feature_order = sparse.feature_order;
    request.dense_feature_order = dense.feature_order;
    double output[4]{};
    require(static_cast<bool>(cm::build_spmm_reference(request, logical_sparse,
        logical_dense, {}, nullptr, output, 4u)), "packed reference failed");
    const double expected[]{11.0, 14.0, 37.0, 44.0};
    for (cm::u32 i = 0u; i < 4u; ++i) require(
        std::fabs(output[i] - expected[i]) < 1.0e-12,
        "X_packed W_packed differs from canonical math");
}

void test_lazy_cpk1_reconstruction(const plan_fixture &fixture) {
    const cm::u32 tile_offsets[]{0u, 2u};
    const cm::u32 blocks[]{0u, 1u}, cell_masks[]{1u, 2u};
    const cm::u32 entry_offsets[]{0u, 1u, 2u};
    const cm::u32 gene_masks[]{3u, 3u}, value_offsets[]{0u, 2u, 4u};
    const float values[]{2.0f, 1.0f, 4.0f, 3.0f};
    cellpack::persistent_packing_payload_view payload;
    payload.payload_schema_version = cellpack::persistent_packing_payload_schema_version;
    payload.payload_kind = cellpack::persistent_packing_payload_kind;
    payload.payload_identity = 0xc0303ull;
    payload.plan = fixture.view();
    auto &tiles = payload.tiles;
    tiles.feature_block_geometry_identity = fixture.view().feature_block_geometry_identity;
    tiles.global_row_begin = 0u;
    tiles.full_row_count = 2u;
    tiles.row_count = 2u;
    tiles.feature_count = 4u;
    tiles.feature_block_count = 2u;
    tiles.tile_row_width = 2u;
    tiles.tile_count = 1u;
    tiles.nnz_count = 4u;
    tiles.tile_block_count = 2u;
    tiles.row_block_entry_count = 2u;
    tiles.value_size_bytes = sizeof(float);
    tiles.feature_axis_fingerprint = canonical_order().feature_axis_identity;
    tiles.feature_axis_fingerprint_version = 1u;
    tiles.row_domain_identity = 0x303ull;
    tiles.tile_block_offsets = tile_offsets;
    tiles.tile_block_ids = blocks;
    tiles.tile_block_cell_masks = cell_masks;
    tiles.block_row_entry_offsets = entry_offsets;
    tiles.row_block_gene_masks = gene_masks;
    tiles.row_block_value_offsets = value_offsets;
    tiles.values = values;
    cm::u32 rows[3]{}, features[4]{}, cursors[2]{};
    float reconstructed[4]{};
    cm::execution_csr_view result;
    require(static_cast<bool>(cm::materialize_execution_csr_from_cpk1_host(payload,
        {3u, 4u, sizeof(reconstructed), 2u,
         rows, features, reconstructed, cursors}, &result)),
        "lazy CPK1 reconstruction failed");
    const cm::u32 expected_features[]{0u, 1u, 2u, 3u};
    require(std::memcmp(features, expected_features, sizeof(features)) == 0
            && std::memcmp(reconstructed, values, sizeof(values)) == 0
            && rows[0] == 0u && rows[1] == 2u && rows[2] == 4u,
        "lazy CPK1 reconstruction changed packed CSR order or values");
}

} // namespace

int main() {
    const plan_fixture fixture;
    const cm::execution_csr_view sparse = test_ordered_plan_adaptation(fixture);
    const cm::packed_dense_operand_view dense = test_dense_pack(fixture);
    test_packed_math(sparse, dense);
    test_lazy_cpk1_reconstruction(fixture);
    std::cout << "cpMathExecutionCsrTest passed\n";
    return 0;
}
