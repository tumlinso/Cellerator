/*
 * CP-BP-06 Phase B exact device-construction tests. The Phase A host builder
 * is the oracle; CUDA output must match every offset, id, mask, and value byte.
 * Native target: Tesla V100 sm_70.
 */

#include "CellPack/cell_block_records_cuda.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace cp = ::cellpack;

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellPackCellBlockRecordsCudaTest: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cellPackCellBlockRecordsCudaTest: %s: %s\n",
            message, cudaGetErrorString(error));
        std::exit(1);
    }
}

void require_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::fprintf(stderr, "cellPackCellBlockRecordsCudaTest: %s: %s (index=%llu)\n",
            message, status.message, static_cast<unsigned long long>(status.index));
        std::exit(1);
    }
}

template <typename T>
class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t count) { allocate(count); }
    ~device_buffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    void allocate(std::size_t count) {
        count_ = count;
        if (count != 0u) require_cuda(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }
    T *get() const { return pointer_; }
    std::size_t size() const { return count_; }

private:
    T *pointer_ = nullptr;
    std::size_t count_ = 0u;
};

cp::frozen_packing_plan make_plan(
    const std::vector<cp::u32> &permutation,
    const std::vector<cp::u32> &block_offsets,
    cp::u32 maximum_width,
    cp::u32 row_count) {
    std::vector<cp::u32> inverse(permutation.size());
    std::vector<cp::u32> feature_to_block(permutation.size());
    std::vector<cp::u32> feature_to_local(permutation.size());
    for (cp::u32 execution = 0u; execution < permutation.size(); ++execution) {
        inverse[permutation[execution]] = execution;
    }
    for (cp::u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (cp::u32 execution = block_offsets[block]; execution < block_offsets[block + 1u]; ++execution) {
            const cp::u32 feature = permutation[execution];
            feature_to_block[feature] = block;
            feature_to_local[feature] = execution - block_offsets[block];
        }
    }
    const std::vector<cp::u32> row_group_offsets{0u, row_count};
    cp::frozen_packing_plan_build_view build;
    build.row_count = row_count;
    build.feature_count = static_cast<cp::u32>(permutation.size());
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = static_cast<cp::u32>(block_offsets.size() - 1u);
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_group_offsets.data();
    build.maximum_feature_block_width = maximum_width;
    build.row_group_width = row_count;
    build.identity.feature_axis_fingerprint = 0x12345678u;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0xabcdu;
    build.identity.evaluation_source_identity = 0x777u;
    build.cost_policy_identity = 0x999u;
    cp::frozen_packing_plan result;
    require_status(cp::freeze_packing_plan(build, &result), "freeze test plan");
    return result;
}

struct host_source {
    std::vector<cp::u32> rows, blocks, locals, features;
    std::vector<unsigned char> values;
    cp::u32 full_rows = 0u, features_count = 0u, value_size = 0u;
    cp::u64 global_row_begin = 0u;

    cp::ordered_plan_partition_view view() const {
        cp::ordered_plan_partition_view result;
        result.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
        result.global_row_begin = global_row_begin;
        result.full_row_count = full_rows;
        result.row_count = static_cast<cp::u32>(rows.size() - 1u);
        result.feature_count = features_count;
        result.nnz_count = static_cast<cp::u32>(features.size());
        result.value_size_bytes = value_size;
        result.feature_axis_fingerprint = 0x12345678u;
        result.feature_axis_fingerprint_version = 1u;
        result.row_domain_identity = 0xabcdu;
        result.row_offsets = rows.data();
        result.block_ids = blocks.empty() ? nullptr : blocks.data();
        result.local_feature_ids = locals.empty() ? nullptr : locals.data();
        result.canonical_feature_ids = features.empty() ? nullptr : features.data();
        result.values = values.empty() ? nullptr : values.data();
        return result;
    }
};

struct host_records {
    std::vector<cp::u32> rows, blocks, masks, values_offsets;
    std::vector<unsigned char> values;
    cp::cell_block_record_view view{};
};

host_records build_host_oracle(
    const cp::frozen_packing_plan &plan,
    const cp::ordered_plan_partition_view &source,
    cp::cell_block_record_requirements *required) {
    require_status(cp::query_cell_block_record_requirements_host(plan, source, required),
        "query host oracle requirements");
    host_records result;
    result.rows.resize(required->row_record_offset_count);
    result.blocks.resize(required->record_count);
    result.masks.resize(required->record_count);
    result.values_offsets.resize(required->record_value_offset_count);
    result.values.resize(required->value_bytes);
    cp::cell_block_record_buffers buffers{
        result.rows.size(), result.blocks.size(), result.values_offsets.size(),
        result.values.size(), result.rows.data(), result.blocks.data(), result.masks.data(),
        result.values_offsets.data(), result.values.data()};
    require_status(cp::build_cell_block_records_host(plan, source, buffers, &result.view),
        "build host oracle");
    return result;
}

void run_exact_case(const cp::frozen_packing_plan &plan, const host_source &host) {
    const cp::ordered_plan_partition_view source = host.view();
    cp::cell_block_record_requirements exact;
    const host_records oracle = build_host_oracle(plan, source, &exact);
    cp::cell_block_record_cuda_requirements scratch;
    require_status(cp::query_cell_block_record_cuda_requirements(source.nnz_count, &scratch),
        "query CUDA scratch");

    device_buffer<cp::u32> d_rows(host.rows.size()), d_blocks(host.blocks.size());
    device_buffer<cp::u32> d_locals(host.locals.size()), d_features(host.features.size());
    device_buffer<unsigned char> d_source_values(host.values.size());
    device_buffer<cp::u32> d_output_rows(exact.row_record_offset_count);
    device_buffer<cp::u32> d_output_blocks(exact.record_count), d_output_masks(exact.record_count);
    device_buffer<cp::u32> d_output_value_offsets(exact.record_value_offset_count);
    device_buffer<unsigned char> d_output_values(exact.value_bytes);
    device_buffer<cp::u32> d_flags(scratch.entry_prefix_count), d_indices(scratch.entry_prefix_count);
    device_buffer<unsigned char> d_cub(scratch.cub_temporary_bytes);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    require_cuda(cudaMemcpyAsync(d_rows.get(), host.rows.data(), host.rows.size() * sizeof(cp::u32),
        cudaMemcpyHostToDevice, stream), "upload row offsets");
    if (source.nnz_count != 0u) {
        require_cuda(cudaMemcpyAsync(d_blocks.get(), host.blocks.data(), host.blocks.size() * sizeof(cp::u32),
            cudaMemcpyHostToDevice, stream), "upload block ids");
        require_cuda(cudaMemcpyAsync(d_locals.get(), host.locals.data(), host.locals.size() * sizeof(cp::u32),
            cudaMemcpyHostToDevice, stream), "upload local ids");
        require_cuda(cudaMemcpyAsync(d_features.get(), host.features.data(), host.features.size() * sizeof(cp::u32),
            cudaMemcpyHostToDevice, stream), "upload canonical ids");
        require_cuda(cudaMemcpyAsync(d_source_values.get(), host.values.data(), host.values.size(),
            cudaMemcpyHostToDevice, stream), "upload values");
    }

    cp::ordered_plan_partition_view device_source = source;
    device_source.row_offsets = d_rows.get();
    device_source.block_ids = d_blocks.get();
    device_source.local_feature_ids = d_locals.get();
    device_source.canonical_feature_ids = d_features.get();
    device_source.values = d_source_values.get();
    cp::cell_block_record_cuda_workspace_view workspace{
        scratch.entry_prefix_count, d_flags.get(), d_indices.get(),
        d_cub.get(), scratch.cub_temporary_bytes};
    cp::cell_block_record_buffers output{
        exact.row_record_offset_count, exact.record_count, exact.record_value_offset_count,
        exact.value_bytes, d_output_rows.get(), d_output_blocks.get(), d_output_masks.get(),
        d_output_value_offsets.get(), d_output_values.get()};
    cp::cell_block_record_view device_records;
    require_status(cp::build_cell_block_records_cuda(
        plan, device_source, exact.record_count, workspace, output, stream, &device_records),
        "build CUDA records");
    require(device_records.row_record_offsets == d_output_rows.get()
            && device_records.values == d_output_values.get()
            && device_records.feature_block_geometry_identity == plan.feature_block_geometry_identity(),
        "CUDA result view does not expose caller-owned device storage or identity");

    host_records actual;
    actual.rows.resize(exact.row_record_offset_count);
    actual.blocks.resize(exact.record_count);
    actual.masks.resize(exact.record_count);
    actual.values_offsets.resize(exact.record_value_offset_count);
    actual.values.resize(exact.value_bytes);
    require_cuda(cudaMemcpyAsync(actual.rows.data(), d_output_rows.get(), actual.rows.size() * sizeof(cp::u32),
        cudaMemcpyDeviceToHost, stream), "download row offsets");
    if (exact.record_count != 0u) {
        require_cuda(cudaMemcpyAsync(actual.blocks.data(), d_output_blocks.get(), actual.blocks.size() * sizeof(cp::u32),
            cudaMemcpyDeviceToHost, stream), "download block ids");
        require_cuda(cudaMemcpyAsync(actual.masks.data(), d_output_masks.get(), actual.masks.size() * sizeof(cp::u32),
            cudaMemcpyDeviceToHost, stream), "download masks");
    }
    require_cuda(cudaMemcpyAsync(actual.values_offsets.data(), d_output_value_offsets.get(),
        actual.values_offsets.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost, stream),
        "download value offsets");
    if (!actual.values.empty()) require_cuda(cudaMemcpyAsync(actual.values.data(), d_output_values.get(),
        actual.values.size(), cudaMemcpyDeviceToHost, stream), "download values");
    require_cuda(cudaStreamSynchronize(stream), "synchronize exact case");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");

    require(actual.rows == oracle.rows, "CPU/CUDA row-record offsets differ");
    require(actual.blocks == oracle.blocks, "CPU/CUDA block ids differ");
    require(actual.masks == oracle.masks, "CPU/CUDA gene masks differ");
    require(actual.values_offsets == oracle.values_offsets, "CPU/CUDA value offsets differ");
    require(actual.values == oracle.values, "CPU/CUDA value bytes differ");

    actual.view = device_records;
    actual.view.row_record_offsets = actual.rows.data();
    actual.view.record_block_ids = actual.blocks.data();
    actual.view.record_gene_masks = actual.masks.data();
    actual.view.record_value_offsets = actual.values_offsets.data();
    actual.view.values = actual.values.data();
    require_status(cp::validate_cell_block_record_view_host(plan, actual.view),
        "validate downloaded CUDA records");
    std::vector<cp::u32> decoded_rows(host.rows.size()), decoded_features(host.features.size());
    std::vector<unsigned char> decoded_values(host.values.size());
    cp::cell_block_decode_buffers decode{
        decoded_rows.size(), decoded_features.size(), decoded_values.size(),
        decoded_rows.data(), decoded_features.data(), decoded_values.data()};
    cp::decoded_cell_block_partition_view decoded;
    require_status(cp::decode_cell_block_records_host(plan, actual.view, decode, &decoded),
        "decode downloaded CUDA records");
    require(decoded_rows == host.rows && decoded_features == host.features
            && decoded_values == host.values, "CUDA records did not decode to the ordered source");
}

void test_fixture_and_empty() {
    const cp::frozen_packing_plan plan = make_plan(
        {3u, 1u, 5u, 0u, 4u, 2u}, {0u, 2u, 5u, 6u}, 3u, 8u);
    host_source fixture;
    fixture.rows = {0u, 3u, 3u, 6u, 12u};
    fixture.blocks = {0u, 1u, 1u, 0u, 1u, 2u, 0u, 0u, 1u, 1u, 1u, 2u};
    fixture.locals = {0u, 0u, 1u, 1u, 2u, 0u, 0u, 1u, 0u, 1u, 2u, 0u};
    fixture.features = {3u, 5u, 0u, 1u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 2u};
    fixture.values.resize(fixture.features.size() * 3u);
    for (std::size_t i = 0u; i < fixture.values.size(); ++i) fixture.values[i] = static_cast<unsigned char>(i * 17u + 3u);
    fixture.full_rows = 8u;
    fixture.features_count = 6u;
    fixture.value_size = 3u;
    fixture.global_row_begin = 2u;
    run_exact_case(plan, fixture);

    host_source empty;
    empty.rows = {0u, 0u, 0u, 0u};
    empty.full_rows = 8u;
    empty.features_count = 6u;
    empty.value_size = 7u;
    empty.global_row_begin = 3u;
    run_exact_case(plan, empty);
}

void test_maximum_mask_bit_and_validation() {
    std::vector<cp::u32> permutation(32u);
    for (cp::u32 i = 0u; i < permutation.size(); ++i) permutation[i] = i;
    const cp::frozen_packing_plan plan = make_plan(permutation, {0u, 32u}, 32u, 1u);
    host_source source;
    source.rows = {0u, 2u};
    source.blocks = {0u, 0u};
    source.locals = {0u, 31u};
    source.features = {0u, 31u};
    source.values = {0xa5u, 0x5au};
    source.full_rows = 1u;
    source.features_count = 32u;
    source.value_size = 1u;
    run_exact_case(plan, source);

    cp::cell_block_record_cuda_requirements ignored;
    require(cp::query_cell_block_record_cuda_requirements(std::numeric_limits<cp::u32>::max(), &ignored).code
            == cp::validation_code::integer_overflow,
        "signed CUB count overflow was accepted");
    require(cp::query_cell_block_record_cuda_requirements(1u, nullptr).code
            == cp::validation_code::null_pointer,
        "null CUDA requirements output was accepted");

    const cp::ordered_plan_partition_view host_view = source.view();
    cp::cell_block_record_requirements exact;
    require_status(cp::query_cell_block_record_requirements_host(plan, host_view, &exact),
        "query validation fixture");
    cp::ordered_plan_partition_view metadata_only = host_view;
    cp::cell_block_record_cuda_workspace_view workspace;
    cp::cell_block_record_buffers buffers;
    cp::cell_block_record_view output;
    require(cp::build_cell_block_records_cuda(plan, metadata_only, 0u, workspace, buffers,
            nullptr, &output).code == cp::validation_code::invalid_matrix_view,
        "zero expected record count was accepted for nonempty source");
    require(cp::build_cell_block_records_cuda(plan, metadata_only, exact.record_count,
            workspace, buffers, nullptr, nullptr).code == cp::validation_code::null_pointer,
        "null CUDA result view was accepted");
}

int main() {
    int devices = 0;
    require_cuda(cudaGetDeviceCount(&devices), "query CUDA devices");
    require(devices > 0, "no CUDA device available");
    test_fixture_and_empty();
    test_maximum_mask_bit_and_validation();
    std::fprintf(stdout, "cellPackCellBlockRecordsCudaTest: passed on device 0\n");
    return 0;
}
