#include "Cellerator/geometry/local_cell_ordering_cuda.hh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackLocalCellOrderingTest: " << message << '\n';
        std::exit(1);
    }
}

void require_cuda(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "cellPackLocalCellOrderingTest: " << message << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

template<class T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;
    explicit device_buffer(std::size_t count_) : count(count_) {
        if (count != 0u) require_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc failed");
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
};

template<class T>
void upload(device_buffer<T> &destination, const std::vector<T> &source) {
    if (!source.empty()) require_cuda(cudaMemcpy(destination.data, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice), "upload failed");
}

template<class T>
std::vector<T> download(const device_buffer<T> &source) {
    std::vector<T> result(source.count);
    if (!result.empty()) require_cuda(cudaMemcpy(result.data(), source.data,
        result.size() * sizeof(T), cudaMemcpyDeviceToHost), "download failed");
    return result;
}

struct record_fixture {
    std::vector<u32> row_offsets;
    std::vector<u32> block_ids;
    std::vector<u32> masks;
    std::vector<u32> value_offsets;

    record_fixture() {
        row_offsets.push_back(0u);
        value_offsets.push_back(0u);
        for (u32 row = 0u; row < 16u; ++row) {
            const bool first_cluster = (row & 1u) == 0u;
            block_ids.push_back(first_cluster ? 0u : 6u);
            block_ids.push_back(first_cluster ? 1u : 7u);
            masks.push_back(1u);
            masks.push_back(1u);
            value_offsets.push_back(value_offsets.back() + 1u);
            value_offsets.push_back(value_offsets.back() + 1u);
            row_offsets.push_back(static_cast<u32>(block_ids.size()));
        }
        row_offsets.push_back(static_cast<u32>(block_ids.size())); // empty tail row
    }

    cellpack::cell_block_record_view view() const {
        cellpack::cell_block_record_view records;
        records.record_schema_version = cellpack::cell_block_record_schema_version;
        records.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
        records.geometry_identity_version = cellpack::feature_block_geometry_identity_version;
        records.feature_block_geometry_identity = 0x1111222233334444ull;
        records.global_row_begin = 40u;
        records.full_row_count = 100u;
        records.row_count = static_cast<u32>(row_offsets.size() - 1u);
        records.feature_count = 64u;
        records.feature_block_count = 8u;
        records.nnz_count = static_cast<u32>(value_offsets.back());
        records.record_count = static_cast<u32>(block_ids.size());
        records.value_size_bytes = sizeof(float);
        records.feature_axis_fingerprint = 0x777u;
        records.feature_axis_fingerprint_version = 1u;
        records.row_domain_identity = 0x5555666677778888ull;
        records.row_record_offsets = row_offsets.data();
        records.record_block_ids = block_ids.data();
        records.record_gene_masks = masks.data();
        records.record_value_offsets = value_offsets.data();
        return records;
    }
};

struct host_order {
    std::vector<u64> primary;
    std::vector<u32> secondary, active, nnz, permutation, inverse;
    cellpack::local_cell_order_view view{};
};

host_order run_host(
    const cellpack::cell_block_record_view &records,
    const cellpack::local_cell_order_config &config) {
    host_order result;
    const std::size_t rows = records.row_count;
    result.primary.resize(rows);
    result.secondary.resize(rows);
    result.active.resize(rows);
    result.nnz.resize(rows);
    result.permutation.resize(rows);
    result.inverse.resize(rows);
    cellpack::local_cell_order_buffers buffers;
    buffers.row_capacity = rows;
    buffers.primary_keys = result.primary.data();
    buffers.secondary_keys = result.secondary.data();
    buffers.active_block_counts = result.active.data();
    buffers.row_nnz_counts = result.nnz.data();
    buffers.row_permutation = result.permutation.data();
    buffers.inverse_row_permutation = result.inverse.data();
    require(static_cast<bool>(cellpack::build_local_cell_order_host(
        records, config, buffers, &result.view)), "host order failed");
    return result;
}

cellpack::local_cell_order_metrics metrics(
    const cellpack::cell_block_record_view &records,
    const host_order &order) {
    std::vector<u32> epochs(records.feature_block_count);
    cellpack::local_cell_order_metric_workspace workspace;
    workspace.block_epoch_capacity = epochs.size();
    workspace.block_epochs = epochs.data();
    cellpack::local_cell_order_metrics result;
    require(static_cast<bool>(cellpack::evaluate_local_cell_order_metrics_host(
        records, order.view, workspace, &result)), "metric evaluation failed");
    return result;
}

void test_host_contract_and_baselines() {
    const record_fixture fixture;
    const auto records = fixture.view();
    cellpack::local_cell_order_requirements required;
    require(static_cast<bool>(cellpack::query_local_cell_order_requirements_host(
        records, &required)), "requirements query failed");
    require(required.row_capacity == 17u && required.block_epoch_capacity == 8u,
        "requirements are wrong");

    cellpack::local_cell_order_config config;
    config.window_size = 8u;
    config.group_width = 4u;
    const host_order inferred = run_host(records, config);
    const host_order repeated = run_host(records, config);
    require(inferred.primary == repeated.primary && inferred.permutation == repeated.permutation,
        "inferred ordering is not deterministic");
    for (u32 execution = 0u; execution < records.row_count; ++execution) {
        const u32 row = inferred.permutation[execution];
        require(row / config.window_size == execution / config.window_size,
            "row crossed its local window");
        require(inferred.inverse[row] == execution, "inverse map is wrong");
    }
    const auto inferred_metrics = metrics(records, inferred);
    require(inferred_metrics.total_group_block_union_references == 8u,
        "inferred group union is wrong");
    require(inferred_metrics.block_id_metadata_bytes == 8u * sizeof(u32),
        "metadata-byte metric is wrong");

    config.kind = cellpack::local_cell_order_kind::original;
    const host_order original = run_host(records, config);
    require(metrics(records, original).total_group_block_union_references == 16u,
        "original baseline union is wrong");
    config.kind = cellpack::local_cell_order_kind::row_nnz_descending;
    const host_order row_nnz = run_host(records, config);
    require(metrics(records, row_nnz).total_group_block_union_references == 16u,
        "row-NNZ baseline union is wrong");
    config.kind = cellpack::local_cell_order_kind::deterministic_random;
    const host_order random_first = run_host(records, config);
    const host_order random_second = run_host(records, config);
    require(random_first.permutation == random_second.permutation,
        "random baseline is not seed deterministic");

    host_order tampered = inferred;
    tampered.permutation[0] = tampered.permutation[8];
    tampered.view.row_permutation = tampered.permutation.data();
    require(!cellpack::validate_local_cell_order_view_host(records, tampered.view),
        "cross-window permutation tamper was accepted");
}

void test_cuda_exact_agreement() {
    const record_fixture fixture;
    const auto host_records = fixture.view();
    device_buffer<u32> d_row_offsets(fixture.row_offsets.size());
    device_buffer<u32> d_block_ids(fixture.block_ids.size());
    device_buffer<u32> d_value_offsets(fixture.value_offsets.size());
    upload(d_row_offsets, fixture.row_offsets);
    upload(d_block_ids, fixture.block_ids);
    upload(d_value_offsets, fixture.value_offsets);
    auto device_records = host_records;
    device_records.row_record_offsets = d_row_offsets.data;
    device_records.record_block_ids = d_block_ids.data;
    device_records.record_gene_masks = nullptr;
    device_records.record_value_offsets = d_value_offsets.data;

    const std::vector<cellpack::local_cell_order_kind> kinds{
        cellpack::local_cell_order_kind::inferred_minhash,
        cellpack::local_cell_order_kind::original,
        cellpack::local_cell_order_kind::deterministic_random,
        cellpack::local_cell_order_kind::row_nnz_descending};
    for (const auto kind : kinds) {
        cellpack::local_cell_order_config config;
        config.kind = kind;
        config.window_size = 8u;
        config.group_width = 4u;
        const host_order reference = run_host(host_records, config);
        cellpack::local_cell_order_cuda_requirements required;
        require(static_cast<bool>(cellpack::query_local_cell_order_cuda_requirements(
            host_records.row_count, config, &required)), "CUDA requirements failed");
        const std::size_t rows = host_records.row_count;
        device_buffer<u64> d_primary(rows), d_primary_gathered(rows), d_primary_sorted(rows);
        device_buffer<u32> d_secondary(rows), d_active(rows), d_nnz(rows), d_permutation(rows),
            d_inverse(rows), d_secondary_sorted(rows), d_row_scratch(rows),
            d_window_offsets(required.window_offset_capacity);
        device_buffer<unsigned char> d_cub(required.cub_temporary_bytes);
        cellpack::local_cell_order_buffers buffers;
        buffers.row_capacity = rows;
        buffers.primary_keys = d_primary.data;
        buffers.secondary_keys = d_secondary.data;
        buffers.active_block_counts = d_active.data;
        buffers.row_nnz_counts = d_nnz.data;
        buffers.row_permutation = d_permutation.data;
        buffers.inverse_row_permutation = d_inverse.data;
        cellpack::local_cell_order_cuda_workspace workspace;
        workspace.row_capacity = rows;
        workspace.window_offset_capacity = required.window_offset_capacity;
        workspace.cub_temporary_bytes = required.cub_temporary_bytes;
        workspace.primary_gathered = d_primary_gathered.data;
        workspace.primary_sorted = d_primary_sorted.data;
        workspace.secondary_sorted = d_secondary_sorted.data;
        workspace.row_scratch = d_row_scratch.data;
        workspace.window_offsets = d_window_offsets.data;
        workspace.cub_temporary_storage = d_cub.data;
        cellpack::local_cell_order_view device_view;
        require(static_cast<bool>(cellpack::build_local_cell_order_cuda(
            device_records, config, buffers, workspace, nullptr, &device_view)),
            "CUDA order enqueue failed");
        require_cuda(cudaDeviceSynchronize(), "CUDA order synchronization failed");
        const auto primary = download(d_primary);
        const auto secondary = download(d_secondary);
        const auto active = download(d_active);
        const auto nnz = download(d_nnz);
        const auto permutation = download(d_permutation);
        const auto inverse = download(d_inverse);
        require(primary == reference.primary && secondary == reference.secondary
            && active == reference.active && nnz == reference.nnz
            && permutation == reference.permutation && inverse == reference.inverse,
            "CPU/CUDA local order differs");
        auto host_view = device_view;
        host_view.primary_keys = primary.data();
        host_view.secondary_keys = secondary.data();
        host_view.active_block_counts = active.data();
        host_view.row_nnz_counts = nnz.data();
        host_view.row_permutation = permutation.data();
        host_view.inverse_row_permutation = inverse.data();
        require(static_cast<bool>(cellpack::validate_local_cell_order_view_host(
            host_records, host_view)), "downloaded CUDA view failed host validation");
    }
}

} // namespace

int main() {
    test_host_contract_and_baselines();
    test_cuda_exact_agreement();
    std::cout << "cellPackLocalCellOrderingTest: PASS\n";
    return 0;
}
