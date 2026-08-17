#include "CellPack/local_cell_ordering_cuda.hh"

#include "benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

struct options {
    u32 rows = 65536u;
    u32 window = 1024u;
    u32 group = 32u;
    u32 clusters = 32u;
    u32 blocks_per_row = 16u;
    u32 feature_blocks = 2048u;
    u32 warmup = 1u;
    u32 repeats = 5u;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_step(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

u32 parse_u32(const char *text, const char *label) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (text == nullptr || *text == '\0' || end == nullptr || *end != '\0'
        || value > std::numeric_limits<u32>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + label);
    }
    return static_cast<u32>(value);
}

options parse_options(int argc, char **argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto next = [&](const char *label) {
            if (++index >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[index];
        };
        if (argument == "--rows") result.rows = parse_u32(next("--rows"), "--rows");
        else if (argument == "--window") result.window = parse_u32(next("--window"), "--window");
        else if (argument == "--group") result.group = parse_u32(next("--group"), "--group");
        else if (argument == "--clusters") result.clusters = parse_u32(next("--clusters"), "--clusters");
        else if (argument == "--blocks-row") result.blocks_per_row = parse_u32(next("--blocks-row"), "--blocks-row");
        else if (argument == "--feature-blocks") result.feature_blocks = parse_u32(next("--feature-blocks"), "--feature-blocks");
        else if (argument == "--warmup") result.warmup = parse_u32(next("--warmup"), "--warmup");
        else if (argument == "--repeats") result.repeats = parse_u32(next("--repeats"), "--repeats");
        else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: cellPackLocalCellOrderingBench [--rows N] [--window N] "
                "[--group N] [--clusters N] [--blocks-row N] [--feature-blocks N] "
                "[--warmup N] [--repeats N]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown argument: " + argument);
    }
    require(result.rows != 0u && result.window != 0u && result.group != 0u
        && result.clusters != 0u && result.blocks_per_row != 0u
        && result.repeats != 0u, "benchmark dimensions must be nonzero");
    require(static_cast<u64>(result.clusters) * result.blocks_per_row <= result.feature_blocks,
        "cluster supports exceed feature-block domain");
    require(static_cast<u64>(result.rows) * result.blocks_per_row <= std::numeric_limits<u32>::max(),
        "record count exceeds u32");
    return result;
}

template<class T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;
    explicit device_buffer(std::size_t count_) : count(count_) {
        if (count != 0u) cuda_step(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc failed");
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
};

template<class T>
void upload(device_buffer<T> &destination, const std::vector<T> &source) {
    if (!source.empty()) cuda_step(cudaMemcpy(destination.data, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice), "upload failed");
}

struct fixture {
    std::vector<u32> row_offsets, block_ids, value_offsets;
    cellpack::cell_block_record_view records{};
};

fixture make_fixture(const options &settings) {
    fixture result;
    result.row_offsets.resize(static_cast<std::size_t>(settings.rows) + 1u);
    result.block_ids.resize(static_cast<std::size_t>(settings.rows) * settings.blocks_per_row);
    result.value_offsets.resize(result.block_ids.size() + 1u);
    for (u32 row = 0u; row < settings.rows; ++row) {
        const u32 begin = row * settings.blocks_per_row;
        result.row_offsets[row] = begin;
        const u32 cluster = row % settings.clusters;
        for (u32 entry = 0u; entry < settings.blocks_per_row; ++entry) {
            result.block_ids[begin + entry] = cluster * settings.blocks_per_row + entry;
            result.value_offsets[begin + entry] = begin + entry;
        }
    }
    result.row_offsets[settings.rows] = static_cast<u32>(result.block_ids.size());
    result.value_offsets[result.block_ids.size()] = static_cast<u32>(result.block_ids.size());
    result.records.record_schema_version = cellpack::cell_block_record_schema_version;
    result.records.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
    result.records.geometry_identity_version = cellpack::feature_block_geometry_identity_version;
    result.records.feature_block_geometry_identity = 0x123456789abcdef0ull;
    result.records.full_row_count = settings.rows;
    result.records.row_count = settings.rows;
    result.records.feature_count = settings.feature_blocks * 16u;
    result.records.feature_block_count = settings.feature_blocks;
    result.records.nnz_count = static_cast<u32>(result.block_ids.size());
    result.records.record_count = static_cast<u32>(result.block_ids.size());
    result.records.value_size_bytes = sizeof(float);
    result.records.feature_axis_fingerprint = 0x1234u;
    result.records.feature_axis_fingerprint_version = 1u;
    result.records.row_domain_identity = 0x5678u;
    result.records.row_record_offsets = result.row_offsets.data();
    result.records.record_block_ids = result.block_ids.data();
    result.records.record_value_offsets = result.value_offsets.data();
    return result;
}

struct host_output {
    std::vector<u64> primary;
    std::vector<u32> secondary, active, nnz, permutation, inverse;
    cellpack::local_cell_order_view view{};
};

host_output run_host(
    const cellpack::cell_block_record_view &records,
    const cellpack::local_cell_order_config &config) {
    host_output result;
    result.primary.resize(records.row_count);
    result.secondary.resize(records.row_count);
    result.active.resize(records.row_count);
    result.nnz.resize(records.row_count);
    result.permutation.resize(records.row_count);
    result.inverse.resize(records.row_count);
    cellpack::local_cell_order_buffers buffers{records.row_count,
        result.primary.data(), result.secondary.data(), result.active.data(),
        result.nnz.data(), result.permutation.data(), result.inverse.data()};
    require(static_cast<bool>(cellpack::build_local_cell_order_host(
        records, config, buffers, &result.view)), "host ordering failed");
    return result;
}

cellpack::local_cell_order_metrics measure(
    const cellpack::cell_block_record_view &records,
    const host_output &output) {
    std::vector<u32> epochs(records.feature_block_count);
    cellpack::local_cell_order_metric_workspace workspace{epochs.size(), epochs.data()};
    cellpack::local_cell_order_metrics result;
    require(static_cast<bool>(cellpack::evaluate_local_cell_order_metrics_host(
        records, output.view, workspace, &result)), "metric evaluation failed");
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const options settings = parse_options(argc, argv);
        const cellerator::bench::benchmark_mutex_guard mutex("cellpack-local-cell-ordering", 0);
        fixture data = make_fixture(settings);
        cellpack::local_cell_order_config config;
        config.window_size = settings.window;
        config.group_width = settings.group;

        const auto cpu_begin = std::chrono::steady_clock::now();
        const host_output reference = run_host(data.records, config);
        const auto cpu_end = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

        const auto inferred_metrics = measure(data.records, reference);
        config.kind = cellpack::local_cell_order_kind::original;
        const auto original_metrics = measure(data.records, run_host(data.records, config));
        config.kind = cellpack::local_cell_order_kind::deterministic_random;
        const auto random_metrics = measure(data.records, run_host(data.records, config));
        config.kind = cellpack::local_cell_order_kind::row_nnz_descending;
        const auto nnz_metrics = measure(data.records, run_host(data.records, config));
        config.kind = cellpack::local_cell_order_kind::inferred_minhash;

        device_buffer<u32> d_row_offsets(data.row_offsets.size());
        device_buffer<u32> d_block_ids(data.block_ids.size());
        device_buffer<u32> d_value_offsets(data.value_offsets.size());
        upload(d_row_offsets, data.row_offsets);
        upload(d_block_ids, data.block_ids);
        upload(d_value_offsets, data.value_offsets);
        auto device_records = data.records;
        device_records.row_record_offsets = d_row_offsets.data;
        device_records.record_block_ids = d_block_ids.data;
        device_records.record_value_offsets = d_value_offsets.data;

        cellpack::local_cell_order_cuda_requirements required;
        require(static_cast<bool>(cellpack::query_local_cell_order_cuda_requirements(
            settings.rows, config, &required)), "CUDA requirements failed");
        device_buffer<u64> d_primary(settings.rows), d_primary_gathered(settings.rows),
            d_primary_sorted(settings.rows);
        device_buffer<u32> d_secondary(settings.rows), d_active(settings.rows), d_nnz(settings.rows),
            d_permutation(settings.rows), d_inverse(settings.rows),
            d_secondary_sorted(settings.rows), d_row_scratch(settings.rows),
            d_window_offsets(required.window_offset_capacity);
        device_buffer<unsigned char> d_cub(required.cub_temporary_bytes);
        cellpack::local_cell_order_buffers buffers{settings.rows, d_primary.data,
            d_secondary.data, d_active.data, d_nnz.data, d_permutation.data, d_inverse.data};
        cellpack::local_cell_order_cuda_workspace workspace{settings.rows,
            required.window_offset_capacity, required.cub_temporary_bytes,
            d_primary_gathered.data, d_primary_sorted.data, d_secondary_sorted.data,
            d_row_scratch.data, d_window_offsets.data, d_cub.data};
        cellpack::local_cell_order_view device_view;
        for (u32 repeat = 0u; repeat < settings.warmup; ++repeat) {
            require(static_cast<bool>(cellpack::build_local_cell_order_cuda(
                device_records, config, buffers, workspace, nullptr, &device_view)),
                "CUDA warmup enqueue failed");
        }
        cuda_step(cudaDeviceSynchronize(), "CUDA warmup failed");
        cudaEvent_t start = nullptr, stop = nullptr;
        cuda_step(cudaEventCreate(&start), "event creation failed");
        cuda_step(cudaEventCreate(&stop), "event creation failed");
        std::vector<float> samples;
        for (u32 repeat = 0u; repeat < settings.repeats; ++repeat) {
            cuda_step(cudaEventRecord(start), "start record failed");
            require(static_cast<bool>(cellpack::build_local_cell_order_cuda(
                device_records, config, buffers, workspace, nullptr, &device_view)),
                "CUDA timed enqueue failed");
            cuda_step(cudaEventRecord(stop), "stop record failed");
            cuda_step(cudaEventSynchronize(stop), "timed synchronization failed");
            float elapsed = 0.0f;
            cuda_step(cudaEventElapsedTime(&elapsed, start, stop), "elapsed time failed");
            samples.push_back(elapsed);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::sort(samples.begin(), samples.end());
        std::vector<u32> gpu_permutation(settings.rows);
        cuda_step(cudaMemcpy(gpu_permutation.data(), d_permutation.data,
            gpu_permutation.size() * sizeof(u32), cudaMemcpyDeviceToHost), "permutation download failed");
        require(gpu_permutation == reference.permutation, "CPU/CUDA permutation mismatch");

        auto print_metric = [](const char *label, const cellpack::local_cell_order_metrics &metric) {
            std::cout << label << "_union_refs=" << metric.total_group_block_union_references
                << " " << label << "_metadata_bytes=" << metric.block_id_metadata_bytes << "\n";
        };
        std::cout << "rows=" << settings.rows << " window=" << settings.window
            << " group=" << settings.group << " blocks_per_row=" << settings.blocks_per_row << "\n";
        std::cout << "cpu_ms=" << cpu_ms << " cuda_median_ms=" << samples[samples.size() / 2u]
            << " cuda_scratch_bytes=" << required.total_temporary_bytes << " exact_agreement=1\n";
        print_metric("inferred", inferred_metrics);
        print_metric("original", original_metrics);
        print_metric("random", random_metrics);
        print_metric("row_nnz", nnz_metrics);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "cellPackLocalCellOrderingBench: " << error.what() << '\n';
        return 1;
    }
}
