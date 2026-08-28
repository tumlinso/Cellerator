#include "Cellerator/geometry/feature_weighted_row_reduction_cuda.hh"
#include "benchmark_mutex.hh"
#include <Cellerator/compute/operators/sparse/ops.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace cp = cellpack;
namespace sparse_ops = cellerator::compute::sparse::ops;
using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

constexpr cp::u32 rows = 65536u;
constexpr cp::u32 features = 32768u;
constexpr cp::u32 block_width = 16u;
constexpr cp::u32 tile_width = 32u;
constexpr cp::u32 row_blocks = 2u;
constexpr cp::u32 nnz_per_row = block_width * row_blocks;
constexpr int warmups = 3;
constexpr int repeats = 11;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackFeatureWeightedRowReductionBench: " << message << '\n';
        std::exit(1);
    }
}

void check_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackFeatureWeightedRowReductionBench: " << message << ": "
                  << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

void check_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::cerr << "cellPackFeatureWeightedRowReductionBench: " << message << ": "
                  << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;
    explicit device_array(std::size_t count = 0u) : size(count) {
        if (count != 0u) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)),
                "cudaMalloc");
        }
    }
    ~device_array() { if (data != nullptr) cudaFree(data); }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    check(device.size >= host.size(), "upload capacity");
    if (!host.empty()) {
        check_cuda(cudaMemcpy(device.data, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice), "upload");
    }
}

storage_t stored(float value) { return static_cast<storage_t>(value); }

struct host_case {
    std::string name;
    cp::u32 sharing_groups = 1u;
    std::vector<cp::u32> csr_offsets, csr_columns;
    std::vector<storage_t> csr_values;
    std::vector<cp::u32> tile_offsets, tile_blocks, tile_cell_masks,
        entry_offsets, gene_masks, value_offsets;
    std::vector<storage_t> tile_values;
    std::vector<cp::u32> row_permutation;
    cp::warp_tile_view tiles{};
};

float value_for(cp::u32 row, cp::u32 feature) {
    const int code = static_cast<int>((row * 17u + feature * 13u) % 31u) - 15;
    return static_cast<float>(code) * 0.0625f;
}

host_case make_case(const char *name, cp::u32 sharing_groups) {
    host_case result;
    result.name = name;
    result.sharing_groups = sharing_groups;
    result.csr_offsets.resize(static_cast<std::size_t>(rows) + 1u);
    result.row_permutation.resize(rows);
    std::iota(result.row_permutation.begin(), result.row_permutation.end(), 0u);
    const cp::u32 block_count = features / block_width;
    const cp::u32 tile_count = rows / tile_width;
    result.tile_offsets.reserve(static_cast<std::size_t>(tile_count) + 1u);
    result.tile_offsets.push_back(0u);
    result.entry_offsets.push_back(0u);
    result.value_offsets.push_back(0u);
    result.csr_columns.reserve(static_cast<std::size_t>(rows) * nnz_per_row);
    result.csr_values.reserve(static_cast<std::size_t>(rows) * nnz_per_row);
    result.gene_masks.reserve(static_cast<std::size_t>(rows) * row_blocks);
    result.tile_values.reserve(static_cast<std::size_t>(rows) * nnz_per_row);

    for (cp::u32 tile = 0u; tile < tile_count; ++tile) {
        std::map<cp::u32, cp::u32> masks;
        std::vector<std::vector<cp::u32>> lane_blocks(tile_width);
        const cp::u32 descriptor_span = sharing_groups * row_blocks;
        const cp::u32 base = (tile * descriptor_span) % (block_count - descriptor_span);
        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 group = lane % sharing_groups;
            lane_blocks[lane] = {base + group * row_blocks,
                base + group * row_blocks + 1u};
            for (cp::u32 block : lane_blocks[lane]) masks[block] |= 1u << lane;
        }
        for (const auto &descriptor : masks) {
            result.tile_blocks.push_back(descriptor.first);
            result.tile_cell_masks.push_back(descriptor.second);
            for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
                if ((descriptor.second & (1u << lane)) == 0u) continue;
                result.gene_masks.push_back(0xffffu);
                const cp::u32 row = tile * tile_width + lane;
                const cp::u32 feature_begin = descriptor.first * block_width;
                for (cp::u32 local = 0u; local < block_width; ++local) {
                    result.tile_values.push_back(stored(value_for(row, feature_begin + local)));
                }
                result.value_offsets.push_back(
                    static_cast<cp::u32>(result.tile_values.size()));
            }
            result.entry_offsets.push_back(static_cast<cp::u32>(result.gene_masks.size()));
        }
        result.tile_offsets.push_back(static_cast<cp::u32>(result.tile_blocks.size()));

        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 row = tile * tile_width + lane;
            for (cp::u32 block : lane_blocks[lane]) {
                const cp::u32 feature_begin = block * block_width;
                for (cp::u32 local = 0u; local < block_width; ++local) {
                    const cp::u32 feature = feature_begin + local;
                    result.csr_columns.push_back(feature);
                    result.csr_values.push_back(stored(value_for(row, feature)));
                }
            }
            result.csr_offsets[row + 1u] = static_cast<cp::u32>(result.csr_columns.size());
        }
    }

    result.tiles.tile_schema_version = cp::warp_tile_schema_version;
    result.tiles.record_schema_version = cp::cell_block_record_schema_version;
    result.tiles.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
    result.tiles.geometry_identity_version = cp::feature_block_geometry_identity_version;
    result.tiles.order_schema_version = cp::local_cell_order_schema_version;
    result.tiles.tile_identity = 0x100000000ull + sharing_groups;
    result.tiles.feature_block_geometry_identity = 0x200000000ull;
    result.tiles.ordering_identity = 0x300000000ull + sharing_groups;
    result.tiles.full_row_count = rows;
    result.tiles.row_count = rows;
    result.tiles.feature_count = features;
    result.tiles.feature_block_count = block_count;
    result.tiles.tile_row_width = tile_width;
    result.tiles.tile_count = tile_count;
    result.tiles.nnz_count = static_cast<cp::u32>(result.tile_values.size());
    result.tiles.tile_block_count = static_cast<cp::u32>(result.tile_blocks.size());
    result.tiles.row_block_entry_count = static_cast<cp::u32>(result.gene_masks.size());
    result.tiles.value_size_bytes = sizeof(storage_t);
    result.tiles.feature_axis_fingerprint = 0x400000000ull;
    result.tiles.feature_axis_fingerprint_version = 1u;
    result.tiles.row_domain_identity = 0x500000000ull;
    result.tiles.tile_block_offsets = result.tile_offsets.data();
    result.tiles.tile_block_ids = result.tile_blocks.data();
    result.tiles.tile_block_cell_masks = result.tile_cell_masks.data();
    result.tiles.block_row_entry_offsets = result.entry_offsets.data();
    result.tiles.row_block_gene_masks = result.gene_masks.data();
    result.tiles.row_block_value_offsets = result.value_offsets.data();
    result.tiles.values = result.tile_values.data();
    return result;
}

cp::frozen_packing_plan make_plan() {
    const cp::u32 block_count = features / block_width;
    std::vector<cp::u32> permutation(features), inverse(features), feature_to_block(features),
        feature_to_local(features), offsets(block_count + 1u);
    std::iota(permutation.begin(), permutation.end(), 0u);
    std::iota(inverse.begin(), inverse.end(), 0u);
    for (cp::u32 block = 0u; block < block_count; ++block) {
        offsets[block] = block * block_width;
        for (cp::u32 local = 0u; local < block_width; ++local) {
            const cp::u32 feature = block * block_width + local;
            feature_to_block[feature] = block;
            feature_to_local[feature] = local;
        }
    }
    offsets.back() = features;
    const cp::u32 row_offsets[] = {0u, rows};
    cp::frozen_packing_plan_build_view build;
    build.row_count = rows;
    build.feature_count = features;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = block_count;
    build.feature_block_offsets = offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = block_width;
    build.row_group_width = rows;
    build.identity.feature_axis_fingerprint = 0x400000000ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x500000000ull;
    build.identity.evaluation_source_identity = 0x600000000ull;
    build.cost_policy_identity = 0x700000000ull;
    cp::frozen_packing_plan plan;
    check_status(cp::freeze_packing_plan(build, &plan), "freeze benchmark plan");
    return plan;
}

struct timing {
    float minimum_ms = 0.0f;
    float median_ms = 0.0f;
    float mean_ms = 0.0f;
};

template<typename Launch>
timing time_cuda(cudaStream_t stream, Launch launch) {
    cudaEvent_t begin = nullptr, end = nullptr;
    check_cuda(cudaEventCreate(&begin), "create begin event");
    check_cuda(cudaEventCreate(&end), "create end event");
    for (int i = 0; i < warmups; ++i) launch();
    check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");
    std::vector<float> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        check_cuda(cudaEventRecord(begin, stream), "record begin");
        launch();
        check_cuda(cudaEventRecord(end, stream), "record end");
        check_cuda(cudaEventSynchronize(end), "time synchronize");
        float elapsed = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed, begin, end), "elapsed time");
        samples.push_back(elapsed);
    }
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    std::sort(samples.begin(), samples.end());
    timing result;
    result.minimum_ms = samples.front();
    result.median_ms = samples[samples.size() / 2u];
    result.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0f)
        / static_cast<float>(samples.size());
    return result;
}

void run_case(const cp::frozen_packing_plan &plan, host_case host,
    const std::vector<compute_t> &weights, cudaStream_t stream) {
    host.tiles.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    const std::vector<cp::u32> plan_offsets(plan.feature_block_offsets(),
        plan.feature_block_offsets() + plan.feature_block_count() + 1u);
    const std::vector<cp::u32> plan_permutation(plan.feature_permutation(),
        plan.feature_permutation() + plan.feature_count());
    device_array<cp::u32> d_plan_offsets(plan_offsets.size()),
        d_plan_permutation(plan_permutation.size()), d_order(host.row_permutation.size()),
        d_tile_offsets(host.tile_offsets.size()), d_tile_blocks(host.tile_blocks.size()),
        d_cell_masks(host.tile_cell_masks.size()), d_entry_offsets(host.entry_offsets.size()),
        d_gene_masks(host.gene_masks.size()), d_value_offsets(host.value_offsets.size()),
        d_csr_offsets(host.csr_offsets.size()), d_csr_columns(host.csr_columns.size());
    device_array<storage_t> d_tile_values(host.tile_values.size()),
        d_csr_values(host.csr_values.size());
    device_array<compute_t> d_weights(weights.size());
    device_array<accum_t> d_packed_output(rows), d_csr_output(rows);
    upload(d_plan_offsets, plan_offsets);
    upload(d_plan_permutation, plan_permutation);
    upload(d_order, host.row_permutation);
    upload(d_tile_offsets, host.tile_offsets);
    upload(d_tile_blocks, host.tile_blocks);
    upload(d_cell_masks, host.tile_cell_masks);
    upload(d_entry_offsets, host.entry_offsets);
    upload(d_gene_masks, host.gene_masks);
    upload(d_value_offsets, host.value_offsets);
    upload(d_tile_values, host.tile_values);
    upload(d_csr_offsets, host.csr_offsets);
    upload(d_csr_columns, host.csr_columns);
    upload(d_csr_values, host.csr_values);
    upload(d_weights, weights);

    auto device_tiles = host.tiles;
    device_tiles.tile_block_offsets = d_tile_offsets.data;
    device_tiles.tile_block_ids = d_tile_blocks.data;
    device_tiles.tile_block_cell_masks = d_cell_masks.data;
    device_tiles.block_row_entry_offsets = d_entry_offsets.data;
    device_tiles.row_block_gene_masks = d_gene_masks.data;
    device_tiles.row_block_value_offsets = d_value_offsets.data;
    device_tiles.values = d_tile_values.data;
    auto input = cp::make_feature_weighted_row_reduction_view(plan, host.tiles,
        0x800000000ull + host.sharing_groups, weights.size(), weights.data());
    input.plan.feature_block_offsets = d_plan_offsets.data;
    input.plan.feature_permutation = d_plan_permutation.data;
    input.tiles = device_tiles;
    input.feature_weights = d_weights.data;
    cp::local_cell_order_view order;
    order.order_schema_version = cp::local_cell_order_schema_version;
    order.signature_algorithm_version = cp::local_cell_signature_algorithm_version;
    order.kind = cp::local_cell_order_kind::original;
    order.window_size = 1024u;
    order.group_width = tile_width;
    order.ordering_identity = host.tiles.ordering_identity;
    order.full_row_count = rows;
    order.row_count = rows;
    order.feature_block_count = features / block_width;
    order.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    order.row_domain_identity = plan.identity().row_domain_identity;
    order.row_permutation = d_order.data;
    cp::feature_weighted_row_reduction_buffers buffers{rows, d_packed_output.data};
    cp::feature_weighted_row_reduction_result_view result;

    const auto packed_timing = time_cuda(stream, [&] {
        check_status(cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            input, order, buffers, stream, &result), "launch packed reduction");
    });

    cellerator::runtime::execution_context ctx;
    ctx.device = 0;
    ctx.stream = stream;
    const auto csr_timing = time_cuda(stream, [&] {
        sparse_ops::base::csr_spmv_fwd_f16_f32(ctx, d_csr_offsets.data,
            d_csr_columns.data, reinterpret_cast<const __half *>(d_csr_values.data),
            rows, reinterpret_cast<const float *>(d_weights.data),
            reinterpret_cast<float *>(d_csr_output.data));
    });

    std::vector<accum_t> packed(rows), csr(rows);
    check_cuda(cudaMemcpy(packed.data(), d_packed_output.data, rows * sizeof(accum_t),
        cudaMemcpyDeviceToHost), "download packed output");
    check_cuda(cudaMemcpy(csr.data(), d_csr_output.data, rows * sizeof(accum_t),
        cudaMemcpyDeviceToHost), "download CSR output");
    for (cp::u32 row = 0u; row < rows; ++row) {
        check(cp::feature_weighted_row_reduction_within_tolerance(csr[row], packed[row]),
            "packed/CSR numerical mismatch");
    }

    const std::uint64_t nnz = host.tile_values.size();
    const std::uint64_t packed_metadata_bytes =
        (host.tile_offsets.size() + host.tile_blocks.size() + host.tile_cell_masks.size()
            + host.entry_offsets.size() + host.gene_masks.size()
            + host.value_offsets.size() + host.row_permutation.size()
            + plan_offsets.size() + plan_permutation.size()) * sizeof(cp::u32);
    const std::uint64_t packed_total_bytes = packed_metadata_bytes
        + host.tile_values.size() * sizeof(storage_t)
        + weights.size() * sizeof(compute_t) + rows * sizeof(accum_t);
    const std::uint64_t csr_total_bytes =
        (host.csr_offsets.size() + host.csr_columns.size()) * sizeof(cp::u32)
        + host.csr_values.size() * sizeof(storage_t)
        + weights.size() * sizeof(compute_t) + rows * sizeof(accum_t);
    const double packed_seconds = packed_timing.median_ms * 1.0e-3;
    const double csr_seconds = csr_timing.median_ms * 1.0e-3;
    std::cout << "regime=" << host.name
              << " groups_per_tile=" << host.sharing_groups
              << " tile_blocks=" << host.tiles.tile_block_count
              << " row_entries=" << host.tiles.row_block_entry_count << '\n'
              << "  packed_ms min/median/mean=" << packed_timing.minimum_ms << '/'
              << packed_timing.median_ms << '/' << packed_timing.mean_ms
              << " nnz_s=" << static_cast<double>(nnz) / packed_seconds
              << " effective_GB_s=" << packed_total_bytes / packed_seconds / 1.0e9
              << " bytes_per_nnz=" << static_cast<double>(packed_total_bytes) / nnz
              << " scratch_bytes=0 launches=1\n"
              << "  csr_custom_ms min/median/mean=" << csr_timing.minimum_ms << '/'
              << csr_timing.median_ms << '/' << csr_timing.mean_ms
              << " nnz_s=" << static_cast<double>(nnz) / csr_seconds
              << " effective_GB_s=" << csr_total_bytes / csr_seconds / 1.0e9
              << " bytes_per_nnz=" << static_cast<double>(csr_total_bytes) / nnz
              << " scratch_bytes=0 launches=1\n";
}

template<typename Storage, typename Compute, typename Accum>
int run_configured_benchmark() {
    if constexpr (!(std::is_same<Storage, __half>::value
            && std::is_same<Compute, float>::value
            && std::is_same<Accum, float>::value)) {
        std::cout << "cellPackFeatureWeightedRowReductionBench: configured precision has "
                     "no type-equivalent existing Cellerator CSR baseline; benchmark skipped\n";
        return 0;
    }
    cellerator::bench::benchmark_mutex_guard mutex(
        "cellPackFeatureWeightedRowReductionBench", 0);
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "get device count");
    check(device_count > 0, "no CUDA device");
    check_cuda(cudaSetDevice(0), "set device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "get device properties");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const auto plan = make_plan();
    std::vector<compute_t> weights(features);
    for (cp::u32 feature = 0u; feature < features; ++feature) {
        weights[feature] = 0.25f + static_cast<float>(feature % 97u) * 0.0078125f;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "cellPackFeatureWeightedRowReductionBench\n"
              << "device=" << properties.name << " sm=" << properties.major
              << properties.minor << " rows=" << rows << " features=" << features
              << " nnz=" << static_cast<std::uint64_t>(rows) * nnz_per_row
              << " tile_width=" << tile_width << " block_width=" << block_width
              << " warmups=" << warmups << " repeats=" << repeats << '\n'
              << "scope=device-resident inputs/outputs; setup, allocation, transfers, and "
                 "synchronization excluded from event intervals\n"
              << "cuSPARSE=not run: configured storage is f16 while the existing Cellerator "
                 "cuSPARSE SpMV wrapper requires f32 values\n";
    run_case(plan, make_case("high_occupancy", 1u), weights, stream);
    run_case(plan, make_case("medium_occupancy", 8u), weights, stream);
    run_case(plan, make_case("low_occupancy", 32u), weights, stream);
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}

} // namespace

int main() {
    return run_configured_benchmark<storage_t, compute_t, accum_t>();
}
