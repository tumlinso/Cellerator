// CP-BP-12 calibration benchmark, not a kernel-optimization claim.
// Compared path: CP-BP-09 direct packed tiles versus Cellerator's maintained
// f16-storage/f32-compute CSR SpMV on Tesla V100 sm_70. Matrix: rows
// {8192,32768}, block widths {8,16,32}, blocks/row {1,2}, sharing groups/tile
// {1,4,8,16,32}; 3 warmups and 11 measured resident launches. The command,
// results, numerical rule, and toolchain context are emitted as artifacts.
// This file stays slightly above the preferred 600-line threshold because the
// paired fixture, measurement contract, and artifact writer must remain one
// auditable benchmark translation unit; none is reusable runtime code.

#include "Cellerator/geometry/feature_weighted_row_reduction_cuda.hh"
#include "Cellerator/geometry/hardware_cost_model.hh"
#include "benchmark_mutex.hh"
#include <Cellerator/compute/operators/sparse/ops.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace cp = cellpack;
namespace sparse_ops = cellerator::compute::sparse::ops;
using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

constexpr cp::u32 feature_count = 32768u;
constexpr cp::u32 tile_width = 32u;
constexpr int warmups = 3;
constexpr int repeats = 11;
constexpr cp::u64 campaign_id = 0x4350425031320001ull;
constexpr cp::u64 operation_id = 0x4657524544554345ull;
constexpr cp::u64 cost_policy_id = 0x4350423132504f4cull;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackHardwareCostAutotuneBench: " << message << '\n';
        std::exit(1);
    }
}

void check_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackHardwareCostAutotuneBench: " << message << ": "
                  << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

void check_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::cerr << "cellPackHardwareCostAutotuneBench: " << message << ": "
                  << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

cp::u64 hash_mix(cp::u64 hash, cp::u64 value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<unsigned char>(value >> (byte * 8u));
        hash *= 1099511628211ull;
    }
    return hash;
}

cp::u64 scenario_identity(cp::u32 rows, cp::u32 width, cp::u32 row_blocks,
    cp::u32 sharing_groups) {
    cp::u64 hash = 1469598103934665603ull;
    hash = hash_mix(hash, rows);
    hash = hash_mix(hash, width);
    hash = hash_mix(hash, row_blocks);
    return hash_mix(hash, sharing_groups);
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

float value_for(cp::u32 row, cp::u32 feature) {
    const int code = static_cast<int>((row * 17u + feature * 13u) % 31u) - 15;
    return static_cast<float>(code) * 0.0625f;
}

struct host_case {
    cp::u32 rows = 0u, block_width = 0u, row_blocks = 0u, sharing_groups = 0u;
    cp::u64 scenario_id = 0u;
    std::vector<cp::u32> csr_offsets, csr_columns, tile_offsets, tile_blocks,
        tile_cell_masks, entry_offsets, gene_masks, value_offsets, row_permutation;
    std::vector<storage_t> csr_values, tile_values;
    cp::warp_tile_view tiles{};
};

host_case make_case(cp::u32 rows, cp::u32 block_width, cp::u32 row_blocks,
    cp::u32 sharing_groups) {
    host_case result;
    result.rows = rows;
    result.block_width = block_width;
    result.row_blocks = row_blocks;
    result.sharing_groups = sharing_groups;
    result.scenario_id = scenario_identity(rows, block_width, row_blocks, sharing_groups);
    result.csr_offsets.resize(static_cast<std::size_t>(rows) + 1u);
    result.row_permutation.resize(rows);
    std::iota(result.row_permutation.begin(), result.row_permutation.end(), 0u);
    const cp::u32 block_count = feature_count / block_width;
    const cp::u32 tile_count = rows / tile_width;
    result.tile_offsets.push_back(0u);
    result.entry_offsets.push_back(0u);
    result.value_offsets.push_back(0u);
    const std::size_t nnz = static_cast<std::size_t>(rows) * row_blocks * block_width;
    result.csr_columns.reserve(nnz);
    result.csr_values.reserve(nnz);
    result.gene_masks.reserve(static_cast<std::size_t>(rows) * row_blocks);
    result.tile_values.reserve(nnz);
    const cp::u32 gene_mask = block_width == 32u
        ? 0xffffffffu : ((cp::u32{1u} << block_width) - 1u);

    for (cp::u32 tile = 0u; tile < tile_count; ++tile) {
        std::map<cp::u32, cp::u32> masks;
        std::vector<std::vector<cp::u32>> lane_blocks(tile_width);
        const cp::u32 descriptor_span = sharing_groups * row_blocks;
        const cp::u32 base = (tile * descriptor_span) % (block_count - descriptor_span);
        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 group = lane % sharing_groups;
            for (cp::u32 slot = 0u; slot < row_blocks; ++slot) {
                const cp::u32 block = base + group * row_blocks + slot;
                lane_blocks[lane].push_back(block);
                masks[block] |= cp::u32{1u} << lane;
            }
        }
        for (const auto &descriptor : masks) {
            result.tile_blocks.push_back(descriptor.first);
            result.tile_cell_masks.push_back(descriptor.second);
            for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
                if ((descriptor.second & (cp::u32{1u} << lane)) == 0u) continue;
                result.gene_masks.push_back(gene_mask);
                const cp::u32 row = tile * tile_width + lane;
                const cp::u32 begin = descriptor.first * block_width;
                for (cp::u32 local = 0u; local < block_width; ++local) {
                    result.tile_values.push_back(stored(value_for(row, begin + local)));
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
                const cp::u32 begin = block * block_width;
                for (cp::u32 local = 0u; local < block_width; ++local) {
                    const cp::u32 feature = begin + local;
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
    result.tiles.tile_identity = hash_mix(result.scenario_id, 1u);
    result.tiles.feature_block_geometry_identity = hash_mix(result.scenario_id, 2u);
    result.tiles.ordering_identity = hash_mix(result.scenario_id, 3u);
    result.tiles.full_row_count = rows;
    result.tiles.row_count = rows;
    result.tiles.feature_count = feature_count;
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

cp::frozen_packing_plan make_plan(const host_case &host) {
    const cp::u32 block_count = feature_count / host.block_width;
    std::vector<cp::u32> permutation(feature_count), inverse(feature_count),
        feature_to_block(feature_count), feature_to_local(feature_count),
        offsets(block_count + 1u);
    std::iota(permutation.begin(), permutation.end(), 0u);
    std::iota(inverse.begin(), inverse.end(), 0u);
    for (cp::u32 block = 0u; block < block_count; ++block) {
        offsets[block] = block * host.block_width;
        for (cp::u32 local = 0u; local < host.block_width; ++local) {
            const cp::u32 feature = block * host.block_width + local;
            feature_to_block[feature] = block;
            feature_to_local[feature] = local;
        }
    }
    offsets.back() = feature_count;
    const cp::u32 row_offsets[] = {0u, host.rows};
    cp::frozen_packing_plan_build_view build;
    build.row_count = host.rows;
    build.feature_count = feature_count;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = block_count;
    build.feature_block_offsets = offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = host.block_width;
    build.row_group_width = host.rows;
    build.identity.feature_axis_fingerprint = 0x400000000ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x500000000ull;
    build.identity.evaluation_source_identity = host.scenario_id;
    build.cost_policy_identity = cost_policy_id;
    cp::frozen_packing_plan plan;
    check_status(cp::freeze_packing_plan(build, &plan), "freeze benchmark plan");
    return plan;
}

struct timing { float minimum_ms = 0.0f, median_ms = 0.0f, mean_ms = 0.0f; };

template<typename Launch>
timing time_cuda(cudaStream_t stream, Launch launch) {
    cudaEvent_t begin = nullptr, end = nullptr;
    check_cuda(cudaEventCreate(&begin), "create event");
    check_cuda(cudaEventCreate(&end), "create event");
    for (int i = 0; i < warmups; ++i) launch();
    check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");
    std::vector<float> samples;
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
        / samples.size();
    return result;
}

struct measured_pair {
    cp::hardware_cost_observation packed{}, csr{};
    timing packed_time{}, csr_time{};
};

measured_pair measure_case(host_case host, const std::vector<compute_t> &weights,
    cudaStream_t stream, cp::u64 hardware_id, cp::u64 toolchain_id) {
    const auto plan = make_plan(host);
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
    device_array<accum_t> d_packed_output(host.rows), d_csr_output(host.rows);
    upload(d_plan_offsets, plan_offsets); upload(d_plan_permutation, plan_permutation);
    upload(d_order, host.row_permutation); upload(d_tile_offsets, host.tile_offsets);
    upload(d_tile_blocks, host.tile_blocks); upload(d_cell_masks, host.tile_cell_masks);
    upload(d_entry_offsets, host.entry_offsets); upload(d_gene_masks, host.gene_masks);
    upload(d_value_offsets, host.value_offsets); upload(d_tile_values, host.tile_values);
    upload(d_csr_offsets, host.csr_offsets); upload(d_csr_columns, host.csr_columns);
    upload(d_csr_values, host.csr_values); upload(d_weights, weights);

    auto device_tiles = host.tiles;
    device_tiles.tile_block_offsets = d_tile_offsets.data;
    device_tiles.tile_block_ids = d_tile_blocks.data;
    device_tiles.tile_block_cell_masks = d_cell_masks.data;
    device_tiles.block_row_entry_offsets = d_entry_offsets.data;
    device_tiles.row_block_gene_masks = d_gene_masks.data;
    device_tiles.row_block_value_offsets = d_value_offsets.data;
    device_tiles.values = d_tile_values.data;
    auto input = cp::make_feature_weighted_row_reduction_view(plan, host.tiles,
        hash_mix(host.scenario_id, 8u), weights.size(), weights.data());
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
    order.full_row_count = host.rows;
    order.row_count = host.rows;
    order.feature_block_count = feature_count / host.block_width;
    order.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    order.row_domain_identity = plan.identity().row_domain_identity;
    order.row_permutation = d_order.data;
    cp::feature_weighted_row_reduction_buffers buffers{host.rows, d_packed_output.data};
    cp::feature_weighted_row_reduction_result_view result;

    measured_pair measured;
    measured.packed_time = time_cuda(stream, [&] {
        check_status(cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            input, order, buffers, stream, &result), "launch packed reduction");
    });
    cellerator::runtime::execution_context ctx;
    ctx.device = 0;
    ctx.stream = stream;
    measured.csr_time = time_cuda(stream, [&] {
        sparse_ops::base::csr_spmv_fwd_f16_f32(ctx, d_csr_offsets.data,
            d_csr_columns.data, reinterpret_cast<const __half *>(d_csr_values.data),
            host.rows, reinterpret_cast<const float *>(d_weights.data),
            reinterpret_cast<float *>(d_csr_output.data));
    });

    std::vector<accum_t> packed(host.rows), csr(host.rows);
    check_cuda(cudaMemcpy(packed.data(), d_packed_output.data,
        host.rows * sizeof(accum_t), cudaMemcpyDeviceToHost), "download packed");
    check_cuda(cudaMemcpy(csr.data(), d_csr_output.data,
        host.rows * sizeof(accum_t), cudaMemcpyDeviceToHost), "download CSR");
    cp::u64 mismatches = 0u;
    for (cp::u32 row = 0u; row < host.rows; ++row) {
        if (!cp::feature_weighted_row_reduction_within_tolerance(csr[row], packed[row])) {
            ++mismatches;
        }
    }
    check(mismatches == 0u, "packed/CSR numerical mismatch");

    const cp::u64 io_bytes = weights.size() * sizeof(compute_t)
        + host.rows * sizeof(accum_t);
    const cp::u64 packed_metadata = (host.tile_offsets.size() + host.tile_blocks.size()
        + host.tile_cell_masks.size() + host.entry_offsets.size()
        + host.gene_masks.size() + host.value_offsets.size()
        + host.row_permutation.size() + plan_offsets.size() + plan_permutation.size())
        * sizeof(cp::u32);
    const cp::u64 csr_metadata = (host.csr_offsets.size() + host.csr_columns.size())
        * sizeof(cp::u32);
    cp::hardware_cost_shape common;
    common.row_count = host.rows;
    common.feature_count = feature_count;
    common.tile_row_width = tile_width;
    common.feature_block_width = host.block_width;
    common.nnz_count = host.tile_values.size();
    common.tile_count = host.tiles.tile_count;
    common.tile_block_count = host.tiles.tile_block_count;
    common.row_block_entry_count = host.tiles.row_block_entry_count;
    common.input_output_bytes = io_bytes;
    common.index_width_bytes = sizeof(cp::u32);
    common.alignment_bytes = alignof(cp::u32);
    const auto partition = host.scenario_id % 5u == 0u
        ? cp::hardware_cost_partition::held_out
        : cp::hardware_cost_partition::calibration;
    auto make_observation = [&](cp::hardware_execution_path path, cp::u64 suffix,
                                const timing &elapsed, cp::u64 metadata,
                                cp::u64 payload) {
        cp::hardware_cost_observation observation;
        observation.path = path;
        observation.partition = partition;
        observation.campaign_identity = campaign_id;
        observation.configuration_identity = hash_mix(host.scenario_id, suffix);
        observation.hardware_identity = hardware_id;
        observation.toolchain_identity = toolchain_id;
        observation.operation_identity = operation_id;
        observation.cost_policy_identity = cost_policy_id;
        observation.shape = common;
        observation.shape.metadata_bytes = metadata;
        observation.shape.payload_bytes = payload;
        observation.shape.estimated_memory_transactions =
            (metadata + payload + observation.shape.input_output_bytes + 31u) / 32u;
        observation.median_elapsed_nanoseconds = static_cast<cp::u64>(
            std::llround(elapsed.median_ms * 1.0e6));
        observation.correctness_items = host.rows;
        observation.correctness_mismatches = mismatches;
        observation.warmup_count = warmups;
        observation.repeat_count = repeats;
        observation.launches_per_repeat = 1u;
        return observation;
    };
    measured.packed = make_observation(cp::hardware_execution_path::direct_warp_tiles,
        1u, measured.packed_time, packed_metadata,
        host.tile_values.size() * sizeof(storage_t));
    measured.csr = make_observation(cp::hardware_execution_path::csr_fallback,
        2u, measured.csr_time, csr_metadata,
        host.csr_values.size() * sizeof(storage_t));
    return measured;
}

std::string json_escape(const std::string &value) {
    std::string result;
    for (char character : value) {
        if (character == '"' || character == '\\') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

void write_text(const std::filesystem::path &path, const std::string &text) {
    std::ofstream output(path);
    check(static_cast<bool>(output), "open output artifact");
    output << text;
    check(static_cast<bool>(output), "write output artifact");
}

template<typename Storage, typename Compute, typename Accum>
int run_benchmark(const std::filesystem::path &output_dir) {
    if constexpr (!(std::is_same<Storage, __half>::value
            && std::is_same<Compute, float>::value
            && std::is_same<Accum, float>::value)) {
        std::cout << "cellPackHardwareCostAutotuneBench: configured precision has no "
                     "type-equivalent maintained CSR baseline; skipped\n";
        return 0;
    }
    cellerator::bench::benchmark_mutex_guard mutex(
        "cellPackHardwareCostAutotuneBench", 0);
    check_cuda(cudaSetDevice(0), "set device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "get device properties");
    const cp::u64 hardware_id = hash_mix(hash_mix(hash_mix(1469598103934665603ull,
        properties.major), properties.minor), properties.totalGlobalMem);
    const cp::u64 toolchain_id = hash_mix(hash_mix(1469598103934665603ull,
        CUDART_VERSION), __CUDACC_VER_MAJOR__ * 10000u
        + __CUDACC_VER_MINOR__ * 100u + __CUDACC_VER_BUILD__);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    std::vector<compute_t> weights(feature_count);
    for (cp::u32 feature = 0u; feature < feature_count; ++feature) {
        weights[feature] = 0.25f + static_cast<float>(feature % 97u) * 0.0078125f;
    }

    std::vector<measured_pair> pairs;
    std::vector<cp::hardware_cost_observation> observations;
    const cp::u32 row_options[] = {8192u, 32768u};
    const cp::u32 width_options[] = {8u, 16u, 32u};
    const cp::u32 block_options[] = {1u, 2u};
    const cp::u32 sharing_options[] = {1u, 4u, 8u, 16u, 32u};
    for (cp::u32 rows : row_options) for (cp::u32 width : width_options)
        for (cp::u32 row_blocks : block_options)
            for (cp::u32 sharing : sharing_options) {
                auto pair = measure_case(make_case(rows, width, row_blocks, sharing),
                    weights, stream, hardware_id, toolchain_id);
                std::cout << "rows=" << rows << " width=" << width
                          << " blocks_per_row=" << row_blocks
                          << " sharing_groups=" << sharing
                          << " packed_ms=" << pair.packed_time.median_ms
                          << " csr_ms=" << pair.csr_time.median_ms << '\n';
                observations.push_back(pair.packed);
                observations.push_back(pair.csr);
                pairs.push_back(pair);
            }
    check_cuda(cudaStreamDestroy(stream), "destroy stream");

    cp::hardware_cost_fit_config fit;
    fit.campaign_identity = campaign_id;
    fit.hardware_identity = hardware_id;
    fit.toolchain_identity = toolchain_id;
    fit.operation_identity = operation_id;
    fit.supported_feature_block_width_mask = cp::hardware_cost_block_width_bit(8u)
        | cp::hardware_cost_block_width_bit(16u)
        | cp::hardware_cost_block_width_bit(32u);
    fit.ridge_regularization = 1.0e-3;
    cp::hardware_cost_model model;
    check_status(cp::fit_hardware_cost_model(observations.data(), observations.size(),
        fit, &model), "fit measured model");
    std::vector<cp::hardware_cost_prediction_error> errors(observations.size());
    cp::hardware_cost_validation_buffers buffers{errors.size(), errors.data()};
    cp::hardware_cost_validation_report report;
    check_status(cp::evaluate_hardware_cost_model(model, observations.data(),
        observations.size(), buffers, &report), "validate measured model");
    check(report.direct_held_out.observation_count != 0u
        && report.csr_held_out.observation_count != 0u, "held-out denominator");
    check(report.direct_held_out.mean_absolute_percentage_error <= 0.50,
        "direct held-out MAPE exceeds recorded v1 bound");
    check(report.csr_held_out.mean_absolute_percentage_error <= 0.50,
        "CSR held-out MAPE exceeds recorded v1 bound");

    cp::u32 packed_wins = 0u, csr_wins = 0u;
    cp::u64 selected_storage_bytes = 0u, selected_measured_nanoseconds = 0u,
        storage_only_bytes = 0u, storage_only_measured_nanoseconds = 0u;
    for (const auto &pair : pairs) {
        cp::hardware_cost_candidate candidates[2];
        candidates[0].candidate_identity = pair.packed.configuration_identity;
        candidates[0].cost_policy_identity = cost_policy_id;
        candidates[0].path = pair.packed.path;
        candidates[0].shape = pair.packed.shape;
        candidates[0].storage_bytes = pair.packed.shape.metadata_bytes
            + pair.packed.shape.payload_bytes;
        candidates[1].candidate_identity = pair.csr.configuration_identity;
        candidates[1].cost_policy_identity = cost_policy_id;
        candidates[1].path = pair.csr.path;
        candidates[1].shape = pair.csr.shape;
        candidates[1].storage_bytes = pair.csr.shape.metadata_bytes
            + pair.csr.shape.payload_bytes;
        cp::hardware_autotune_config tune;
        tune.supported_feature_block_width_mask = fit.supported_feature_block_width_mask;
        tune.storage_byte_weight = 1.0;
        tune.runtime_nanosecond_weight = 1.0;
        cp::hardware_autotune_result selected;
        check_status(cp::select_hardware_cost_candidate(model, candidates, 2u,
            tune, &selected), "select paired path");
        const bool selected_packed =
            selected.path == cp::hardware_execution_path::direct_warp_tiles;
        selected_packed ? ++packed_wins : ++csr_wins;
        selected_storage_bytes += selected.storage_bytes;
        selected_measured_nanoseconds += selected_packed
            ? pair.packed.median_elapsed_nanoseconds
            : pair.csr.median_elapsed_nanoseconds;
        tune.runtime_nanosecond_weight = 0.0;
        check_status(cp::select_hardware_cost_candidate(model, candidates, 2u,
            tune, &selected), "select storage-only paired path");
        const bool storage_packed =
            selected.path == cp::hardware_execution_path::direct_warp_tiles;
        storage_only_bytes += selected.storage_bytes;
        storage_only_measured_nanoseconds += storage_packed
            ? pair.packed.median_elapsed_nanoseconds
            : pair.csr.median_elapsed_nanoseconds;
    }

    std::filesystem::create_directories(output_dir / "impl_a");
    std::filesystem::create_directories(output_dir / "impl_b");
    std::ostringstream csv;
    csv << "config_id,path,partition,rows,width,blocks_per_row,sharing_groups,nnz,"
           "metadata_bytes,payload_bytes,io_bytes,median_ns,correctness_items,mismatches\n";
    for (std::size_t index = 0; index < pairs.size(); ++index) {
        const cp::u32 sharing = sharing_options[index % 5u];
        for (const auto *observation : {&pairs[index].packed, &pairs[index].csr}) {
            csv << observation->configuration_identity << ','
                << (observation->path == cp::hardware_execution_path::direct_warp_tiles
                    ? "direct_tiles" : "csr") << ','
                << (observation->partition == cp::hardware_cost_partition::held_out
                    ? "held_out" : "calibration") << ','
                << observation->shape.row_count << ','
                << observation->shape.feature_block_width << ','
                << observation->shape.row_block_entry_count / observation->shape.row_count
                << ',' << sharing << ',' << observation->shape.nnz_count << ','
                << observation->shape.metadata_bytes << ','
                << observation->shape.payload_bytes << ','
                << observation->shape.input_output_bytes << ','
                << observation->median_elapsed_nanoseconds << ','
                << observation->correctness_items << ','
                << observation->correctness_mismatches << '\n';
        }
    }
    write_text(output_dir / "hardware_cost_observations.csv", csv.str());
    std::ostringstream config_json;
    config_json << "{\n  \"schema_version\": 1,\n  \"campaign_identity\": "
                << campaign_id << ",\n  \"device\": \"" << json_escape(properties.name)
                << "\",\n  \"sm\": \"" << properties.major << properties.minor
                << "\",\n  \"cuda_runtime\": " << CUDART_VERSION
                << ",\n  \"rows\": [8192,32768],\n  \"feature_block_widths\": "
                   "[8,16,32],\n  \"blocks_per_row\": [1,2],\n  \"sharing_groups_per_tile\": "
                   "[1,4,8,16,32],\n  \"warmups\": 3,\n  \"repeats\": 11,\n  \"timing_scope\": "
                   "\"device-resident one-launch kernel; setup, transfers, allocation, and "
                   "synchronization excluded\",\n  \"tolerance\": "
                   "\"feature_weighted_row_reduction_within_tolerance v1\",\n  \"mutex\": "
                   "\"repository benchmark mutex plus external CP-BP-12 GPU lock\"\n}\n";
    write_text(output_dir / "compare_config.json", config_json.str());
    write_text(output_dir / "impl_a" / "run_config.json", config_json.str());
    write_text(output_dir / "impl_b" / "run_config.json", config_json.str());
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(6)
            << "{\n  \"status\": \"passed\",\n  \"scenario_count\": " << pairs.size()
            << ",\n  \"observation_count\": " << observations.size()
            << ",\n  \"model_identity\": " << model.model_identity
            << ",\n  \"direct_heldout_count\": "
            << report.direct_held_out.observation_count
            << ",\n  \"direct_heldout_mape\": "
            << report.direct_held_out.mean_absolute_percentage_error
            << ",\n  \"csr_heldout_count\": " << report.csr_held_out.observation_count
            << ",\n  \"csr_heldout_mape\": "
            << report.csr_held_out.mean_absolute_percentage_error
            << ",\n  \"lambda_storage_1_runtime_1_packed_wins\": " << packed_wins
            << ",\n  \"lambda_storage_1_runtime_1_csr_wins\": " << csr_wins
            << ",\n  \"lambda_storage_1_runtime_1_selected_bytes\": "
            << selected_storage_bytes
            << ",\n  \"lambda_storage_1_runtime_1_measured_ns\": "
            << selected_measured_nanoseconds
            << ",\n  \"storage_only_selected_bytes\": " << storage_only_bytes
            << ",\n  \"storage_only_measured_ns\": "
            << storage_only_measured_nanoseconds
            << ",\n  \"correctness_mismatches\": 0\n}\n";
    write_text(output_dir / "summary.json", summary.str());
    write_text(output_dir / "impl_a" / "results.json", summary.str());
    write_text(output_dir / "impl_b" / "results.json", summary.str());
    std::ostringstream text_summary;
    text_summary << "CP-BP-12 measured V100 model: PASSED\n"
                 << "Scenarios: " << pairs.size() << ", observations: "
                 << observations.size() << "\nDirect held-out MAPE: "
                 << report.direct_held_out.mean_absolute_percentage_error
                 << "\nCSR held-out MAPE: "
                 << report.csr_held_out.mean_absolute_percentage_error
                 << "\nStorage+runtime lambda (1,1) path wins: packed=" << packed_wins
                 << ", CSR=" << csr_wins
                 << "\nSelected measured totals: bytes=" << selected_storage_bytes
                 << ", ns=" << selected_measured_nanoseconds
                 << "\nStorage-only measured totals: bytes=" << storage_only_bytes
                 << ", ns=" << storage_only_measured_nanoseconds
                 << "\nCorrectness mismatches: 0\n";
    write_text(output_dir / "summary.txt", text_summary.str());
    std::cout << text_summary.str();
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::filesystem::path output = "cp-bp12-compare";
    if (argc == 3 && std::string(argv[1]) == "--output-dir") output = argv[2];
    else if (argc != 1) {
        std::cerr << "usage: cellPackHardwareCostAutotuneBench "
                     "[--output-dir PATH]\n";
        return 2;
    }
    return run_benchmark<storage_t, compute_t, accum_t>(output);
}
