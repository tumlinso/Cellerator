#include "Cellerator/geometry/apply_plan.hh"

#include "benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

struct options {
    u32 rows = 65536u;
    u32 features = 30000u;
    u32 nnz_per_row = 32u;
    u32 block_width = 16u;
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
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto next = [&](const char *label) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (argument == "--rows") result.rows = parse_u32(next("--rows"), "--rows");
        else if (argument == "--features") result.features = parse_u32(next("--features"), "--features");
        else if (argument == "--nnz-row") result.nnz_per_row = parse_u32(next("--nnz-row"), "--nnz-row");
        else if (argument == "--block-width") result.block_width = parse_u32(next("--block-width"), "--block-width");
        else if (argument == "--warmup") result.warmup = parse_u32(next("--warmup"), "--warmup");
        else if (argument == "--repeats") result.repeats = parse_u32(next("--repeats"), "--repeats");
        else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: cellPackApplyPlanBench [--rows N] [--features N] "
                "[--nnz-row N] [--block-width N] [--warmup N] [--repeats N]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown argument: " + argument);
    }
    require(result.rows != 0u && result.features != 0u, "rows/features must be nonzero");
    require(result.nnz_per_row != 0u && result.nnz_per_row <= result.features,
        "nnz-row must be in the feature-axis range");
    require(result.block_width != 0u && result.repeats != 0u,
        "block width/repeats must be nonzero");
    require(static_cast<u64>(result.rows) * result.nnz_per_row <= static_cast<u64>(INT_MAX),
        "benchmark NNZ exceeds the CUB signed-count limit");
    return result;
}

template<class T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;

    device_buffer() = default;
    explicit device_buffer(std::size_t count_) : count(count_) {
        if (count != 0u) cuda_step(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc failed");
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
};

struct fixture {
    std::vector<u32> rows, features;
    std::vector<float> values;
};

fixture make_fixture(const options &settings) {
    fixture result;
    result.rows.reserve(static_cast<std::size_t>(settings.rows) + 1u);
    result.features.reserve(static_cast<std::size_t>(settings.rows) * settings.nnz_per_row);
    result.values.reserve(result.features.capacity());
    result.rows.push_back(0u);
    std::vector<u32> row_features(settings.nnz_per_row);
    for (u32 row = 0u; row < settings.rows; ++row) {
        for (u32 entry = 0u; entry < settings.nnz_per_row; ++entry) {
            row_features[entry] = static_cast<u32>((static_cast<u64>(row) * 131u
                + static_cast<u64>(entry) * 977u) % settings.features);
        }
        std::sort(row_features.begin(), row_features.end());
        row_features.erase(std::unique(row_features.begin(), row_features.end()), row_features.end());
        for (u32 feature : row_features) {
            result.features.push_back(feature);
            result.values.push_back(static_cast<float>((row & 1023u) * 0.001 + feature * 0.00001));
        }
        result.rows.push_back(static_cast<u32>(result.features.size()));
        row_features.resize(settings.nnz_per_row);
    }
    return result;
}

cellpack::frozen_packing_plan make_plan(const options &settings) {
    std::vector<u32> permutation(settings.features), inverse(settings.features);
    const u32 shift = settings.features / 3u;
    for (u32 execution = 0u; execution < settings.features; ++execution) {
        permutation[execution] = (execution + shift) % settings.features;
        inverse[permutation[execution]] = execution;
    }
    std::vector<u32> block_offsets;
    for (u32 begin = 0u; begin < settings.features; begin += settings.block_width) {
        block_offsets.push_back(begin);
    }
    block_offsets.push_back(settings.features);
    std::vector<u32> feature_to_block(settings.features), feature_to_local(settings.features);
    for (u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (u32 execution = block_offsets[block]; execution < block_offsets[block + 1u]; ++execution) {
            const u32 canonical = permutation[execution];
            feature_to_block[canonical] = block;
            feature_to_local[canonical] = execution - block_offsets[block];
        }
    }
    constexpr u32 row_group_width = 128u;
    std::vector<u32> row_groups;
    for (u32 begin = 0u; begin < settings.rows; begin += row_group_width) row_groups.push_back(begin);
    row_groups.push_back(settings.rows);
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = settings.rows;
    build.feature_count = settings.features;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = static_cast<u32>(block_offsets.size() - 1u);
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = static_cast<u32>(row_groups.size() - 1u);
    build.row_group_offsets = row_groups.data();
    build.maximum_feature_block_width = settings.block_width;
    build.row_group_width = row_group_width;
    build.identity.feature_axis_fingerprint = 0x4350425030354641ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x435042503035524full;
    build.identity.evaluation_source_identity = 0x4350425030354556ull;
    build.cost_policy_identity = 0x435042503035434full;
    cellpack::frozen_packing_plan result;
    const cellpack::validation_result status = cellpack::freeze_packing_plan(build, &result);
    require(static_cast<bool>(status), status.message);
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const options settings = parse_options(argc, argv);
        cellerator::bench::benchmark_mutex_guard mutex("cellPackApplyPlanBench", 0);
        cuda_step(cudaSetDevice(0), "cudaSetDevice failed");
        fixture source_data = make_fixture(settings);
        cellpack::frozen_packing_plan plan = make_plan(settings);
        cellpack::plan_application_context context;
        context.full_row_count = settings.rows;
        context.feature_count = settings.features;
        context.feature_axis_fingerprint = plan.identity().feature_axis_fingerprint;
        context.feature_axis_fingerprint_version = plan.identity().feature_axis_fingerprint_version;
        context.row_domain_identity = plan.identity().row_domain_identity;
        cellpack::plan_application_source_view source;
        source.row_count = settings.rows;
        source.feature_count = settings.features;
        source.nnz_count = static_cast<u32>(source_data.features.size());
        source.value_size_bytes = sizeof(float);
        source.row_offsets = source_data.rows.data();
        source.canonical_feature_ids = source_data.features.data();
        source.values = source_data.values.data();
        require(static_cast<bool>(cellpack::validate_plan_application_source_host(plan, context, source)),
            "source validation failed");

        std::vector<u64> host_keys(source.nnz_count);
        std::vector<u32> host_order(source.nnz_count), host_blocks(source.nnz_count);
        std::vector<u32> host_locals(source.nnz_count), host_features(source.nnz_count);
        std::vector<u32> host_rows(source_data.rows.size());
        std::vector<float> host_values(source.nnz_count);
        const cellpack::plan_application_host_workspace_view host_workspace{
            source.nnz_count, host_keys.data(), host_order.data()};
        const cellpack::plan_application_buffers host_buffers{
            host_rows.size(), host_blocks.size(), host_values.size() * sizeof(float),
            host_rows.data(), host_blocks.data(), host_locals.data(), host_features.data(), host_values.data()};
        cellpack::ordered_plan_partition_view host_result;
        const auto host_begin = std::chrono::steady_clock::now();
        require(static_cast<bool>(cellpack::apply_frozen_plan_host(
            plan, context, source, host_workspace, host_buffers, &host_result)), "host application failed");
        const double host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_begin).count();

        cellpack::plan_application_cuda_requirements required;
        require(static_cast<bool>(cellpack::query_plan_application_cuda_requirements(
            source.row_count, source.nnz_count, &required)), "CUDA requirements query failed");
        device_buffer<u32> d_rows(source_data.rows.size()), d_features(source.nnz_count);
        device_buffer<float> d_values(source.nnz_count);
        device_buffer<u32> d_map_block(plan.feature_count()), d_map_local(plan.feature_count());
        device_buffer<u64> d_keys_in(source.nnz_count), d_keys_out(source.nnz_count);
        device_buffer<u32> d_order_in(source.nnz_count), d_order_out(source.nnz_count);
        device_buffer<unsigned char> d_cub(required.cub_temporary_bytes);
        device_buffer<u32> d_out_rows(source_data.rows.size()), d_out_blocks(source.nnz_count);
        device_buffer<u32> d_out_locals(source.nnz_count), d_out_features(source.nnz_count);
        device_buffer<float> d_out_values(source.nnz_count);
        cuda_step(cudaMemcpy(d_rows.data, source_data.rows.data(), source_data.rows.size() * sizeof(u32), cudaMemcpyHostToDevice), "row upload failed");
        cuda_step(cudaMemcpy(d_features.data, source_data.features.data(), source.nnz_count * sizeof(u32), cudaMemcpyHostToDevice), "feature upload failed");
        cuda_step(cudaMemcpy(d_values.data, source_data.values.data(), source.nnz_count * sizeof(float), cudaMemcpyHostToDevice), "value upload failed");
        cuda_step(cudaMemcpy(d_map_block.data, plan.feature_to_block(), plan.feature_count() * sizeof(u32), cudaMemcpyHostToDevice), "block-map upload failed");
        cuda_step(cudaMemcpy(d_map_local.data, plan.feature_to_local(), plan.feature_count() * sizeof(u32), cudaMemcpyHostToDevice), "local-map upload failed");
        cellpack::plan_application_source_view device_source = source;
        device_source.row_offsets = d_rows.data;
        device_source.canonical_feature_ids = d_features.data;
        device_source.values = d_values.data;
        const cellpack::plan_application_device_feature_view device_plan{
            plan.feature_count(), plan.feature_block_count(), d_map_block.data, d_map_local.data};
        const cellpack::plan_application_cuda_workspace_view cuda_workspace{
            source.nnz_count, d_keys_in.data, d_keys_out.data, d_order_in.data, d_order_out.data,
            d_cub.data, required.cub_temporary_bytes};
        const cellpack::plan_application_buffers cuda_buffers{
            source_data.rows.size(), source.nnz_count, source.nnz_count * sizeof(float),
            d_out_rows.data, d_out_blocks.data, d_out_locals.data, d_out_features.data, d_out_values.data};
        cellpack::ordered_plan_partition_view device_result;
        cudaEvent_t begin = nullptr, end = nullptr;
        cuda_step(cudaEventCreate(&begin), "event create failed");
        cuda_step(cudaEventCreate(&end), "event create failed");
        std::vector<float> timings;
        for (u32 iteration = 0u; iteration < settings.warmup + settings.repeats; ++iteration) {
            cuda_step(cudaEventRecord(begin), "event record failed");
            require(static_cast<bool>(cellpack::apply_frozen_plan_cuda(
                plan, context, device_source, device_plan, cuda_workspace, cuda_buffers, nullptr,
                &device_result)), "CUDA application failed");
            cuda_step(cudaEventRecord(end), "event record failed");
            cuda_step(cudaEventSynchronize(end), "event synchronize failed");
            float elapsed = 0.0f;
            cuda_step(cudaEventElapsedTime(&elapsed, begin, end), "event elapsed failed");
            if (iteration >= settings.warmup) timings.push_back(elapsed);
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);

        std::vector<u32> check_features(source.nnz_count);
        std::vector<float> check_values(source.nnz_count);
        cuda_step(cudaMemcpy(check_features.data(), d_out_features.data, source.nnz_count * sizeof(u32), cudaMemcpyDeviceToHost), "feature download failed");
        cuda_step(cudaMemcpy(check_values.data(), d_out_values.data, source.nnz_count * sizeof(float), cudaMemcpyDeviceToHost), "value download failed");
        require(check_features == host_features && check_values == host_values,
            "CUDA application differs from the host reference");
        const float minimum_ms = *std::min_element(timings.begin(), timings.end());
        const double mean_ms = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        const std::size_t source_bytes = source_data.rows.size() * sizeof(u32)
            + source_data.features.size() * sizeof(u32) + source_data.values.size() * sizeof(float);
        const std::size_t output_bytes = source_data.rows.size() * sizeof(u32)
            + source_data.features.size() * (3u * sizeof(u32) + sizeof(float));
        std::cout << "application: cp_bp_05_cub_segmented_radix_sort\n";
        std::cout << "architecture: sm_70\n";
        std::cout << "rows: " << settings.rows << "\n";
        std::cout << "features: " << settings.features << "\n";
        std::cout << "nnz: " << source.nnz_count << "\n";
        std::cout << "value_size_bytes: " << source.value_size_bytes << "\n";
        std::cout << "block_width: " << settings.block_width << "\n";
        std::cout << "source_bytes: " << source_bytes << "\n";
        std::cout << "output_bytes: " << output_bytes << "\n";
        std::cout << "cub_temporary_bytes: " << required.cub_temporary_bytes << "\n";
        std::cout << "total_temporary_bytes: " << required.total_temporary_bytes << "\n";
        std::cout << "host_reference_ms: " << host_ms << "\n";
        std::cout << "cuda_minimum_ms: " << minimum_ms << "\n";
        std::cout << "cuda_mean_ms: " << mean_ms << "\n";
        std::cout << "transfers_in_timed_region: 0\n";
        std::cout << "synchronizations_in_api: 0\n";
        std::cout << "specialized_short_row_sort: deferred_pending_evidence\n";
    } catch (const std::exception &error) {
        std::cerr << "cellPackApplyPlanBench: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
