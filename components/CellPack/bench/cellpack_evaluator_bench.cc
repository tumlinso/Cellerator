#include <CellPack/evaluator.hh>

#include "benchmark_mutex.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct bench_options {
    cellpack::u32 rows = 100000u;
    cellpack::u32 features = 20000u;
    cellpack::u32 nnz_per_row = 32u;
    cellpack::u32 row_group_width = 128u;
    cellpack::u32 feature_block_width = 128u;
    cellpack::u32 warmup = 1u;
    cellpack::u32 repeats = 5u;
};

struct fixture {
    std::vector<cellpack::u32> row_offsets;
    std::vector<cellpack::u32> feature_ids;
    std::vector<cellpack::u32> row_permutation;
    std::vector<cellpack::u32> inverse_row_permutation;
    std::vector<cellpack::u32> feature_permutation;
    std::vector<cellpack::u32> inverse_feature_permutation;
    std::vector<cellpack::u32> row_group_offsets;
    std::vector<cellpack::u32> feature_block_offsets;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

cellpack::u32 parse_u32(const char *text, const char *label) {
    if (text == nullptr || *text == '\0') throw std::invalid_argument(std::string("missing value for ") + label);
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == nullptr || *end != '\0' || value > 0xfffffffful) {
        throw std::invalid_argument(std::string("invalid uint32 value for ") + label);
    }
    return static_cast<cellpack::u32>(value);
}

bench_options parse_args(int argc, char **argv) {
    bench_options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto next = [&](const char *label) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (argument == "--rows") options.rows = parse_u32(next("--rows"), "--rows");
        else if (argument == "--features") options.features = parse_u32(next("--features"), "--features");
        else if (argument == "--nnz-row") options.nnz_per_row = parse_u32(next("--nnz-row"), "--nnz-row");
        else if (argument == "--row-group-width") options.row_group_width = parse_u32(next("--row-group-width"), "--row-group-width");
        else if (argument == "--feature-block-width") options.feature_block_width = parse_u32(next("--feature-block-width"), "--feature-block-width");
        else if (argument == "--warmup") options.warmup = parse_u32(next("--warmup"), "--warmup");
        else if (argument == "--repeats") options.repeats = parse_u32(next("--repeats"), "--repeats");
        else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: cellPackEvaluatorBench [--rows N] [--features N] [--nnz-row N] "
                << "[--row-group-width N] [--feature-block-width N] [--warmup N] [--repeats N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    require(options.rows != 0u, "rows must be nonzero");
    require(options.features != 0u, "features must be nonzero");
    require(options.nnz_per_row != 0u && options.nnz_per_row <= options.features, "nnz per row must be in [1, features]");
    require(options.row_group_width != 0u, "row group width must be nonzero");
    require(options.feature_block_width != 0u, "feature block width must be nonzero");
    require(options.repeats != 0u, "repeats must be nonzero");
    const cellpack::u64 nnz = static_cast<cellpack::u64>(options.rows) * options.nnz_per_row;
    require(nnz <= 0xffffffffull, "benchmark nnz exceeds CellPack uint32 support");
    return options;
}

void build_boundaries(cellpack::u32 count, cellpack::u32 width, std::vector<cellpack::u32> *out) {
    out->push_back(0u);
    for (cellpack::u32 begin = width; begin < count; begin += width) out->push_back(begin);
    out->push_back(count);
}

void rotate_permutation(cellpack::u32 count, cellpack::u32 shift, std::vector<cellpack::u32> *permutation) {
    permutation->resize(count);
    std::iota(permutation->begin(), permutation->end(), 0u);
    if (count != 0u) std::rotate(permutation->begin(), permutation->begin() + (shift % count), permutation->end());
}

fixture generate_fixture(const bench_options &options) {
    fixture data;
    data.row_offsets.reserve(static_cast<std::size_t>(options.rows) + 1u);
    data.feature_ids.reserve(static_cast<std::size_t>(options.rows) * options.nnz_per_row);
    data.row_offsets.push_back(0u);
    std::vector<cellpack::u32> row_features(options.nnz_per_row);
    for (cellpack::u32 row = 0u; row < options.rows; ++row) {
        const cellpack::u32 begin = static_cast<cellpack::u32>((static_cast<cellpack::u64>(row) * 131u) % options.features);
        for (cellpack::u32 entry = 0u; entry < options.nnz_per_row; ++entry) {
            row_features[entry] = (begin + entry) % options.features;
        }
        std::sort(row_features.begin(), row_features.end());
        data.feature_ids.insert(data.feature_ids.end(), row_features.begin(), row_features.end());
        data.row_offsets.push_back(static_cast<cellpack::u32>(data.feature_ids.size()));
    }
    rotate_permutation(options.rows, options.rows / 3u, &data.row_permutation);
    rotate_permutation(options.features, options.features / 5u, &data.feature_permutation);
    data.inverse_row_permutation.resize(options.rows);
    data.inverse_feature_permutation.resize(options.features);
    require(cellpack::build_inverse_permutation(
                data.row_permutation.data(), options.rows, data.inverse_row_permutation.data()),
            "failed to invert row permutation");
    require(cellpack::build_inverse_permutation(
                data.feature_permutation.data(), options.features, data.inverse_feature_permutation.data()),
            "failed to invert feature permutation");
    build_boundaries(options.rows, options.row_group_width, &data.row_group_offsets);
    build_boundaries(options.features, options.feature_block_width, &data.feature_block_offsets);
    return data;
}

cellpack::packing_plan_view make_plan(const bench_options &options, const fixture &data) {
    cellpack::packing_plan_view plan;
    plan.row_count = options.rows;
    plan.feature_count = options.features;
    plan.row_permutation = data.row_permutation.data();
    plan.inverse_row_permutation = data.inverse_row_permutation.data();
    plan.feature_permutation = data.feature_permutation.data();
    plan.inverse_feature_permutation = data.inverse_feature_permutation.data();
    plan.row_group_count = static_cast<cellpack::u32>(data.row_group_offsets.size() - 1u);
    plan.row_group_offsets = data.row_group_offsets.data();
    plan.feature_block_count = static_cast<cellpack::u32>(data.feature_block_offsets.size() - 1u);
    plan.feature_block_offsets = data.feature_block_offsets.data();
    return plan;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const bench_options options = parse_args(argc, argv);
        cellerator::bench::benchmark_mutex_guard mutex("cellPackEvaluatorBench");
        fixture data = generate_fixture(options);
        const cellpack::csr_support_view source{
            options.rows,
            options.features,
            static_cast<cellpack::u32>(data.feature_ids.size()),
            data.row_offsets.data(),
            data.feature_ids.data()
        };
        const cellpack::packing_plan_view plan = make_plan(options, data);
        const clock_type::time_point prepare_begin = clock_type::now();
        cellpack::prepared_csr_support prepared;
        cellpack::validation_result status = cellpack::prepare_csr_support(source, &prepared);
        require(static_cast<bool>(status), status.message);
        const clock_type::time_point prepare_end = clock_type::now();
        cellpack::packing_evaluation_requirements requirements;
        status = cellpack::query_packing_evaluation_requirements(prepared, plan, &requirements);
        require(static_cast<bool>(status), status.message);

        std::vector<cellpack::packing_evaluation_entry> workspace(requirements.workspace_entry_capacity);
        std::vector<cellpack::occupied_tile_occupancy> tiles(requirements.occupied_tile_capacity);
        std::vector<cellpack::u32> row_active(requirements.execution_row_capacity);
        std::vector<cellpack::row_group_occupancy> row_groups(requirements.row_group_capacity);
        const cellpack::packing_evaluation_workspace_view workspace_view{
            workspace.data(), static_cast<cellpack::u32>(workspace.size())};
        const cellpack::packing_occupancy_buffers buffers{
            tiles.data(), static_cast<cellpack::u32>(tiles.size()),
            row_active.data(), static_cast<cellpack::u32>(row_active.size()),
            row_groups.data(), static_cast<cellpack::u32>(row_groups.size())};

        cellpack::packing_occupancy_result result;
        for (cellpack::u32 i = 0u; i < options.warmup; ++i) {
            status = cellpack::evaluate_packing_plan(prepared, plan, workspace_view, buffers, &result);
            require(static_cast<bool>(status), status.message);
        }
        double total_ms = 0.0;
        for (cellpack::u32 i = 0u; i < options.repeats; ++i) {
            const clock_type::time_point begin = clock_type::now();
            status = cellpack::evaluate_packing_plan(prepared, plan, workspace_view, buffers, &result);
            const clock_type::time_point end = clock_type::now();
            require(static_cast<bool>(status), status.message);
            total_ms += std::chrono::duration<double, std::milli>(end - begin).count();
        }

        const cellpack::u64 source_bytes = static_cast<cellpack::u64>(data.row_offsets.size()) * sizeof(cellpack::u32)
            + static_cast<cellpack::u64>(data.feature_ids.size()) * sizeof(cellpack::u32);
        const double mean_nnz_per_tile = result.occupied_tile_count == 0u
            ? 0.0
            : static_cast<double>(result.totals.total_nnz) / result.occupied_tile_count;
        std::cout << "evaluator: host_reference_exact\n";
        std::cout << "source_bytes: " << source_bytes << "\n";
        std::cout << "rows: " << options.rows << "\n";
        std::cout << "features: " << options.features << "\n";
        std::cout << "nnz: " << source.nnz_count << "\n";
        std::cout << "row_group_width: " << options.row_group_width << "\n";
        std::cout << "feature_block_width: " << options.feature_block_width << "\n";
        std::cout << "source_prepare_ms: "
                  << std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count() << "\n";
        std::cout << "evaluation_ms_mean: " << (total_ms / options.repeats) << "\n";
        std::cout << "temporary_workspace_bytes: " << requirements.temporary_workspace_bytes << "\n";
        std::cout << "output_buffer_bytes: " << requirements.output_buffer_bytes << "\n";
        std::cout << "logical_tiles: " << result.totals.logical_tile_count << "\n";
        std::cout << "occupied_tiles: " << result.totals.occupied_tile_count << "\n";
        std::cout << "empty_tiles: " << result.totals.empty_tile_count << "\n";
        std::cout << "nnz_per_occupied_tile_mean: " << mean_nnz_per_tile << "\n";
        std::cout << "dense_padding_if_occupied_tiles_dense: " << result.totals.dense_padding << "\n";
        std::cout << "future_cuda_route: cub_radix_sort_run_length_reduce\n";
    } catch (const std::exception &error) {
        std::cerr << "cellPackEvaluatorBench: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
