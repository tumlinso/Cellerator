#include <Cellerator/compute/neighbors/forward_neighbors.hh>
#include "benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fn = ::cellerator::compute::neighbors::forward_neighbors;

namespace {

struct bench_config {
    std::string backend = "dense";
    std::int64_t rows = 32768;
    std::int64_t queries = 2048;
    std::int64_t latent_dim = 64;
    std::int64_t top_k = 16;
    std::int64_t shard_count = 4;
    std::int64_t warmup = 2;
    std::int64_t iters = 6;
    std::string out_dir;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

std::int64_t parse_i64(const char *value, const char *label) {
    char *end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (value == nullptr || *value == '\0' || end == nullptr || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
    return static_cast<std::int64_t>(parsed);
}

bench_config parse_args(int argc, char **argv) {
    bench_config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *label) -> const char * {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (arg == "--backend") {
            cfg.backend = require_value("--backend");
        } else if (arg == "--rows") {
            cfg.rows = parse_i64(require_value("--rows"), "--rows");
        } else if (arg == "--queries") {
            cfg.queries = parse_i64(require_value("--queries"), "--queries");
        } else if (arg == "--latent-dim") {
            cfg.latent_dim = parse_i64(require_value("--latent-dim"), "--latent-dim");
        } else if (arg == "--top-k") {
            cfg.top_k = parse_i64(require_value("--top-k"), "--top-k");
        } else if (arg == "--shards") {
            cfg.shard_count = parse_i64(require_value("--shards"), "--shards");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_i64(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_i64(require_value("--iters"), "--iters");
        } else if (arg == "--out-dir") {
            cfg.out_dir = require_value("--out-dir");
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: forwardNeighborsBench [--backend dense|dense-ann|blocked-sparse|blocked-sparse-ann|sliced-sparse|sliced-sparse-ann] "
                << "[--rows N] [--queries N] [--latent-dim N] [--top-k N] [--shards N] "
                << "[--warmup N] [--iters N] [--out-dir DIR]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }
    require(
        cfg.backend == "dense"
        || cfg.backend == "dense-ann"
        || cfg.backend == "blocked-sparse"
        || cfg.backend == "blocked-sparse-ann"
        || cfg.backend == "sliced-sparse"
        || cfg.backend == "sliced-sparse-ann",
        "backend must be dense, dense-ann, blocked-sparse, blocked-sparse-ann, sliced-sparse, or sliced-sparse-ann");
    require(cfg.rows > 0 && cfg.queries > 0 && cfg.latent_dim > 0 && cfg.top_k > 0, "benchmark dimensions must be > 0");
    return cfg;
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start, const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

fn::ForwardNeighborOwnedRecordBatch make_records(const bench_config &cfg) {
    fn::ForwardNeighborOwnedRecordBatch records;
    records.cell_indices.resize(static_cast<std::size_t>(cfg.rows));
    records.developmental_time.resize(static_cast<std::size_t>(cfg.rows));
    records.embryo_ids.assign_fill(static_cast<std::size_t>(cfg.rows), static_cast<std::int64_t>(0));
    records.dense_values.resize(static_cast<std::size_t>(cfg.rows) * static_cast<std::size_t>(cfg.latent_dim));
    records.dense_cols = cfg.latent_dim;

    for (std::int64_t row = 0; row < cfg.rows; ++row) {
        records.cell_indices[static_cast<std::size_t>(row)] = row;
        records.developmental_time[static_cast<std::size_t>(row)] =
            cfg.rows == 1 ? 0.0f : static_cast<float>(row) / static_cast<float>(cfg.rows - 1);
        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.latent_dim);
        for (std::int64_t dim = 0; dim < cfg.latent_dim; ++dim) {
            const float angle = static_cast<float>((row + 1) * (dim + 3)) * 0.00173f;
            const float signal = std::sin(angle) + 0.5f * std::cos(angle * 0.5f);
            const float gate = ((row + dim) % 7) < 3 ? 0.0f : 1.0f;
            records.dense_values[base + static_cast<std::size_t>(dim)] = gate * signal;
        }
    }
    return records;
}

fn::ForwardNeighborOwnedQueryBatch make_queries(
    const fn::ForwardNeighborOwnedRecordBatch &records,
    const bench_config &cfg) {
    fn::ForwardNeighborOwnedQueryBatch query;
    query.cell_indices.resize(static_cast<std::size_t>(cfg.queries));
    query.developmental_time.resize(static_cast<std::size_t>(cfg.queries));
    query.embryo_ids.assign_fill(static_cast<std::size_t>(cfg.queries), static_cast<std::int64_t>(0));
    query.dense_values.resize(static_cast<std::size_t>(cfg.queries) * static_cast<std::size_t>(cfg.latent_dim));
    query.dense_cols = cfg.latent_dim;

    const std::int64_t begin = cfg.rows / 8;
    const std::int64_t span = std::max<std::int64_t>(1, cfg.rows / 16);
    for (std::int64_t row = 0; row < cfg.queries; ++row) {
        const std::int64_t src = begin + (row % span);
        query.cell_indices[static_cast<std::size_t>(row)] = records.cell_indices[static_cast<std::size_t>(src)];
        query.developmental_time[static_cast<std::size_t>(row)] = records.developmental_time[static_cast<std::size_t>(src)];
        std::memcpy(
            query.dense_values.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.latent_dim),
            records.dense_values.data() + static_cast<std::size_t>(src) * static_cast<std::size_t>(cfg.latent_dim),
            static_cast<std::size_t>(cfg.latent_dim) * sizeof(float));
    }
    return query;
}

fn::ForwardNeighborSearchConfig make_search_config(const bench_config &cfg) {
    fn::ForwardNeighborSearchConfig search;
    if (cfg.backend == "dense") {
        search.backend = fn::ForwardNeighborBackend::dense_exact_windowed;
        search.similarity = fn::ForwardNeighborSimilarity::dense_cosine;
    } else if (cfg.backend == "dense-ann") {
        search.backend = fn::ForwardNeighborBackend::dense_ann_windowed;
        search.similarity = fn::ForwardNeighborSimilarity::dense_cosine;
    } else if (cfg.backend == "blocked-sparse") {
        search.backend = fn::ForwardNeighborBackend::sparse_exact_blocked_ell;
        search.similarity = fn::ForwardNeighborSimilarity::sparse_blocked_affinity;
    } else if (cfg.backend == "blocked-sparse-ann") {
        search.backend = fn::ForwardNeighborBackend::sparse_ann_blocked_ell;
        search.similarity = fn::ForwardNeighborSimilarity::sparse_blocked_affinity;
    } else if (cfg.backend == "sliced-sparse") {
        search.backend = fn::ForwardNeighborBackend::sparse_exact_sliced_ell;
        search.similarity = fn::ForwardNeighborSimilarity::sparse_blocked_affinity;
    } else {
        search.backend = fn::ForwardNeighborBackend::sparse_ann_sliced_ell;
        search.similarity = fn::ForwardNeighborSimilarity::sparse_blocked_affinity;
    }
    search.embryo_policy = fn::ForwardNeighborEmbryoPolicy::any_embryo;
    search.top_k = cfg.top_k;
    search.candidate_k = cfg.top_k;
    search.query_block_rows = 512;
    search.index_block_rows = 16384;
    search.ann_probe_list_count = 8;
    search.time_window.min_delta = 0.0f;
    search.time_window.max_delta = 0.125f;
    return search;
}

void write_results(const bench_config &cfg, double build_ms, double search_ms, double checksum) {
    if (cfg.out_dir.empty()) return;
    std::filesystem::create_directories(cfg.out_dir);
    std::ofstream out(cfg.out_dir + "/results.json");
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"backend\": \"" << cfg.backend << "\",\n"
        << "  \"rows\": " << cfg.rows << ",\n"
        << "  \"queries\": " << cfg.queries << ",\n"
        << "  \"latent_dim\": " << cfg.latent_dim << ",\n"
        << "  \"top_k\": " << cfg.top_k << ",\n"
        << "  \"build_ms\": " << build_ms << ",\n"
        << "  \"steady_search_ms\": " << search_ms << ",\n"
        << "  \"checksum\": " << checksum << "\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("forwardNeighborsBench");
    const bench_config cfg = parse_args(argc, argv);

    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "forwardNeighborsBench requires at least one visible CUDA device");

    const auto build_start = std::chrono::steady_clock::now();
    const fn::ForwardNeighborOwnedRecordBatch records = make_records(cfg);
    fn::ForwardNeighborBuildConfig build_config;
    build_config.target_shard_count = cfg.shard_count;
    build_config.max_rows_per_segment = std::max<std::int64_t>(1, cfg.rows / std::max<std::int64_t>(1, cfg.shard_count));
    build_config.ann_rows_per_list = std::max<std::int64_t>(1, build_config.max_rows_per_segment / 8);
    const fn::ForwardNeighborIndex index = fn::build_forward_neighbor_index(records.view(), build_config);
    const fn::ForwardNeighborOwnedQueryBatch query = make_queries(records, cfg);
    const fn::ForwardNeighborSearchConfig search = make_search_config(cfg);
    fn::ForwardNeighborSearchExecutor executor;
    const auto build_stop = std::chrono::steady_clock::now();

    for (std::int64_t iter = 0; iter < cfg.warmup; ++iter) {
        (void) executor.search_future_neighbors(index, query.view(), search);
    }
    cudaDeviceSynchronize();

    fn::ForwardNeighborSearchResult result;
    const auto search_start = std::chrono::steady_clock::now();
    for (std::int64_t iter = 0; iter < cfg.iters; ++iter) {
        result = executor.search_future_neighbors(index, query.view(), search);
    }
    cudaDeviceSynchronize();
    const auto search_stop = std::chrono::steady_clock::now();

    double checksum = 0.0;
    if (!result.neighbor_cell_indices.empty()) checksum += static_cast<double>(result.neighbor_cell_indices[0]);
    if (!result.neighbor_similarity.empty()) checksum += static_cast<double>(result.neighbor_similarity[0]);

    const double build_ms = elapsed_ms(build_start, build_stop);
    const double steady_ms = elapsed_ms(search_start, search_stop) / static_cast<double>(cfg.iters);
    write_results(cfg, build_ms, steady_ms, checksum);

    std::cout << std::fixed << std::setprecision(3)
              << "backend=" << cfg.backend
              << " shards=" << index.shard_count()
              << " build_ms=" << build_ms
              << " steady_ms=" << steady_ms
              << " checksum=" << checksum
              << "\n";
    return 0;
}
