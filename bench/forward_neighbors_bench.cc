#include "../src/compute/neighbors/forward_neighbors/forwardNeighbors.hh"
#include "benchmark_mutex.hh"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fn = ::cellerator::compute::neighbors::forward_neighbors;

namespace {

struct scoped_nvtx_range {
    explicit scoped_nvtx_range(const char *label) { nvtxRangePushA(label); }
    ~scoped_nvtx_range() { nvtxRangePop(); }
};

struct bench_config {
    std::string scenario = "cross-pair";
    std::int64_t rows = 262144;
    std::int64_t queries = 8192;
    std::int64_t latent_dim = 64;
    std::int64_t top_k = 16;
    std::int64_t shard_count = 4;
    std::int64_t query_block_rows = 1024;
    std::int64_t warmup = 2;
    std::int64_t iters = 8;
    bool eager_upload = false;
    std::string out_dir;
};

struct routing_summary {
    std::size_t block_count = 0u;
    double mean_shards_per_block = 0.0;
    double mean_devices_per_block = 0.0;
    std::size_t first_block_shards = 0u;
    std::size_t first_block_devices = 0u;
};

struct scenario_window {
    float query_begin_time = 0.0f;
    float query_end_time = 0.0f;
    float max_delta = 0.0f;
    std::size_t expected_min_routes = 0u;
    std::size_t expected_max_routes = 0u;
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
        if (arg == "--scenario") {
            cfg.scenario = require_value("--scenario");
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
        } else if (arg == "--query-block-rows") {
            cfg.query_block_rows = parse_i64(require_value("--query-block-rows"), "--query-block-rows");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_i64(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_i64(require_value("--iters"), "--iters");
        } else if (arg == "--out-dir") {
            cfg.out_dir = require_value("--out-dir");
        } else if (arg == "--eager-upload") {
            cfg.eager_upload = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: forwardNeighborsBench [--scenario single-shard|one-pair|cross-pair] "
                << "[--rows N] [--queries N] [--latent-dim N] [--top-k N] [--shards N] "
                << "[--query-block-rows N] [--warmup N] [--iters N] [--eager-upload] [--out-dir DIR]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }

    require(cfg.rows > 0, "rows must be > 0");
    require(cfg.queries > 0, "queries must be > 0");
    require(cfg.latent_dim > 1, "latent_dim must be > 1");
    require(cfg.top_k > 0 && cfg.top_k <= 32, "top_k must be in [1, 32]");
    require(cfg.shard_count > 0, "shards must be > 0");
    require(cfg.query_block_rows > 0, "query_block_rows must be > 0");
    require(cfg.warmup >= 0, "warmup must be >= 0");
    require(cfg.iters > 0, "iters must be > 0");
    if (cfg.scenario != "single-shard" && cfg.scenario != "one-pair" && cfg.scenario != "cross-pair") {
        throw std::invalid_argument("scenario must be single-shard, one-pair, or cross-pair");
    }
    return cfg;
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start, const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void sync_all_visible_devices() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) return;
    for (int device = 0; device < device_count; ++device) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
}

int pair_id_for_device(int device_id) {
    if (device_id == 0 || device_id == 2) return 0;
    if (device_id == 1 || device_id == 3) return 1;
    return device_id;
}

std::vector<fn::ForwardNeighborShardSummary> collect_shard_summaries(const fn::ForwardNeighborIndex &index) {
    std::vector<fn::ForwardNeighborShardSummary> shards;
    shards.reserve(index.shard_count());
    for (std::size_t shard = 0; shard < index.shard_count(); ++shard) {
        shards.push_back(index.shard_summary(shard));
    }
    return shards;
}

scenario_window resolve_window(
    const bench_config &cfg,
    const std::vector<fn::ForwardNeighborShardSummary> &shards) {
    require(!shards.empty(), "forward-neighbor benchmark requires at least one realized shard");

    if (cfg.scenario == "single-shard") {
        const auto &shard = shards.front();
        const float width = std::max(1.0e-5f, shard.time_end - shard.time_begin);
        const float query_begin = shard.time_begin + 0.20f * width;
        const float query_end = shard.time_begin + 0.40f * width;
        float upper = shard.time_begin + 0.70f * width;
        if (shards.size() > 1u) upper = std::min(upper, shards[1].time_begin - 1.0e-5f);
        require(upper > query_end, "single-shard benchmark window does not fit inside the realized shard");
        return scenario_window{ query_begin, query_end, upper - query_end, 1u, 1u };
    }

    const bool want_cross_pair = cfg.scenario == "cross-pair";
    for (std::size_t shard = 0; shard + 1u < shards.size(); ++shard) {
        const bool cross_pair = pair_id_for_device(shards[shard].device_id) != pair_id_for_device(shards[shard + 1u].device_id);
        if (cross_pair != want_cross_pair) continue;

        const auto &lhs = shards[shard];
        const auto &rhs = shards[shard + 1u];
        const float lhs_width = std::max(1.0e-5f, lhs.time_end - lhs.time_begin);
        const float rhs_width = std::max(1.0e-5f, rhs.time_end - rhs.time_begin);
        const float query_begin = lhs.time_end - 0.18f * lhs_width;
        const float query_end = lhs.time_end - 0.06f * lhs_width;
        float upper = rhs.time_begin + 0.35f * rhs_width;
        if (shard + 2u < shards.size()) upper = std::min(upper, shards[shard + 2u].time_begin - 1.0e-5f);
        require(upper > query_end, "adjacent-shard benchmark window does not fit inside the realized shard pair");
        return scenario_window{ query_begin, query_end, upper - query_end, 2u, 2u };
    }

    throw std::runtime_error("could not find a realized shard pair matching the requested scenario");
}

fn::ForwardNeighborRecordBatch make_records(const bench_config &cfg) {
    fn::ForwardNeighborRecordBatch batch;
    batch.latent_dim = cfg.latent_dim;
    batch.cell_indices.resize(static_cast<std::size_t>(cfg.rows));
    batch.developmental_time.resize(static_cast<std::size_t>(cfg.rows));
    batch.embryo_ids.assign_fill(static_cast<std::size_t>(cfg.rows), static_cast<std::int64_t>(0));
    batch.latent_unit.resize(static_cast<std::size_t>(cfg.rows) * static_cast<std::size_t>(cfg.latent_dim));

    for (std::int64_t row = 0; row < cfg.rows; ++row) {
        const float t = cfg.rows == 1 ? 0.0f : static_cast<float>(row) / static_cast<float>(cfg.rows - 1);
        const float angle = 6.28318530718f * t;
        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.latent_dim);
        batch.cell_indices[static_cast<std::size_t>(row)] = row;
        batch.developmental_time[static_cast<std::size_t>(row)] = t;
        batch.latent_unit[base] = std::cos(angle);
        batch.latent_unit[base + 1u] = std::sin(angle);
        for (std::int64_t dim = 2; dim < cfg.latent_dim; ++dim) {
            const float harmonic = std::sin((static_cast<float>(row + 1) * static_cast<float>(dim + 3)) * 0.000173f);
            const float trend = (static_cast<float>((row + dim) % 29) / 29.0f) - 0.5f;
            batch.latent_unit[base + static_cast<std::size_t>(dim)] = 0.125f * harmonic + 0.05f * trend;
        }
    }
    return batch;
}

fn::ForwardNeighborQueryBatch make_query_batch(
    const fn::ForwardNeighborRecordBatch &records,
    const bench_config &cfg,
    const scenario_window &window) {
    fn::ForwardNeighborQueryBatch query;
    query.latent_dim = cfg.latent_dim;
    query.cell_indices.resize(static_cast<std::size_t>(cfg.queries));
    query.developmental_time.resize(static_cast<std::size_t>(cfg.queries));
    query.embryo_ids.assign_fill(static_cast<std::size_t>(cfg.queries), static_cast<std::int64_t>(0));
    query.latent_unit.resize(static_cast<std::size_t>(cfg.queries) * static_cast<std::size_t>(cfg.latent_dim));

    const std::int64_t begin_row = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(window.query_begin_time * static_cast<float>(cfg.rows - 1)),
        0,
        cfg.rows - 1);
    const std::int64_t end_row = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(window.query_end_time * static_cast<float>(cfg.rows - 1)),
        begin_row + 1,
        cfg.rows);
    const std::int64_t span = std::max<std::int64_t>(1, end_row - begin_row);

    for (std::int64_t row = 0; row < cfg.queries; ++row) {
        const std::int64_t src = begin_row + (row % span);
        const std::size_t src_base = static_cast<std::size_t>(src) * static_cast<std::size_t>(cfg.latent_dim);
        const std::size_t dst_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.latent_dim);
        query.cell_indices[static_cast<std::size_t>(row)] = records.cell_indices[static_cast<std::size_t>(src)];
        query.developmental_time[static_cast<std::size_t>(row)] = records.developmental_time[static_cast<std::size_t>(src)];
        std::memcpy(
            query.latent_unit.data() + dst_base,
            records.latent_unit.data() + src_base,
            static_cast<std::size_t>(cfg.latent_dim) * sizeof(float));
    }
    return query;
}

routing_summary summarize_routing(const fn::ForwardNeighborRoutingPlan &plan) {
    routing_summary summary;
    summary.block_count = plan.block_query_begin.size();
    if (summary.block_count == 0u || plan.block_route_offsets.size() < summary.block_count + 1u) return summary;

    double total_shards = 0.0;
    double total_devices = 0.0;
    for (std::size_t block = 0; block < summary.block_count; ++block) {
        const std::uint32_t route_begin = plan.block_route_offsets[block];
        const std::uint32_t route_end = plan.block_route_offsets[block + 1u];
        const std::size_t shard_count = static_cast<std::size_t>(route_end - route_begin);
        std::size_t device_count = 0u;
        int seen_devices[4] = {-1, -1, -1, -1};
        for (std::uint32_t route = route_begin; route < route_end; ++route) {
            const int device_id = plan.route_device_ids[route];
            bool seen = false;
            for (int &slot : seen_devices) {
                if (slot == device_id) {
                    seen = true;
                    break;
                }
                if (slot < 0) {
                    slot = device_id;
                    ++device_count;
                    seen = true;
                    break;
                }
            }
            if (!seen) ++device_count;
        }
        if (block == 0u) {
            summary.first_block_shards = shard_count;
            summary.first_block_devices = device_count;
        }
        total_shards += static_cast<double>(shard_count);
        total_devices += static_cast<double>(device_count);
    }
    summary.mean_shards_per_block = total_shards / static_cast<double>(summary.block_count);
    summary.mean_devices_per_block = total_devices / static_cast<double>(summary.block_count);
    return summary;
}

void write_results(
    const bench_config &cfg,
    const routing_summary &routing,
    std::size_t shard_count,
    const char *status,
    double build_ms,
    double route_ms,
    double steady_ms,
    double collect_ms,
    double checksum) {
    if (cfg.out_dir.empty()) return;
    std::filesystem::create_directories(cfg.out_dir);
    {
        std::ofstream config_out(cfg.out_dir + "/run_config.json");
        config_out
            << "{\n"
            << "  \"scenario\": \"" << cfg.scenario << "\",\n"
            << "  \"rows\": " << cfg.rows << ",\n"
            << "  \"queries\": " << cfg.queries << ",\n"
            << "  \"latent_dim\": " << cfg.latent_dim << ",\n"
            << "  \"top_k\": " << cfg.top_k << ",\n"
            << "  \"shard_count_target\": " << cfg.shard_count << ",\n"
            << "  \"query_block_rows\": " << cfg.query_block_rows << ",\n"
            << "  \"warmup\": " << cfg.warmup << ",\n"
            << "  \"repeats\": " << cfg.iters << ",\n"
            << "  \"eager_upload\": " << (cfg.eager_upload ? "true" : "false") << "\n"
            << "}\n";
    }
    {
        std::ofstream results_out(cfg.out_dir + "/results.json");
        results_out << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"status\": \"" << status << "\",\n"
            << "  \"correctness\": \"scenario-routed\",\n"
            << "  \"primary_metric_ms\": " << steady_ms << ",\n"
            << "  \"phases\": {\n"
            << "    \"build_index\": " << build_ms << ",\n"
            << "    \"plan_routes\": " << route_ms << ",\n"
            << "    \"steady_state_search\": " << steady_ms << ",\n"
            << "    \"collect\": " << collect_ms << "\n"
            << "  },\n"
            << "  \"checksum\": " << checksum << ",\n"
            << "  \"routing\": {\n"
            << "    \"realized_shards\": " << shard_count << ",\n"
            << "    \"block_count\": " << routing.block_count << ",\n"
            << "    \"mean_shards_per_block\": " << routing.mean_shards_per_block << ",\n"
            << "    \"mean_devices_per_block\": " << routing.mean_devices_per_block << ",\n"
            << "    \"first_block_shards\": " << routing.first_block_shards << ",\n"
            << "    \"first_block_devices\": " << routing.first_block_devices << "\n"
            << "  }\n"
            << "}\n";
    }
}

} // namespace

int main(int argc, char **argv) {
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("forwardNeighborsBench");
    const bench_config cfg = parse_args(argc, argv);

    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "forwardNeighborsBench requires at least one visible CUDA device");

    const auto build_start = std::chrono::steady_clock::now();
    fn::ForwardNeighborRecordBatch records;
    fn::ForwardNeighborIndex index;
    fn::ForwardNeighborQueryBatch query;
    scenario_window window;
    {
        scoped_nvtx_range range("forward_neighbors_build");
        records = make_records(cfg);

        fn::ForwardNeighborBuildConfig build_config;
        build_config.target_shard_count = cfg.shard_count;
        build_config.max_rows_per_segment = std::max<std::int64_t>(1, cfg.rows / std::max<std::int64_t>(1, cfg.shard_count));
        build_config.ann_rows_per_list = std::max<std::int64_t>(1, build_config.max_rows_per_segment / 8);
        build_config.eager_device_upload = cfg.eager_upload;
        index = fn::build_forward_neighbor_index(records, build_config);
        window = resolve_window(cfg, collect_shard_summaries(index));
        query = make_query_batch(records, cfg, window);
    }
    sync_all_visible_devices();
    const auto build_stop = std::chrono::steady_clock::now();

    fn::ForwardNeighborSearchConfig search_config;
    search_config.backend = fn::ForwardNeighborBackend::exact_windowed;
    search_config.embryo_policy = fn::ForwardNeighborEmbryoPolicy::any_embryo;
    search_config.top_k = cfg.top_k;
    search_config.candidate_k = cfg.top_k;
    search_config.query_block_rows = cfg.query_block_rows;
    search_config.index_block_rows = std::max<std::int64_t>(cfg.rows / std::max<std::int64_t>(1, cfg.shard_count), cfg.query_block_rows);
    search_config.time_window.min_delta = 0.0f;
    search_config.time_window.max_delta = window.max_delta;

    fn::ForwardNeighborExecutorConfig executor_config;
    executor_config.max_resident_shards_per_device = 2;
    fn::ForwardNeighborSearchExecutor executor(executor_config);

    const auto route_start = std::chrono::steady_clock::now();
    fn::ForwardNeighborRoutingPlan plan;
    routing_summary routing;
    {
        scoped_nvtx_range range("forward_neighbors_route_plan");
        plan = executor.plan_future_neighbor_routes(index, query, search_config);
        routing = summarize_routing(plan);
    }
    const auto route_stop = std::chrono::steady_clock::now();

    require(index.shard_count() >= 1u, "forward-neighbor benchmark built zero shards");
    require(routing.block_count >= 1u, "forward-neighbor benchmark produced no query blocks");
    require(routing.first_block_shards >= window.expected_min_routes, "forward-neighbor benchmark routed fewer shards than expected");
    require(routing.first_block_shards <= window.expected_max_routes, "forward-neighbor benchmark routed more shards than expected");

    for (std::int64_t iter = 0; iter < cfg.warmup; ++iter) {
        scoped_nvtx_range range("forward_neighbors_warmup");
        (void) executor.search_future_neighbors(index, query, search_config);
    }
    sync_all_visible_devices();

    fn::ForwardNeighborSearchResult last_result;
    const auto steady_start = std::chrono::steady_clock::now();
    {
        scoped_nvtx_range range("forward_neighbors_search_loop");
        for (std::int64_t iter = 0; iter < cfg.iters; ++iter) {
            last_result = executor.search_future_neighbors(index, query, search_config);
        }
    }
    sync_all_visible_devices();
    const auto steady_stop = std::chrono::steady_clock::now();

    const auto collect_start = std::chrono::steady_clock::now();
    double checksum = 0.0;
    if (!last_result.neighbor_cell_indices.empty()) {
        checksum += static_cast<double>(last_result.neighbor_cell_indices[0]);
    }
    if (!last_result.neighbor_similarity.empty()) {
        checksum += static_cast<double>(last_result.neighbor_similarity[0]);
    }
    checksum += static_cast<double>(routing.first_block_shards * 100 + routing.first_block_devices);
    const auto collect_stop = std::chrono::steady_clock::now();

    const double build_ms = elapsed_ms(build_start, build_stop);
    const double route_ms = elapsed_ms(route_start, route_stop);
    const double steady_ms = elapsed_ms(steady_start, steady_stop) / static_cast<double>(cfg.iters);
    const double collect_ms = elapsed_ms(collect_start, collect_stop);

    write_results(cfg, routing, index.shard_count(), "ok", build_ms, route_ms, steady_ms, collect_ms, checksum);

    std::cout << std::fixed << std::setprecision(3)
              << "scenario=" << cfg.scenario
              << " realized_shards=" << index.shard_count()
              << " first_block_shards=" << routing.first_block_shards
              << " first_block_devices=" << routing.first_block_devices
              << " steady_ms=" << steady_ms
              << " checksum=" << checksum
              << "\n";
    return 0;
}
