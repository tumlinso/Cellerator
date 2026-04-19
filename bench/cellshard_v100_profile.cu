#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "benchmark_mutex.hh"
#include "../extern/CellShard/include/CellShard/CellShard.hh"
#include "../extern/CellShard/src/convert/compressed_from_coo_raw.cuh"
#include <Cellerator/ingest/compressed_parts.cuh>

namespace {

namespace cs = ::cellshard;
namespace csb = ::cellshard::bucket;
namespace csd = ::cellshard::distributed;
namespace csi = ::cellerator::ingest::mtx;

struct config {
    unsigned int build_device = 0;
    unsigned int parts = 32;
    unsigned int rows_per_part = 32768;
    unsigned int cols = 65536;
    unsigned int avg_nnz_per_row = 128;
    unsigned int shards = 8;
    unsigned int build_repeats = 8;
    unsigned int upload_repeats = 4;
    unsigned int bucket_repeats = 8;
    unsigned int bucket_count = 128;
    unsigned int seed = 7;
};

struct scoped_nvtx_range {
    explicit scoped_nvtx_range(const char *label) { nvtxRangePushA(label); }
    ~scoped_nvtx_range() { nvtxRangePop(); }
};

static int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --device N             Device for the single-GPU build benchmark. Default: 0\n"
                 "  --parts N              Number of physical parts. Default: 32\n"
                 "  --rows-per-partition N Rows in each partition. Default: 32768\n"
                 "  --cols N               Matrix columns. Default: 65536\n"
                 "  --avg-nnz-row N        Average nnz/row inside each partition. Default: 128\n"
                 "  --shards N             Logical shard count. Default: 8\n"
                 "  --build-repeats N      Repeats for device-only COO->compressed. Default: 8\n"
                 "  --upload-repeats N     Repeats for multi-GPU hot upload. Default: 4\n"
                 "  --bucket-repeats N     Repeats for resident multi-GPU shard rebuild. Default: 8\n"
                 "  --bucket-count N       Requested row buckets for shard rebuild. Default: 128\n"
                 "  --seed N               RNG seed. Default: 7\n",
                 argv0);
}

static int parse_u32(const char *text, unsigned int *value) {
    char *end = 0;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != 0 || parsed > 0xfffffffful) return 0;
    *value = (unsigned int) parsed;
    return 1;
}

static int parse_args(int argc, char **argv, config *cfg) {
    int i = 1;
    while (i < argc) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->build_device)) return 0;
        } else if (std::strcmp(argv[i], "--parts") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->parts)) return 0;
        } else if (std::strcmp(argv[i], "--rows-per-partition") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->rows_per_part)) return 0;
        } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->cols)) return 0;
        } else if (std::strcmp(argv[i], "--avg-nnz-row") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->avg_nnz_per_row)) return 0;
        } else if (std::strcmp(argv[i], "--shards") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->shards)) return 0;
        } else if (std::strcmp(argv[i], "--build-repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->build_repeats)) return 0;
        } else if (std::strcmp(argv[i], "--upload-repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->upload_repeats)) return 0;
        } else if (std::strcmp(argv[i], "--bucket-repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->bucket_repeats)) return 0;
        } else if (std::strcmp(argv[i], "--bucket-count") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->bucket_count)) return 0;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->seed)) return 0;
        } else {
            return 0;
        }
        ++i;
    }
    return cfg->parts != 0 && cfg->rows_per_part != 0 && cfg->cols != 0 && cfg->shards != 0;
}

static void build_row_counts(std::vector<unsigned int> *counts,
                             unsigned int rows,
                             unsigned int cols,
                             unsigned int avg_nnz_per_row,
                             std::mt19937 *rng) {
    std::vector<float> weights(rows, 0.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    unsigned long long assigned = 0;
    const unsigned long long target_nnz = (unsigned long long) rows * (unsigned long long) avg_nnz_per_row;
    unsigned int i = 0;

    counts->assign(rows, 0u);
    for (i = 0; i < rows; ++i) {
        const float heavy = unit(*rng) < 0.15f ? 6.0f + unit(*rng) * 4.0f : 0.5f + unit(*rng) * 1.5f;
        weights[i] = heavy;
    }

    {
        float weight_sum = 0.0f;
        for (float w : weights) weight_sum += w;
        for (i = 0; i < rows; ++i) {
            unsigned int count = (unsigned int) ((target_nnz * (unsigned long long) (weights[i] * 1024.0f / weight_sum)) >> 10);
            if (count > cols) count = cols;
            (*counts)[i] = count;
            assigned += count;
        }
    }

    while (assigned < target_nnz) {
        const unsigned int row = (*rng)() % rows;
        if ((*counts)[row] < cols) {
            ++(*counts)[row];
            ++assigned;
        }
    }
    while (assigned > target_nnz) {
        const unsigned int row = (*rng)() % rows;
        if ((*counts)[row] != 0) {
            --(*counts)[row];
            --assigned;
        }
    }
}

static cs::sparse::compressed *make_compressed_part(unsigned int rows,
                                                    unsigned int cols,
                                                    unsigned int avg_nnz_per_row,
                                                    unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<unsigned int> row_counts;
    unsigned long long nnz = 0;
    unsigned int row = 0;
    auto *part = new cs::sparse::compressed;

    build_row_counts(&row_counts, rows, cols, avg_nnz_per_row, &rng);
    for (row = 0; row < rows; ++row) nnz += row_counts[row];

    cs::sparse::init(part, rows, cols, (cs::types::nnz_t) nnz, cs::sparse::compressed_by_row);
    if (!cs::sparse::allocate(part)) {
        delete part;
        return 0;
    }

    part->majorPtr[0] = 0;
    for (row = 0; row < rows; ++row) {
        const unsigned int count = row_counts[row];
        const unsigned int begin = part->majorPtr[row];
        const unsigned int max_start = cols > count ? cols - count : 0;
        const unsigned int start = max_start != 0 ? (rng() % (max_start + 1u)) : 0u;
        unsigned int j = 0;

        part->majorPtr[row + 1] = begin + count;
        for (j = 0; j < count; ++j) {
            part->minorIdx[begin + j] = start + j;
            part->val[begin + j] = __float2half((float) (1u + ((row + j) % 11u)));
        }
    }
    return part;
}

static int prepare_unsorted_triplets(csi::compressed_workspace *ws,
                                     unsigned int rows,
                                     unsigned int cols,
                                     unsigned int avg_nnz_per_row,
                                     unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<unsigned int> row_counts;
    unsigned int row = 0;
    unsigned int write_pos = 0;
    unsigned int nnz = 0;

    build_row_counts(&row_counts, rows, cols, avg_nnz_per_row, &rng);
    for (row = 0; row < rows; ++row) nnz += row_counts[row];

    if (!csi::reserve(ws, rows, cols, nnz)) return 0;

    for (row = 0; row < rows; ++row) {
        const unsigned int count = row_counts[row];
        const unsigned int max_start = cols > count ? cols - count : 0;
        const unsigned int start = max_start != 0 ? (rng() % (max_start + 1u)) : 0u;
        unsigned int j = 0;

        for (j = 0; j < count; ++j) {
            ws->h_row_idx[write_pos] = row;
            ws->h_col_idx[write_pos] = start + j;
            ws->h_in_val[write_pos] = __float2half((float) (1u + ((row + j) % 7u)));
            ++write_pos;
        }
    }

    if (nnz > 1) {
        unsigned int i = 0;
        for (i = nnz - 1; i > 0; --i) {
            const unsigned int j = rng() % (i + 1u);
            const unsigned int tmp_row = ws->h_row_idx[i];
            const unsigned int tmp_col = ws->h_col_idx[i];
            const __half tmp_val = ws->h_in_val[i];
            ws->h_row_idx[i] = ws->h_row_idx[j];
            ws->h_col_idx[i] = ws->h_col_idx[j];
            ws->h_in_val[i] = ws->h_in_val[j];
            ws->h_row_idx[j] = tmp_row;
            ws->h_col_idx[j] = tmp_col;
            ws->h_in_val[j] = tmp_val;
        }
    }
    return 1;
}

static std::size_t total_device_stage_bytes(const cs::sharded<cs::sparse::compressed> *view) {
    std::size_t total = 0;
    unsigned long shard = 0;
    for (shard = 0; shard < view->num_shards; ++shard) total += cs::device::device_shard_bytes(view, shard);
    return total;
}

static unsigned long long total_shard_nnz(const cs::sharded<cs::sparse::compressed> *view) {
    unsigned long long total = 0;
    unsigned long shard = 0;
    for (shard = 0; shard < view->num_shards; ++shard) total += (unsigned long long) cs::nnz_in_shard(view, shard);
    return total;
}

static int pin_loaded_parts(cs::sharded<cs::sparse::compressed> *view) {
    unsigned long part = 0;
    for (part = 0; part < view->num_parts; ++part) {
        if (view->parts[part] == 0) return 0;
        if (!cs::sparse::pin(view->parts[part])) return 0;
    }
    return 1;
}

static void unpin_loaded_parts(cs::sharded<cs::sparse::compressed> *view) {
    unsigned long part = 0;
    for (part = 0; part < view->num_parts; ++part) {
        if (view->parts[part] != 0) cs::sparse::unpin(view->parts[part]);
    }
}

static int build_host_matrix(const config &cfg, cs::sharded<cs::sparse::compressed> *built) {
    unsigned int part = 0;

    cs::init(built);
    {
        scoped_nvtx_range range("generate_parts");
        for (part = 0; part < cfg.parts; ++part) {
            cs::sparse::compressed *matrix_part = make_compressed_part(cfg.rows_per_part,
                                                                       cfg.cols,
                                                                       cfg.avg_nnz_per_row,
                                                                       cfg.seed + 17u * part);
            if (matrix_part == 0) return 0;
            if (!cs::append_part(built, matrix_part)) return 0;
        }
    }
    if (!cs::set_equal_shards(built, cfg.shards)) return 0;
    if (!pin_loaded_parts(built)) return 0;
    return 1;
}

static void release_uploaded_fleet(csd::device_fleet<cs::sparse::compressed> *fleet,
                                   const csd::shard_map *map,
                                   const cs::sharded<cs::sparse::compressed> *view) {
    unsigned int slot = 0;
    for (slot = 0; slot < fleet->count; ++slot) {
        unsigned long shard = 0;
        for (shard = 0; shard < view->num_shards; ++shard) {
            if (map->device_slot == 0 || shard >= map->shard_count) continue;
            if ((unsigned int) map->device_slot[shard] != slot) continue;
            (void) cs::device::release_shard(fleet->states + slot, view, shard);
        }
    }
}

static int setup_multi_gpu_runtime(csd::local_context *ctx,
                                   csd::device_fleet<cs::sparse::compressed> *fleet,
                                   csd::shard_map *map,
                                   const cs::sharded<cs::sparse::compressed> *view) {
    csd::init(ctx);
    csd::init(fleet);
    csd::init(map);

    if (!check_cuda(csd::discover_local(ctx, 1, cudaStreamNonBlocking), "discover_local")) return 0;
    if (ctx->device_count == 0) {
        std::fprintf(stderr, "No CUDA devices visible.\n");
        return 0;
    }
    if (!check_cuda(csd::enable_peer_access(ctx), "enable_peer_access")) return 0;
    if (!csd::reserve(fleet, ctx->device_count)) return 0;
    if (!csd::reserve_parts(fleet, view->num_parts)) return 0;
    if (!csd::assign_shards_by_bytes(map, view, ctx)) return 0;
    return 1;
}

static int run_device_triplet_build_benchmark(const config &cfg) {
    csi::compressed_workspace ws;
    const unsigned int rows = cfg.rows_per_part;
    const unsigned int cols = cfg.cols;
    const unsigned int cdim = rows;
    const unsigned int udim = cols;
    const unsigned int nnz = rows * cfg.avg_nnz_per_row;
    std::chrono::steady_clock::time_point t0;
    unsigned int iter = 0;

    csi::init(&ws);
    if (!csi::setup(&ws, (int) cfg.build_device, (cudaStream_t) 0)) return 0;
    if (!prepare_unsorted_triplets(&ws, rows, cols, cfg.avg_nnz_per_row, cfg.seed ^ 0x9e3779b9u)) return 0;
    if (!csi::reserve(&ws, cdim, udim, nnz)) return 0;

    if (nnz != 0) {
        if (!check_cuda(cudaMemcpyAsync(ws.d_row_idx, ws.h_row_idx, (std::size_t) nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, ws.stream), "copy rows")) return 0;
        if (!check_cuda(cudaMemcpyAsync(ws.d_col_idx, ws.h_col_idx, (std::size_t) nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, ws.stream), "copy cols")) return 0;
        if (!check_cuda(cudaMemcpyAsync(ws.d_in_val, ws.h_in_val, (std::size_t) nnz * sizeof(__half), cudaMemcpyHostToDevice, ws.stream), "copy vals")) return 0;
    }
    if (!check_cuda(cudaStreamSynchronize(ws.stream), "sync build input")) return 0;
    if (!cs::convert::build_compressed_from_coo_sorted_raw(cdim,
                                                           udim,
                                                           nnz,
                                                           ws.d_row_idx,
                                                           ws.d_col_idx,
                                                           ws.d_in_val,
                                                           ws.d_major_ptr,
                                                           ws.d_sort_cax,
                                                           ws.d_minor_idx,
                                                           ws.d_out_val,
                                                           ws.d_permutation,
                                                           ws.d_tmp,
                                                           ws.d_sort_bytes,
                                                           ws.stream)) {
        if (!cs::convert::build_compressed_from_coo_raw(cdim,
                                                        nnz,
                                                        ws.d_row_idx,
                                                        ws.d_col_idx,
                                                        ws.d_in_val,
                                                        ws.d_major_ptr,
                                                        ws.d_heads,
                                                        ws.d_minor_idx,
                                                        ws.d_out_val,
                                                        ws.d_tmp,
                                                        ws.d_scan_bytes,
                                                        ws.stream)) return 0;
    }
    if (!check_cuda(cudaStreamSynchronize(ws.stream), "sync build warmup")) return 0;

    t0 = std::chrono::steady_clock::now();
    {
        scoped_nvtx_range range("device_coo_to_compressed_loop");
        for (iter = 0; iter < cfg.build_repeats; ++iter) {
            if (!cs::convert::build_compressed_from_coo_sorted_raw(cdim,
                                                                   udim,
                                                                   nnz,
                                                                   ws.d_row_idx,
                                                                   ws.d_col_idx,
                                                                   ws.d_in_val,
                                                                   ws.d_major_ptr,
                                                                   ws.d_sort_cax,
                                                                   ws.d_minor_idx,
                                                                   ws.d_out_val,
                                                                   ws.d_permutation,
                                                                   ws.d_tmp,
                                                                   ws.d_sort_bytes,
                                                                   ws.stream)) {
                if (!cs::convert::build_compressed_from_coo_raw(cdim,
                                                                nnz,
                                                                ws.d_row_idx,
                                                                ws.d_col_idx,
                                                                ws.d_in_val,
                                                                ws.d_major_ptr,
                                                                ws.d_heads,
                                                                ws.d_minor_idx,
                                                                ws.d_out_val,
                                                                ws.d_tmp,
                                                                ws.d_scan_bytes,
                                                                ws.stream)) {
                    csi::clear(&ws);
                    return 0;
                }
            }
        }
    }
    if (!check_cuda(cudaStreamSynchronize(ws.stream), "sync build loop")) {
        csi::clear(&ws);
        return 0;
    }

    {
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double approx_bytes = (double) cfg.build_repeats * (double) nnz * (double) (sizeof(unsigned int) * 4u + sizeof(__half) * 2u);
        std::printf("device_coo_to_compressed: device=%u repeats=%u rows=%u cols=%u nnz=%u time_ms=%.3f approx_device_gib=%.3f approx_gib_per_s=%.3f\n",
                    cfg.build_device,
                    cfg.build_repeats,
                    rows,
                    cols,
                    nnz,
                    ms,
                    approx_bytes / (1024.0 * 1024.0 * 1024.0),
                    (approx_bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0));
    }

    csi::clear(&ws);
    return 1;
}

static int run_multi_gpu_upload_benchmark(const config &cfg) {
    cs::sharded<cs::sparse::compressed> built;
    csd::local_context ctx;
    csd::device_fleet<cs::sparse::compressed> fleet;
    csd::shard_map map;
    std::size_t staged_bytes = 0;
    unsigned int iter = 0;
    int ok = 0;

    if (!build_host_matrix(cfg, &built)) {
        cs::clear(&built);
        return 0;
    }
    if (!setup_multi_gpu_runtime(&ctx, &fleet, &map, &built)) goto done;

    staged_bytes = total_device_stage_bytes(&built);
    if (!check_cuda(csd::stage_all_shards_on_owners(&fleet, &ctx, &map, &built, 0, 0), "warm stage_all_shards_on_owners")) goto done;
    if (!check_cuda(csd::synchronize(&ctx), "warm synchronize")) goto done;
    release_uploaded_fleet(&fleet, &map, &built);

    {
        const auto t0 = std::chrono::steady_clock::now();
        scoped_nvtx_range range("multi_gpu_hot_upload_loop");
        for (iter = 0; iter < cfg.upload_repeats; ++iter) {
            if (!check_cuda(csd::stage_all_shards_on_owners(&fleet, &ctx, &map, &built, 0, 0), "stage_all_shards_on_owners")) goto done;
            if (!check_cuda(csd::synchronize(&ctx), "synchronize upload")) goto done;
            release_uploaded_fleet(&fleet, &map, &built);
        }

        {
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double gib = (double) cfg.upload_repeats * (double) staged_bytes / (1024.0 * 1024.0 * 1024.0);
            std::printf("multi_gpu_hot_upload: devices=%u repeats=%u parts=%u shards=%lu staged_gib=%.3f time_ms=%.3f staged_gib_per_s=%.3f\n",
                        ctx.device_count,
                        cfg.upload_repeats,
                        cfg.parts,
                        built.num_shards,
                        gib,
                        ms,
                        gib / (ms / 1000.0));
        }
    }

    ok = 1;

done:
    release_uploaded_fleet(&fleet, &map, &built);
    csd::clear(&fleet);
    csd::clear(&map);
    csd::clear(&ctx);
    unpin_loaded_parts(&built);
    cs::clear(&built);
    return ok;
}

static int run_multi_gpu_bucket_benchmark(const config &cfg) {
    cs::sharded<cs::sparse::compressed> built;
    csd::local_context ctx;
    csd::device_fleet<cs::sparse::compressed> fleet;
    csd::shard_map map;
    std::vector<csb::sharded_major_bucket_workspace> workspaces;
    unsigned long long nnz_per_pass = 0;
    int ok = 0;

    if (!build_host_matrix(cfg, &built)) {
        cs::clear(&built);
        return 0;
    }
    if (!setup_multi_gpu_runtime(&ctx, &fleet, &map, &built)) goto done;
    if (!check_cuda(csd::stage_all_shards_on_owners(&fleet, &ctx, &map, &built, 0, 0), "stage owners for bucket benchmark")) goto done;
    if (!check_cuda(csd::synchronize(&ctx), "synchronize staged shards")) goto done;

    workspaces.resize(ctx.device_count);
    for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
        csb::init(&workspaces[slot]);
        if (!csb::setup(&workspaces[slot], ctx.device_ids[slot])) {
            std::fprintf(stderr, "Failed to set up bucket workspace on device %d\n", ctx.device_ids[slot]);
            goto done;
        }
    }

    nnz_per_pass = total_shard_nnz(&built);
    for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
        unsigned long shard = 0;
        csb::sharded_major_bucket_result warm_result;
        for (shard = 0; shard < built.num_shards; ++shard) {
            if ((unsigned int) map.device_slot[shard] != slot) continue;
            if (!csb::build_bucketed_shard_major_view(fleet.states + slot,
                                                     &built,
                                                     shard,
                                                     (cs::types::idx_t) cfg.bucket_count,
                                                     &workspaces[slot],
                                                     &warm_result)) goto done;
        }
        if (!check_cuda(cudaSetDevice(ctx.device_ids[slot]), "set device bucket warmup")) goto done;
        if (!check_cuda(cudaStreamSynchronize(workspaces[slot].base.stream), "sync bucket warmup")) goto done;
    }

    {
        const auto t0 = std::chrono::steady_clock::now();
        scoped_nvtx_range range("multi_gpu_bucket_loop");
        for (unsigned int iter = 0; iter < cfg.bucket_repeats; ++iter) {
            std::atomic<int> first_error(0);
            std::vector<std::thread> workers;
            workers.reserve(ctx.device_count);

            for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
                workers.emplace_back([&, slot]() {
                    unsigned long shard = 0;
                    csb::sharded_major_bucket_result result;

                    for (shard = 0; shard < built.num_shards; ++shard) {
                        if (first_error.load() != 0) return;
                        if ((unsigned int) map.device_slot[shard] != slot) continue;
                        if (!csb::build_bucketed_shard_major_view(fleet.states + slot,
                                                                 &built,
                                                                 shard,
                                                                 (cs::types::idx_t) cfg.bucket_count,
                                                                 &workspaces[slot],
                                                                 &result)) {
                            first_error.store(1);
                            return;
                        }
                    }
                    if (cudaSetDevice(ctx.device_ids[slot]) != cudaSuccess ||
                        cudaStreamSynchronize(workspaces[slot].base.stream) != cudaSuccess) {
                        first_error.store(1);
                    }
                });
            }
            for (std::thread &worker : workers) worker.join();
            if (first_error.load() != 0) goto done;
        }

        {
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double nnz_per_s = ((double) cfg.bucket_repeats * (double) nnz_per_pass) / (ms / 1000.0);
            const double approx_bytes = (double) cfg.bucket_repeats * (double) nnz_per_pass * (double) (sizeof(unsigned int) + sizeof(__half)) * 2.0;
            std::printf("multi_gpu_bucket_rebuild: devices=%u repeats=%u shards=%lu nnz_per_pass=%llu time_ms=%.3f approx_device_gib=%.3f approx_nnz_per_s=%.3f\n",
                        ctx.device_count,
                        cfg.bucket_repeats,
                        built.num_shards,
                        nnz_per_pass,
                        ms,
                        approx_bytes / (1024.0 * 1024.0 * 1024.0),
                        nnz_per_s);
        }
    }

    ok = 1;

done:
    for (csb::sharded_major_bucket_workspace &ws : workspaces) csb::clear(&ws);
    release_uploaded_fleet(&fleet, &map, &built);
    csd::clear(&fleet);
    csd::clear(&map);
    csd::clear(&ctx);
    unpin_loaded_parts(&built);
    cs::clear(&built);
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("cellshardV100Profile");
    config cfg;
    int device_count = 0;

    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 2;
    }

    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) return 1;

    std::printf("config: build_device=%u visible_devices=%d parts=%u rows_per_part=%u cols=%u avg_nnz_row=%u shards=%u build_repeats=%u upload_repeats=%u bucket_repeats=%u bucket_count=%u\n",
                cfg.build_device,
                device_count,
                cfg.parts,
                cfg.rows_per_part,
                cfg.cols,
                cfg.avg_nnz_per_row,
                cfg.shards,
                cfg.build_repeats,
                cfg.upload_repeats,
                cfg.bucket_repeats,
                cfg.bucket_count);

    if (!run_device_triplet_build_benchmark(cfg)) return 1;
    if (!run_multi_gpu_upload_benchmark(cfg)) return 1;
    if (!run_multi_gpu_bucket_benchmark(cfg)) return 1;
    return 0;
}
