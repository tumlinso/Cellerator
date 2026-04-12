#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <atomic>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "benchmark_mutex.hh"
#include "../extern/CellShard/src/CellShard.hh"
#include "../src/compute/preprocess/preprocess.cuh"

namespace {

namespace cs = ::cellshard;
namespace csd = ::cellshard::distributed;
namespace csv = ::cellshard::device;
namespace crp = ::cellerator::compute::preprocess;

struct config {
    unsigned int parts = 32;
    unsigned int rows_per_part = 32768;
    unsigned int cols = 32768;
    unsigned int avg_nnz_per_row = 128;
    unsigned int shards = 8;
    unsigned int repeats = 1;
    unsigned int seed = 7;
    float target_sum = 10000.0f;
    float min_counts = 500.0f;
    unsigned int min_genes = 200;
    float max_mito_fraction = 0.2f;
    float min_gene_sum = 1.0f;
    float min_gene_detected = 5.0f;
    float min_gene_variance = 0.01f;
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
                 "  --parts N                Physical parts. Default: 32\n"
                 "  --rows-per-part N        Rows per part. Default: 32768\n"
                 "  --cols N                 Number of genes. Default: 32768\n"
                 "  --avg-nnz-row N          Average nnz per row. Default: 128\n"
                 "  --shards N               Logical shard count. Default: 8\n"
                 "  --repeats N              Full preprocess repeats. Default: 1\n"
                 "  --seed N                 RNG seed. Default: 7\n"
                 "  --target-sum F           Library size target. Default: 10000\n"
                 "  --min-counts F           Cell filter minimum counts. Default: 500\n"
                 "  --min-genes N            Cell filter minimum genes. Default: 200\n"
                 "  --max-mito-fraction F    Cell filter max mito fraction. Default: 0.2\n"
                 "  --min-gene-sum F         Gene filter minimum sum. Default: 1\n"
                 "  --min-gene-detected F    Gene filter minimum detected cells. Default: 5\n"
                 "  --min-gene-variance F    Gene filter minimum variance. Default: 0.01\n",
                 argv0);
}

static int parse_u32(const char *text, unsigned int *value) {
    char *end = 0;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != 0 || parsed > 0xfffffffful) return 0;
    *value = (unsigned int) parsed;
    return 1;
}

static int parse_f32(const char *text, float *value) {
    char *end = 0;
    float parsed = std::strtof(text, &end);
    if (text == end || *end != 0) return 0;
    *value = parsed;
    return 1;
}

static int parse_args(int argc, char **argv, config *cfg) {
    int i = 1;
    while (i < argc) {
        if (std::strcmp(argv[i], "--parts") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->parts)) return 0;
        } else if (std::strcmp(argv[i], "--rows-per-part") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->rows_per_part)) return 0;
        } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->cols)) return 0;
        } else if (std::strcmp(argv[i], "--avg-nnz-row") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->avg_nnz_per_row)) return 0;
        } else if (std::strcmp(argv[i], "--shards") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->shards)) return 0;
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->repeats)) return 0;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->seed)) return 0;
        } else if (std::strcmp(argv[i], "--target-sum") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->target_sum)) return 0;
        } else if (std::strcmp(argv[i], "--min-counts") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_counts)) return 0;
        } else if (std::strcmp(argv[i], "--min-genes") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->min_genes)) return 0;
        } else if (std::strcmp(argv[i], "--max-mito-fraction") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->max_mito_fraction)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-sum") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_sum)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-detected") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_detected)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-variance") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_variance)) return 0;
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
            part->val[begin + j] = __float2half((float) (1u + ((row + j) % 23u)));
        }
    }
    return part;
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
    for (unsigned long part = 0; part < view->num_parts; ++part) {
        if (view->parts[part] != 0) cs::sparse::unpin(view->parts[part]);
    }
}

static int build_host_matrix(const config &cfg, cs::sharded<cs::sparse::compressed> *built) {
    cs::init(built);
    for (unsigned int part = 0; part < cfg.parts; ++part) {
        cs::sparse::compressed *matrix_part = make_compressed_part(cfg.rows_per_part,
                                                                   cfg.cols,
                                                                   cfg.avg_nnz_per_row,
                                                                   cfg.seed + 17u * part);
        if (matrix_part == 0) return 0;
        if (!cs::append_part(built, matrix_part)) return 0;
    }
    if (!cs::set_equal_shards(built, cfg.shards)) return 0;
    return pin_loaded_parts(built);
}

static void release_uploaded_fleet(csd::device_fleet<cs::sparse::compressed> *fleet,
                                   const csd::shard_map *map,
                                   const cs::sharded<cs::sparse::compressed> *view) {
    for (unsigned int slot = 0; slot < fleet->count; ++slot) {
        for (unsigned long shard = 0; shard < view->num_shards; ++shard) {
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
    if (ctx->device_count == 0) return 0;
    if (!check_cuda(csd::enable_peer_access(ctx), "enable_peer_access")) return 0;
#if CELLSHARD_HAS_NCCL
    if (ctx->comms != 0) {
        if (csd::init_local_nccl(ctx) != ncclSuccess) {
            std::fprintf(stderr, "Warning: NCCL init failed, host reduction fallback will be used.\n");
        }
    }
#endif
    if (!csd::reserve(fleet, ctx->device_count)) return 0;
    if (!csd::reserve_parts(fleet, view->num_parts)) return 0;
    if (!csd::assign_shards_by_bytes(map, view, ctx)) return 0;
    return 1;
}

static int max_part_rows(const cs::sharded<cs::sparse::compressed> *view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view->num_parts; ++part) best = std::max(best, view->part_rows[part]);
    return (int) best;
}

static int max_part_nnz(const cs::sharded<cs::sparse::compressed> *view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view->num_parts; ++part) best = std::max(best, view->part_nnz[part]);
    return (int) best;
}

static int count_owned_shards(const csd::shard_map *map, unsigned int slot) {
    int count = 0;
    for (unsigned long shard = 0; shard < map->shard_count; ++shard) {
        if ((unsigned int) map->device_slot[shard] == slot) ++count;
    }
    return count;
}

static int run_preprocess_benchmark(const config &cfg) {
    cs::sharded<cs::sparse::compressed> built;
    csd::local_context ctx;
    csd::device_fleet<cs::sparse::compressed> fleet;
    csd::shard_map map;
    crp::distributed_workspace workspaces;
    std::vector<unsigned char> gene_flags;
    std::vector<float> host_gene_sum;
    std::vector<unsigned char> host_keep_genes;
    double qc_norm_ms = 0.0;
    double gene_ms = 0.0;
    int kept_genes = 0;
    double gene_sum_checksum = 0.0;
    int ok = 0;

    cs::init(&built);
    csd::init(&ctx);
    csd::init(&fleet);
    csd::init(&map);
    crp::init(&workspaces);

    if (!build_host_matrix(cfg, &built)) goto done;
    if (!setup_multi_gpu_runtime(&ctx, &fleet, &map, &built)) goto done;
    if (!check_cuda(csd::stage_all_shards_on_owners(&fleet, &ctx, &map, &built, 0, 0), "stage_all_shards_on_owners")) goto done;
    if (!check_cuda(csd::synchronize(&ctx), "synchronize staged shards")) goto done;

    if (!crp::setup(&workspaces, &ctx)) goto done;
    if (!crp::reserve(&workspaces, &ctx, cfg.cols, (unsigned int) max_part_rows(&built), (unsigned int) max_part_nnz(&built))) goto done;

    gene_flags.assign(cfg.cols, 0u);
    for (unsigned int gene = 0; gene < cfg.cols; ++gene) {
        if (gene < 64u || (gene % 97u) == 0u) gene_flags[gene] = (unsigned char) crp::gene_flag_mito;
    }
    if (!crp::upload_gene_flags(&workspaces, cfg.cols, gene_flags.data())) goto done;
    if (!crp::synchronize(&workspaces)) goto done;

    {
        const crp::cell_filter_params cell_filter = {
            cfg.min_counts,
            cfg.min_genes,
            cfg.max_mito_fraction
        };
        const crp::gene_filter_params gene_filter = {
            cfg.min_gene_sum,
            cfg.min_gene_detected,
            cfg.min_gene_variance
        };

        for (unsigned int iter = 0; iter < cfg.repeats; ++iter) {
            auto t_qc0 = std::chrono::steady_clock::now();
            std::atomic<int> worker_failed(0);
            if (!crp::zero_gene_metrics(&workspaces, cfg.cols)) goto done;

            {
                scoped_nvtx_range range("scrna_qc_normalize_loop");
                std::vector<std::thread> workers;
                workers.reserve(ctx.device_count);
                for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
                    workers.emplace_back([&, slot]() {
                        crp::device_workspace *ws = workspaces.devices + slot;
                        if (worker_failed.load(std::memory_order_relaxed) != 0) return;
                        if (cudaSetDevice(ctx.device_ids[slot]) != cudaSuccess) {
                            worker_failed.store(1, std::memory_order_relaxed);
                            return;
                        }
                        for (unsigned long shard = 0; shard < built.num_shards; ++shard) {
                            if ((unsigned int) map.device_slot[shard] != slot) continue;
                            const unsigned long begin = cs::first_part_in_shard(&built, shard);
                            const unsigned long end = cs::last_part_in_shard(&built, shard);
                            for (unsigned long part = begin; part < end; ++part) {
                                csv::compressed_view part_view;
                                if (worker_failed.load(std::memory_order_relaxed) != 0) return;
                                if (!crp::bind_uploaded_part_view(&part_view, &built, fleet.states[slot].parts + part, part)) {
                                    worker_failed.store(1, std::memory_order_relaxed);
                                    return;
                                }
                                if (!crp::preprocess_part_inplace(&part_view, ws, cell_filter, cfg.target_sum, 0)) {
                                    worker_failed.store(1, std::memory_order_relaxed);
                                    return;
                                }
                            }
                        }
                        (void) cudaSetDevice(ctx.device_ids[slot]);
                        (void) cudaStreamSynchronize(ws->stream);
                    });
                }
                for (std::thread &worker : workers) worker.join();
                if (worker_failed.load(std::memory_order_relaxed) != 0) goto done;
            }

            {
                auto t_qc1 = std::chrono::steady_clock::now();
                qc_norm_ms += std::chrono::duration<double, std::milli>(t_qc1 - t_qc0).count();
            }

            {
                auto t_gene0 = std::chrono::steady_clock::now();
                scoped_nvtx_range range("scrna_gene_reduce_loop");
                if (!crp::allreduce_gene_metrics(&workspaces, &ctx, cfg.cols)) goto done;
                for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
                    if (count_owned_shards(&map, slot) == 0) continue;
                    if (!crp::build_gene_filter_mask(workspaces.devices + slot,
                                                     cfg.cols,
                                                     gene_filter,
                                                     0)) goto done;
                }
                if (!crp::synchronize(&workspaces)) goto done;
                {
                    auto t_gene1 = std::chrono::steady_clock::now();
                    gene_ms += std::chrono::duration<double, std::milli>(t_gene1 - t_gene0).count();
                }
            }
        }
    }

    host_gene_sum.resize(cfg.cols);
    host_keep_genes.resize(cfg.cols);
    {
        unsigned int summary_slot = 0;
        for (; summary_slot < ctx.device_count; ++summary_slot) {
            if (count_owned_shards(&map, summary_slot) != 0) break;
        }
        if (summary_slot >= ctx.device_count) goto done;
        if (!check_cuda(cudaSetDevice(ctx.device_ids[summary_slot]), "cudaSetDevice summary")) goto done;
        if (!check_cuda(cudaMemcpy(host_gene_sum.data(),
                                   workspaces.devices[summary_slot].d_gene_sum,
                                   (std::size_t) cfg.cols * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy host gene sum")) goto done;
        if (!check_cuda(cudaMemcpy(host_keep_genes.data(),
                                   workspaces.devices[summary_slot].d_keep_genes,
                                   (std::size_t) cfg.cols * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy host keep genes")) goto done;
    }

    kept_genes = 0;
    gene_sum_checksum = 0.0;
    for (unsigned int gene = 0; gene < cfg.cols; ++gene) {
        kept_genes += host_keep_genes[gene] != 0;
        gene_sum_checksum += (double) host_gene_sum[gene];
    }

    std::printf("scrna_preprocess: devices=%u parts=%u rows=%lu cols=%u nnz=%lu repeats=%u qc_norm_ms=%.3f gene_ms=%.3f kept_genes=%d gene_sum_checksum=%.6f\n",
                ctx.device_count,
                cfg.parts,
                built.rows,
                cfg.cols,
                built.nnz,
                cfg.repeats,
                qc_norm_ms / (double) cfg.repeats,
                gene_ms / (double) cfg.repeats,
                kept_genes,
                gene_sum_checksum);

    ok = 1;

done:
    crp::clear(&workspaces);
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
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("scrnaPreprocessBench");
    config cfg;

    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 2;
    }

    std::printf("config: parts=%u rows_per_part=%u cols=%u avg_nnz_row=%u shards=%u repeats=%u target_sum=%.1f min_counts=%.1f min_genes=%u max_mito_fraction=%.3f\n",
                cfg.parts,
                cfg.rows_per_part,
                cfg.cols,
                cfg.avg_nnz_per_row,
                cfg.shards,
                cfg.repeats,
                cfg.target_sum,
                cfg.min_counts,
                cfg.min_genes,
                cfg.max_mito_fraction);

    return run_preprocess_benchmark(cfg) ? 0 : 1;
}
