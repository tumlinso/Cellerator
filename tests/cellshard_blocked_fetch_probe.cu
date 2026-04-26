#include "../extern/CellShard/include/CellShard/CellShard.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace cs = ::cellshard;

namespace {

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s --dataset PATH --cache-root PATH [--partition N] [--prefetch-only]\n",
                 argv0);
}

unsigned long parse_ul(const char *text, const char *flag) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (text == nullptr || *text == '\0' || end == nullptr || *end != '\0') {
        std::fprintf(stderr, "invalid integer for %s: %s\n", flag, text != nullptr ? text : "(null)");
        std::exit(2);
    }
    return value;
}

unsigned long find_partition_shard(const cs::sharded<cs::sparse::blocked_ell> *m,
                                   unsigned long partition_id) {
    unsigned long shard_id = 0ul;
    if (m == nullptr) return std::numeric_limits<unsigned long>::max();
    for (shard_id = 0ul; shard_id < m->num_shards; ++shard_id) {
        if (partition_id >= cs::first_partition_in_shard(m, shard_id)
            && partition_id < cs::last_partition_in_shard(m, shard_id)) {
            return shard_id;
        }
    }
    return std::numeric_limits<unsigned long>::max();
}

std::uint64_t count_bucketed_partition_actual_nnz(const cs::bucketed_blocked_ell_partition *part) {
    std::uint64_t actual_nnz = 0u;
    if (part == nullptr) return 0u;
    for (std::uint32_t segment = 0u; segment < part->segment_count; ++segment) {
        const cs::sparse::blocked_ell *seg = part->segments + segment;
        const std::uint32_t block_size = seg->block_size;
        const std::uint32_t width_blocks = cs::sparse::ell_width_blocks(seg);
        for (std::uint32_t row = 0u; row < seg->rows; ++row) {
            const std::uint32_t row_block = row / block_size;
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const cs::types::idx_t block_col = seg->blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == cs::sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                    const ::real::storage_t value =
                        seg->val[(std::size_t) row * seg->ell_cols + (std::size_t) slot * block_size + col_in_block];
                    if (__half2float(value) != 0.0f) ++actual_nnz;
                }
            }
        }
    }
    return actual_nnz;
}

} // namespace

int main(int argc, char **argv) {
    const char *dataset = nullptr;
    const char *cache_root = nullptr;
    unsigned long partition_id = 0ul;
    int prefetch_only = 0;
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    unsigned long shard_id = std::numeric_limits<unsigned long>::max();
    cs::bucketed_blocked_ell_partition exec_part;
    int ok = 0;

    cs::init(&matrix);
    cs::init(&storage);
    cs::init(&exec_part);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            dataset = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-root") == 0 && i + 1 < argc) {
            cache_root = argv[++i];
        } else if (std::strcmp(argv[i], "--partition") == 0 && i + 1 < argc) {
            partition_id = parse_ul(argv[++i], "--partition");
        } else if (std::strcmp(argv[i], "--prefetch-only") == 0) {
            prefetch_only = 1;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (dataset == nullptr || cache_root == nullptr) {
        usage(argv[0]);
        return 2;
    }

    if (!cs::load_header(dataset, &matrix, &storage)) {
        std::fprintf(stderr, "probe: load_header failed for %s\n", dataset);
        goto done;
    }
    if (partition_id >= matrix.num_partitions) {
        std::fprintf(stderr,
                     "probe: partition %lu out of range for %lu partitions\n",
                     partition_id,
                     matrix.num_partitions);
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, cache_root)) {
        std::fprintf(stderr, "probe: bind_dataset_h5_cache failed for %s\n", cache_root);
        goto done;
    }
    shard_id = find_partition_shard(&matrix, partition_id);
    if (shard_id == std::numeric_limits<unsigned long>::max()) {
        std::fprintf(stderr, "probe: failed to locate shard for partition %lu\n", partition_id);
        goto done;
    }

    std::fprintf(stderr,
                 "probe: rows=%lu cols=%lu partitions=%lu shards=%lu partition=%lu shard=%lu\n",
                 (unsigned long) matrix.rows,
                 (unsigned long) matrix.cols,
                 matrix.num_partitions,
                 matrix.num_shards,
                 partition_id,
                 shard_id);
    if (cs::fetch_dataset_blocked_ell_h5_pack_partition(&exec_part, &matrix, &storage, partition_id)) {
        std::fprintf(stderr,
                     "probe: pack partition %lu nnz=%llu actual_nnz=%llu segments=%u rows=%u cols=%u\n",
                     partition_id,
                     (unsigned long long) exec_part.nnz,
                     (unsigned long long) count_bucketed_partition_actual_nnz(&exec_part),
                     exec_part.segment_count,
                     exec_part.rows,
                     exec_part.cols);
    } else {
        std::fprintf(stderr, "probe: pack partition %lu fetch failed\n", partition_id);
    }
    if (!cs::prefetch_dataset_blocked_ell_h5_shard_cache(&matrix, &storage, shard_id)) {
        std::fprintf(stderr, "probe: prefetch shard %lu failed\n", shard_id);
        goto done;
    }
    std::fprintf(stderr, "probe: prefetch shard %lu ok\n", shard_id);

    if (prefetch_only) {
        ok = 1;
        goto done;
    }

    if (!cs::fetch_dataset_blocked_ell_h5_partition(&matrix, &storage, partition_id)) {
        std::fprintf(stderr, "probe: fetch partition %lu failed\n", partition_id);
        goto done;
    }
    if (matrix.parts[partition_id] == nullptr) {
        std::fprintf(stderr, "probe: fetch partition %lu returned null part\n", partition_id);
        goto done;
    }
    std::fprintf(stderr,
                 "probe: fetch partition %lu ok rows=%u cols=%u nnz=%llu block=%u ell=%u\n",
                 partition_id,
                 matrix.parts[partition_id]->rows,
                 matrix.parts[partition_id]->cols,
                 (unsigned long long) matrix.parts[partition_id]->nnz,
                 matrix.parts[partition_id]->block_size,
                 matrix.parts[partition_id]->ell_cols);
    ok = 1;

done:
    cs::clear(&exec_part);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok ? 0 : 1;
}
