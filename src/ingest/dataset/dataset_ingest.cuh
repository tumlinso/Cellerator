#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "dataset_manifest.cuh"
#include "dataset_partition.cuh"
#include "../common/barcode_table.cuh"
#include "../common/feature_table.cuh"
#include "../common/metadata_table.cuh"
#include "../h5ad/h5ad_reader.cuh"
#include "../mtx/mtx_reader.cuh"
#include "../mtx/compressed_parts.cuh"
#include "../../../extern/CellShard/src/convert/blocked_ell_from_compressed.cuh"
#include "../../../extern/CellShard/src/bucket/operators/blocked_ell_bucket.cuh"
#include "../../../extern/CellShard/src/disk/csh5.cuh"

namespace cellerator {
namespace ingest {
namespace dataset {

using ::cellshard::clear;
using ::cellshard::find_offset_span;
using ::cellshard::init;
using ::cellshard::sharded;
namespace sparse = ::cellshard::sparse;

// Conversion knobs for MTX-dataset ingest.
struct mtx_convert_options {
    unsigned long max_part_nnz;
    unsigned long convert_window_bytes;
    unsigned long target_shard_bytes;
    std::size_t reader_bytes;
    std::string cache_root;
};

// Metadata-only init with throughput-oriented defaults.
static inline void init(mtx_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->convert_window_bytes = 1ul << 30ul;
    opts->target_shard_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->cache_root.clear();
}

// Small manifest access helpers.
static inline const char *dataset_id_at(const manifest *m, unsigned int idx) {
    return common::get(&m->dataset_ids, idx);
}

static inline const char *matrix_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->matrix_paths, idx);
}

static inline unsigned int format_at(const manifest *m, unsigned int idx) {
    return idx < m->count ? m->formats[idx] : source_unknown;
}

static inline const char *feature_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->feature_paths, idx);
}

static inline const char *barcode_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->barcode_paths, idx);
}

static inline const char *metadata_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->metadata_paths, idx);
}

static inline const char *matrix_source_at(const manifest *m, unsigned int idx) {
    return common::get(&m->matrix_sources, idx);
}

static inline int allow_processed_at(const manifest *m, unsigned int idx) {
    return idx < m->count && m->allow_processed != 0 && m->allow_processed[idx] != 0u;
}

static inline int build_dataset_h5_output_path(const char *out_dir,
                                              char *out_path,
                                              std::size_t out_cap) {
    if (std::snprintf(out_path, out_cap, "%s/dataset.csh5", out_dir) <= 0) return 0;
    return 1;
}

struct dataset_h5_convert_options {
    unsigned long max_part_nnz;
    unsigned long convert_window_bytes;
    unsigned long target_shard_bytes;
    std::size_t reader_bytes;
    std::string cache_root;
    int device;
    cudaStream_t stream;
};

static inline void init(dataset_h5_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->convert_window_bytes = 1ul << 30ul;
    opts->target_shard_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->cache_root.clear();
    opts->device = 0;
    opts->stream = (cudaStream_t) 0;
}

struct dataset_dataset_plan {
    unsigned int manifest_idx;
    unsigned int dataset_idx;
    mtx::header header;
    std::vector<unsigned long> row_offsets;
    std::vector<unsigned long> part_rows;
    std::vector<unsigned long> part_nnz;
    std::vector<unsigned long> part_bytes;
    std::vector<unsigned long> part_aux;
    std::vector<std::string> spool_paths;
    std::vector<std::uint32_t> feature_to_global;
    unsigned long global_row_begin;
    unsigned long global_part_begin;
};

static inline std::size_t standard_csr_bytes(unsigned long rows, unsigned long nnz) {
    return (std::size_t) (rows + 1ul) * sizeof(cellshard::types::ptr_t)
        + (std::size_t) nnz * sizeof(cellshard::types::idx_t)
        + (std::size_t) nnz * sizeof(::real::storage_t);
}

static inline int convert_coo_part_to_blocked_ell_auto(const sparse::coo *src,
                                                       std::uint32_t global_cols,
                                                       const std::uint32_t *feature_to_global,
                                                       sparse::blocked_ell *dst,
                                                       cellshard::convert::blocked_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 8u, 16u, 32u };
    sparse::clear(dst);
    sparse::init(dst);
    return cellshard::convert::blocked_ell_from_coo_auto(src,
                                                         global_cols,
                                                         feature_to_global,
                                                         candidates,
                                                         sizeof(candidates) / sizeof(candidates[0]),
                                                         dst,
                                                         tune);
}

static inline cellshard::dataset_text_column_view as_text_view(const common::text_column *c) {
    cellshard::dataset_text_column_view view;
    view.count = c != 0 ? c->count : 0u;
    view.bytes = c != 0 ? c->bytes : 0u;
    view.offsets = c != 0 ? c->offsets : 0;
    view.data = c != 0 ? c->data : 0;
    return view;
}

static inline int scan_source_row_nnz(const manifest *m,
                                      unsigned int idx,
                                      mtx::header *header,
                                      unsigned long **row_nnz_out,
                                      std::size_t reader_bytes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return mtx::scan_row_nnz(matrix_path_at(m, idx), header, row_nnz_out, reader_bytes);
    }
    if (format == source_h5ad) {
        h5ad::selected_matrix_info info;
        if (!h5ad::probe_selected_matrix(matrix_path_at(m, idx), matrix_source_at(m, idx), &info, &error)) return 0;
        if (info.processed_like && !allow_processed_at(m, idx)) return 0;
        return h5ad::scan_row_nnz(matrix_path_at(m, idx), matrix_source_at(m, idx), header, row_nnz_out, &error);
    }
    return 0;
}

static inline int load_source_barcodes(const manifest *m,
                                       unsigned int idx,
                                       common::barcode_table *barcodes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return barcode_path_at(m, idx) != 0 && *barcode_path_at(m, idx) != 0
            && common::load_lines(barcode_path_at(m, idx), barcodes);
    }
    if (format == source_h5ad) {
        return h5ad::load_barcodes(matrix_path_at(m, idx), barcodes, &error);
    }
    return 0;
}

static inline int load_source_features(const manifest *m,
                                       unsigned int idx,
                                       common::feature_table *features) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return feature_path_at(m, idx) != 0 && *feature_path_at(m, idx) != 0
            && common::load_tsv(feature_path_at(m, idx), features, 0);
    }
    if (format == source_h5ad) {
        return h5ad::load_feature_table(matrix_path_at(m, idx), matrix_source_at(m, idx), features, &error);
    }
    return 0;
}

static inline int load_source_part_window_coo(const manifest *m,
                                              unsigned int idx,
                                              const mtx::header *header,
                                              const unsigned long *row_offsets,
                                              const unsigned long *part_nnz,
                                              unsigned long num_parts,
                                              unsigned long part_begin,
                                              unsigned long part_end,
                                              sharded<sparse::coo> *out,
                                              std::size_t reader_bytes) {
    std::string error;
    const unsigned int format = format_at(m, idx);
    if (format == source_mtx || format == source_tenx_mtx) {
        return mtx::load_part_window_coo(matrix_path_at(m, idx),
                                         header,
                                         row_offsets,
                                         part_nnz,
                                         num_parts,
                                         part_begin,
                                         part_end,
                                         out,
                                         reader_bytes);
    }
    if (format == source_h5ad) {
        return h5ad::load_part_window_coo(matrix_path_at(m, idx),
                                          matrix_source_at(m, idx),
                                          header,
                                          row_offsets,
                                          part_nnz,
                                          num_parts,
                                          part_begin,
                                          part_end,
                                          out,
                                          &error);
    }
    return 0;
}

static inline int blocked_ell_to_canonical_coo(const sparse::blocked_ell *part,
                                               sparse::coo *out) {
    const std::uint32_t block_size = part != nullptr ? part->block_size : 0u;
    const std::uint32_t width_blocks = part != nullptr ? cellshard::sparse::ell_width_blocks(part) : 0u;
    std::size_t emitted = 0u;
    if (part == nullptr || out == nullptr) return 0;
    sparse::clear(out);
    sparse::init(out, part->rows, part->cols, part->nnz);
    if (!sparse::allocate(out)) return 0;
    for (std::uint32_t row = 0u; row < part->rows; ++row) {
        const std::uint32_t row_block = row / block_size;
        for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
            const cellshard::types::idx_t block_col = part->blockColIdx[(std::size_t) row_block * width_blocks + slot];
            if (block_col == cellshard::sparse::blocked_ell_invalid_col) continue;
            for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                const std::uint32_t col = (std::uint32_t) block_col * block_size + col_in_block;
                const ::real::storage_t value =
                    part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block];
                if (__half2float(value) == 0.0f || col >= part->cols) continue;
                if (emitted >= out->nnz) {
                    sparse::clear(out);
                    return 0;
                }
                out->rowIdx[emitted] = row;
                out->colIdx[emitted] = col;
                out->val[emitted] = value;
                ++emitted;
            }
        }
    }
    if (emitted != out->nnz) {
        sparse::clear(out);
        return 0;
    }
    return 1;
}

struct shard_column_signature {
    std::uint32_t canonical_col;
    std::uint32_t support;
    std::uint64_t hash_a;
    std::uint64_t hash_b;
    std::uint32_t min_row_block;
};

static inline std::uint64_t mix_signature(std::uint64_t seed, std::uint64_t value) {
    const unsigned char *ptr = reinterpret_cast<const unsigned char *>(&value);
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        seed ^= (std::uint64_t) ptr[i];
        seed *= 1099511628211ull;
    }
    return seed;
}

static inline int build_shard_column_maps(const std::vector<sparse::blocked_ell> &parts,
                                          std::uint32_t cols,
                                          std::vector<std::uint32_t> *exec_to_canonical,
                                          std::vector<std::uint32_t> *canonical_to_exec) {
    std::vector<shard_column_signature> signatures;
    std::uint32_t global_row_block = 0u;
    if (exec_to_canonical == nullptr || canonical_to_exec == nullptr) return 0;
    exec_to_canonical->clear();
    canonical_to_exec->clear();
    exec_to_canonical->resize(cols, 0u);
    canonical_to_exec->resize(cols, 0u);
    signatures.resize((std::size_t) cols);
    for (std::uint32_t col = 0u; col < cols; ++col) {
        signatures[(std::size_t) col].canonical_col = col;
        signatures[(std::size_t) col].support = 0u;
        signatures[(std::size_t) col].hash_a = 1469598103934665603ull;
        signatures[(std::size_t) col].hash_b = 1099511628211ull;
        signatures[(std::size_t) col].min_row_block = std::numeric_limits<std::uint32_t>::max();
    }
    for (const sparse::blocked_ell &part : parts) {
        const std::uint32_t row_block_count = cellshard::sparse::row_block_count(&part);
        const std::uint32_t width_blocks = cellshard::sparse::ell_width_blocks(&part);
        for (std::uint32_t row_block = 0u; row_block < row_block_count; ++row_block, ++global_row_block) {
            const std::uint32_t rows_in_block = std::min<std::uint32_t>(part.block_size, part.rows - row_block * part.block_size);
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const cellshard::types::idx_t block_col = part.blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == cellshard::sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < part.block_size; ++col_in_block) {
                    const std::uint32_t col = (std::uint32_t) block_col * part.block_size + col_in_block;
                    bool seen = false;
                    if (col >= cols) continue;
                    for (std::uint32_t row_in_block = 0u; row_in_block < rows_in_block; ++row_in_block) {
                        const std::size_t offset =
                            (std::size_t) (row_block * part.block_size + row_in_block) * part.ell_cols
                            + (std::size_t) slot * part.block_size + col_in_block;
                        if (__half2float(part.val[offset]) != 0.0f) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) continue;
                    signatures[(std::size_t) col].support += 1u;
                    signatures[(std::size_t) col].hash_a =
                        mix_signature(signatures[(std::size_t) col].hash_a, (std::uint64_t) global_row_block + 1u);
                    signatures[(std::size_t) col].hash_b =
                        mix_signature(signatures[(std::size_t) col].hash_b, ((std::uint64_t) global_row_block + 1u) * 1315423911ull);
                    signatures[(std::size_t) col].min_row_block =
                        std::min(signatures[(std::size_t) col].min_row_block, global_row_block);
                }
            }
        }
    }
    std::stable_sort(signatures.begin(),
                     signatures.end(),
                     [](const shard_column_signature &lhs, const shard_column_signature &rhs) {
                         if (lhs.support == 0u || rhs.support == 0u) {
                             if (lhs.support != rhs.support) return lhs.support > rhs.support;
                         }
                         if (lhs.min_row_block != rhs.min_row_block) return lhs.min_row_block < rhs.min_row_block;
                         if (lhs.hash_a != rhs.hash_a) return lhs.hash_a < rhs.hash_a;
                         if (lhs.hash_b != rhs.hash_b) return lhs.hash_b < rhs.hash_b;
                         if (lhs.support != rhs.support) return lhs.support > rhs.support;
                         return lhs.canonical_col < rhs.canonical_col;
                     });
    for (std::uint32_t exec_col = 0u; exec_col < cols; ++exec_col) {
        const std::uint32_t canonical_col = signatures[(std::size_t) exec_col].canonical_col;
        (*exec_to_canonical)[exec_col] = canonical_col;
        (*canonical_to_exec)[canonical_col] = exec_col;
    }
    return 1;
}

static inline int choose_bucket_count_for_part_host_exact(const sparse::blocked_ell *part,
                                                          std::uint32_t *bucket_count_out) {
    const std::uint32_t row_blocks = part != nullptr ? cellshard::sparse::row_block_count(part) : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, row_blocks);
    cellshard::bucketed_blocked_ell_partition trial;
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();
    if (part == nullptr || bucket_count_out == nullptr) return 0;
    cellshard::init(&trial);
    for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
        std::uint64_t bytes = 0u;
        cellshard::clear(&trial);
        cellshard::init(&trial);
        if (!cellshard::build_bucketed_blocked_ell_partition(&trial, part, buckets, &bytes)) {
            cellshard::clear(&trial);
            return 0;
        }
        if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
            best_bytes = bytes;
            best_buckets = buckets;
        }
    }
    cellshard::clear(&trial);
    *bucket_count_out = best_buckets;
    return 1;
}

static inline std::uint64_t estimate_bucketed_bytes_from_gpu_plan(const sparse::blocked_ell *part,
                                                                  const cellshard::bucket::blocked_ell_bucket_plan &plan,
                                                                  std::uint32_t bucket_count) {
    const std::uint32_t row_block_count = (std::uint32_t) plan.row_block_order.size();
    std::uint64_t bytes = 0u;
    if (part == nullptr || row_block_count == 0u || bucket_count == 0u) return 0u;
    for (std::uint32_t bucket = 0u; bucket < bucket_count; ++bucket) {
        const std::uint32_t rb_begin = (bucket * row_block_count) / bucket_count;
        const std::uint32_t rb_end = ((bucket + 1u) * row_block_count) / bucket_count;
        std::uint32_t seg_rows = 0u;
        std::uint32_t seg_width = 0u;
        if (rb_end <= rb_begin) continue;
        seg_width = plan.row_block_width_sorted[(std::size_t) rb_end - 1u];
        for (std::uint32_t pos = rb_begin; pos < rb_end; ++pos) {
            const std::uint32_t rb = plan.row_block_order[pos];
            const std::uint32_t row_begin = rb * part->block_size;
            seg_rows += row_begin < part->rows ? std::min<std::uint32_t>(part->block_size, part->rows - row_begin) : 0u;
        }
        bytes += (std::uint64_t) (rb_end - rb_begin) * seg_width * sizeof(cellshard::types::idx_t);
        bytes += (std::uint64_t) seg_rows * seg_width * part->block_size * sizeof(::real::storage_t);
    }
    bytes += (std::uint64_t) part->rows * sizeof(std::uint32_t) * 2u;
    bytes += (std::uint64_t) (bucket_count + 1u) * sizeof(std::uint32_t);
    return bytes;
}

static inline int choose_bucket_count_for_part(const sparse::blocked_ell *part,
                                               int device,
                                               std::uint32_t *bucket_count_out) {
    static thread_local cellshard::bucket::blocked_ell_bucket_workspace ws;
    static thread_local int ws_ready = 0;
    cellshard::bucket::blocked_ell_bucket_plan plan;
    const std::uint32_t row_blocks = part != nullptr ? cellshard::sparse::row_block_count(part) : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, row_blocks);
    const std::uint64_t original_bytes =
        (std::uint64_t) cellshard::packed_bytes((const sparse::blocked_ell *) nullptr,
                                                part != nullptr ? part->rows : 0u,
                                                part != nullptr ? part->cols : 0u,
                                                part != nullptr ? part->nnz : 0u,
                                                part != nullptr ? (unsigned long) cellshard::partition_aux(part) : 0ul,
                                                sizeof(::real::storage_t));
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = original_bytes;

    if (part == nullptr || bucket_count_out == nullptr) return 0;
    if (row_blocks <= 1u) {
        *bucket_count_out = 1u;
        return 1;
    }
    if (!ws_ready) {
        ws_ready = cellshard::bucket::setup(&ws, device) ? 1 : 0;
    }
    if (ws_ready && cellshard::bucket::build_plan(part, &ws, &plan)) {
        for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
            const std::uint64_t candidate = estimate_bucketed_bytes_from_gpu_plan(part, plan, buckets);
            if (candidate < best_bytes || (candidate == best_bytes && buckets < best_buckets)) {
                best_bytes = candidate;
                best_buckets = buckets;
            }
        }
        *bucket_count_out = best_buckets;
        return 1;
    }
    return choose_bucket_count_for_part_host_exact(part, bucket_count_out);
}

static inline int build_bucketed_optimized_shard(const std::vector<sparse::blocked_ell> &parts,
                                                 std::uint32_t cols,
                                                 int device,
                                                 cellshard::bucketed_blocked_ell_shard *out) {
    std::vector<std::uint32_t> exec_to_canonical_cols;
    std::vector<std::uint32_t> canonical_to_exec_cols;
    std::uint32_t local_rows = 0u;
    std::uint32_t local_nnz = 0u;
    sparse::coo canonical_coo;
    sparse::blocked_ell permuted;

    if (out == nullptr) return 0;
    cellshard::clear(out);
    cellshard::init(out);
    if (!build_shard_column_maps(parts, cols, &exec_to_canonical_cols, &canonical_to_exec_cols)) return 0;

    out->rows = 0u;
    out->cols = cols;
    out->nnz = 0u;
    out->partition_count = (std::uint32_t) parts.size();
    out->partitions = parts.empty()
        ? nullptr
        : (cellshard::bucketed_blocked_ell_partition *) std::calloc(parts.size(), sizeof(cellshard::bucketed_blocked_ell_partition));
    out->partition_row_offsets = (std::uint32_t *) std::calloc(parts.size() + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_cols = cols != 0u ? (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t)) : nullptr;
    out->canonical_to_exec_cols = cols != 0u ? (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t)) : nullptr;
    if ((out->partition_count != 0u && (out->partitions == nullptr || out->partition_row_offsets == nullptr))
        || (cols != 0u && (out->exec_to_canonical_cols == nullptr || out->canonical_to_exec_cols == nullptr))) {
        cellshard::clear(out);
        return 0;
    }
    if (cols != 0u) {
        std::memcpy(out->exec_to_canonical_cols, exec_to_canonical_cols.data(), (std::size_t) cols * sizeof(std::uint32_t));
        std::memcpy(out->canonical_to_exec_cols, canonical_to_exec_cols.data(), (std::size_t) cols * sizeof(std::uint32_t));
    }
    sparse::init(&canonical_coo);
    sparse::init(&permuted);
    for (std::size_t part_i = 0; part_i < parts.size(); ++part_i) {
        std::uint32_t bucket_count = 1u;
        std::uint64_t bucketed_bytes = 0u;
        cellshard::bucketed_blocked_ell_partition *bucketed = out->partitions + part_i;
        cellshard::init(bucketed);
        out->partition_row_offsets[part_i] = local_rows;
        if (!blocked_ell_to_canonical_coo(&parts[part_i], &canonical_coo)
            || !cellshard::convert::blocked_ell_from_coo(&canonical_coo,
                                                         cols,
                                                         canonical_to_exec_cols.data(),
                                                         parts[part_i].block_size,
                                                         &permuted)
            || !choose_bucket_count_for_part(&permuted, device, &bucket_count)
            || !cellshard::build_bucketed_blocked_ell_partition(bucketed, &permuted, bucket_count, &bucketed_bytes)) {
            sparse::clear(&canonical_coo);
            sparse::clear(&permuted);
            cellshard::clear(out);
            return 0;
        }
        if (cols != 0u) {
            bucketed->exec_to_canonical_cols = (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t));
            bucketed->canonical_to_exec_cols = (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t));
            if (bucketed->exec_to_canonical_cols == nullptr || bucketed->canonical_to_exec_cols == nullptr) {
                sparse::clear(&canonical_coo);
                sparse::clear(&permuted);
                cellshard::clear(out);
                return 0;
            }
            std::memcpy(bucketed->exec_to_canonical_cols, out->exec_to_canonical_cols, (std::size_t) cols * sizeof(std::uint32_t));
            std::memcpy(bucketed->canonical_to_exec_cols, out->canonical_to_exec_cols, (std::size_t) cols * sizeof(std::uint32_t));
        }
        local_rows += bucketed->rows;
        local_nnz += bucketed->nnz;
        out->nnz += bucketed->nnz;
        sparse::clear(&canonical_coo);
        sparse::init(&canonical_coo);
        sparse::clear(&permuted);
        sparse::init(&permuted);
    }
    out->partition_row_offsets[parts.size()] = local_rows;
    out->rows = local_rows;
    (void) local_nnz;
    sparse::clear(&canonical_coo);
    sparse::clear(&permuted);
    return 1;
}

static inline const std::string *find_spool_path_for_part(const std::vector<dataset_dataset_plan> &plans,
                                                          unsigned long global_part_id) {
    for (const dataset_dataset_plan &plan : plans) {
        if (global_part_id < plan.global_part_begin) continue;
        const unsigned long local_part = global_part_id - plan.global_part_begin;
        if (local_part < (unsigned long) plan.spool_paths.size()) return &plan.spool_paths[(std::size_t) local_part];
    }
    return nullptr;
}

static inline std::string build_ingest_spool_root(const char *out_path,
                                                  const std::string &cache_root) {
    namespace fs = std::filesystem;
    const fs::path out = out_path != 0 ? fs::path(out_path) : fs::path("dataset.csh5");
    const std::string stem = out.filename().string();
    if (!cache_root.empty()) {
        return (fs::path(cache_root) / (stem + ".ingest_spool")).string();
    }
    if (!out.parent_path().empty()) {
        return (out.parent_path() / (stem + ".ingest_spool")).string();
    }
    return stem + ".ingest_spool";
}

static inline int prepare_ingest_spool_root(const std::string &root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (root.empty()) return 0;
    fs::remove_all(root, ec);
    ec.clear();
    return fs::create_directories(root, ec) || (!ec && fs::exists(root, ec));
}

static inline std::string build_ingest_spool_part_path(const std::string &root,
                                                       unsigned long global_part_id) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "part.%08lu.bell", global_part_id);
    return (std::filesystem::path(root) / filename).string();
}

static inline int convert_manifest_dataset_to_hdf5(const manifest *m,
                                                  const char *out_path,
                                                  const dataset_h5_convert_options *opts) {
    namespace fs = std::filesystem;
    std::vector<dataset_dataset_plan> plans;
    common::text_column dataset_ids;
    common::text_column matrix_paths;
    common::text_column feature_paths;
    common::text_column barcode_paths;
    common::text_column metadata_paths;
    common::text_column global_barcodes;
    common::text_column global_feature_ids;
    common::text_column global_feature_names;
    common::text_column global_feature_types;
    std::vector<std::uint32_t> dataset_formats;
    std::vector<std::uint64_t> dataset_row_begin;
    std::vector<std::uint64_t> dataset_row_end;
    std::vector<std::uint64_t> dataset_rows;
    std::vector<std::uint64_t> dataset_cols;
    std::vector<std::uint64_t> dataset_nnz;
    std::vector<std::uint32_t> cell_dataset_ids;
    std::vector<std::uint64_t> cell_local_indices;
    std::vector<std::uint32_t> feature_dataset_ids;
    std::vector<std::uint64_t> feature_local_indices;
    std::vector<std::uint64_t> dataset_feature_offsets;
    std::vector<std::uint32_t> dataset_feature_to_global;
    std::vector<std::uint64_t> part_rows;
    std::vector<std::uint64_t> part_nnz;
    std::vector<std::uint32_t> part_axes;
    std::vector<std::uint64_t> part_aux;
    std::vector<std::uint64_t> part_row_offsets;
    std::vector<std::uint32_t> part_dataset_ids;
    std::vector<std::uint32_t> part_codec_ids;
    std::vector<std::uint64_t> part_bytes;
    std::vector<std::uint64_t> shard_offsets;
    std::unordered_map<std::string, std::uint32_t> feature_map;
    cellshard::dataset_codec_descriptor codec;
    cellshard::dataset_layout_view layout;
    cellshard::dataset_dataset_table_view dataset_view;
    cellshard::dataset_provenance_view provenance_view;
    unsigned int manifest_i = 0;
    unsigned int dataset_idx = 0;
    unsigned long global_rows = 0;
    unsigned long global_parts = 0;
    std::string spool_root;
    int spool_ready = 0;
    partition shard_plan;
    int ok = 0;

    if (m == 0 || out_path == 0 || opts == 0) return 0;

    common::init(&dataset_ids);
    common::init(&matrix_paths);
    common::init(&feature_paths);
    common::init(&barcode_paths);
    common::init(&metadata_paths);
    common::init(&global_barcodes);
    common::init(&global_feature_ids);
    common::init(&global_feature_names);
    common::init(&global_feature_types);
    init(&shard_plan);
    part_row_offsets.push_back(0ull);
    spool_root = build_ingest_spool_root(out_path, opts->cache_root);
    if (!prepare_ingest_spool_root(spool_root)) goto done;
    spool_ready = 1;

    for (manifest_i = 0; manifest_i < m->count; ++manifest_i) {
        common::barcode_table barcodes;
        common::feature_table features;
        mtx::header header;
        unsigned long *row_nnz = 0;
        unsigned long *row_offsets = 0;
        unsigned long *part_nnz_raw = 0;
        unsigned long num_parts = 0;
        dataset_dataset_plan plan;
        unsigned long local_part = 0;
        unsigned int feature_i = 0;

        if (format_at(m, manifest_i) != source_mtx
            && format_at(m, manifest_i) != source_tenx_mtx
            && format_at(m, manifest_i) != source_h5ad) continue;

        common::init(&barcodes);
        common::init(&features);
        mtx::init(&header);
        // Dataset planning is intentionally CPU-heavy: full MTX scan, row
        // partitioning, then barcode/feature ingest before windowed conversion.
        if (!scan_source_row_nnz(m, manifest_i, &header, &row_nnz, opts->reader_bytes)) {
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!mtx::plan_row_partitions_by_nnz(row_nnz, header.rows, opts->max_part_nnz, &row_offsets, &num_parts)) {
            std::free(row_nnz);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!mtx::build_part_nnz_from_row_nnz(row_nnz, row_offsets, num_parts, &part_nnz_raw)) {
            std::free(row_nnz);
            std::free(row_offsets);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }

        if (!load_source_barcodes(m, manifest_i, &barcodes)) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!load_source_features(m, manifest_i, &features)) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (common::count(&barcodes) != header.rows || common::count(&features) != header.cols) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }

        plan.manifest_idx = manifest_i;
        plan.dataset_idx = dataset_idx;
        plan.header = header;
        plan.global_row_begin = global_rows;
        plan.global_part_begin = global_parts;
        plan.row_offsets.assign(row_offsets, row_offsets + num_parts + 1ul);
        plan.part_nnz.assign(part_nnz_raw, part_nnz_raw + num_parts);
        plan.part_rows.resize((std::size_t) num_parts);
        plan.part_bytes.resize((std::size_t) num_parts);
        plan.part_aux.resize((std::size_t) num_parts);
        plan.spool_paths.resize((std::size_t) num_parts);
        plan.feature_to_global.resize((std::size_t) header.cols);

        for (local_part = 0; local_part < num_parts; ++local_part) {
            const unsigned long rows = row_offsets[local_part + 1ul] - row_offsets[local_part];
            plan.part_rows[local_part] = rows;
            plan.part_bytes[local_part] = (unsigned long) standard_csr_bytes(rows, part_nnz_raw[local_part]);
            part_rows.push_back((std::uint64_t) rows);
            part_nnz.push_back((std::uint64_t) part_nnz_raw[local_part]);
            part_axes.push_back(0u);
            part_aux.push_back(0ull);
            part_dataset_ids.push_back((std::uint32_t) dataset_idx);
            part_codec_ids.push_back(0u);
            part_bytes.push_back((std::uint64_t) plan.part_bytes[local_part]);
            ++global_parts;
        }

        for (local_part = 0; local_part < num_parts; ++local_part) {
            global_rows += plan.part_rows[local_part];
            part_row_offsets.push_back((std::uint64_t) global_rows);
        }

        if (!common::append(&dataset_ids, dataset_id_at(m, manifest_i), std::strlen(dataset_id_at(m, manifest_i)))) goto done;
        if (!common::append(&matrix_paths, matrix_path_at(m, manifest_i), std::strlen(matrix_path_at(m, manifest_i)))) goto done;
        if (!common::append(&feature_paths,
                            feature_path_at(m, manifest_i) != 0 ? feature_path_at(m, manifest_i) : "",
                            std::strlen(feature_path_at(m, manifest_i) != 0 ? feature_path_at(m, manifest_i) : ""))) goto done;
        if (!common::append(&barcode_paths,
                            barcode_path_at(m, manifest_i) != 0 ? barcode_path_at(m, manifest_i) : "",
                            std::strlen(barcode_path_at(m, manifest_i) != 0 ? barcode_path_at(m, manifest_i) : ""))) goto done;
        if (!common::append(&metadata_paths, metadata_path_at(m, manifest_i) != 0 ? metadata_path_at(m, manifest_i) : "", std::strlen(metadata_path_at(m, manifest_i) != 0 ? metadata_path_at(m, manifest_i) : ""))) goto done;
        dataset_formats.push_back(format_at(m, manifest_i));
        dataset_row_begin.push_back((std::uint64_t) plan.global_row_begin);
        dataset_row_end.push_back((std::uint64_t) (plan.global_row_begin + header.rows));
        dataset_rows.push_back((std::uint64_t) header.rows);
        dataset_cols.push_back((std::uint64_t) header.cols);
        dataset_nnz.push_back((std::uint64_t) header.nnz_file);

        // Global feature identity is merged through a host hash table. That is
        // appropriate for offline ingest even though it is not cheap.
        for (feature_i = 0; feature_i < common::count(&features); ++feature_i) {
            const char *feature_id = common::id(&features, feature_i);
            const char *feature_name = common::name(&features, feature_i);
            const char *feature_type = common::type(&features, feature_i);
            std::string key = std::string(feature_id != 0 ? feature_id : "")
                + "\t" + std::string(feature_name != 0 ? feature_name : "")
                + "\t" + std::string(feature_type != 0 ? feature_type : "");
            std::unordered_map<std::string, std::uint32_t>::const_iterator hit = feature_map.find(key);
            std::uint32_t global_feature = 0u;

            if (hit == feature_map.end()) {
                global_feature = (std::uint32_t) feature_dataset_ids.size();
                feature_map.insert(std::make_pair(key, global_feature));
                if (!common::append(&global_feature_ids, feature_id != 0 ? feature_id : "", std::strlen(feature_id != 0 ? feature_id : ""))) goto done;
                if (!common::append(&global_feature_names, feature_name != 0 ? feature_name : "", std::strlen(feature_name != 0 ? feature_name : ""))) goto done;
                if (!common::append(&global_feature_types, feature_type != 0 ? feature_type : "", std::strlen(feature_type != 0 ? feature_type : ""))) goto done;
                feature_dataset_ids.push_back((std::uint32_t) dataset_idx);
                feature_local_indices.push_back((std::uint64_t) feature_i);
            } else {
                global_feature = hit->second;
            }
            plan.feature_to_global[feature_i] = global_feature;
            dataset_feature_to_global.push_back(global_feature);
        }
        dataset_feature_offsets.push_back((std::uint64_t) dataset_feature_to_global.size() - (std::uint64_t) header.cols);

        for (feature_i = 0; feature_i < common::count(&barcodes); ++feature_i) {
            const char *barcode = common::get(&barcodes, feature_i);
            if (!common::append(&global_barcodes, barcode != 0 ? barcode : "", std::strlen(barcode != 0 ? barcode : ""))) goto done;
            cell_dataset_ids.push_back((std::uint32_t) dataset_idx);
            cell_local_indices.push_back((std::uint64_t) feature_i);
        }

        plans.push_back(plan);
        ++dataset_idx;

        std::free(row_nnz);
        std::free(row_offsets);
        std::free(part_nnz_raw);
        common::clear(&barcodes);
        common::clear(&features);
    }

    if (plans.empty()) goto done;
    dataset_feature_offsets.push_back((std::uint64_t) dataset_feature_to_global.size());
    for (manifest_i = 0; manifest_i < plans.size(); ++manifest_i) {
        partition windows;
        sharded<sparse::coo> window_view;
        sparse::blocked_ell blocked_part;
        unsigned long window_i = 0;

        init(&windows);
        init(&window_view);
        sparse::init(&blocked_part);
        if (!build_by_bytes(&windows,
                            plans[manifest_i].part_rows.data(),
                            plans[manifest_i].part_bytes.data(),
                            (unsigned long) plans[manifest_i].part_rows.size(),
                            opts->convert_window_bytes)) {
            clear(&windows);
            goto done;
        }

        for (window_i = 0; window_i < windows.count; ++window_i) {
            unsigned long local_part = 0;
            if (!load_source_part_window_coo(m,
                                             plans[manifest_i].manifest_idx,
                                             &plans[manifest_i].header,
                                             plans[manifest_i].row_offsets.data(),
                                             plans[manifest_i].part_nnz.data(),
                                             (unsigned long) plans[manifest_i].part_rows.size(),
                                             windows.ranges[window_i].part_begin,
                                             windows.ranges[window_i].part_end,
                                             &window_view,
                                             opts->reader_bytes)) {
                clear(&windows);
                clear(&window_view);
                sparse::clear(&blocked_part);
                goto done;
            }

            for (local_part = 0; local_part < window_view.num_partitions; ++local_part) {
                const unsigned long global_part_id = plans[manifest_i].global_part_begin + windows.ranges[window_i].part_begin + local_part;
                cellshard::convert::blocked_ell_tune_result tune = {};
                if (!convert_coo_part_to_blocked_ell_auto(window_view.parts[local_part],
                                                          (std::uint32_t) feature_dataset_ids.size(),
                                                          plans[manifest_i].feature_to_global.data(),
                                                          &blocked_part,
                                                          &tune)) {
                    clear(&windows);
                    clear(&window_view);
                    sparse::clear(&blocked_part);
                    goto done;
                }
                plans[manifest_i].part_aux[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    cellshard::sparse::pack_blocked_ell_aux(blocked_part.block_size, cellshard::sparse::ell_width_blocks(&blocked_part));
                plans[manifest_i].part_bytes[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    (unsigned long) cellshard::packed_blocked_ell_bytes(blocked_part.rows, blocked_part.ell_cols, blocked_part.block_size, sizeof(::real::storage_t));
                plans[manifest_i].spool_paths[(std::size_t) (windows.ranges[window_i].part_begin + local_part)] =
                    build_ingest_spool_part_path(spool_root, global_part_id);
                part_aux[(std::size_t) global_part_id] = plans[manifest_i].part_aux[(std::size_t) (windows.ranges[window_i].part_begin + local_part)];
                part_bytes[(std::size_t) global_part_id] = (std::uint64_t) plans[manifest_i].part_bytes[(std::size_t) (windows.ranges[window_i].part_begin + local_part)];
                if (plans[manifest_i].spool_paths[(std::size_t) (windows.ranges[window_i].part_begin + local_part)].empty()
                    || !cellshard::store(plans[manifest_i].spool_paths[(std::size_t) (windows.ranges[window_i].part_begin + local_part)].c_str(), &blocked_part)) {
                    clear(&windows);
                    clear(&window_view);
                    sparse::clear(&blocked_part);
                    goto done;
                }
                sparse::clear(&blocked_part);
                sparse::init(&blocked_part);
            }
            clear(&window_view);
            init(&window_view);
        }

        clear(&windows);
        sparse::clear(&blocked_part);
    }
    if (!build_blocked_ell_shards(&shard_plan,
                                  (const unsigned long *) part_rows.data(),
                                  (const unsigned long *) part_nnz.data(),
                                  (const unsigned long *) part_aux.data(),
                                  (const unsigned long *) part_bytes.data(),
                                  (unsigned long) part_rows.size(),
                                  opts->target_shard_bytes)) goto done;
    shard_offsets.resize((std::size_t) shard_plan.count + 1u, 0ull);
    for (manifest_i = 0; manifest_i < shard_plan.count; ++manifest_i) {
        shard_offsets[manifest_i] = (std::uint64_t) shard_plan.ranges[manifest_i].row_begin;
    }
    shard_offsets[shard_plan.count] = (std::uint64_t) global_rows;

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = (std::uint64_t) global_rows;
    layout.cols = (std::uint64_t) feature_dataset_ids.size();
    layout.nnz = 0u;
    for (manifest_i = 0; manifest_i < part_nnz.size(); ++manifest_i) layout.nnz += part_nnz[manifest_i];
    layout.num_partitions = (std::uint64_t) part_rows.size();
    layout.num_shards = (std::uint64_t) shard_plan.count;
    layout.partition_rows = part_rows.data();
    layout.partition_nnz = part_nnz.data();
    layout.partition_axes = part_axes.data();
    layout.partition_aux = part_aux.data();
    layout.partition_row_offsets = part_row_offsets.data();
    layout.partition_dataset_ids = part_dataset_ids.data();
    layout.partition_codec_ids = part_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    dataset_view.count = dataset_idx;
    dataset_view.dataset_ids = as_text_view(&dataset_ids);
    dataset_view.matrix_paths = as_text_view(&matrix_paths);
    dataset_view.feature_paths = as_text_view(&feature_paths);
    dataset_view.barcode_paths = as_text_view(&barcode_paths);
    dataset_view.metadata_paths = as_text_view(&metadata_paths);
    dataset_view.formats = dataset_formats.data();
    dataset_view.row_begin = dataset_row_begin.data();
    dataset_view.row_end = dataset_row_end.data();
    dataset_view.rows = dataset_rows.data();
    dataset_view.cols = dataset_cols.data();
    dataset_view.nnz = dataset_nnz.data();

    provenance_view.global_barcodes = as_text_view(&global_barcodes);
    provenance_view.cell_dataset_ids = cell_dataset_ids.data();
    provenance_view.cell_local_indices = cell_local_indices.data();
    provenance_view.feature_ids = as_text_view(&global_feature_ids);
    provenance_view.feature_names = as_text_view(&global_feature_names);
    provenance_view.feature_types = as_text_view(&global_feature_types);
    provenance_view.feature_dataset_ids = feature_dataset_ids.data();
    provenance_view.feature_local_indices = feature_local_indices.data();
    provenance_view.dataset_feature_offsets = dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = dataset_feature_to_global.data();

    if (!cellshard::create_dataset_blocked_ell_h5(out_path, &layout, &dataset_view, &provenance_view)) goto done;

    for (unsigned long shard_id = 0; shard_id < shard_plan.count; ++shard_id) {
        const shard_range &range = shard_plan.ranges[shard_id];
        std::vector<sparse::blocked_ell> shard_parts;
        cellshard::bucketed_blocked_ell_shard optimized_shard;

        cellshard::init(&optimized_shard);
        shard_parts.resize((std::size_t) (range.part_end - range.part_begin));
        for (sparse::blocked_ell &part : shard_parts) sparse::init(&part);
        for (unsigned long global_part_id = range.part_begin; global_part_id < range.part_end; ++global_part_id) {
            const std::string *spool_path = find_spool_path_for_part(plans, global_part_id);
            const std::size_t local_part = (std::size_t) (global_part_id - range.part_begin);
            if (spool_path == nullptr
                || spool_path->empty()
                || !cellshard::load(spool_path->c_str(), &shard_parts[local_part])) {
                for (sparse::blocked_ell &part : shard_parts) sparse::clear(&part);
                cellshard::clear(&optimized_shard);
                goto done;
            }
        }
        if (!build_bucketed_optimized_shard(shard_parts,
                                            (std::uint32_t) layout.cols,
                                            opts->device,
                                            &optimized_shard)
            || !cellshard::append_bucketed_blocked_ell_shard_h5(out_path, shard_id, &optimized_shard)) {
            for (sparse::blocked_ell &part : shard_parts) sparse::clear(&part);
            cellshard::clear(&optimized_shard);
            goto done;
        }
        for (sparse::blocked_ell &part : shard_parts) sparse::clear(&part);
        cellshard::clear(&optimized_shard);
    }

    if (!opts->cache_root.empty()
        && !cellshard::warm_dataset_blocked_ell_h5_cache(out_path, opts->cache_root.c_str())) {
        goto done;
    }

    ok = 1;

done:
    if (ok && spool_ready) {
        std::error_code ec;
        fs::remove_all(spool_root, ec);
    }
    clear(&shard_plan);
    common::clear(&dataset_ids);
    common::clear(&matrix_paths);
    common::clear(&feature_paths);
    common::clear(&barcode_paths);
    common::clear(&metadata_paths);
    common::clear(&global_barcodes);
    common::clear(&global_feature_ids);
    common::clear(&global_feature_names);
    common::clear(&global_feature_types);
    return ok;
}

// Default MTX-dataset conversion path. This emits one portable HDF5-backed
// dataset container in out_dir/dataset.csh5.
static inline int convert_manifest_mtx_dataset_to_hdf5(const manifest *m,
                                                      const char *out_path,
                                                      const dataset_h5_convert_options *opts) {
    return convert_manifest_dataset_to_hdf5(m, out_path, opts);
}

static inline int convert_manifest_mtx_series(const manifest *m,
                                              const char *out_dir,
                                              const mtx_convert_options *opts) {
    dataset_h5_convert_options h5_opts;
    char out_path[4096];

    if (m == 0 || out_dir == 0 || opts == 0) return 0;
    init(&h5_opts);
    h5_opts.max_part_nnz = opts->max_part_nnz;
    h5_opts.convert_window_bytes = opts->convert_window_bytes;
    h5_opts.target_shard_bytes = opts->target_shard_bytes;
    h5_opts.reader_bytes = opts->reader_bytes;
    h5_opts.cache_root = opts->cache_root;
    if (!build_dataset_h5_output_path(out_dir, out_path, sizeof(out_path))) return 0;
    return convert_manifest_dataset_to_hdf5(m, out_path, &h5_opts);
}

} // namespace dataset
} // namespace ingest
} // namespace cellerator
