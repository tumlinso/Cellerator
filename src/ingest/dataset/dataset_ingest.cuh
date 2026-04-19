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
#include "../../../extern/CellShard/src/convert/blocked_ell_from_coo_cuda.cuh"
#include "../../../extern/CellShard/src/convert/sliced_ell_from_compressed.cuh"
#include "../../../extern/CellShard/src/bucket/operators/blocked_ell_bucket.cuh"
#include "../../../extern/CellShard/include/CellShard/io/csh5/api.cuh"

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
    std::string working_root;
};

// Metadata-only init with throughput-oriented defaults.
static inline void init(mtx_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->convert_window_bytes = 1ul << 30ul;
    opts->target_shard_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->cache_root.clear();
    opts->working_root.clear();
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
    std::string working_root;
    int device;
    cudaStream_t stream;
};

static inline void init(dataset_h5_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->convert_window_bytes = 1ul << 30ul;
    opts->target_shard_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->cache_root.clear();
    opts->working_root.clear();
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
                                                       int device,
                                                       sparse::blocked_ell *dst,
                                                       cellshard::convert::blocked_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 4u, 8u, 16u, 32u };
    static thread_local cellshard::convert::blocked_ell_from_coo_cuda_workspace cuda_ws;
    static thread_local int cuda_ws_ready = 0;
    sparse::clear(dst);
    sparse::init(dst);
    if (!cuda_ws_ready) {
        cellshard::convert::init(&cuda_ws);
        cuda_ws_ready = 1;
    }
    if (cuda_ws.device != device && !cellshard::convert::setup(&cuda_ws, device)) {
        cuda_ws.device = -1;
    }
    if (cuda_ws.device == device
        && cellshard::convert::blocked_ell_from_coo_cuda_auto(src,
                                                              global_cols,
                                                              feature_to_global,
                                                              candidates,
                                                              sizeof(candidates) / sizeof(candidates[0]),
                                                              dst,
                                                              device,
                                                              tune,
                                                              &cuda_ws,
                                                              cuda_ws.stream)) {
        return 1;
    }
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

#include "internal/source_load_part.hh"

static inline int convert_compressed_part_to_sliced_ell_auto(const sparse::compressed *src,
                                                             int device,
                                                             sparse::sliced_ell *dst,
                                                             cellshard::convert::sliced_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 8u, 16u, 32u, 64u };
    sparse::clear(dst);
    sparse::init(dst);
    if (src == nullptr) return 0;
    if (src->axis == sparse::compressed_by_row
        && cellshard::convert::sliced_ell_from_compressed_cuda_auto(src,
                                                                    candidates,
                                                                    sizeof(candidates) / sizeof(candidates[0]),
                                                                    dst,
                                                                    device,
                                                                    (cudaStream_t) 0,
                                                                    tune)) {
        return 1;
    }
    return cellshard::convert::sliced_ell_from_compressed_auto(src,
                                                               candidates,
                                                               sizeof(candidates) / sizeof(candidates[0]),
                                                               dst,
                                                               tune);
}

static inline int convert_coo_part_to_sliced_ell_auto(const sparse::coo *src,
                                                      sparse::sliced_ell *dst,
                                                      cellshard::convert::sliced_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 8u, 16u, 32u, 64u };
    std::unique_ptr<cellshard::types::u32[]> row_nnz;
    std::unique_ptr<cellshard::types::u32[]> slice_row_offsets;
    std::unique_ptr<cellshard::types::u32[]> slice_widths;
    std::unique_ptr<cellshard::types::u32[]> row_write_cursor;
    cellshard::convert::sliced_ell_tune_result local_tune{};
    cellshard::types::u32 slice_count = 0u;

    if (src == nullptr || dst == nullptr) return 0;
    row_nnz.reset(src->rows != 0u ? new cellshard::types::u32[src->rows]() : nullptr);
    if (src->rows != 0u && !row_nnz) return 0;
    for (cellshard::types::nnz_t i = 0u; i < src->nnz; ++i) {
        const cellshard::types::u32 row = src->rowIdx[i];
        if (row >= src->rows) return 0;
        row_nnz[row] += 1u;
    }
    if (!cellshard::convert::choose_sliced_ell_slice_rows(row_nnz.get(),
                                                          src->rows,
                                                          src->nnz,
                                                          candidates,
                                                          sizeof(candidates) / sizeof(candidates[0]),
                                                          &local_tune)) return 0;
    if (!cellshard::convert::detail::build_uniform_sliced_layout_(row_nnz.get(),
                                                                  src->rows,
                                                                  local_tune.slice_rows == 0u ? 1u : local_tune.slice_rows,
                                                                  &slice_row_offsets,
                                                                  &slice_widths,
                                                                  &slice_count)) return 0;
    sparse::clear(dst);
    sparse::init(dst, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(dst, slice_count, slice_row_offsets.get(), slice_widths.get())) return 0;
    row_write_cursor.reset(dst->rows != 0u ? new cellshard::types::u32[dst->rows]() : nullptr);
    if (dst->rows != 0u && !row_write_cursor) {
        sparse::clear(dst);
        return 0;
    }
    for (cellshard::types::u32 row = 0u; row < dst->rows; ++row) {
        const cellshard::types::u32 slice = sparse::find_slice(dst, row);
        const cellshard::types::u32 row_begin = slice < dst->slice_count ? dst->slice_row_offsets[slice] : 0u;
        row_write_cursor[row] = (cellshard::types::u32) (sparse::slice_slot_base(dst, slice)
            + (std::size_t) (row - row_begin) * (std::size_t) dst->slice_widths[slice]);
    }
    for (cellshard::types::nnz_t i = 0u; i < src->nnz; ++i) {
        const cellshard::types::u32 row = src->rowIdx[i];
        dst->col_idx[row_write_cursor[row]] = src->colIdx[i];
        dst->val[row_write_cursor[row]] = src->val[i];
        row_write_cursor[row] += 1u;
    }
    if (tune != nullptr) *tune = local_tune;
    return 1;
}

static inline int blocked_ell_to_canonical_coo(const sparse::blocked_ell *part,
                                               sparse::coo *out) {
    const std::uint32_t block_size = part != nullptr ? part->block_size : 0u;
    const std::uint32_t width_blocks = part != nullptr ? cellshard::sparse::ell_width_blocks(part) : 0u;
    std::size_t actual_nnz = 0u;
    std::size_t emitted = 0u;
    if (part == nullptr || out == nullptr) return 0;
    sparse::clear(out);
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
                ++actual_nnz;
            }
        }
    }
    sparse::init(out, part->rows, part->cols, (cellshard::types::nnz_t) actual_nnz);
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

#include "internal/layout_build_part.hh"

#include "internal/dataset_convert_part.hh"

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
    h5_opts.working_root = opts->working_root;
    if (!build_dataset_h5_output_path(out_dir, out_path, sizeof(out_path))) return 0;
    return convert_manifest_dataset_to_hdf5(m, out_path, &h5_opts);
}

} // namespace dataset
} // namespace ingest
} // namespace cellerator
