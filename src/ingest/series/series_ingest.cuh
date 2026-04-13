#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "series_manifest.cuh"
#include "series_partition.cuh"
#include "../common/barcode_table.cuh"
#include "../common/feature_table.cuh"
#include "../mtx/mtx_reader.cuh"
#include "../mtx/compressed_parts.cuh"
#include "../../../extern/CellShard/src/convert/blocked_ell_from_compressed.cuh"
#include "../../../extern/CellShard/src/sharded/series_h5.cuh"

namespace cellerator {
namespace ingest {
namespace series {

using ::cellshard::clear;
using ::cellshard::find_offset_span;
using ::cellshard::init;
using ::cellshard::sharded;
namespace sparse = ::cellshard::sparse;

// Conversion knobs for MTX-series ingest.
struct mtx_convert_options {
    unsigned long max_part_nnz;
    unsigned long max_window_bytes;
    std::size_t reader_bytes;
};

// Metadata-only init with throughput-oriented defaults.
static inline void init(mtx_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->max_window_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
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

static inline int build_series_h5_output_path(const char *out_dir,
                                              char *out_path,
                                              std::size_t out_cap) {
    if (std::snprintf(out_path, out_cap, "%s/series.csh5", out_dir) <= 0) return 0;
    return 1;
}

struct series_h5_convert_options {
    unsigned long max_part_nnz;
    unsigned long max_window_bytes;
    std::size_t reader_bytes;
    int device;
    cudaStream_t stream;
};

static inline void init(series_h5_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->max_window_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->device = 0;
    opts->stream = (cudaStream_t) 0;
}

struct series_dataset_plan {
    unsigned int manifest_idx;
    unsigned int dataset_idx;
    mtx::header header;
    std::vector<unsigned long> row_offsets;
    std::vector<unsigned long> part_rows;
    std::vector<unsigned long> part_nnz;
    std::vector<unsigned long> part_bytes;
    std::vector<unsigned long> part_aux;
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

static inline cellshard::series_text_column_view as_text_view(const common::text_column *c) {
    cellshard::series_text_column_view view;
    view.count = c != 0 ? c->count : 0u;
    view.bytes = c != 0 ? c->bytes : 0u;
    view.offsets = c != 0 ? c->offsets : 0;
    view.data = c != 0 ? c->data : 0;
    return view;
}

static inline int convert_manifest_mtx_series_to_hdf5(const manifest *m,
                                                      const char *out_path,
                                                      const series_h5_convert_options *opts) {
    std::vector<series_dataset_plan> plans;
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
    cellshard::series_codec_descriptor codec;
    cellshard::series_layout_view layout;
    cellshard::series_dataset_table_view dataset_view;
    cellshard::series_provenance_view provenance_view;
    unsigned int manifest_i = 0;
    unsigned int dataset_idx = 0;
    unsigned long global_rows = 0;
    unsigned long global_parts = 0;
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

    for (manifest_i = 0; manifest_i < m->count; ++manifest_i) {
        common::barcode_table barcodes;
        common::feature_table features;
        mtx::header header;
        unsigned long *row_nnz = 0;
        unsigned long *row_offsets = 0;
        unsigned long *part_nnz_raw = 0;
        unsigned long num_parts = 0;
        series_dataset_plan plan;
        unsigned long local_part = 0;
        unsigned int feature_i = 0;

        if (format_at(m, manifest_i) != source_mtx && format_at(m, manifest_i) != source_tenx_mtx) continue;

        common::init(&barcodes);
        common::init(&features);
        mtx::init(&header);
        // Dataset planning is intentionally CPU-heavy: full MTX scan, row
        // partitioning, then barcode/feature ingest before windowed conversion.
        if (!mtx::scan_row_nnz(matrix_path_at(m, manifest_i), &header, &row_nnz, opts->reader_bytes)) {
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

        if (barcode_path_at(m, manifest_i) == 0 || *barcode_path_at(m, manifest_i) == 0) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (feature_path_at(m, manifest_i) == 0 || *feature_path_at(m, manifest_i) == 0) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!common::load_lines(barcode_path_at(m, manifest_i), &barcodes)) {
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            common::clear(&barcodes);
            common::clear(&features);
            goto done;
        }
        if (!common::load_tsv(feature_path_at(m, manifest_i), &features, 0)) {
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
        if (!common::append(&feature_paths, feature_path_at(m, manifest_i), std::strlen(feature_path_at(m, manifest_i)))) goto done;
        if (!common::append(&barcode_paths, barcode_path_at(m, manifest_i), std::strlen(barcode_path_at(m, manifest_i)))) goto done;
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
                            opts->max_window_bytes)) {
            clear(&windows);
            goto done;
        }

        for (window_i = 0; window_i < windows.count; ++window_i) {
            unsigned long local_part = 0;
            if (!mtx::load_part_window_coo(matrix_path_at(m, plans[manifest_i].manifest_idx),
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
                part_aux[(std::size_t) global_part_id] = plans[manifest_i].part_aux[(std::size_t) (windows.ranges[window_i].part_begin + local_part)];
                part_bytes[(std::size_t) global_part_id] = (std::uint64_t) plans[manifest_i].part_bytes[(std::size_t) (windows.ranges[window_i].part_begin + local_part)];
                sparse::clear(&blocked_part);
                sparse::init(&blocked_part);
            }
            clear(&window_view);
            init(&window_view);
        }

        clear(&windows);
        sparse::clear(&blocked_part);
    }
    if (!build_by_bytes(&shard_plan,
                        (const unsigned long *) part_rows.data(),
                        (const unsigned long *) part_bytes.data(),
                        (unsigned long) part_rows.size(),
                        opts->max_window_bytes)) goto done;
    shard_offsets.resize((std::size_t) shard_plan.count + 1u, 0ull);
    for (manifest_i = 0; manifest_i < shard_plan.count; ++manifest_i) {
        shard_offsets[manifest_i] = (std::uint64_t) shard_plan.ranges[manifest_i].row_begin;
    }
    shard_offsets[shard_plan.count] = (std::uint64_t) global_rows;

    codec.codec_id = 0u;
    codec.family = cellshard::series_codec_family_blocked_ell;
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

    if (!cellshard::create_series_blocked_ell_h5(out_path, &layout, &dataset_view, &provenance_view)) goto done;

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
                            opts->max_window_bytes)) {
            clear(&windows);
            goto done;
        }

        for (window_i = 0; window_i < windows.count; ++window_i) {
            unsigned long local_part = 0;
            if (!mtx::load_part_window_coo(matrix_path_at(m, plans[manifest_i].manifest_idx),
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
                unsigned long global_part_id = plans[manifest_i].global_part_begin + windows.ranges[window_i].part_begin + local_part;
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
                if (!cellshard::append_blocked_ell_partition_h5(out_path, global_part_id, &blocked_part)) {
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

    ok = 1;

done:
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

// Default MTX-series conversion path. This emits one portable HDF5-backed
// series container in out_dir/series.csh5.
static inline int convert_manifest_mtx_series(const manifest *m,
                                              const char *out_dir,
                                              const mtx_convert_options *opts) {
    series_h5_convert_options h5_opts;
    char out_path[4096];

    if (m == 0 || out_dir == 0 || opts == 0) return 0;
    init(&h5_opts);
    h5_opts.max_part_nnz = opts->max_part_nnz;
    h5_opts.max_window_bytes = opts->max_window_bytes;
    h5_opts.reader_bytes = opts->reader_bytes;
    if (!build_series_h5_output_path(out_dir, out_path, sizeof(out_path))) return 0;
    return convert_manifest_mtx_series_to_hdf5(m, out_path, &h5_opts);
}

} // namespace series
} // namespace ingest
} // namespace cellerator
