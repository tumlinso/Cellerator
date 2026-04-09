#include "../extern/CellShard/src/CellShard.hh"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    cellshard::series_text_column_view view() const {
        cellshard::series_text_column_view out;
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? 0 : offsets.data();
        out.data = data.empty() ? 0 : data.data();
        return out;
    }
};

static owned_text_column make_column(const std::vector<const char *> &values) {
    owned_text_column col;
    std::size_t i = 0;
    std::uint32_t cursor = 0;

    col.offsets.resize(values.size() + 1u, 0u);
    for (i = 0; i < values.size(); ++i) {
        const char *value = values[i] != 0 ? values[i] : "";
        const std::size_t len = std::strlen(value);
        col.offsets[i] = cursor;
        col.data.insert(col.data.end(), value, value + len);
        col.data.push_back(0);
        cursor += (std::uint32_t) len + 1u;
    }
    col.offsets[values.size()] = cursor;
    return col;
}

static int populate_part(cellshard::sparse::compressed *part,
                         unsigned int rows,
                         unsigned int cols,
                         const std::vector<unsigned int> &row_ptr,
                         const std::vector<unsigned int> &col_idx,
                         const std::vector<float> &values) {
    std::size_t i = 0;

    cellshard::sparse::init(part, rows, cols, (cellshard::types::nnz_t) values.size(), cellshard::sparse::compressed_by_row);
    if (!cellshard::sparse::allocate(part)) return 0;
    std::memcpy(part->majorPtr, row_ptr.data(), row_ptr.size() * sizeof(unsigned int));
    std::memcpy(part->minorIdx, col_idx.data(), col_idx.size() * sizeof(unsigned int));
    for (i = 0; i < values.size(); ++i) part->val[i] = __float2half(values[i]);
    return 1;
}

static int check_part(const cellshard::sparse::compressed *part,
                      const std::vector<unsigned int> &row_ptr,
                      const std::vector<unsigned int> &col_idx,
                      const std::vector<float> &values) {
    std::size_t i = 0;
    if (part == 0) return 0;
    for (i = 0; i < row_ptr.size(); ++i) {
        if (part->majorPtr[i] != row_ptr[i]) return 0;
    }
    for (i = 0; i < col_idx.size(); ++i) {
        if (part->minorIdx[i] != col_idx[i]) return 0;
        if (__half2float(part->val[i]) != values[i]) return 0;
    }
    return 1;
}

} // namespace

int main() {
    const std::string out_path = "/tmp/cellshard_series_test.csh5";
    const std::string cache_dir = "/tmp/cellshard_series_cache";
    const std::string cache_part0 = cache_dir + "/part.0.cscache";
    const std::string cache_part1 = cache_dir + "/part.1.cscache";
    cellshard::sparse::compressed part0;
    cellshard::sparse::compressed part1;
    cellshard::sharded<cellshard::sparse::compressed> loaded;
    cellshard::shard_storage storage;
    owned_text_column dataset_ids = make_column({"embryo_1_exon", "embryo_1_intron"});
    owned_text_column matrix_paths = make_column({"/tmp/exon.mtx", "/tmp/intron.mtx"});
    owned_text_column feature_paths = make_column({"/tmp/features.tsv", "/tmp/features.tsv"});
    owned_text_column barcode_paths = make_column({"/tmp/barcodes_exon.tsv", "/tmp/barcodes_intron.tsv"});
    owned_text_column metadata_paths = make_column({"", ""});
    owned_text_column global_barcodes = make_column({"bc0", "bc1", "bc2"});
    owned_text_column feature_ids = make_column({"g0", "g1", "g2"});
    owned_text_column feature_names = make_column({"Gene0", "Gene1", "Gene2"});
    owned_text_column feature_types = make_column({"gene", "gene", "gene"});
    std::vector<std::uint32_t> dataset_formats = { 2u, 2u };
    std::vector<std::uint64_t> dataset_row_begin = { 0u, 2u };
    std::vector<std::uint64_t> dataset_row_end = { 2u, 3u };
    std::vector<std::uint64_t> dataset_rows = { 2u, 1u };
    std::vector<std::uint64_t> dataset_cols = { 3u, 3u };
    std::vector<std::uint64_t> dataset_nnz = { 3u, 1u };
    std::vector<std::uint32_t> cell_dataset_ids = { 0u, 0u, 1u };
    std::vector<std::uint64_t> cell_local_indices = { 0u, 1u, 0u };
    std::vector<std::uint32_t> feature_dataset_ids = { 0u, 0u, 0u };
    std::vector<std::uint64_t> feature_local_indices = { 0u, 1u, 2u };
    std::vector<std::uint64_t> dataset_feature_offsets = { 0u, 3u, 6u };
    std::vector<std::uint32_t> dataset_feature_to_global = { 0u, 1u, 2u, 0u, 1u, 2u };
    std::vector<std::uint64_t> part_rows = { 2u, 1u };
    std::vector<std::uint64_t> part_nnz = { 3u, 1u };
    std::vector<std::uint32_t> part_axes = { (std::uint32_t) cellshard::sparse::compressed_by_row, (std::uint32_t) cellshard::sparse::compressed_by_row };
    std::vector<std::uint64_t> part_row_offsets = { 0u, 2u, 3u };
    std::vector<std::uint32_t> part_dataset_ids = { 0u, 1u };
    std::vector<std::uint32_t> part_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u, 3u };
    cellshard::series_codec_descriptor codec;
    cellshard::series_layout_view layout;
    cellshard::series_dataset_table_view dataset_view;
    cellshard::series_provenance_view provenance_view;
    int rc = 1;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_part(&part0, 2u, 3u, {0u, 2u, 3u}, {0u, 2u, 1u}, {1.0f, 2.0f, 3.0f})) goto done;
    if (!populate_part(&part1, 1u, 3u, {0u, 1u}, {2u}, {4.0f})) goto done;

    codec.codec_id = 0u;
    codec.family = cellshard::series_codec_family_standard_csr;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 3u;
    layout.cols = 3u;
    layout.nnz = 4u;
    layout.num_parts = 2u;
    layout.num_shards = 2u;
    layout.part_rows = part_rows.data();
    layout.part_nnz = part_nnz.data();
    layout.part_axes = part_axes.data();
    layout.part_row_offsets = part_row_offsets.data();
    layout.part_dataset_ids = part_dataset_ids.data();
    layout.part_codec_ids = part_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    dataset_view.count = 2u;
    dataset_view.dataset_ids = dataset_ids.view();
    dataset_view.matrix_paths = matrix_paths.view();
    dataset_view.feature_paths = feature_paths.view();
    dataset_view.barcode_paths = barcode_paths.view();
    dataset_view.metadata_paths = metadata_paths.view();
    dataset_view.formats = dataset_formats.data();
    dataset_view.row_begin = dataset_row_begin.data();
    dataset_view.row_end = dataset_row_end.data();
    dataset_view.rows = dataset_rows.data();
    dataset_view.cols = dataset_cols.data();
    dataset_view.nnz = dataset_nnz.data();

    provenance_view.global_barcodes = global_barcodes.view();
    provenance_view.cell_dataset_ids = cell_dataset_ids.data();
    provenance_view.cell_local_indices = cell_local_indices.data();
    provenance_view.feature_ids = feature_ids.view();
    provenance_view.feature_names = feature_names.view();
    provenance_view.feature_types = feature_types.view();
    provenance_view.feature_dataset_ids = feature_dataset_ids.data();
    provenance_view.feature_local_indices = feature_local_indices.data();
    provenance_view.dataset_feature_offsets = dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = dataset_feature_to_global.data();

    if (!cellshard::create_series_compressed_h5(out_path.c_str(), &layout, &dataset_view, &provenance_view)) goto done;
    if (!cellshard::append_standard_csr_part_h5(out_path.c_str(), 0u, &part0)) goto done;
    if (!cellshard::append_standard_csr_part_h5(out_path.c_str(), 1u, &part1)) goto done;

    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) goto done;
    if (loaded.rows != 3u || loaded.cols != 3u || loaded.nnz != 4u) goto done;
    if (storage.backend != cellshard::shard_storage_backend_series_h5) goto done;
    std::remove(cache_part0.c_str());
    std::remove(cache_part1.c_str());
    ::rmdir(cache_dir.c_str());
    if (!cellshard::bind_series_h5_part_cache(&storage, cache_dir.c_str())) goto done;
    if (!cellshard::prefetch_series_compressed_h5_shard_to_cache(&loaded, &storage, 0u)) goto done;
    if (!cellshard::prefetch_series_compressed_h5_shard_to_cache(&loaded, &storage, 1u)) goto done;
    if (::access(cache_part0.c_str(), R_OK) != 0) goto done;
    if (::access(cache_part1.c_str(), R_OK) != 0) goto done;
    if (!cellshard::fetch_part(&loaded, &storage, 0u)) goto done;
    if (!cellshard::drop_part(&loaded, 0u)) goto done;
    if (!cellshard::fetch_shard(&loaded, &storage, 1u)) goto done;
    if (!check_part(loaded.parts[0], {0u, 2u, 3u}, {0u, 2u, 1u}, {1.0f, 2.0f, 3.0f})) goto done;
    if (!check_part(loaded.parts[1], {0u, 1u}, {2u}, {4.0f})) goto done;

    rc = 0;

done:
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(cache_part0.c_str());
    std::remove(cache_part1.c_str());
    ::rmdir(cache_dir.c_str());
    std::remove(out_path.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "cellShardSeriesH5Test failed\n");
    }
    return rc;
}
