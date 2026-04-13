#include "../extern/CellShard/src/CellShard.hh"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
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

struct owned_observation_metadata_column {
    std::string name;
    std::uint32_t type = cellshard::series_observation_metadata_type_none;
    owned_text_column text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;

    cellshard::series_observation_metadata_column_view view() const {
        cellshard::series_observation_metadata_column_view out{};
        out.name = name.c_str();
        out.type = type;
        out.text_values = text_values.view();
        out.float32_values = float32_values.empty() ? nullptr : float32_values.data();
        out.uint8_values = uint8_values.empty() ? nullptr : uint8_values.data();
        return out;
    }
};

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

static int populate_blocked_ell_part(cellshard::sparse::blocked_ell *part,
                                     unsigned int rows,
                                     unsigned int cols,
                                     unsigned int block_size,
                                     unsigned int ell_cols,
                                     const std::vector<unsigned int> &block_idx,
                                     const std::vector<float> &values) {
    std::size_t i = 0;

    cellshard::sparse::init(part,
                            rows,
                            cols,
                            (cellshard::types::nnz_t) values.size(),
                            block_size,
                            ell_cols);
    if (!cellshard::sparse::allocate(part)) return 0;
    std::memcpy(part->blockColIdx, block_idx.data(), block_idx.size() * sizeof(unsigned int));
    for (i = 0; i < values.size(); ++i) part->val[i] = __float2half(values[i]);
    return 1;
}

static int check_blocked_ell_part(const cellshard::sparse::blocked_ell *part,
                                  const std::vector<unsigned int> &block_idx,
                                  const std::vector<float> &values) {
    std::size_t i = 0;
    if (part == 0) return 0;
    for (i = 0; i < block_idx.size(); ++i) {
        if (part->blockColIdx[i] != block_idx[i]) return 0;
    }
    for (i = 0; i < values.size(); ++i) {
        if (__half2float(part->val[i]) != values[i]) return 0;
    }
    return 1;
}

static int run_blocked_ell_roundtrip_test() {
    const std::string out_path = "/tmp/cellshard_series_blocked_ell_test.csh5";
    const std::string cache_root = "/tmp/cellshard_series_blocked_ell_cache";
    cellshard::sparse::blocked_ell part0;
    cellshard::sparse::blocked_ell part1;
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    cellshard::shard_storage storage;
    std::vector<std::uint64_t> partition_rows = { 2u, 2u };
    std::vector<std::uint64_t> partition_nnz = { 8u, 8u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul),
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 4u };
    cellshard::series_codec_descriptor codec{};
    cellshard::series_layout_view layout{};
    int rc = 1;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_blocked_ell_part(&part0,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {0u, 1u},
                                   {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) goto done;
    if (!populate_blocked_ell_part(&part1,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {1u, 0u},
                                   {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) goto done;

    codec.codec_id = 0u;
    codec.family = cellshard::series_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 4u;
    layout.cols = 4u;
    layout.nnz = 16u;
    layout.num_partitions = 2u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cellshard::create_series_blocked_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 0u, &part0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 1u, &part1)) goto done;

    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) {
        std::fprintf(stderr, "failed to load blocked ell series header\n");
        goto done;
    }
    if (storage.backend != cellshard::shard_storage_backend_series_h5) {
        std::fprintf(stderr, "blocked ell storage backend mismatch\n");
        goto done;
    }
    if (!cellshard::bind_series_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to bind blocked ell cache dir\n");
        goto done;
    }
    if (!cellshard::prefetch_series_blocked_ell_h5_shard_cache(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to prefetch blocked ell shard to cache\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch blocked ell part 0\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[0], {0u, 1u}, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) {
        std::fprintf(stderr, "blocked ell part 0 mismatch after fetch\n");
        goto done;
    }
    if (!cellshard::drop_partition(&loaded, 0u)) {
        std::fprintf(stderr, "failed to drop blocked ell part 0\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 1u)) {
        std::fprintf(stderr, "failed to fetch blocked ell part 1\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[1], {1u, 0u}, {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) {
        std::fprintf(stderr, "blocked ell part 1 mismatch after fetch\n");
        goto done;
    }
    if (!cellshard::drop_all_partitions(&loaded)) {
        std::fprintf(stderr, "failed to drop blocked ell parts\n");
        goto done;
    }
    if (!cellshard::fetch_shard(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch blocked ell shard 0\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[0], {0u, 1u}, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})
        || !check_blocked_ell_part(loaded.parts[1], {1u, 0u}, {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) {
        std::fprintf(stderr, "blocked ell shard payload mismatch after fetch\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_series_h5) {
        cellshard::invalidate_series_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    return rc;
}

} // namespace

int main() {
    const std::string out_path = "/tmp/cellshard_series_test.csh5";
    const std::string cache_root = "/tmp/cellshard_series_cache";
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
    owned_text_column metadata_column_names = make_column({"stage", "batch"});
    owned_text_column metadata_field_values = make_column({"E8", "A", "E9", "A", "E10", "B"});
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
    std::vector<std::uint32_t> metadata_row_offsets = { 0u, 2u, 4u, 6u };
    std::vector<std::uint32_t> metadata_dataset_indices = { 0u };
    std::vector<std::uint64_t> metadata_global_row_begin = { 0u };
    std::vector<std::uint64_t> metadata_global_row_end = { 3u };
    std::vector<std::uint64_t> partition_rows = { 2u, 1u };
    std::vector<std::uint64_t> partition_nnz = { 3u, 1u };
    std::vector<std::uint32_t> partition_axes = { (std::uint32_t) cellshard::sparse::compressed_by_row, (std::uint32_t) cellshard::sparse::compressed_by_row };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 3u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 1u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u, 3u };
    cellshard::series_codec_descriptor codec;
    cellshard::series_layout_view layout{};
    cellshard::series_dataset_table_view dataset_view{};
    cellshard::series_provenance_view provenance_view{};
    cellshard::series_metadata_table_view metadata_table_view{};
    cellshard::series_embedded_metadata_view embedded_metadata_view{};
    cellshard::series_observation_metadata_view observation_metadata_view{};
    cellshard::series_browse_cache_view browse_view{};
    owned_observation_metadata_column obs_day_label;
    owned_observation_metadata_column obs_day;
    owned_observation_metadata_column obs_postnatal;
    std::vector<cellshard::series_observation_metadata_column_view> observation_columns;
    std::vector<std::uint32_t> browse_feature_indices = { 0u, 2u };
    std::vector<float> browse_gene_sum = { 1.0f, 6.0f };
    std::vector<float> browse_gene_detected = { 1.0f, 2.0f };
    std::vector<float> browse_gene_sq_sum = { 1.0f, 20.0f };
    std::vector<float> browse_dataset_mean = { 0.5f, 1.0f, 0.0f, 4.0f };
    std::vector<float> browse_shard_mean = { 0.5f, 1.0f, 0.0f, 4.0f };
    std::vector<std::uint32_t> browse_part_sample_offsets = { 0u, 2u, 4u };
    std::vector<std::uint64_t> browse_part_sample_rows = { 0u, 1u, 2u, std::numeric_limits<std::uint64_t>::max() };
    std::vector<float> browse_partition_sample_values = {
        1.0f, 2.0f,
        0.0f, 0.0f,
        0.0f, 4.0f,
        0.0f, 0.0f
    };
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
    layout.num_partitions = 2u;
    layout.num_shards = 2u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = partition_axes.data();
    layout.partition_aux = nullptr;
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
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

    metadata_table_view.rows = 3u;
    metadata_table_view.cols = 2u;
    metadata_table_view.column_names = metadata_column_names.view();
    metadata_table_view.field_values = metadata_field_values.view();
    metadata_table_view.row_offsets = metadata_row_offsets.data();

    embedded_metadata_view.count = 1u;
    embedded_metadata_view.dataset_indices = metadata_dataset_indices.data();
    embedded_metadata_view.global_row_begin = metadata_global_row_begin.data();
    embedded_metadata_view.global_row_end = metadata_global_row_end.data();
    embedded_metadata_view.tables = &metadata_table_view;

    obs_day_label.name = "embryonic_day_label";
    obs_day_label.type = cellshard::series_observation_metadata_type_text;
    obs_day_label.text_values = make_column({"E8.5", "E8.5", "P0"});
    obs_day.name = "embryonic_day";
    obs_day.type = cellshard::series_observation_metadata_type_float32;
    obs_day.float32_values = {8.5f, 8.5f, std::numeric_limits<float>::quiet_NaN()};
    obs_postnatal.name = "is_postnatal";
    obs_postnatal.type = cellshard::series_observation_metadata_type_uint8;
    obs_postnatal.uint8_values = {0u, 0u, 1u};
    observation_columns = {obs_day_label.view(), obs_day.view(), obs_postnatal.view()};
    observation_metadata_view.rows = 3u;
    observation_metadata_view.cols = (std::uint32_t) observation_columns.size();
    observation_metadata_view.columns = observation_columns.data();

    browse_view.selected_feature_count = 2u;
    browse_view.selected_feature_indices = browse_feature_indices.data();
    browse_view.gene_sum = browse_gene_sum.data();
    browse_view.gene_detected = browse_gene_detected.data();
    browse_view.gene_sq_sum = browse_gene_sq_sum.data();
    browse_view.dataset_count = 2u;
    browse_view.dataset_feature_mean = browse_dataset_mean.data();
    browse_view.shard_count = 2u;
    browse_view.shard_feature_mean = browse_shard_mean.data();
    browse_view.partition_count = 2u;
    browse_view.sample_rows_per_partition = 2u;
    browse_view.partition_sample_row_offsets = browse_part_sample_offsets.data();
    browse_view.partition_sample_global_rows = browse_part_sample_rows.data();
    browse_view.partition_sample_values = browse_partition_sample_values.data();

    if (!cellshard::create_series_compressed_h5(out_path.c_str(), &layout, &dataset_view, &provenance_view)) goto done;
    if (!cellshard::append_standard_csr_partition_h5(out_path.c_str(), 0u, &part0)) goto done;
    if (!cellshard::append_standard_csr_partition_h5(out_path.c_str(), 1u, &part1)) goto done;
    if (!cellshard::append_series_embedded_metadata_h5(out_path.c_str(), &embedded_metadata_view)) {
        std::fprintf(stderr, "failed to append embedded metadata\n");
        goto done;
    }
    if (!cellshard::append_series_observation_metadata_h5(out_path.c_str(), &observation_metadata_view)) {
        std::fprintf(stderr, "failed to append observation metadata\n");
        goto done;
    }
    if (!cellshard::append_series_browse_cache_h5(out_path.c_str(), &browse_view)) {
        std::fprintf(stderr, "failed to append browse cache\n");
        goto done;
    }

    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) {
        std::fprintf(stderr, "failed to reload header after side-domain append\n");
        goto done;
    }
    if (loaded.rows != 3u || loaded.cols != 3u || loaded.nnz != 4u) {
        std::fprintf(stderr, "reloaded header mismatch\n");
        goto done;
    }
    if (storage.backend != cellshard::shard_storage_backend_series_h5) {
        std::fprintf(stderr, "storage backend mismatch after side-domain append\n");
        goto done;
    }
    if (!cellshard::bind_series_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to bind series h5 part cache\n");
        goto done;
    }
    if (!cellshard::prefetch_series_compressed_h5_shard_cache(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to prefetch shard 0 to cache\n");
        goto done;
    }
    if (!cellshard::prefetch_series_compressed_h5_shard_cache(&loaded, &storage, 1u)) {
        std::fprintf(stderr, "failed to prefetch shard 1 to cache\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch part 0 from series file\n");
        goto done;
    }
    if (!cellshard::drop_partition(&loaded, 0u)) {
        std::fprintf(stderr, "failed to drop part 0 after fetch\n");
        goto done;
    }
    if (!cellshard::fetch_shard(&loaded, &storage, 1u)) {
        std::fprintf(stderr, "failed to fetch shard 1 from series file\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to refetch part 0 from series file\n");
        goto done;
    }
    if (!check_part(loaded.parts[0], {0u, 2u, 3u}, {0u, 2u, 1u}, {1.0f, 2.0f, 3.0f})) {
        std::fprintf(stderr, "part 0 payload mismatch after side-domain append\n");
        goto done;
    }
    if (!check_part(loaded.parts[1], {0u, 1u}, {2u}, {4.0f})) {
        std::fprintf(stderr, "part 1 payload mismatch after side-domain append\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_series_h5) {
        cellshard::invalidate_series_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "cellShardSeriesH5Test failed\n");
    }
    if (run_blocked_ell_roundtrip_test() != 0) {
        std::fprintf(stderr, "cellShardSeriesH5Test blocked ell roundtrip failed\n");
        return 1;
    }
    return 0;
}
