#include "../src/_apps/series_workbench.hh"

#include "../extern/CellShard/src/CellShard.hh"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace wb = ::cellerator::apps::workbench;

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

owned_text_column make_column(const std::vector<const char *> &values) {
    owned_text_column col;
    std::uint32_t cursor = 0;
    col.offsets.resize(values.size() + 1u, 0u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const char *value = values[i] != nullptr ? values[i] : "";
        const std::size_t len = std::strlen(value);
        col.offsets[i] = cursor;
        col.data.insert(col.data.end(), value, value + len);
        col.data.push_back(0);
        cursor += (std::uint32_t) len + 1u;
    }
    col.offsets[values.size()] = cursor;
    return col;
}

bool write_file(const std::string &path, const char *text) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const bool ok = std::fputs(text, f) >= 0;
    std::fclose(f);
    return ok;
}

bool populate_part(cellshard::sparse::compressed *part,
                   unsigned int rows,
                   unsigned int cols,
                   const std::vector<unsigned int> &row_ptr,
                   const std::vector<unsigned int> &col_idx,
                   const std::vector<float> &values) {
    cellshard::sparse::init(part, rows, cols, (cellshard::types::nnz_t) values.size(), cellshard::sparse::compressed_by_row);
    if (!cellshard::sparse::allocate(part)) return false;
    std::memcpy(part->majorPtr, row_ptr.data(), row_ptr.size() * sizeof(unsigned int));
    std::memcpy(part->minorIdx, col_idx.data(), col_idx.size() * sizeof(unsigned int));
    for (std::size_t i = 0; i < values.size(); ++i) part->val[i] = __float2half(values[i]);
    return true;
}

} // namespace

int main() {
    const std::string base = "/tmp/cellerator_series_workbench";
    const std::string manifest_path = base + ".manifest.tsv";
    const std::string matrix_path = base + ".matrix.mtx";
    const std::string feature_path = base + ".features.tsv";
    const std::string barcode_path = base + ".barcodes.tsv";
    const std::string metadata_path = base + ".metadata.tsv";
    const std::string series_path = base + ".series.csh5";
    cellshard::sparse::compressed part;
    cellshard::series_codec_descriptor codec;
    cellshard::series_layout_view layout;
    cellshard::series_dataset_table_view dataset_view;
    cellshard::series_provenance_view provenance_view;
    owned_text_column dataset_ids = make_column({"sample_a"});
    owned_text_column matrix_paths = make_column({matrix_path.c_str()});
    owned_text_column feature_paths = make_column({feature_path.c_str()});
    owned_text_column barcode_paths = make_column({barcode_path.c_str()});
    owned_text_column metadata_paths = make_column({metadata_path.c_str()});
    owned_text_column global_barcodes = make_column({"bc0", "bc1"});
    owned_text_column feature_ids = make_column({"g0", "g1", "g2"});
    owned_text_column feature_names = make_column({"MT-CO1", "GeneB", "GeneC"});
    owned_text_column feature_types = make_column({"gene", "gene", "gene"});
    std::vector<std::uint32_t> dataset_formats = { (std::uint32_t) cellerator::ingest::series::source_mtx };
    std::vector<std::uint64_t> dataset_row_begin = { 0u };
    std::vector<std::uint64_t> dataset_row_end = { 2u };
    std::vector<std::uint64_t> dataset_rows = { 2u };
    std::vector<std::uint64_t> dataset_cols = { 3u };
    std::vector<std::uint64_t> dataset_nnz = { 3u };
    std::vector<std::uint64_t> part_rows = { 2u };
    std::vector<std::uint64_t> part_nnz = { 3u };
    std::vector<std::uint32_t> part_axes = { (std::uint32_t) cellshard::sparse::compressed_by_row };
    std::vector<std::uint64_t> part_row_offsets = { 0u, 2u };
    std::vector<std::uint32_t> part_dataset_ids = { 0u };
    std::vector<std::uint32_t> part_codec_ids = { 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u };
    std::vector<std::uint32_t> cell_dataset_ids = { 0u, 0u };
    std::vector<std::uint64_t> cell_local_indices = { 0u, 1u };
    std::vector<std::uint32_t> feature_dataset_ids = { 0u, 0u, 0u };
    std::vector<std::uint64_t> feature_local_indices = { 0u, 1u, 2u };
    std::vector<std::uint64_t> dataset_feature_offsets = { 0u, 3u };
    std::vector<std::uint32_t> dataset_feature_to_global = { 0u, 1u, 2u };

    std::remove(manifest_path.c_str());
    std::remove(matrix_path.c_str());
    std::remove(feature_path.c_str());
    std::remove(barcode_path.c_str());
    std::remove(metadata_path.c_str());
    std::remove(series_path.c_str());

    if (!write_file(matrix_path,
                    "%%MatrixMarket matrix coordinate integer general\n"
                    "2 3 3\n"
                    "1 1 5\n"
                    "1 3 1\n"
                    "2 2 7\n")) return 1;
    if (!write_file(feature_path, "g0\tMT-CO1\tgene\ng1\tGeneB\tgene\ng2\tGeneC\tgene\n")) return 1;
    if (!write_file(barcode_path, "bc0\nbc1\n")) return 1;
    if (!write_file(metadata_path, "stage\tbatch\nE8\tA\nE9\tA\n")) return 1;
    if (!write_file(manifest_path,
                    ("dataset\tpath\tformat\tfeatures\tbarcodes\tmetadata\trows\tcols\tnnz\n"
                     "sample_a\t" + matrix_path + "\tmtx\t" + feature_path + "\t" + barcode_path + "\t" + metadata_path + "\t2\t3\t3\n").c_str())) return 1;

    wb::manifest_inspection inspection = wb::inspect_manifest_tsv(manifest_path);
    if (!inspection.ok || inspection.sources.size() != 1u) return 1;
    if (inspection.sources[0].rows != 2u || inspection.sources[0].feature_count != 3u || inspection.sources[0].barcode_count != 2u) return 1;

    wb::ingest_policy policy;
    policy.max_part_nnz = 2u;
    policy.max_window_bytes = 256u << 20u;
    policy.output_path = series_path;
    wb::ingest_plan plan = wb::plan_series_ingest(inspection.sources, policy);
    if (!plan.ok || plan.parts.size() != 2u || plan.shards.empty()) return 1;

    cellshard::sparse::init(&part);
    if (!populate_part(&part, 2u, 3u, {0u, 2u, 3u}, {0u, 2u, 1u}, {5.0f, 1.0f, 7.0f})) return 1;

    codec.codec_id = 0u;
    codec.family = cellshard::series_codec_family_standard_csr;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 2u;
    layout.cols = 3u;
    layout.nnz = 3u;
    layout.num_parts = 1u;
    layout.num_shards = 1u;
    layout.part_rows = part_rows.data();
    layout.part_nnz = part_nnz.data();
    layout.part_axes = part_axes.data();
    layout.part_row_offsets = part_row_offsets.data();
    layout.part_dataset_ids = part_dataset_ids.data();
    layout.part_codec_ids = part_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    dataset_view.count = 1u;
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

    if (!cellshard::create_series_compressed_h5(series_path.c_str(), &layout, &dataset_view, &provenance_view)) return 1;
    if (!cellshard::append_standard_csr_part_h5(series_path.c_str(), 0u, &part)) return 1;

    wb::series_summary summary = wb::summarize_series_csh5(series_path);
    if (!summary.ok) return 1;
    if (summary.rows != 2u || summary.cols != 3u || summary.datasets.size() != 1u || summary.parts.size() != 1u) return 1;
    if (summary.feature_names.size() != 3u || summary.feature_names[0] != "MT-CO1") return 1;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        wb::preprocess_config preprocess;
        preprocess.device = 0;
        preprocess.min_counts = 0.0f;
        preprocess.min_genes = 0u;
        preprocess.max_mito_fraction = 1.0f;
        preprocess.min_gene_sum = 0.0f;
        preprocess.min_detected_cells = 0.0f;
        preprocess.min_variance = 0.0f;
        wb::preprocess_summary preprocess_summary = wb::run_preprocess_pass(series_path, preprocess);
        if (!preprocess_summary.ok) return 1;
        if (preprocess_summary.parts_processed != 1u || preprocess_summary.kept_genes == 0u) return 1;
    }

    cellshard::sparse::clear(&part);
    std::remove(manifest_path.c_str());
    std::remove(matrix_path.c_str());
    std::remove(feature_path.c_str());
    std::remove(barcode_path.c_str());
    std::remove(metadata_path.c_str());
    std::remove(series_path.c_str());
    return 0;
}
