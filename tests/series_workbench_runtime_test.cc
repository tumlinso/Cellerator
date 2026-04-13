#include "../src/workbench/series_workbench.hh"

#include "../extern/CellShard/src/CellShard.hh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace wb = ::cellerator::apps::workbench;
namespace fs = std::filesystem;

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
    const std::string converted_series_path = base + ".converted.series.csh5";
    const std::string builder_dir = base + ".builder";
    const std::string exported_manifest_path = base + ".builder.manifest.tsv";
    cellshard::sparse::compressed part;
    cellshard::series_codec_descriptor codec;
    cellshard::series_layout_view layout{};
    cellshard::series_dataset_table_view dataset_view{};
    cellshard::series_provenance_view provenance_view{};
    cellshard::series_metadata_table_view metadata_table_view{};
    cellshard::series_embedded_metadata_view embedded_metadata_view{};
    cellshard::series_observation_metadata_view observation_metadata_view{};
    cellshard::series_browse_cache_view browse_view{};
    owned_text_column dataset_ids = make_column({"sample_a"});
    owned_text_column matrix_paths = make_column({matrix_path.c_str()});
    owned_text_column feature_paths = make_column({feature_path.c_str()});
    owned_text_column barcode_paths = make_column({barcode_path.c_str()});
    owned_text_column metadata_paths = make_column({metadata_path.c_str()});
    owned_text_column global_barcodes = make_column({"bc0", "bc1"});
    owned_text_column feature_ids = make_column({"g0", "g1", "g2"});
    owned_text_column feature_names = make_column({"MT-CO1", "GeneB", "GeneC"});
    owned_text_column feature_types = make_column({"gene", "gene", "gene"});
    owned_text_column metadata_column_names = make_column({"day", "embryo_id", "cell_id"});
    owned_text_column metadata_field_values = make_column({"E8.5", "embryo_1", "bc0", "P0", "embryo_1", "bc1"});
    owned_observation_metadata_column obs_day;
    owned_observation_metadata_column obs_embryo;
    owned_observation_metadata_column obs_cell;
    owned_observation_metadata_column obs_day_label;
    owned_observation_metadata_column obs_day_numeric;
    owned_observation_metadata_column obs_postnatal;
    std::vector<cellshard::series_observation_metadata_column_view> observation_columns;
    std::vector<std::uint32_t> dataset_formats = { (std::uint32_t) cellerator::ingest::series::source_mtx };
    std::vector<std::uint64_t> dataset_row_begin = { 0u };
    std::vector<std::uint64_t> dataset_row_end = { 2u };
    std::vector<std::uint64_t> dataset_rows = { 2u };
    std::vector<std::uint64_t> dataset_cols = { 3u };
    std::vector<std::uint64_t> dataset_nnz = { 3u };
    std::vector<std::uint64_t> partition_rows = { 2u };
    std::vector<std::uint64_t> partition_nnz = { 3u };
    std::vector<std::uint32_t> partition_axes = { (std::uint32_t) cellshard::sparse::compressed_by_row };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u };
    std::vector<std::uint32_t> cell_dataset_ids = { 0u, 0u };
    std::vector<std::uint64_t> cell_local_indices = { 0u, 1u };
    std::vector<std::uint32_t> feature_dataset_ids = { 0u, 0u, 0u };
    std::vector<std::uint64_t> feature_local_indices = { 0u, 1u, 2u };
    std::vector<std::uint64_t> dataset_feature_offsets = { 0u, 3u };
    std::vector<std::uint32_t> dataset_feature_to_global = { 0u, 1u, 2u };
    std::vector<std::uint32_t> metadata_row_offsets = { 0u, 3u, 6u };
    std::vector<std::uint32_t> metadata_dataset_indices = { 0u };
    std::vector<std::uint64_t> metadata_global_row_begin = { 0u };
    std::vector<std::uint64_t> metadata_global_row_end = { 2u };
    std::vector<std::uint32_t> browse_feature_indices = { 0u, 1u };
    std::vector<float> browse_gene_sum = { 5.0f, 7.0f };
    std::vector<float> browse_gene_detected = { 1.0f, 1.0f };
    std::vector<float> browse_gene_sq_sum = { 25.0f, 49.0f };
    std::vector<float> browse_dataset_mean = { 2.5f, 3.5f };
    std::vector<float> browse_shard_mean = { 2.5f, 3.5f };
    std::vector<std::uint32_t> browse_part_sample_offsets = { 0u, 2u };
    std::vector<std::uint64_t> browse_part_sample_rows = { 0u, 1u };
    std::vector<float> browse_partition_sample_values = { 5.0f, 0.0f, 0.0f, 7.0f };

    std::remove(manifest_path.c_str());
    std::remove(matrix_path.c_str());
    std::remove(feature_path.c_str());
    std::remove(barcode_path.c_str());
    std::remove(metadata_path.c_str());
    std::remove(series_path.c_str());
    std::remove(converted_series_path.c_str());
    std::remove(exported_manifest_path.c_str());
    std::error_code dir_ec;
    fs::remove_all(builder_dir, dir_ec);
    fs::create_directories(builder_dir, dir_ec);
    if (dir_ec) return 1;

    if (!write_file(matrix_path,
                    "%%MatrixMarket matrix coordinate integer general\n"
                    "2 3 3\n"
                    "1 1 5\n"
                    "1 3 1\n"
                    "2 2 7\n")) return 1;
    if (!write_file(feature_path, "g0\tMT-CO1\tgene\ng1\tGeneB\tgene\ng2\tGeneC\tgene\n")) return 1;
    if (!write_file(barcode_path, "bc0\nbc1\n")) return 1;
    if (!write_file(metadata_path, "day\tembryo_id\tcell_id\nE8.5\tembryo_1\tbc0\nP0\tembryo_1\tbc1\n")) return 1;
    if (!write_file(manifest_path,
                    ("dataset\tpath\tformat\tfeatures\tbarcodes\tmetadata\trows\tcols\tnnz\n"
                     "sample_a\t" + matrix_path + "\tmtx\t" + feature_path + "\t" + barcode_path + "\t" + metadata_path + "\t2\t3\t3\n").c_str())) return 1;
    if (!write_file(builder_dir + "/matrix.mtx",
                    "%%MatrixMarket matrix coordinate integer general\n"
                    "2 3 3\n"
                    "1 1 5\n"
                    "1 3 1\n"
                    "2 2 7\n")) return 1;
    if (!write_file(builder_dir + "/features.tsv", "g0\tMT-CO1\tgene\ng1\tGeneB\tgene\ng2\tGeneC\tgene\n")) return 1;
    if (!write_file(builder_dir + "/barcodes.tsv", "bc0\nbc1\n")) return 1;
    if (!write_file(builder_dir + "/metadata.tsv", "day\tembryo_id\tcell_id\nE8.5\tembryo_1\tbc0\nP0\tembryo_1\tbc1\n")) return 1;

    wb::manifest_inspection inspection = wb::inspect_manifest_tsv(manifest_path);
    if (!inspection.ok || inspection.sources.size() != 1u) return 1;
    if (inspection.sources[0].rows != 2u || inspection.sources[0].feature_count != 3u || inspection.sources[0].barcode_count != 2u) return 1;

    std::vector<wb::issue> builder_issues;
    const std::vector<wb::draft_dataset> drafts = wb::discover_dataset_drafts(builder_dir, &builder_issues);
    if (drafts.size() != 1u) return 1;
    if (drafts[0].matrix_path != builder_dir + "/matrix.mtx") return 1;
    if (drafts[0].feature_path != builder_dir + "/features.tsv") return 1;
    if (drafts[0].barcode_path != builder_dir + "/barcodes.tsv") return 1;
    if (drafts[0].metadata_path != builder_dir + "/metadata.tsv") return 1;

    const std::vector<wb::source_entry> builder_sources = wb::sources_from_dataset_drafts(drafts);
    if (builder_sources.size() != 1u) return 1;
    wb::manifest_inspection builder_inspection = wb::inspect_source_entries(builder_sources, "<builder>");
    if (!builder_inspection.ok || builder_inspection.sources.size() != 1u) return 1;
    if (builder_inspection.sources[0].rows != 2u || builder_inspection.sources[0].cols != 3u || builder_inspection.sources[0].nnz != 3u) return 1;

    std::vector<wb::issue> export_issues;
    if (!wb::export_manifest_tsv(exported_manifest_path, drafts, (std::size_t) 8u << 20u, &export_issues)) return 1;
    wb::manifest_inspection exported_inspection = wb::inspect_manifest_tsv(exported_manifest_path);
    if (!exported_inspection.ok || exported_inspection.sources.size() != 1u) return 1;
    if (exported_inspection.sources[0].matrix_path != builder_dir + "/matrix.mtx") return 1;

    wb::ingest_policy policy;
    policy.max_part_nnz = 2u;
    policy.max_window_bytes = 256u << 20u;
    policy.output_path = series_path;
    wb::ingest_plan plan = wb::plan_series_ingest(inspection.sources, policy);
    if (!plan.ok || plan.parts.size() != 2u || plan.shards.empty()) return 1;
    if (plan.parts[0].execution_bytes == 0u || plan.shards[0].execution_bytes == 0u) return 1;
    if (plan.parts[0].preferred_format == wb::execution_format::unknown) return 1;

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
    layout.num_partitions = 1u;
    layout.num_shards = 1u;
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

    metadata_table_view.rows = 2u;
    metadata_table_view.cols = 2u;
    metadata_table_view.column_names = metadata_column_names.view();
    metadata_table_view.field_values = metadata_field_values.view();
    metadata_table_view.row_offsets = metadata_row_offsets.data();

    embedded_metadata_view.count = 1u;
    embedded_metadata_view.dataset_indices = metadata_dataset_indices.data();
    embedded_metadata_view.global_row_begin = metadata_global_row_begin.data();
    embedded_metadata_view.global_row_end = metadata_global_row_end.data();
    embedded_metadata_view.tables = &metadata_table_view;

    obs_day.name = "day";
    obs_day.type = cellshard::series_observation_metadata_type_text;
    obs_day.text_values = make_column({"E8.5", "P0"});
    obs_embryo.name = "embryo_id";
    obs_embryo.type = cellshard::series_observation_metadata_type_text;
    obs_embryo.text_values = make_column({"embryo_1", "embryo_1"});
    obs_cell.name = "cell_id";
    obs_cell.type = cellshard::series_observation_metadata_type_text;
    obs_cell.text_values = make_column({"bc0", "bc1"});
    obs_day_label.name = "embryonic_day_label";
    obs_day_label.type = cellshard::series_observation_metadata_type_text;
    obs_day_label.text_values = make_column({"E8.5", "P0"});
    obs_day_numeric.name = "embryonic_day";
    obs_day_numeric.type = cellshard::series_observation_metadata_type_float32;
    obs_day_numeric.float32_values = {8.5f, std::numeric_limits<float>::quiet_NaN()};
    obs_postnatal.name = "is_postnatal";
    obs_postnatal.type = cellshard::series_observation_metadata_type_uint8;
    obs_postnatal.uint8_values = {0u, 1u};
    observation_columns = {
        obs_day.view(),
        obs_embryo.view(),
        obs_cell.view(),
        obs_day_label.view(),
        obs_day_numeric.view(),
        obs_postnatal.view()
    };
    observation_metadata_view.rows = 2u;
    observation_metadata_view.cols = (std::uint32_t) observation_columns.size();
    observation_metadata_view.columns = observation_columns.data();

    browse_view.selected_feature_count = 2u;
    browse_view.selected_feature_indices = browse_feature_indices.data();
    browse_view.gene_sum = browse_gene_sum.data();
    browse_view.gene_detected = browse_gene_detected.data();
    browse_view.gene_sq_sum = browse_gene_sq_sum.data();
    browse_view.dataset_count = 1u;
    browse_view.dataset_feature_mean = browse_dataset_mean.data();
    browse_view.shard_count = 1u;
    browse_view.shard_feature_mean = browse_shard_mean.data();
    browse_view.partition_count = 1u;
    browse_view.sample_rows_per_partition = 2u;
    browse_view.partition_sample_row_offsets = browse_part_sample_offsets.data();
    browse_view.partition_sample_global_rows = browse_part_sample_rows.data();
    browse_view.partition_sample_values = browse_partition_sample_values.data();

    if (!cellshard::create_series_compressed_h5(series_path.c_str(), &layout, &dataset_view, &provenance_view)) return 1;
    if (!cellshard::append_standard_csr_partition_h5(series_path.c_str(), 0u, &part)) return 1;
    if (!cellshard::append_series_embedded_metadata_h5(series_path.c_str(), &embedded_metadata_view)) return 1;
    if (!cellshard::append_series_observation_metadata_h5(series_path.c_str(), &observation_metadata_view)) return 1;
    if (!cellshard::append_series_browse_cache_h5(series_path.c_str(), &browse_view)) return 1;

    wb::series_summary summary = wb::summarize_series_csh5(series_path);
    if (!summary.ok) return 1;
    if (summary.rows != 2u || summary.cols != 3u || summary.datasets.size() != 1u || summary.partitions.size() != 1u) return 1;
    if (summary.feature_names.size() != 3u || summary.feature_names[0] != "MT-CO1") return 1;
    if (summary.embedded_metadata.size() != 1u) return 1;
    if (summary.embedded_metadata[0].column_names.size() != 3u || summary.embedded_metadata[0].column_names[0] != "day") return 1;
    wb::embedded_metadata_table loaded_metadata = wb::load_embedded_metadata_table(series_path, 0u);
    if (!loaded_metadata.available) return 1;
    if (loaded_metadata.field_values.size() != 6u || loaded_metadata.field_values[0] != "E8.5" || loaded_metadata.field_values[3] != "P0") return 1;
    if (loaded_metadata.row_offsets.size() != 3u || loaded_metadata.row_offsets[1] != 3u) return 1;
    if (!summary.observation_metadata.available || summary.observation_metadata.rows != 2u || summary.observation_metadata.cols != 6u) return 1;
    if (summary.observation_metadata.columns.size() != 6u || summary.observation_metadata.columns[3].name != "embryonic_day_label") return 1;
    wb::observation_metadata_table loaded_observation = wb::load_observation_metadata_table(series_path);
    if (!loaded_observation.available || loaded_observation.columns.size() != 6u) return 1;
    if (loaded_observation.columns[0].text_values.size() != 2u || loaded_observation.columns[0].text_values[0] != "E8.5") return 1;
    if (loaded_observation.columns[3].text_values[1] != "P0") return 1;
    if (loaded_observation.columns[4].float32_values.size() != 2u || loaded_observation.columns[4].float32_values[0] != 8.5f) return 1;
    if (!std::isnan(loaded_observation.columns[4].float32_values[1])) return 1;
    if (loaded_observation.columns[5].uint8_values.size() != 2u || loaded_observation.columns[5].uint8_values[1] != 1u) return 1;
    if (!summary.browse.available || summary.browse.selected_feature_indices.size() != 2u) return 1;
    if (summary.browse.selected_feature_names.size() != 2u || summary.browse.selected_feature_names[0] != "MT-CO1") return 1;

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
        if (preprocess_summary.partitions_processed != 1u || preprocess_summary.kept_genes == 0u) return 1;
    }

    if (device_count >= 4) {
        policy.output_path = converted_series_path;
        policy.embed_metadata = true;
        policy.build_browse_cache = true;
        policy.browse_top_features = 2u;
        policy.browse_sample_rows_per_partition = 2u;
        wb::ingest_plan converted_plan = wb::plan_series_ingest(inspection.sources, policy);
        if (!converted_plan.ok) {
            std::fprintf(stderr, "converted_plan not ok\n");
            return 1;
        }
        wb::conversion_report report = wb::convert_plan_to_series_csh5(converted_plan);
        if (!report.ok) {
            std::fprintf(stderr, "convert_plan_to_series_csh5 failed\n");
            for (const wb::issue &entry : report.issues) {
                std::fprintf(stderr, "%s %s: %s\n",
                             wb::severity_name(entry.severity).c_str(),
                             entry.scope.c_str(),
                             entry.message.c_str());
            }
            return 1;
        }
        wb::series_summary converted_summary = wb::summarize_series_csh5(converted_series_path);
        if (!converted_summary.ok) {
            std::fprintf(stderr, "converted summary not ok\n");
            return 1;
        }
        if (converted_summary.embedded_metadata.empty()) {
            std::fprintf(stderr, "converted summary missing embedded metadata\n");
            return 1;
        }
        if (!converted_summary.observation_metadata.available || converted_summary.observation_metadata.cols < 3u) {
            std::fprintf(stderr, "converted summary missing observation metadata\n");
            return 1;
        }
        if (!converted_summary.browse.available || converted_summary.browse.selected_feature_indices.size() != 2u) {
            std::fprintf(stderr, "converted summary missing browse cache\n");
            return 1;
        }
        if (converted_summary.partitions.empty() || converted_summary.partitions[0].execution_format == 0u) {
            std::fprintf(stderr, "converted summary missing execution metadata\n");
            return 1;
        }
    }

    cellshard::sparse::clear(&part);
    std::remove(manifest_path.c_str());
    std::remove(matrix_path.c_str());
    std::remove(feature_path.c_str());
    std::remove(barcode_path.c_str());
    std::remove(metadata_path.c_str());
    std::remove(series_path.c_str());
    std::remove(converted_series_path.c_str());
    return 0;
}
