#include <Cellerator/compute/preprocess.cuh>
#include <Cellerator/workbench/dataset_workbench.hh>

#include "../extern/CellShard/include/CellShard/CellShard.hh"

#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace wb = ::cellerator::apps::workbench;
namespace cpre = ::cellerator::compute::preprocess;
namespace fs = std::filesystem;

static_assert(cpre::classify_backend(cpre::preprocess_operator_cell_metrics,
                                     static_cast<const cellshard::device::blocked_ell_view *>(nullptr)).backend
                  == cpre::preprocess_backend_custom_kernel,
              "blocked-ell cell metrics should stay on the custom-kernel path");
static_assert(cpre::classify_backend(cpre::preprocess_operator_gene_metrics,
                                     static_cast<const cellshard::device::blocked_ell_view *>(nullptr)).layout
                  == cpre::preprocess_layout_native_blocked_ell,
              "blocked-ell gene metrics should report a native blocked-ell path");
static_assert(cpre::classify_backend(cpre::preprocess_operator_gene_metrics,
                                     static_cast<const cellshard::device::compressed_view *>(nullptr)).backend
                  == cpre::preprocess_backend_cusparse_spmv,
              "compressed gene metrics should remain the cuSPARSE SpMV temporary-analysis path");
static_assert(cpre::classify_backend(cpre::preprocess_operator_normalize_log1p,
                                     static_cast<const cellshard::device::compressed_view *>(nullptr)).layout
                  == cpre::preprocess_layout_temporary_csr_analysis,
              "compressed normalize/log1p should stay classified as a CSR temporary-analysis path");

namespace {

#define CHECK_OR_FAIL(cond, message) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "%s\n", message); \
            return 1; \
        } \
    } while (0)

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    cellshard::dataset_text_column_view view() const {
        cellshard::dataset_text_column_view out;
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
    std::uint32_t type = cellshard::dataset_observation_metadata_type_none;
    owned_text_column text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;

    cellshard::dataset_observation_metadata_column_view view() const {
        cellshard::dataset_observation_metadata_column_view out{};
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

hid_t make_vlen_string_type() {
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, H5T_VARIABLE);
    return type;
}

bool write_string_attr(hid_t obj, const char *name, const char *value) {
    hid_t type = make_vlen_string_type();
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    const char *src = value;
    const bool ok = attr >= 0 && H5Awrite(attr, type, &src) >= 0;
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    return ok;
}

bool write_u64_attr_1d(hid_t obj, const char *name, const std::vector<std::uint64_t> &values) {
    hsize_t dims[1] = {values.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
    const bool ok = attr >= 0 && (values.empty() || H5Awrite(attr, H5T_NATIVE_UINT64, values.data()) >= 0);
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    return ok;
}

template<typename T>
bool write_numeric_dataset(hid_t parent, const char *name, hid_t dtype, const std::vector<T> &values) {
    hsize_t dims[1] = {values.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(parent, name, dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const bool ok = dset >= 0 && (values.empty() || H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    if (dset >= 0) H5Dclose(dset);
    if (space >= 0) H5Sclose(space);
    return ok;
}

bool write_string_dataset(hid_t parent, const char *name, const std::vector<const char *> &values) {
    hsize_t dims[1] = {values.size()};
    hid_t type = make_vlen_string_type();
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const bool ok = dset >= 0 && (values.empty() || H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    if (dset >= 0) H5Dclose(dset);
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    return ok;
}

bool write_csr_group(hid_t parent,
                     const char *name,
                     std::uint64_t rows,
                     std::uint64_t cols,
                     const std::vector<std::uint64_t> &indptr,
                     const std::vector<std::uint64_t> &indices,
                     const std::vector<float> &values) {
    hid_t group = H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const bool ok = group >= 0
        && write_string_attr(group, "encoding-type", "csr_matrix")
        && write_u64_attr_1d(group, "shape", {rows, cols})
        && write_numeric_dataset(group, "indptr", H5T_NATIVE_UINT64, indptr)
        && write_numeric_dataset(group, "indices", H5T_NATIVE_UINT64, indices)
        && write_numeric_dataset(group, "data", H5T_NATIVE_FLOAT, values);
    if (group >= 0) H5Gclose(group);
    return ok;
}

bool write_csc_group(hid_t parent,
                     const char *name,
                     std::uint64_t rows,
                     std::uint64_t cols,
                     const std::vector<std::uint64_t> &indptr,
                     const std::vector<std::uint64_t> &indices,
                     const std::vector<float> &values) {
    hid_t group = H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const bool ok = group >= 0
        && write_string_attr(group, "encoding-type", "csc_matrix")
        && write_u64_attr_1d(group, "shape", {rows, cols})
        && write_numeric_dataset(group, "indptr", H5T_NATIVE_UINT64, indptr)
        && write_numeric_dataset(group, "indices", H5T_NATIVE_UINT64, indices)
        && write_numeric_dataset(group, "data", H5T_NATIVE_FLOAT, values);
    if (group >= 0) H5Gclose(group);
    return ok;
}

bool write_minimal_h5ad(const std::string &path) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t obs = (hid_t) -1;
    hid_t var = (hid_t) -1;
    hid_t raw = (hid_t) -1;
    hid_t raw_var = (hid_t) -1;
    hid_t layers = (hid_t) -1;
    bool ok = false;

    if (file < 0) return false;
    if (!write_string_attr(file, "encoding-type", "anndata")) goto done;
    if (!write_string_attr(file, "encoding-version", "0.1.0")) goto done;
    if (!write_csr_group(file,
                         "X",
                         2u,
                         3u,
                         {0u, 2u, 3u},
                         {0u, 2u, 1u},
                         {0.5f, 1.25f, 2.0f})) goto done;

    raw = H5Gcreate2(file, "raw", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (raw < 0) goto done;
    if (!write_csr_group(raw,
                         "X",
                         2u,
                         3u,
                         {0u, 2u, 3u},
                         {0u, 2u, 1u},
                         {5.0f, 1.0f, 7.0f})) goto done;
    raw_var = H5Gcreate2(raw, "var", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (raw_var < 0) goto done;
    if (!write_string_attr(raw_var, "_index", "gene_id")) goto done;
    if (!write_string_dataset(raw_var, "gene_id", {"g0", "g1", "g2"})) goto done;
    if (!write_string_dataset(raw_var, "gene_name", {"MT-CO1", "GeneB", "GeneC"})) goto done;
    if (!write_string_dataset(raw_var, "feature_type", {"gene", "gene", "gene"})) goto done;

    layers = H5Gcreate2(file, "layers", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (layers < 0) goto done;
    if (!write_csr_group(layers,
                         "counts",
                         2u,
                         3u,
                         {0u, 2u, 3u},
                         {0u, 2u, 1u},
                         {5.0f, 1.0f, 7.0f})) goto done;
    if (!write_csc_group(layers,
                         "counts_csc",
                         2u,
                         3u,
                         {0u, 1u, 2u, 3u},
                         {0u, 1u, 0u},
                         {5.0f, 7.0f, 1.0f})) goto done;

    obs = H5Gcreate2(file, "obs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (obs < 0) goto done;
    if (!write_string_attr(obs, "_index", "_index")) goto done;
    if (!write_string_dataset(obs, "_index", {"bc0", "bc1"})) goto done;
    if (!write_string_dataset(obs, "batch", {"batch0", "batch1"})) goto done;
    if (!write_numeric_dataset(obs, "quality", H5T_NATIVE_FLOAT, std::vector<float>{1.0f, 2.5f})) goto done;

    var = H5Gcreate2(file, "var", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (var < 0) goto done;
    if (!write_string_attr(var, "_index", "gene_id")) goto done;
    if (!write_string_dataset(var, "gene_id", {"g0", "g1", "g2"})) goto done;
    if (!write_string_dataset(var, "gene_name", {"MT-CO1", "GeneB", "GeneC"})) goto done;
    if (!write_string_dataset(var, "feature_type", {"gene", "gene", "gene"})) goto done;

    ok = true;

done:
    if (var >= 0) H5Gclose(var);
    if (obs >= 0) H5Gclose(obs);
    if (layers >= 0) H5Gclose(layers);
    if (raw_var >= 0) H5Gclose(raw_var);
    if (raw >= 0) H5Gclose(raw);
    if (file >= 0) H5Fclose(file);
    return ok;
}

bool populate_blocked_ell_part(cellshard::sparse::blocked_ell *part,
                               unsigned int rows,
                               unsigned int cols,
                               unsigned int block_size,
                               unsigned int ell_cols,
                               const std::vector<unsigned int> &block_idx,
                               const std::vector<float> &values) {
    cellshard::sparse::init(part,
                            rows,
                            cols,
                            (cellshard::types::nnz_t) values.size(),
                            block_size,
                            ell_cols);
    if (!cellshard::sparse::allocate(part)) return false;
    std::memcpy(part->blockColIdx, block_idx.data(), block_idx.size() * sizeof(unsigned int));
    for (std::size_t i = 0; i < values.size(); ++i) part->val[i] = __float2half(values[i]);
    return true;
}

} // namespace

int main() {
    const std::string base = "/tmp/cellerator_dataset_workbench";
    const std::string manifest_path = base + ".manifest.tsv";
    const std::string matrix_path = base + ".matrix.mtx";
    const std::string feature_path = base + ".features.tsv";
    const std::string barcode_path = base + ".barcodes.tsv";
    const std::string metadata_path = base + ".metadata.tsv";
    const std::string dataset_path = base + ".dataset.csh5";
    const std::string converted_dataset_path = base + ".converted.dataset.csh5";
    const std::string converted_cache_root = base + ".converted.cache";
    const std::string converted_filtered_output_path = base + ".converted.filtered.dataset.csh5";
    const std::string finalized_dataset_path = base + ".finalized.dataset.csh5";
    const std::string finalized_cache_root = base + ".finalized.cache";
    const std::string preprocess_cache_root = base + ".preprocess.cache";
    const std::string derived_dataset_path = base + ".derived.dataset.csh5";
    const std::string derived_cache_root = base + ".derived.cache";
    const std::string builder_dir = base + ".builder";
    const std::string exported_manifest_path = base + ".builder.manifest.tsv";
    const std::string h5ad_path = base + ".input.h5ad";
    const std::string h5ad_manifest_path = base + ".h5ad.manifest.tsv";
    const std::string h5ad_dataset_path = base + ".h5ad.dataset.csh5";
    const std::string h5ad_cache_root = base + ".h5ad.cache";
    cellshard::sparse::blocked_ell part;
    cellshard::dataset_codec_descriptor codec;
    cellshard::dataset_layout_view layout{};
    cellshard::dataset_dataset_table_view dataset_view{};
    cellshard::dataset_provenance_view provenance_view{};
    cellshard::dataset_metadata_table_view metadata_table_view{};
    cellshard::dataset_embedded_metadata_view embedded_metadata_view{};
    cellshard::dataset_observation_metadata_view observation_metadata_view{};
    cellshard::dataset_feature_metadata_view feature_metadata_view{};
    cellshard::dataset_browse_cache_view browse_view{};
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
    std::vector<cellshard::dataset_observation_metadata_column_view> observation_columns;
    owned_observation_metadata_column feature_short_name;
    std::vector<cellshard::dataset_observation_metadata_column_view> feature_columns;
    std::vector<std::uint32_t> dataset_formats = { (std::uint32_t) cellerator::ingest::dataset::source_mtx };
    std::vector<std::uint64_t> dataset_row_begin = { 0u };
    std::vector<std::uint64_t> dataset_row_end = { 2u };
    std::vector<std::uint64_t> dataset_rows = { 2u };
    std::vector<std::uint64_t> dataset_cols = { 3u };
    std::vector<std::uint64_t> dataset_nnz = { 6u };
    std::vector<std::uint64_t> partition_rows = { 2u };
    std::vector<std::uint64_t> partition_nnz = { 6u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(1u, 3ul)
    };
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
    std::remove(dataset_path.c_str());
    std::remove(converted_dataset_path.c_str());
    std::remove(converted_filtered_output_path.c_str());
    std::remove(finalized_dataset_path.c_str());
    std::remove(derived_dataset_path.c_str());
    std::remove(exported_manifest_path.c_str());
    std::remove(h5ad_path.c_str());
    std::remove(h5ad_manifest_path.c_str());
    std::remove(h5ad_dataset_path.c_str());
    std::error_code dir_ec;
    fs::remove_all(builder_dir, dir_ec);
    fs::remove_all(converted_cache_root, dir_ec);
    fs::remove_all(finalized_cache_root, dir_ec);
    fs::remove_all(derived_cache_root, dir_ec);
    fs::remove_all(h5ad_cache_root, dir_ec);
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
    if (!write_minimal_h5ad(h5ad_path)) return 1;

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
    policy.convert_window_bytes = 256u << 20u;
    policy.target_shard_bytes = 256u << 20u;
    policy.output_path = dataset_path;
    wb::ingest_plan plan = wb::plan_dataset_ingest(inspection.sources, policy);
    if (!plan.ok || plan.parts.size() != 2u || plan.shards.empty()) return 1;
    if (plan.parts[0].execution_bytes == 0u || plan.shards[0].execution_bytes == 0u) return 1;
    if (plan.parts[0].preferred_format == wb::execution_format::unknown) return 1;

    cellshard::sparse::init(&part);
    if (!populate_blocked_ell_part(&part,
                                   2u,
                                   3u,
                                   1u,
                                   3u,
                                   {0u, 1u, 2u, 0u, 1u, 2u},
                                   {5.0f, 9.0f, 1.0f, 8.0f, 7.0f, 6.0f})) {
        return 1;
    }

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 2u;
    layout.cols = 3u;
    layout.nnz = 6u;
    layout.num_partitions = 1u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = nullptr;
    layout.partition_aux = partition_aux.data();
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
    metadata_table_view.cols = 3u;
    metadata_table_view.column_names = metadata_column_names.view();
    metadata_table_view.field_values = metadata_field_values.view();
    metadata_table_view.row_offsets = metadata_row_offsets.data();

    embedded_metadata_view.count = 1u;
    embedded_metadata_view.dataset_indices = metadata_dataset_indices.data();
    embedded_metadata_view.global_row_begin = metadata_global_row_begin.data();
    embedded_metadata_view.global_row_end = metadata_global_row_end.data();
    embedded_metadata_view.tables = &metadata_table_view;

    obs_day.name = "day";
    obs_day.type = cellshard::dataset_observation_metadata_type_text;
    obs_day.text_values = make_column({"E8.5", "P0"});
    obs_embryo.name = "embryo_id";
    obs_embryo.type = cellshard::dataset_observation_metadata_type_text;
    obs_embryo.text_values = make_column({"embryo_1", "embryo_1"});
    obs_cell.name = "cell_id";
    obs_cell.type = cellshard::dataset_observation_metadata_type_text;
    obs_cell.text_values = make_column({"bc0", "bc1"});
    obs_day_label.name = "embryonic_day_label";
    obs_day_label.type = cellshard::dataset_observation_metadata_type_text;
    obs_day_label.text_values = make_column({"E8.5", "P0"});
    obs_day_numeric.name = "embryonic_day";
    obs_day_numeric.type = cellshard::dataset_observation_metadata_type_float32;
    obs_day_numeric.float32_values = {8.5f, std::numeric_limits<float>::quiet_NaN()};
    obs_postnatal.name = "is_postnatal";
    obs_postnatal.type = cellshard::dataset_observation_metadata_type_uint8;
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

    feature_short_name.name = "gene_short_name";
    feature_short_name.type = cellshard::dataset_observation_metadata_type_text;
    feature_short_name.text_values = make_column({"MT-CO1", "GeneB", "GeneC"});
    feature_columns = {
        feature_short_name.view()
    };
    feature_metadata_view.cols = 3u;
    feature_metadata_view.annotation_count = (std::uint32_t) feature_columns.size();
    feature_metadata_view.annotations = feature_columns.data();

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

    if (!cellshard::create_dataset_blocked_ell_h5(dataset_path.c_str(), &layout, &dataset_view, &provenance_view)) return 1;
    {
        cellshard::bucketed_blocked_ell_shard optimized_shard;
        cellshard::bucketed_blocked_ell_partition bucketed_part;
        cellshard::init(&optimized_shard);
        cellshard::init(&bucketed_part);
        optimized_shard.partition_count = 1u;
        optimized_shard.rows = part.rows;
        optimized_shard.cols = part.cols;
        optimized_shard.nnz = part.nnz;
        optimized_shard.partitions =
            (cellshard::bucketed_blocked_ell_partition *) std::calloc(1u, sizeof(cellshard::bucketed_blocked_ell_partition));
        optimized_shard.partition_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
        optimized_shard.exec_to_canonical_cols =
            (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
        optimized_shard.canonical_to_exec_cols =
            (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
        const bool wrote_shard =
            optimized_shard.partitions != nullptr
            && optimized_shard.partition_row_offsets != nullptr
            && optimized_shard.exec_to_canonical_cols != nullptr
            && optimized_shard.canonical_to_exec_cols != nullptr
            && cellshard::build_bucketed_blocked_ell_partition(&bucketed_part, &part, 1u, nullptr) != 0;
        if (wrote_shard) {
            bucketed_part.exec_to_canonical_cols =
                (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
            bucketed_part.canonical_to_exec_cols =
                (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
        }
        if (wrote_shard && (bucketed_part.exec_to_canonical_cols == nullptr || bucketed_part.canonical_to_exec_cols == nullptr)) {
            cellshard::clear(&bucketed_part);
            cellshard::clear(&optimized_shard);
            return 1;
        }
        if (wrote_shard) {
            optimized_shard.partitions[0] = bucketed_part;
            std::memset(&bucketed_part, 0, sizeof(bucketed_part));
            optimized_shard.partition_row_offsets[0] = 0u;
            optimized_shard.partition_row_offsets[1] = part.rows;
            for (std::uint32_t col = 0u; col < part.cols; ++col) {
                optimized_shard.partitions[0].exec_to_canonical_cols[col] = col;
                optimized_shard.partitions[0].canonical_to_exec_cols[col] = col;
                optimized_shard.exec_to_canonical_cols[col] = col;
                optimized_shard.canonical_to_exec_cols[col] = col;
            }
        }
        const bool appended = wrote_shard && cellshard::append_bucketed_blocked_ell_shard_h5(dataset_path.c_str(), 0u, &optimized_shard);
        cellshard::clear(&bucketed_part);
        cellshard::clear(&optimized_shard);
        if (!appended) return 1;
    }
    if (!cellshard::append_dataset_embedded_metadata_h5(dataset_path.c_str(), &embedded_metadata_view)) return 1;
    if (!cellshard::append_dataset_observation_metadata_h5(dataset_path.c_str(), &observation_metadata_view)) return 1;
    if (!cellshard::append_dataset_feature_metadata_h5(dataset_path.c_str(), &feature_metadata_view)) return 1;
    if (!cellshard::append_dataset_browse_cache_h5(dataset_path.c_str(), &browse_view)) return 1;

    wb::dataset_summary summary = wb::summarize_dataset_csh5(dataset_path);
    if (!summary.ok) return 1;
    if (summary.rows != 2u || summary.cols != 3u || summary.datasets.size() != 1u || summary.partitions.size() != 1u) return 1;
    if (summary.feature_names.size() != 3u || summary.feature_names[0] != "MT-CO1") return 1;
    if (summary.embedded_metadata.size() != 1u) return 1;
    if (summary.embedded_metadata[0].column_names.size() != 3u || summary.embedded_metadata[0].column_names[0] != "day") return 1;
    wb::embedded_metadata_table loaded_metadata = wb::load_embedded_metadata_table(dataset_path, 0u);
    if (!loaded_metadata.available) return 1;
    if (loaded_metadata.field_values.size() != 6u || loaded_metadata.field_values[0] != "E8.5" || loaded_metadata.field_values[3] != "P0") return 1;
    if (loaded_metadata.row_offsets.size() != 3u || loaded_metadata.row_offsets[1] != 3u) return 1;
    if (!summary.observation_metadata.available || summary.observation_metadata.rows != 2u || summary.observation_metadata.cols != 6u) return 1;
    if (summary.observation_metadata.columns.size() != 6u || summary.observation_metadata.columns[3].name != "embryonic_day_label") return 1;
    wb::observation_metadata_table loaded_observation = wb::load_observation_metadata_table(dataset_path);
    if (!loaded_observation.available || loaded_observation.columns.size() != 6u) return 1;
    if (loaded_observation.columns[0].text_values.size() != 2u || loaded_observation.columns[0].text_values[0] != "E8.5") return 1;
    if (loaded_observation.columns[3].text_values[1] != "P0") return 1;
    if (loaded_observation.columns[4].float32_values.size() != 2u || loaded_observation.columns[4].float32_values[0] != 8.5f) return 1;
    if (!std::isnan(loaded_observation.columns[4].float32_values[1])) return 1;
    if (loaded_observation.columns[5].uint8_values.size() != 2u || loaded_observation.columns[5].uint8_values[1] != 1u) return 1;
    if (!summary.browse.available || summary.browse.selected_feature_indices.size() != 2u) return 1;
    if (summary.browse.selected_feature_names.size() != 2u || summary.browse.selected_feature_names[0] != "MT-CO1") return 1;

    {
        std::vector<std::uint64_t> grouped_rows;
        std::vector<cellshard::exporting::derivation_group_span> row_groups;
        std::vector<wb::issue> derived_issues;
        if (!wb::resolve_observation_group_order(dataset_path,
                                                 "day",
                                                 {"P0", "E8.5"},
                                                 &grouped_rows,
                                                 &row_groups,
                                                 &derived_issues,
                                                 false)) {
            std::fprintf(stderr, "workbench row-group resolution failed\n");
            return 1;
        }
        if (grouped_rows != std::vector<std::uint64_t>({1u, 0u})) {
            std::fprintf(stderr, "workbench row-group order mismatch\n");
            return 1;
        }

        std::vector<std::uint64_t> grouped_features;
        std::vector<cellshard::exporting::derivation_group_span> feature_groups;
        if (!wb::resolve_feature_group_order(dataset_path,
                                             "gene_short_name",
                                             {"GeneC", "MT-CO1"},
                                             &grouped_features,
                                             &feature_groups,
                                             &derived_issues,
                                             false)) {
            std::fprintf(stderr, "workbench feature-group resolution failed\n");
            return 1;
        }
        if (grouped_features != std::vector<std::uint64_t>({2u, 0u})) {
            std::fprintf(stderr, "workbench feature-group order mismatch\n");
            return 1;
        }

        cellshard::exporting::derived_materialization_request derived_request;
        derived_request.output_path = derived_dataset_path;
        derived_request.cache_root = derived_cache_root;
        derived_request.derived_pack_name = "day_grouped";
        derived_request.row_indices = grouped_rows;
        derived_request.feature_indices = grouped_features;
        derived_request.row_groups = row_groups;
        derived_request.feature_groups = feature_groups;
        derived_request.materialize_dataset = true;
        derived_request.materialize_execution_pack = true;

        wb::derived_materialization_report derived_report =
            wb::materialize_derived_dataset(dataset_path, derived_request);
        if (!derived_report.ok
            || !derived_report.result.materialized_execution_pack
            || derived_report.result.rows != 2u
            || derived_report.result.cols != 2u) {
            std::fprintf(stderr, "workbench derived materialization report mismatch\n");
            return 1;
        }
        wb::dataset_summary derived_summary = wb::summarize_dataset_csh5(derived_dataset_path);
        if (!derived_summary.ok || derived_summary.rows != 2u || derived_summary.cols != 2u) {
            std::fprintf(stderr, "workbench derived summary mismatch\n");
            return 1;
        }
        wb::observation_metadata_table derived_obs = wb::load_observation_metadata_table(derived_dataset_path);
        if (!derived_obs.available || derived_obs.columns.size() != 7u
            || derived_obs.columns.back().name != "derived.row_group"
            || derived_obs.columns.back().text_values != std::vector<std::string>({"P0", "E8.5"})) {
            std::fprintf(stderr, "workbench derived observation metadata mismatch\n");
            return 1;
        }
        wb::feature_metadata_table derived_var = wb::load_feature_metadata_table(derived_dataset_path);
        if (!derived_var.available || derived_var.columns.empty()
            || derived_var.columns.back().name != "derived.feature_group") {
            std::fprintf(stderr, "workbench derived feature metadata mismatch\n");
            return 1;
        }
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) device_count = 0;

    policy.output_path = converted_dataset_path;
    policy.cache_dir = converted_cache_root;
    policy.embed_metadata = true;
    policy.build_browse_cache = device_count >= 4;
    policy.browse_top_features = 2u;
    policy.browse_sample_rows_per_partition = 2u;
    wb::ingest_plan converted_plan = wb::plan_dataset_ingest(inspection.sources, policy);
    if (!converted_plan.ok) {
        std::fprintf(stderr, "converted_plan not ok\n");
        return 1;
    }
    wb::conversion_report report = wb::convert_plan_to_dataset_csh5(converted_plan);
    if (!report.ok) {
        std::fprintf(stderr, "convert_plan_to_dataset_csh5 failed\n");
        for (const wb::issue &entry : report.issues) {
            std::fprintf(stderr, "%s %s: %s\n",
                         wb::severity_name(entry.severity).c_str(),
                         entry.scope.c_str(),
                         entry.message.c_str());
        }
        return 1;
    }
    wb::dataset_summary converted_summary = wb::summarize_dataset_csh5(converted_dataset_path);
    if (!converted_summary.ok) {
        std::fprintf(stderr, "converted summary not ok\n");
        return 1;
    }
    if (converted_summary.matrix_format != "sliced_ell") {
        std::fprintf(stderr, "converted summary payload layout mismatch\n");
        return 1;
    }
    if (converted_summary.codecs.size() != 1u
        || converted_summary.codecs[0].family != cellshard::dataset_codec_family_sliced_ell) {
        std::fprintf(stderr, "converted summary codec family mismatch\n");
        return 1;
    }
    if (!converted_summary.runtime_service.available
        || converted_summary.runtime_service.service_mode != cellshard::dataset_runtime_service_mode_owner_hosted
        || converted_summary.runtime_service.live_write_mode != cellshard::dataset_live_write_mode_append_only
        || converted_summary.runtime_service.prefer_pack_delivery != 1u
        || converted_summary.runtime_service.single_reader_coordinator != 1u
        || converted_summary.runtime_service.canonical_generation != 1u
        || converted_summary.runtime_service.pack_generation != 1u
        || converted_summary.runtime_service.active_read_generation != 1u) {
        std::fprintf(stderr, "converted summary runtime service mismatch\n");
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
    if (!cellshard::warm_dataset_sliced_ell_h5_cache(converted_dataset_path.c_str(), converted_cache_root.c_str())) {
        std::fprintf(stderr, "failed to warm converted sliced-ell cache\n");
        return 1;
    }
    {
        cellshard::sharded<cellshard::sparse::sliced_ell> converted_matrix;
        cellshard::shard_storage converted_storage;
        cellshard::bucketed_sliced_ell_partition converted_exec_part;
        cellshard::dataset_execution_view converted_execution{};
        cellshard::dataset_runtime_service_view converted_runtime_service{};
        cellshard::init(&converted_matrix);
        cellshard::init(&converted_storage);
        cellshard::init(&converted_exec_part);
        if (!cellshard::load_dataset_sliced_ell_h5_header(converted_dataset_path.c_str(), &converted_matrix, &converted_storage)
            || !cellshard::bind_dataset_h5_cache(&converted_storage, converted_cache_root.c_str())
            || !cellshard::get_dataset_h5_execution_metadata(&converted_storage, &converted_execution)
            || !cellshard::get_dataset_h5_runtime_service(&converted_storage, &converted_runtime_service)
            || !cellshard::fetch_dataset_sliced_ell_h5_bucketed_partition(&converted_exec_part, &converted_matrix, &converted_storage, 0u)) {
            std::fprintf(stderr, "failed to inspect converted sliced-ell codec runtime\n");
            return 1;
        }
        if (converted_execution.partition_count != converted_plan.parts.size()
            || converted_execution.shard_count != converted_plan.shards.size()) {
            std::fprintf(stderr, "converted execution metadata mismatch\n");
            return 1;
        }
        if (converted_runtime_service.service_mode != cellshard::dataset_runtime_service_mode_owner_hosted
            || converted_runtime_service.active_read_generation != 1u) {
            std::fprintf(stderr, "converted runtime service metadata mismatch\n");
            return 1;
        }
        if (converted_exec_part.cols != 3u
            || converted_exec_part.exec_to_canonical_rows == nullptr
            || converted_exec_part.canonical_to_exec_rows == nullptr) {
            std::fprintf(stderr, "converted execution partition row maps missing\n");
            return 1;
        }
        for (std::uint32_t exec_row = 0u; exec_row < converted_exec_part.rows; ++exec_row) {
            const std::uint32_t canonical_row = converted_exec_part.exec_to_canonical_rows[exec_row];
            if (canonical_row >= converted_exec_part.rows) {
                std::fprintf(stderr, "converted execution partition exec_to_canonical out of range\n");
                return 1;
            }
            if (converted_exec_part.canonical_to_exec_rows[canonical_row] != exec_row) {
                std::fprintf(stderr, "converted execution partition column-map inverse mismatch\n");
                return 1;
            }
        }
        if (converted_storage.backend == cellshard::shard_storage_backend_dataset_h5) {
            cellshard::invalidate_dataset_h5_cache(&converted_storage);
        }
        cellshard::clear(&converted_exec_part);
        cellshard::clear(&converted_storage);
        cellshard::clear(&converted_matrix);
    }

    if (device_count > 0) {
        fs::create_directories(preprocess_cache_root);
        wb::preprocess_config preprocess;
        preprocess.device = 0;
        preprocess.min_counts = 0.0f;
        preprocess.min_genes = 0u;
        preprocess.max_mito_fraction = 1.0f;
        preprocess.min_gene_sum = 0.0f;
        preprocess.min_detected_cells = 0.0f;
        preprocess.min_variance = 0.0f;
        preprocess.cache_dir = preprocess_cache_root;
        preprocess.finalize_after_preprocess = false;
        wb::preprocess_analysis_table cached_analysis = wb::analyze_dataset_preprocess(converted_dataset_path, preprocess);
        CHECK_OR_FAIL(cached_analysis.ok, "cached preprocess analysis failed");
        wb::preprocess_config uncached_preprocess = preprocess;
        uncached_preprocess.enable_sliced_device_cache = false;
        wb::preprocess_analysis_table uncached_analysis = wb::analyze_dataset_preprocess(converted_dataset_path, uncached_preprocess);
        CHECK_OR_FAIL(uncached_analysis.ok, "uncached preprocess analysis failed");
        CHECK_OR_FAIL(cached_analysis.partitions_processed == uncached_analysis.partitions_processed
                          && cached_analysis.rows == uncached_analysis.rows
                          && cached_analysis.cols == uncached_analysis.cols
                          && cached_analysis.nnz == uncached_analysis.nnz
                          && cached_analysis.gene_sum == uncached_analysis.gene_sum
                          && cached_analysis.gene_sq_sum == uncached_analysis.gene_sq_sum
                          && cached_analysis.gene_detected_cells == uncached_analysis.gene_detected_cells
                          && cached_analysis.cell_total_counts == uncached_analysis.cell_total_counts
                          && cached_analysis.cell_mito_counts == uncached_analysis.cell_mito_counts
                          && cached_analysis.cell_max_counts == uncached_analysis.cell_max_counts
                          && cached_analysis.cell_detected_genes == uncached_analysis.cell_detected_genes
                          && cached_analysis.cell_keep == uncached_analysis.cell_keep
                          && cached_analysis.gene_keep == uncached_analysis.gene_keep
                          && cached_analysis.gene_flags == uncached_analysis.gene_flags,
                      "cached and uncached preprocess analysis mismatch");
        {
            wb::preprocess_config finalize_output_preprocess = preprocess;
            finalize_output_preprocess.finalize_after_preprocess = true;
            wb::preprocess_persist_summary finalized_output =
                wb::persist_preprocess_analysis_to_output(converted_dataset_path,
                                                          converted_filtered_output_path,
                                                          cached_analysis,
                                                          finalize_output_preprocess);
            if (!finalized_output.summary.ok) {
                for (const wb::issue &entry : finalized_output.summary.issues) {
                    std::fprintf(stderr, "separate-output sliced finalize issue [%s] %s\n", entry.scope.c_str(), entry.message.c_str());
                }
            }
            CHECK_OR_FAIL(finalized_output.summary.ok, "separate-output sliced finalize failed");
            wb::dataset_summary raw_converted_summary = wb::summarize_dataset_csh5(converted_dataset_path);
            CHECK_OR_FAIL(raw_converted_summary.ok && !raw_converted_summary.preprocess.available,
                          "separate-output finalize unexpectedly mutated raw sliced source");
            wb::dataset_summary filtered_converted_summary = wb::summarize_dataset_csh5(converted_filtered_output_path);
            CHECK_OR_FAIL(filtered_converted_summary.ok && filtered_converted_summary.preprocess.available,
                          "separate-output sliced finalize missing preprocess block");
            CHECK_OR_FAIL(filtered_converted_summary.matrix_format == "sliced_ell",
                          "separate-output sliced finalize matrix format mismatch");
            CHECK_OR_FAIL(filtered_converted_summary.preprocess.processed_matrix_available,
                          "separate-output sliced finalize missing processed matrix flag");
            CHECK_OR_FAIL(filtered_converted_summary.rows == 2u && filtered_converted_summary.cols == 3u,
                          "separate-output sliced finalize shape mismatch");
            {
                cellshard::sharded<cellshard::sparse::sliced_ell> filtered_matrix;
                cellshard::shard_storage filtered_storage;
                cellshard::bucketed_sliced_ell_partition filtered_exec_part;
                cellshard::init(&filtered_matrix);
                cellshard::init(&filtered_storage);
                cellshard::init(&filtered_exec_part);
                CHECK_OR_FAIL(cellshard::load_dataset_sliced_ell_h5_header(converted_filtered_output_path.c_str(),
                                                                           &filtered_matrix,
                                                                           &filtered_storage)
                                  && cellshard::fetch_dataset_sliced_ell_h5_bucketed_partition(&filtered_exec_part,
                                                                                                &filtered_matrix,
                                                                                                &filtered_storage,
                                                                                                0u),
                              "separate-output sliced finalize fetch failed");
                CHECK_OR_FAIL(filtered_exec_part.rows != 0u && filtered_exec_part.cols == 3u
                                  && filtered_exec_part.exec_to_canonical_rows != nullptr
                                  && filtered_exec_part.canonical_to_exec_rows != nullptr,
                              "separate-output sliced finalize partition row-map mismatch");
                for (std::uint32_t exec_row = 0u; exec_row < filtered_exec_part.rows; ++exec_row) {
                    const std::uint32_t canonical_row = filtered_exec_part.exec_to_canonical_rows[exec_row];
                    CHECK_OR_FAIL(canonical_row < filtered_exec_part.rows
                                      && filtered_exec_part.canonical_to_exec_rows[canonical_row] == exec_row,
                                  "separate-output sliced finalize row-map inverse mismatch");
                }
                cellshard::clear(&filtered_exec_part);
                cellshard::clear(&filtered_storage);
                cellshard::clear(&filtered_matrix);
            }
        }
        wb::preprocess_summary preprocess_summary = wb::run_preprocess_pass(converted_dataset_path, preprocess);
        CHECK_OR_FAIL(preprocess_summary.ok, "preprocess pass failed");
        CHECK_OR_FAIL(preprocess_summary.partitions_processed == converted_plan.parts.size() && preprocess_summary.kept_genes != 0u,
                      "preprocess summary partition or kept-gene mismatch");

        wb::dataset_summary preprocessed_summary = wb::summarize_dataset_csh5(converted_dataset_path);
        CHECK_OR_FAIL(preprocessed_summary.ok && preprocessed_summary.preprocess.available,
                      "preprocessed summary missing preprocess block");
        CHECK_OR_FAIL(preprocessed_summary.preprocess.raw_counts_available && !preprocessed_summary.preprocess.processed_matrix_available,
                      "preprocessed summary raw/processed flags mismatch");
        CHECK_OR_FAIL(preprocessed_summary.preprocess.normalized_log1p_metrics && preprocessed_summary.preprocess.kept_genes != 0u,
                      "preprocessed summary normalized metrics mismatch");
        CHECK_OR_FAIL(preprocessed_summary.preprocess.mito_feature_count == 1u,
                      "preprocessed summary mito feature count mismatch");
        CHECK_OR_FAIL(preprocessed_summary.matrix_format == "sliced_ell",
                      "preprocessed summary matrix format mismatch");
        CHECK_OR_FAIL(preprocessed_summary.rows == 2u && preprocessed_summary.cols == 3u && preprocessed_summary.nnz == 3u,
                      "preprocessed summary shape mismatch");
        CHECK_OR_FAIL(preprocessed_summary.dataset_attributes.available,
                      "preprocessed summary missing dataset attributes");
        CHECK_OR_FAIL(std::find(preprocessed_summary.dataset_attributes.keys.begin(),
                                preprocessed_summary.dataset_attributes.keys.end(),
                                "preprocess.pipeline_scope") != preprocessed_summary.dataset_attributes.keys.end(),
                      "preprocessed summary missing pipeline_scope attribute");

        wb::persisted_preprocess_table preprocess_table = wb::load_persisted_preprocess_table(converted_dataset_path);
        CHECK_OR_FAIL(preprocess_table.available, "persisted preprocess table unavailable");
        CHECK_OR_FAIL(preprocess_table.cell_total_counts.size() == 2u && preprocess_table.gene_sum.size() == 3u,
                      "persisted preprocess table dimensions mismatch");
        CHECK_OR_FAIL(preprocess_table.cell_keep.size() == 2u && preprocess_table.cell_keep[0] == 1u,
                      "persisted preprocess cell keep mismatch");
        CHECK_OR_FAIL(preprocess_table.gene_flags.size() == 3u && preprocess_table.gene_flags[0] != 0u,
                      "persisted preprocess gene flags mismatch");

        wb::observation_metadata_table preprocessed_obs = wb::load_observation_metadata_table(converted_dataset_path);
        CHECK_OR_FAIL(preprocessed_obs.available && preprocessed_obs.columns.size() >= 8u,
                      "preprocessed observation metadata unavailable");
        CHECK_OR_FAIL(preprocessed_obs.columns.back().name == "preprocess_keep",
                      "preprocessed observation metadata missing preprocess_keep");

        wb::feature_metadata_table preprocessed_var = wb::load_feature_metadata_table(converted_dataset_path);
        CHECK_OR_FAIL(preprocessed_var.available && preprocessed_var.columns.size() >= 5u,
                      "preprocessed feature metadata unavailable");
        CHECK_OR_FAIL(preprocessed_var.columns.back().name == "preprocess_flags",
                      "preprocessed feature metadata missing preprocess_flags");

        wb::dataset_attribute_table preprocess_attrs = wb::load_dataset_attribute_table(converted_dataset_path);
        CHECK_OR_FAIL(preprocess_attrs.available && !preprocess_attrs.entries.empty(),
                      "preprocessed dataset attributes unavailable");

        wb::preprocess_summary second_preprocess = wb::run_preprocess_pass(converted_dataset_path, preprocess);
        CHECK_OR_FAIL(!second_preprocess.ok, "second preprocess pass unexpectedly succeeded");

        // Blocked in-place finalize coverage was tied to the legacy canonical blocked write path.
        // The active contract under test here is sliced-first finalize plus execution-form blocked runtime.
    }

    if (device_count >= 4) {
        if (converted_summary.browse.available) {
            std::fprintf(stderr, "converted sliced summary should not have browse cache before blocked finalize\n");
            return 1;
        }
        if (converted_summary.partitions.empty() || converted_summary.partitions[0].execution_format == 0u) {
            std::fprintf(stderr, "converted summary missing execution metadata\n");
            return 1;
        }
    }

    if (!write_file(h5ad_manifest_path,
                    ("dataset\tpath\tformat\tfeatures\tbarcodes\tmetadata\tmatrix_source\tallow_processed\trows\tcols\tnnz\n"
                     "sample_h5ad_x\t" + h5ad_path + "\th5ad\t\t\t\tx\t0\t0\t0\t0\n"
                     "sample_h5ad_raw\t" + h5ad_path + "\th5ad\t\t\t\traw_x\t0\t0\t0\t0\n"
                     "sample_h5ad_layer\t" + h5ad_path + "\th5ad\t\t\t\tlayer:counts\t0\t0\t0\t0\n"
                     "sample_h5ad_layer_csc\t" + h5ad_path + "\th5ad\t\t\t\tlayer:counts_csc\t0\t0\t0\t0\n"
                     "sample_h5ad_x_override\t" + h5ad_path + "\th5ad\t\t\t\tx\t1\t0\t0\t0\n").c_str())) return 1;

    wb::manifest_inspection h5ad_inspection = wb::inspect_manifest_tsv(h5ad_manifest_path);
    if (h5ad_inspection.sources.size() != 5u) {
        std::fprintf(stderr, "unexpected h5ad source count\n");
        return 1;
    }
    if (h5ad_inspection.sources[0].probe_ok) {
        std::fprintf(stderr, "processed h5ad X should have been rejected\n");
        return 1;
    }
    if (!h5ad_inspection.sources[1].probe_ok || !h5ad_inspection.sources[2].probe_ok
        || !h5ad_inspection.sources[3].probe_ok || !h5ad_inspection.sources[4].probe_ok) {
        std::fprintf(stderr, "expected h5ad sources did not probe cleanly\n");
        return 1;
    }
    if (h5ad_inspection.sources[1].rows != 2u || h5ad_inspection.sources[1].cols != 3u || h5ad_inspection.sources[1].barcode_count != 2u) {
        std::fprintf(stderr, "raw_x dimensions/barcodes mismatch\n");
        return 1;
    }
    if (h5ad_inspection.sources[1].feature_count != 3u || h5ad_inspection.sources[1].metadata_rows != 2u) {
        std::fprintf(stderr, "raw_x features/metadata mismatch\n");
        return 1;
    }
    if (!h5ad_inspection.sources[1].matrix_sparse || h5ad_inspection.sources[1].matrix_encoding != "csr_matrix") {
        std::fprintf(stderr, "raw_x should have probed as csr_matrix\n");
        return 1;
    }
    if (h5ad_inspection.sources[3].matrix_encoding != "csc_matrix") {
        std::fprintf(stderr, "csc layer did not probe as csc_matrix\n");
        return 1;
    }
    if (h5ad_inspection.sources[1].matrix_count_like != true) {
        std::fprintf(stderr, "raw_x should look count-like\n");
        return 1;
    }

    wb::ingest_policy h5ad_policy;
    h5ad_policy.max_part_nnz = 2u;
    h5ad_policy.convert_window_bytes = 256u << 20u;
    h5ad_policy.target_shard_bytes = 256u << 20u;
    h5ad_policy.output_path = h5ad_dataset_path;
    h5ad_policy.cache_dir = h5ad_cache_root;
    h5ad_policy.embed_metadata = true;
    h5ad_policy.build_browse_cache = false;
    std::vector<wb::source_entry> h5ad_sources;
    h5ad_sources.push_back(h5ad_inspection.sources[3]);
    wb::ingest_plan h5ad_plan = wb::plan_dataset_ingest(h5ad_sources, h5ad_policy);
    if (!h5ad_plan.ok || h5ad_plan.parts.empty() || h5ad_plan.datasets.size() != 1u) {
        std::fprintf(stderr, "h5ad csc plan failed\n");
        return 1;
    }
    wb::conversion_report h5ad_report = wb::convert_plan_to_dataset_csh5(h5ad_plan);
    if (!h5ad_report.ok) {
        std::fprintf(stderr, "h5ad csc conversion failed\n");
        for (const wb::issue &entry : h5ad_report.issues) {
            std::fprintf(stderr, "%s %s: %s\n",
                         wb::severity_name(entry.severity).c_str(),
                         entry.scope.c_str(),
                         entry.message.c_str());
        }
        return 1;
    }
    wb::dataset_summary h5ad_summary = wb::summarize_dataset_csh5(h5ad_dataset_path);
    if (!h5ad_summary.ok || h5ad_summary.rows != 2u || h5ad_summary.cols != 3u) {
        std::fprintf(stderr, "h5ad csc summary mismatch\n");
        return 1;
    }
    if (!h5ad_summary.observation_metadata.available || h5ad_summary.observation_metadata.cols < 2u) {
        std::fprintf(stderr, "h5ad csc observation metadata missing\n");
        return 1;
    }
    wb::observation_metadata_table h5ad_obs = wb::load_observation_metadata_table(h5ad_dataset_path);
    if (!h5ad_obs.available || h5ad_obs.columns.empty()) {
        std::fprintf(stderr, "h5ad csc observation metadata table unavailable\n");
        return 1;
    }
    if (h5ad_obs.columns[0].text_values.empty() || h5ad_obs.columns[0].text_values[0] != "batch0") {
        std::fprintf(stderr, "h5ad csc observation metadata contents mismatch\n");
        return 1;
    }
    if ((h5ad_summary.matrix_format == "sliced_ell"
             && !cellshard::warm_dataset_sliced_ell_h5_cache(h5ad_dataset_path.c_str(), h5ad_cache_root.c_str()))
        || (h5ad_summary.matrix_format == "blocked_ell"
            && !cellshard::warm_dataset_blocked_ell_h5_cache(h5ad_dataset_path.c_str(), h5ad_cache_root.c_str()))) {
        std::fprintf(stderr, "h5ad csc cache warm failed\n");
        return 1;
    }

    cellshard::sparse::clear(&part);
    std::remove(manifest_path.c_str());
    std::remove(matrix_path.c_str());
    std::remove(feature_path.c_str());
    std::remove(barcode_path.c_str());
    std::remove(metadata_path.c_str());
    std::remove(dataset_path.c_str());
    std::remove(converted_dataset_path.c_str());
    std::remove(converted_filtered_output_path.c_str());
    std::remove(finalized_dataset_path.c_str());
    std::remove(derived_dataset_path.c_str());
    std::remove(h5ad_path.c_str());
    std::remove(h5ad_manifest_path.c_str());
    std::remove(h5ad_dataset_path.c_str());
    fs::remove_all(preprocess_cache_root);
    fs::remove_all(converted_cache_root);
    fs::remove_all(finalized_cache_root);
    fs::remove_all(derived_cache_root);
    fs::remove_all(h5ad_cache_root);
    return 0;
}
