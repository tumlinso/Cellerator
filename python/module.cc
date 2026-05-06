#include <Cellerator/preprocess/runtime.hh>

#include <CellShard/export/dataset_export.hh>
#include <CellShard/io/csh5/api.cuh>
#include <CellShard/runtime/device/sharded_device.cuh>
#include <CellShard/runtime/host/sharded_host.cuh>
#include <CellShard/runtime/storage/disk.cuh>

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace cpre = ::cellerator::preprocess;
namespace cse = ::cellshard::exporting;
namespace csd = ::cellshard::device;

namespace cellerator::python_bindings {

namespace {

struct text_column_owner {
    std::vector<std::uint32_t> offsets;
    std::string data;

    cellshard::dataset_text_column_view view() const {
        cellshard::dataset_text_column_view out{};
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? nullptr : offsets.data();
        out.data = data.empty() ? nullptr : data.data();
        return out;
    }
};

text_column_owner make_text_column(const std::vector<std::string> &values) {
    text_column_owner out;
    out.offsets.reserve(values.size() + 1u);
    out.offsets.push_back(0u);
    for (const std::string &value : values) {
        out.data.append(value);
        out.offsets.push_back((std::uint32_t) out.data.size());
    }
    return out;
}

template<typename T>
py::array_t<T> copy_array(const std::vector<T> &values) {
    py::array_t<T> out((py::ssize_t) values.size());
    if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    return out;
}

void require_cuda(cudaError_t err, const char *label) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(err));
    }
}

template<typename T>
void copy_device_to_host(std::vector<T> *dst, std::size_t offset, const T *src, std::size_t count, const char *label) {
    if (count == 0u) return;
    if (src == nullptr) throw std::runtime_error(std::string(label) + " was not produced by native preprocessing");
    if (offset + count > dst->size()) throw std::out_of_range(std::string(label) + " copy exceeds destination");
    require_cuda(cudaMemcpy(dst->data() + offset, src, count * sizeof(T), cudaMemcpyDeviceToHost), label);
}

void add_inplace(std::vector<float> *dst, const std::vector<float> &src) {
    if (dst->size() != src.size()) throw std::runtime_error("gene metric size mismatch while reducing partitions");
    for (std::size_t i = 0; i < src.size(); ++i) (*dst)[i] += src[i];
}

std::vector<const char *> c_strings(const std::vector<std::string> &values) {
    std::vector<const char *> out;
    out.reserve(values.size());
    for (const std::string &value : values) out.push_back(value.c_str());
    return out;
}

} // namespace

struct PreprocessOptions {
    std::string assay = "scrna";
    std::string matrix_orientation = "observations_by_features";
    std::string matrix_state = "raw_counts";
    std::string feature_namespace = "gene_symbol";
    std::string mito_prefix = "MT-";
    float target_sum = 10000.0f;
    float min_counts = 1.0f;
    unsigned int min_features = 1u;
    float max_group_fraction = 1.0f;
    unsigned int fraction_group_index = cpre::qc_group_mt;
    float min_gene_sum = 0.0f;
    float min_detected_cells = 0.0f;
    float min_variance = 0.0f;
    int device = 0;
    bool allow_processed = false;
};

struct AdapterStagePlan {
    std::string layout;
    std::string value_type;
    std::string accumulator_type;
    bool adapt_to_cellshard_first = false;
    bool direct_external_kernels = false;
};

struct PreprocessSession {
    std::string source_path;
    std::string layout;
    PreprocessOptions options;
    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t partitions_processed = 0u;
    std::uint64_t kept_cells = 0u;
    std::uint32_t kept_genes = 0u;
    double gene_sum_checksum = 0.0;
    std::vector<std::string> group_names = {"mt", "ribo", "hb"};
    std::vector<std::uint32_t> feature_group_masks;
    std::vector<std::uint8_t> cell_keep;
    std::vector<std::uint8_t> gene_keep;
    std::vector<float> cell_total_counts;
    std::vector<float> cell_mito_counts;
    std::vector<float> cell_max_counts;
    std::vector<std::uint32_t> cell_detected_genes;
    std::vector<float> cell_group_counts;
    std::vector<float> cell_group_pct;
    std::vector<float> gene_sum;
    std::vector<float> gene_sq_sum;
    std::vector<float> gene_detected_cells;

    py::dict metrics() const {
        py::dict out;
        out["source_path"] = source_path;
        out["layout"] = layout;
        out["shape"] = py::make_tuple((py::ssize_t) rows, (py::ssize_t) cols);
        out["nnz"] = py::int_(nnz);
        out["partitions_processed"] = py::int_(partitions_processed);
        out["kept_cells"] = py::int_(kept_cells);
        out["kept_genes"] = py::int_(kept_genes);
        out["target_sum"] = options.target_sum;
        out["gene_sum_checksum"] = gene_sum_checksum;
        out["cell_total_counts"] = copy_array(cell_total_counts);
        out["cell_keep"] = copy_array(cell_keep);
        out["cell_group_counts"] = copy_array(cell_group_counts);
        out["cell_group_pct"] = copy_array(cell_group_pct);
        out["gene_sum"] = copy_array(gene_sum);
        out["gene_detected_cells"] = copy_array(gene_detected_cells);
        out["gene_keep"] = copy_array(gene_keep);
        out["feature_group_masks"] = copy_array(feature_group_masks);
        out["group_names"] = group_names;
        return out;
    }

    py::dict publish(const std::string &output_path, const std::string &working_root = "") const {
        if (source_path.empty() || output_path.empty()) throw std::invalid_argument("publish requires source and output paths");
        if (cell_keep.empty() || gene_keep.empty()) throw std::runtime_error("preprocess session has no keep masks to publish");

        text_column_owner groups = make_text_column(group_names);
        cellshard::dataset_preprocess_view prep{};
        prep.assay = options.assay.c_str();
        prep.matrix_orientation = options.matrix_orientation.c_str();
        prep.matrix_state = "filtered_raw_counts";
        prep.pipeline_scope = "cellerator_python_gpu_v1";
        prep.raw_matrix_name = "raw_counts";
        prep.active_matrix_name = "filtered_raw_counts";
        prep.feature_namespace = options.feature_namespace.c_str();
        prep.mito_prefix = options.mito_prefix.c_str();
        prep.raw_counts_available = 1u;
        prep.processed_matrix_available = 1u;
        prep.normalized_log1p_metrics = 1u;
        prep.qc_group_count = (std::uint32_t) group_names.size();
        prep.rows = rows;
        prep.cols = (std::uint32_t) cols;
        prep.nnz = nnz;
        prep.partitions_processed = (std::uint32_t) partitions_processed;
        prep.target_sum = options.target_sum;
        prep.min_counts = options.min_counts;
        prep.min_genes = options.min_features;
        prep.max_mito_fraction = options.max_group_fraction;
        prep.min_gene_sum = options.min_gene_sum;
        prep.min_detected_cells = options.min_detected_cells;
        prep.min_variance = options.min_variance;
        prep.kept_cells = (double) kept_cells;
        prep.kept_genes = kept_genes;
        prep.gene_sum_checksum = gene_sum_checksum;
        prep.cell_total_counts = cell_total_counts.empty() ? nullptr : cell_total_counts.data();
        prep.cell_mito_counts = cell_mito_counts.empty() ? nullptr : cell_mito_counts.data();
        prep.cell_max_counts = cell_max_counts.empty() ? nullptr : cell_max_counts.data();
        prep.cell_detected_genes = cell_detected_genes.empty() ? nullptr : cell_detected_genes.data();
        prep.cell_keep = cell_keep.empty() ? nullptr : cell_keep.data();
        prep.qc_group_names = groups.view();
        prep.cell_group_counts = cell_group_counts.empty() ? nullptr : cell_group_counts.data();
        prep.cell_group_pct = cell_group_pct.empty() ? nullptr : cell_group_pct.data();
        prep.gene_sum = gene_sum.empty() ? nullptr : gene_sum.data();
        prep.gene_sq_sum = gene_sq_sum.empty() ? nullptr : gene_sq_sum.data();
        prep.gene_detected_cells = gene_detected_cells.empty() ? nullptr : gene_detected_cells.data();
        prep.gene_keep = gene_keep.empty() ? nullptr : gene_keep.data();
        prep.feature_group_masks = feature_group_masks.empty() ? nullptr : feature_group_masks.data();

        std::uint64_t out_rows = 0u, out_cols = 0u, out_nnz = 0u;
        int ok = 0;
        if (layout == "blocked_ell") {
            ok = cellshard::finalize_preprocessed_blocked_ell_dataset_h5_to_output(
                source_path.c_str(), output_path.c_str(), cell_keep.data(), gene_keep.data(),
                nullptr, nullptr, nullptr, nullptr, &prep,
                working_root.empty() ? nullptr : working_root.c_str(), &out_rows, &out_cols, &out_nnz);
        } else if (layout == "sliced_ell") {
            ok = cellshard::finalize_preprocessed_sliced_ell_dataset_h5_to_output(
                source_path.c_str(), output_path.c_str(), cell_keep.data(), gene_keep.data(),
                nullptr, nullptr, nullptr, nullptr, &prep,
                working_root.empty() ? nullptr : working_root.c_str(), &out_rows, &out_cols, &out_nnz);
        } else {
            throw std::runtime_error("publish only supports blocked_ell and sliced_ell sessions");
        }
        if (!ok) throw std::runtime_error("CellShard failed to publish preprocessed dataset");
        py::dict out;
        out["path"] = output_path;
        out["shape"] = py::make_tuple((py::ssize_t) out_rows, (py::ssize_t) out_cols);
        out["nnz"] = py::int_(out_nnz);
        return out;
    }
};

std::vector<std::uint32_t> compile_default_masks_from_summary(const cse::dataset_summary &summary) {
    std::vector<std::string> modalities(summary.var_names.size(), "rna");
    std::vector<const char *> ids = c_strings(summary.var_ids);
    std::vector<const char *> names = c_strings(summary.var_names);
    std::vector<const char *> types = c_strings(summary.var_types);
    std::vector<const char *> modality_ptrs = c_strings(modalities);
    std::vector<std::uint32_t> masks(summary.var_names.size(), 0u);
    cpre::qc_feature_annotation_view features{
        ids.empty() ? nullptr : ids.data(),
        names.empty() ? nullptr : names.data(),
        types.empty() ? nullptr : types.data(),
        modality_ptrs.empty() ? nullptr : modality_ptrs.data(),
        (std::uint32_t) masks.size()
    };
    if (!cpre::compile_default_qc_feature_group_masks(&features, nullptr, masks.data())) {
        throw std::runtime_error("failed to compile default QC feature groups");
    }
    return masks;
}

void validate_raw_or_throw(const PreprocessOptions &options) {
    cpre::preprocess_state_view state{};
    state.assay = options.assay.c_str();
    state.matrix_orientation = options.matrix_orientation.c_str();
    state.matrix_state = options.matrix_state.c_str();
    state.feature_namespace = options.feature_namespace.c_str();
    state.raw_counts_available = options.allow_processed ? 1u : 1u;
    state.processed_matrix_available = options.allow_processed ? 0u : 0u;
    cpre::status status{};
    if (!cpre::validate_raw_count_state(&state, &status)) throw std::runtime_error(status.message);
}

template<typename MatrixT>
struct loaded_dataset {
    cellshard::sharded<MatrixT> view;
    cellshard::shard_storage storage;

    loaded_dataset() {
        cellshard::init(&view);
        cellshard::init(&storage);
    }

    ~loaded_dataset() {
        cellshard::clear(&storage);
        cellshard::clear(&view);
    }
};

csd::blocked_ell_view make_device_view(const cellshard::sparse::blocked_ell *part,
                                       const csd::partition_record<cellshard::sparse::blocked_ell> &record) {
    csd::blocked_ell_view out{};
    (void) part;
    if (record.view == nullptr) throw std::runtime_error("missing uploaded blocked-ELL device descriptor");
    require_cuda(cudaMemcpy(&out, record.view, sizeof(out), cudaMemcpyDeviceToHost), "copy blocked-ELL device descriptor");
    return out;
}

csd::sliced_ell_view make_device_view(const cellshard::sparse::sliced_ell *part,
                                      const csd::partition_record<cellshard::sparse::sliced_ell> &record) {
    csd::sliced_ell_view out{};
    (void) part;
    if (record.view == nullptr) throw std::runtime_error("missing uploaded Sliced-ELL device descriptor");
    require_cuda(cudaMemcpy(&out, record.view, sizeof(out), cudaMemcpyDeviceToHost), "copy Sliced-ELL device descriptor");
    return out;
}

int preprocess_part(csd::blocked_ell_view *view,
                    cpre::preprocess_workspace *workspace,
                    const cpre::qc_group_config_view *groups,
                    const cpre::cell_qc_filter_params *filter,
                    float target_sum,
                    cpre::part_preprocess_result *out) {
    return cpre::preprocess_blocked_ell_qc_groups_inplace(view, workspace, groups, filter, target_sum, out);
}

int preprocess_part(csd::sliced_ell_view *view,
                    cpre::preprocess_workspace *workspace,
                    const cpre::qc_group_config_view *groups,
                    const cpre::cell_qc_filter_params *filter,
                    float target_sum,
                    cpre::part_preprocess_result *out) {
    return cpre::preprocess_sliced_ell_qc_groups_inplace(view, workspace, groups, filter, target_sum, out);
}

template<typename MatrixT>
PreprocessSession run_dataset_layout(const std::string &path,
                                     const std::string &layout,
                                     const cse::dataset_summary &summary,
                                     const PreprocessOptions &options,
                                     const std::vector<std::uint32_t> &feature_group_masks) {
    if (summary.rows > (std::uint64_t) UINT32_MAX || summary.cols > (std::uint64_t) UINT32_MAX) {
        throw std::runtime_error("v1 Python preprocessing requires per-dataset rows/cols to fit 32-bit native kernels");
    }

    loaded_dataset<MatrixT> loaded;
    if (!cellshard::load_header(path.c_str(), &loaded.view, &loaded.storage)) {
        throw std::runtime_error("failed to load CellShard dataset header");
    }

    cpre::preprocess_workspace workspace;
    cpre::init(&workspace);
    if (!cpre::setup(&workspace, options.device)) throw std::runtime_error("failed to set up Cellerator preprocess workspace");

    cellshard::device::sharded_device<MatrixT> device_state;
    cellshard::device::init(&device_state);

    PreprocessSession session;
    session.source_path = path;
    session.layout = layout;
    session.options = options;
    session.rows = summary.rows;
    session.cols = summary.cols;
    session.nnz = summary.nnz;
    session.feature_group_masks = feature_group_masks;
    session.cell_keep.assign((std::size_t) summary.rows, 0u);
    session.gene_keep.assign((std::size_t) summary.cols, 1u);
    session.cell_total_counts.assign((std::size_t) summary.rows, 0.0f);
    session.cell_mito_counts.assign((std::size_t) summary.rows, 0.0f);
    session.cell_max_counts.assign((std::size_t) summary.rows, 0.0f);
    session.cell_detected_genes.assign((std::size_t) summary.rows, 0u);
    session.cell_group_counts.assign((std::size_t) summary.rows * session.group_names.size(), 0.0f);
    session.cell_group_pct.assign((std::size_t) summary.rows * session.group_names.size(), 0.0f);
    session.gene_sum.assign((std::size_t) summary.cols, 0.0f);
    session.gene_sq_sum.assign((std::size_t) summary.cols, 0.0f);
    session.gene_detected_cells.assign((std::size_t) summary.cols, 0.0f);

    cpre::qc_group_config_view groups{};
    groups.group_count = (unsigned int) session.group_names.size();
    groups.group_names = nullptr;
    groups.feature_group_masks = session.feature_group_masks.empty() ? nullptr : session.feature_group_masks.data();
    cpre::cell_qc_filter_params cell_filter{
        options.min_counts,
        options.min_features,
        options.max_group_fraction,
        options.fraction_group_index
    };

    try {
        for (unsigned long part_id = 0; part_id < loaded.view.num_partitions; ++part_id) {
            if (!cellshard::fetch_partition(&loaded.view, &loaded.storage, part_id)) {
                throw std::runtime_error("failed to fetch CellShard partition " + std::to_string(part_id));
            }
            require_cuda(
                cellshard::device::upload_partition_async(&device_state, &loaded.view, part_id, options.device, workspace.stream),
                "upload CellShard partition to GPU");
            require_cuda(cudaStreamSynchronize(workspace.stream), "synchronize GPU partition upload");
            MatrixT *host_part = loaded.view.parts[part_id];
            auto device_view = make_device_view(host_part, device_state.parts[part_id]);

            cpre::part_preprocess_result result{};
            if (!preprocess_part(&device_view, &workspace, &groups, &cell_filter, options.target_sum, &result)) {
                throw std::runtime_error("native Cellerator preprocessing failed on partition " + std::to_string(part_id));
            }
            require_cuda(cudaStreamSynchronize(workspace.stream), "synchronize native preprocessing");

            const std::size_t row_offset = (std::size_t) loaded.view.partition_offsets[part_id];
            const std::size_t part_rows = (std::size_t) host_part->rows;
            copy_device_to_host(&session.cell_total_counts, row_offset, result.cell.total_counts, part_rows, "cell total counts");
            copy_device_to_host(&session.cell_mito_counts, row_offset, result.cell.mito_counts, part_rows, "cell mito counts");
            copy_device_to_host(&session.cell_max_counts, row_offset, result.cell.max_counts, part_rows, "cell max counts");
            copy_device_to_host(&session.cell_detected_genes, row_offset, result.cell.detected_genes, part_rows, "cell detected genes");
            copy_device_to_host(&session.cell_keep, row_offset, result.cell.keep_cells, part_rows, "cell keep mask");
            copy_device_to_host(&session.cell_group_counts,
                                row_offset * session.group_names.size(),
                                result.cell.cell_group_counts,
                                part_rows * session.group_names.size(),
                                "cell group counts");
            copy_device_to_host(&session.cell_group_pct,
                                row_offset * session.group_names.size(),
                                result.cell.cell_group_pct,
                                part_rows * session.group_names.size(),
                                "cell group pct");

            std::vector<float> part_gene_sum((std::size_t) summary.cols, 0.0f);
            std::vector<float> part_gene_sq_sum((std::size_t) summary.cols, 0.0f);
            std::vector<float> part_gene_detected((std::size_t) summary.cols, 0.0f);
            copy_device_to_host(&part_gene_sum, 0u, result.gene.sum, (std::size_t) summary.cols, "gene sum");
            copy_device_to_host(&part_gene_sq_sum, 0u, result.gene.sq_sum, (std::size_t) summary.cols, "gene sq sum");
            copy_device_to_host(&part_gene_detected, 0u, result.gene.detected_cells, (std::size_t) summary.cols, "gene detected cells");
            add_inplace(&session.gene_sum, part_gene_sum);
            add_inplace(&session.gene_sq_sum, part_gene_sq_sum);
            add_inplace(&session.gene_detected_cells, part_gene_detected);

            require_cuda(cellshard::device::release_partition(&device_state, part_id), "release GPU partition");
            ++session.partitions_processed;
        }
        for (std::uint8_t keep : session.cell_keep) session.kept_cells += keep != 0u ? 1u : 0u;
        for (float value : session.gene_sum) session.gene_sum_checksum += value;
        cpre::gene_filter_params gene_filter{options.min_gene_sum, options.min_detected_cells, options.min_variance};
        if (!cpre::finalize_gene_keep_mask_host(session.gene_sum.data(),
                                                session.gene_sq_sum.data(),
                                                session.gene_detected_cells.data(),
                                                (unsigned int) summary.cols,
                                                (float) session.kept_cells,
                                                &gene_filter,
                                                session.gene_keep.data(),
                                                &session.kept_genes)) {
            throw std::runtime_error("failed to finalize gene keep mask");
        }
    } catch (...) {
        cellshard::device::clear(&device_state);
        cpre::clear(&workspace);
        throw;
    }

    cellshard::device::clear(&device_state);
    cpre::clear(&workspace);
    return session;
}

PreprocessSession preprocess_cellshard(const std::string &path, const PreprocessOptions &options) {
    validate_raw_or_throw(options);
    cse::dataset_summary summary;
    std::string error;
    if (!cse::load_dataset_summary(path.c_str(), &summary, &error)) throw std::runtime_error(error);
    std::vector<std::uint32_t> masks = compile_default_masks_from_summary(summary);
    if (summary.matrix_format == "blocked_ell") {
        return run_dataset_layout<cellshard::sparse::blocked_ell>(path, "blocked_ell", summary, options, masks);
    }
    if (summary.matrix_format == "sliced_ell") {
        return run_dataset_layout<cellshard::sparse::sliced_ell>(path, "sliced_ell", summary, options, masks);
    }
    throw std::runtime_error("Cellerator Python preprocessing supports blocked_ell and sliced_ell .csh5 datasets in v1");
}

AdapterStagePlan plan_cellshard_adapter_stage(const std::string &path,
                                              const std::string &format,
                                              const std::string &matrix_source,
                                              bool allow_processed) {
    cpre::adapter_source_view source{};
    source.path = path.c_str();
    source.format = format.c_str();
    source.matrix_source = matrix_source.c_str();
    source.allow_processed = allow_processed ? 1u : 0u;
    cpre::cellshard_stage_plan native{};
    cpre::status status{};
    if (!cpre::plan_cellshard_adapter_stage(&source, &native, &status)) throw std::runtime_error(status.message);
    AdapterStagePlan out;
    out.layout = native.layout == cpre::native_sparse_blocked_ell ? "blocked_ell"
        : native.layout == cpre::native_sparse_sliced_ell ? "sliced_ell"
        : native.layout == cpre::native_sparse_compressed_fallback ? "compressed_fallback"
        : "unknown";
    out.value_type = native.value_type == cpre::value_precision_fp16 ? "fp16"
        : native.value_type == cpre::value_precision_fp32_accumulator ? "fp32"
        : "unknown";
    out.accumulator_type = native.accumulator_type == cpre::value_precision_fp32_accumulator ? "fp32"
        : native.accumulator_type == cpre::value_precision_fp16 ? "fp16"
        : "unknown";
    out.adapt_to_cellshard_first = native.adapt_to_cellshard_first != 0u;
    out.direct_external_kernels = native.direct_external_kernels != 0u;
    return out;
}

bool validate_raw_count_state(const std::string &assay,
                              const std::string &matrix_orientation,
                              const std::string &matrix_state,
                              const std::string &feature_namespace,
                              bool raw_counts_available,
                              bool processed_matrix_available,
                              bool preprocess_available) {
    cpre::preprocess_state_view state{};
    state.assay = assay.c_str();
    state.matrix_orientation = matrix_orientation.c_str();
    state.matrix_state = matrix_state.c_str();
    state.feature_namespace = feature_namespace.c_str();
    state.raw_counts_available = raw_counts_available ? 1u : 0u;
    state.processed_matrix_available = processed_matrix_available ? 1u : 0u;
    state.preprocess_available = preprocess_available ? 1u : 0u;
    cpre::status status{};
    if (!cpre::validate_raw_count_state(&state, &status)) throw std::runtime_error(status.message);
    return true;
}

std::vector<std::uint32_t> compile_default_qc_feature_group_masks(const std::vector<std::string> &feature_ids,
                                                                  const std::vector<std::string> &feature_names,
                                                                  const std::vector<std::string> &feature_types,
                                                                  const std::vector<std::string> &modalities) {
    const std::size_t count = feature_names.size();
    if ((!feature_ids.empty() && feature_ids.size() != count)
        || (!feature_types.empty() && feature_types.size() != count)
        || (!modalities.empty() && modalities.size() != count)) {
        throw std::invalid_argument("feature metadata arrays must have matching lengths");
    }
    std::vector<const char *> ids = c_strings(feature_ids);
    std::vector<const char *> names = c_strings(feature_names);
    std::vector<const char *> types = c_strings(feature_types);
    std::vector<const char *> modality_ptrs = c_strings(modalities);
    std::vector<std::uint32_t> out(count, 0u);
    cpre::qc_feature_annotation_view features{
        ids.empty() ? nullptr : ids.data(),
        names.empty() ? nullptr : names.data(),
        types.empty() ? nullptr : types.data(),
        modality_ptrs.empty() ? nullptr : modality_ptrs.data(),
        (std::uint32_t) count
    };
    if (!cpre::compile_default_qc_feature_group_masks(&features, nullptr, out.data())) {
        throw std::runtime_error("failed to compile default QC feature group masks");
    }
    return out;
}

std::vector<std::uint32_t> compile_qc_feature_group_masks(const std::vector<std::string> &feature_ids,
                                                         const std::vector<std::string> &feature_names,
                                                         const std::vector<std::string> &feature_types,
                                                         const std::vector<std::string> &modalities,
                                                         const std::vector<py::dict> &rules) {
    const std::size_t count = feature_names.size();
    if ((!feature_ids.empty() && feature_ids.size() != count)
        || (!feature_types.empty() && feature_types.size() != count)
        || (!modalities.empty() && modalities.size() != count)) {
        throw std::invalid_argument("feature metadata arrays must have matching lengths");
    }
    std::vector<std::string> group_names, prefixes, exact_ids, exact_names, rule_types, rule_modalities;
    std::vector<cpre::qc_group_rule_view> native_rules;
    group_names.reserve(rules.size());
    prefixes.reserve(rules.size());
    exact_ids.reserve(rules.size());
    exact_names.reserve(rules.size());
    rule_types.reserve(rules.size());
    rule_modalities.reserve(rules.size());
    native_rules.reserve(rules.size());
    for (const py::dict &rule : rules) {
        group_names.push_back(py::cast<std::string>(rule.contains("group_name") ? rule["group_name"] : py::str("custom")));
        prefixes.push_back(rule.contains("prefix") && !rule["prefix"].is_none() ? py::cast<std::string>(rule["prefix"]) : std::string{});
        exact_ids.push_back(rule.contains("exact_feature_id") && !rule["exact_feature_id"].is_none() ? py::cast<std::string>(rule["exact_feature_id"]) : std::string{});
        exact_names.push_back(rule.contains("exact_feature_name") && !rule["exact_feature_name"].is_none() ? py::cast<std::string>(rule["exact_feature_name"]) : std::string{});
        rule_types.push_back(rule.contains("feature_type") && !rule["feature_type"].is_none() ? py::cast<std::string>(rule["feature_type"]) : std::string{});
        rule_modalities.push_back(rule.contains("modality") && !rule["modality"].is_none() ? py::cast<std::string>(rule["modality"]) : std::string{});
        cpre::qc_group_rule_view native{};
        native.group_index = rule.contains("group_index") ? py::cast<std::uint32_t>(rule["group_index"]) : 0u;
        native.group_name = group_names.back().c_str();
        native.prefix = prefixes.back().empty() ? nullptr : prefixes.back().c_str();
        native.exact_feature_id = exact_ids.back().empty() ? nullptr : exact_ids.back().c_str();
        native.exact_feature_name = exact_names.back().empty() ? nullptr : exact_names.back().c_str();
        native.feature_type = rule_types.back().empty() ? nullptr : rule_types.back().c_str();
        native.modality = rule_modalities.back().empty() ? nullptr : rule_modalities.back().c_str();
        native_rules.push_back(native);
    }
    std::vector<const char *> ids = c_strings(feature_ids);
    std::vector<const char *> names = c_strings(feature_names);
    std::vector<const char *> types = c_strings(feature_types);
    std::vector<const char *> modality_ptrs = c_strings(modalities);
    std::vector<std::uint32_t> out(count, 0u);
    cpre::qc_feature_annotation_view features{
        ids.empty() ? nullptr : ids.data(),
        names.empty() ? nullptr : names.data(),
        types.empty() ? nullptr : types.data(),
        modality_ptrs.empty() ? nullptr : modality_ptrs.data(),
        (std::uint32_t) count
    };
    if (!cpre::compile_qc_feature_group_masks(&features,
                                              native_rules.empty() ? nullptr : native_rules.data(),
                                              (std::uint32_t) native_rules.size(),
                                              nullptr,
                                              out.data())) {
        throw std::runtime_error("failed to compile QC feature group masks");
    }
    return out;
}

} // namespace cellerator::python_bindings

PYBIND11_MODULE(_cellerator, m) {
    using namespace cellerator::python_bindings;
    m.doc() = "GPU-native Cellerator Python bindings for sparse omics preprocessing.";

    py::class_<PreprocessOptions>(m, "PreprocessOptions")
        .def(py::init<>())
        .def_readwrite("assay", &PreprocessOptions::assay)
        .def_readwrite("matrix_orientation", &PreprocessOptions::matrix_orientation)
        .def_readwrite("matrix_state", &PreprocessOptions::matrix_state)
        .def_readwrite("feature_namespace", &PreprocessOptions::feature_namespace)
        .def_readwrite("mito_prefix", &PreprocessOptions::mito_prefix)
        .def_readwrite("target_sum", &PreprocessOptions::target_sum)
        .def_readwrite("min_counts", &PreprocessOptions::min_counts)
        .def_readwrite("min_features", &PreprocessOptions::min_features)
        .def_readwrite("max_group_fraction", &PreprocessOptions::max_group_fraction)
        .def_readwrite("fraction_group_index", &PreprocessOptions::fraction_group_index)
        .def_readwrite("min_gene_sum", &PreprocessOptions::min_gene_sum)
        .def_readwrite("min_detected_cells", &PreprocessOptions::min_detected_cells)
        .def_readwrite("min_variance", &PreprocessOptions::min_variance)
        .def_readwrite("device", &PreprocessOptions::device)
        .def_readwrite("allow_processed", &PreprocessOptions::allow_processed);

    py::class_<AdapterStagePlan>(m, "AdapterStagePlan")
        .def_readonly("layout", &AdapterStagePlan::layout)
        .def_readonly("value_type", &AdapterStagePlan::value_type)
        .def_readonly("accumulator_type", &AdapterStagePlan::accumulator_type)
        .def_readonly("adapt_to_cellshard_first", &AdapterStagePlan::adapt_to_cellshard_first)
        .def_readonly("direct_external_kernels", &AdapterStagePlan::direct_external_kernels);

    py::class_<PreprocessSession>(m, "PreprocessSession")
        .def_readonly("source_path", &PreprocessSession::source_path)
        .def_readonly("layout", &PreprocessSession::layout)
        .def_readonly("rows", &PreprocessSession::rows)
        .def_readonly("cols", &PreprocessSession::cols)
        .def_readonly("nnz", &PreprocessSession::nnz)
        .def_readonly("partitions_processed", &PreprocessSession::partitions_processed)
        .def_readonly("kept_cells", &PreprocessSession::kept_cells)
        .def_readonly("kept_genes", &PreprocessSession::kept_genes)
        .def("metrics", &PreprocessSession::metrics)
        .def("publish", &PreprocessSession::publish, py::arg("output_path"), py::arg("working_root") = "");

    m.def("preprocess_cellshard", &preprocess_cellshard, py::arg("path"), py::arg("options") = PreprocessOptions{});
    m.def("plan_cellshard_adapter_stage",
          &plan_cellshard_adapter_stage,
          py::arg("path"),
          py::arg("format") = "h5ad",
          py::arg("matrix_source") = "counts",
          py::arg("allow_processed") = false);
    m.def("validate_raw_count_state",
          &validate_raw_count_state,
          py::arg("assay") = "scrna",
          py::arg("matrix_orientation") = "observations_by_features",
          py::arg("matrix_state") = "raw_counts",
          py::arg("feature_namespace") = "gene_symbol",
          py::arg("raw_counts_available") = true,
          py::arg("processed_matrix_available") = false,
          py::arg("preprocess_available") = false);
    m.def("compile_default_qc_feature_group_masks",
          [](const std::vector<std::string> &feature_ids,
             const std::vector<std::string> &feature_names,
             const std::vector<std::string> &feature_types,
             const std::vector<std::string> &modalities) {
              return copy_array(compile_default_qc_feature_group_masks(feature_ids, feature_names, feature_types, modalities));
          },
          py::arg("feature_ids"),
          py::arg("feature_names"),
          py::arg("feature_types") = std::vector<std::string>{},
          py::arg("modalities") = std::vector<std::string>{});
    m.def("compile_qc_feature_group_masks",
          [](const std::vector<std::string> &feature_ids,
             const std::vector<std::string> &feature_names,
             const std::vector<std::string> &feature_types,
             const std::vector<std::string> &modalities,
             const std::vector<py::dict> &rules) {
              return copy_array(compile_qc_feature_group_masks(feature_ids, feature_names, feature_types, modalities, rules));
          },
          py::arg("feature_ids"),
          py::arg("feature_names"),
          py::arg("feature_types"),
          py::arg("modalities"),
          py::arg("rules"));
}
