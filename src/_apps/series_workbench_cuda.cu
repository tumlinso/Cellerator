#include "series_workbench.hh"

#include "../compute/preprocess/operators.cuh"
#include "../compute/preprocess/workspace.cuh"
#include "../ingest/series/series_ingest.cuh"

#include "../../extern/CellShard/src/sharded/disk.cuh"
#include "../../extern/CellShard/src/sharded/sharded_device.cuh"
#include "../../extern/CellShard/src/sharded/sharded_host.cuh"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace cellerator::apps::workbench {

namespace cpre = ::cellerator::compute::preprocess;
namespace cseries = ::cellerator::ingest::series;
namespace cs = ::cellshard;
namespace csv = ::cellshard::device;

namespace {

inline void push_issue(std::vector<issue> *issues,
                       issue_severity severity,
                       const std::string &scope,
                       const std::string &message) {
    if (issues == nullptr) return;
    issues->push_back(issue{severity, scope, message});
}

inline bool check_cuda(cudaError_t status,
                       std::vector<issue> *issues,
                       const std::string &scope,
                       const std::string &label) {
    if (status == cudaSuccess) return true;
    push_issue(issues,
               issue_severity::error,
               scope,
               label + ": " + std::string(cudaGetErrorString(status)));
    return false;
}

inline std::string normalized_upper(std::string value) {
    for (char &ch : value) ch = (char) std::toupper((unsigned char) ch);
    return value;
}

inline std::vector<unsigned char> build_gene_flags(const series_summary &summary,
                                                   const preprocess_config &config) {
    std::vector<unsigned char> flags(summary.cols, 0u);
    if (!config.mark_mito_from_feature_names || config.mito_prefix.empty()) return flags;
    const std::string prefix = normalized_upper(config.mito_prefix);
    for (std::size_t i = 0; i < summary.feature_names.size() && i < flags.size(); ++i) {
        std::string name = normalized_upper(summary.feature_names[i]);
        if (name.rfind(prefix, 0) == 0) flags[i] = (unsigned char) cpre::gene_flag_mito;
    }
    return flags;
}

inline unsigned int max_part_rows(const cs::sharded<cs::sparse::compressed> &view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view.num_parts; ++part) best = std::max(best, view.part_rows[part]);
    return (unsigned int) best;
}

inline unsigned int max_part_nnz(const cs::sharded<cs::sparse::compressed> &view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view.num_parts; ++part) best = std::max(best, view.part_nnz[part]);
    return (unsigned int) best;
}

} // namespace

conversion_report convert_plan_to_series_csh5(const ingest_plan &plan) {
    conversion_report report;
    cseries::manifest manifest;
    cseries::series_h5_convert_options options;
    cs::sharded<cs::sparse::compressed> header;
    cs::shard_storage storage;
    cseries::init(&manifest);
    cseries::init(&options);
    cs::init(&header);
    cs::init(&storage);

    report.events.push_back({"validate", "validating ingest plan"});
    if (!plan.ok) {
        push_issue(&report.issues, issue_severity::error, "convert", "ingest plan is not valid");
        return report;
    }
    if (plan.policy.output_path.empty()) {
        push_issue(&report.issues, issue_severity::error, "convert", "output path is empty");
        return report;
    }

    for (const source_entry &source : plan.sources) {
        if (!source.included) continue;
        if (!cseries::append(&manifest,
                             source.dataset_id.c_str(),
                             source.matrix_path.c_str(),
                             source.format,
                             source.feature_path.c_str(),
                             source.barcode_path.c_str(),
                             source.metadata_path.c_str(),
                             source.rows,
                             source.cols,
                             source.nnz)) {
            push_issue(&report.issues, issue_severity::error, "convert", "failed to materialize manifest from ingest plan");
            cseries::clear(&manifest);
            return report;
        }
    }

    options.max_part_nnz = plan.policy.max_part_nnz;
    options.max_window_bytes = plan.policy.max_window_bytes;
    options.reader_bytes = plan.policy.reader_bytes;
    options.device = plan.policy.device;
    options.stream = (cudaStream_t) 0;

    report.events.push_back({"convert", "writing series.csh5"});
    if (!cseries::convert_manifest_mtx_series_to_hdf5(&manifest, plan.policy.output_path.c_str(), &options)) {
        push_issue(&report.issues, issue_severity::error, "convert", "series ingest conversion failed");
        cseries::clear(&manifest);
        return report;
    }

    if (plan.policy.verify_after_write) {
        report.events.push_back({"verify", "loading written header"});
        if (!cs::load_header(plan.policy.output_path.c_str(), &header, &storage)) {
            push_issue(&report.issues, issue_severity::error, "verify", "failed to reopen written series.csh5 header");
            cseries::clear(&manifest);
            cs::clear(&storage);
            cs::clear(&header);
            return report;
        }
        if (header.rows != plan.total_rows || header.num_parts != plan.parts.size() || header.num_shards != plan.shards.size()) {
            push_issue(&report.issues, issue_severity::error, "verify", "written header does not match the planned layout");
            cseries::clear(&manifest);
            cs::clear(&storage);
            cs::clear(&header);
            return report;
        }
    }

    report.events.push_back({"done", "conversion completed"});
    report.ok = true;
    cseries::clear(&manifest);
    cs::clear(&storage);
    cs::clear(&header);
    return report;
}

preprocess_summary run_preprocess_pass(const std::string &path, const preprocess_config &config) {
    preprocess_summary summary;
    cs::sharded<cs::sparse::compressed> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::compressed> device_state;
    cpre::device_workspace workspace;
    series_summary series = summarize_series_csh5(path);
    std::vector<unsigned char> gene_flags;
    std::vector<unsigned char> host_keep_genes;
    std::vector<float> host_gene_sum;
    float kept_cells = 0.0f;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);
    cpre::init(&workspace);

    if (!series.ok) {
        summary.issues = series.issues;
        push_issue(&summary.issues, issue_severity::error, "preprocess", "cannot preprocess an unreadable series.csh5");
        return summary;
    }

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), &summary.issues, "preprocess", "cudaGetDeviceCount")) return summary;
    if (device_count == 0) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "no CUDA devices are available");
        return summary;
    }
    if (config.device < 0 || config.device >= device_count) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "requested CUDA device is out of range");
        return summary;
    }

    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to load series header");
        return summary;
    }
    if (!config.cache_dir.empty() && !cs::bind_series_h5_part_cache(&storage, config.cache_dir.c_str())) {
        push_issue(&summary.issues, issue_severity::warning, "preprocess", "failed to bind requested part cache directory");
    }

    if (!csv::reserve(&device_state, matrix.num_parts)) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to reserve device part state");
        goto done;
    }
    if (!cpre::setup(&workspace, config.device, (cudaStream_t) 0)
        || !cpre::reserve(&workspace, max_part_rows(matrix), (unsigned int) matrix.cols, max_part_nnz(matrix))) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to set up preprocess workspace");
        goto done;
    }

    gene_flags = build_gene_flags(series, config);
    if (!cpre::upload_gene_flags(&workspace, (unsigned int) matrix.cols, gene_flags.empty() ? nullptr : gene_flags.data())
        || !cpre::zero_gene_metrics(&workspace, (unsigned int) matrix.cols)) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to initialize preprocess workspace state");
        goto done;
    }

    for (unsigned long part_id = 0; part_id < matrix.num_parts; ++part_id) {
        bool loaded_here = false;
        csv::compressed_view part_view;
        if (!cs::part_loaded(&matrix, part_id)) {
            if (!cs::fetch_part(&matrix, &storage, part_id)) {
                push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to fetch matrix part from storage");
                goto done;
            }
            loaded_here = true;
        }
        if (!check_cuda(csv::upload_part(&device_state, &matrix, part_id, config.device), &summary.issues, "preprocess", "upload_part")) goto done;
        if (!cpre::bind_uploaded_part_view(&part_view, &matrix, device_state.parts + part_id, part_id)) {
            push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to bind uploaded device part view");
            goto done;
        }
        if (!cpre::preprocess_part_inplace(&part_view, &workspace, config.cell_filter, config.target_sum, nullptr)) {
            push_issue(&summary.issues, issue_severity::error, "preprocess", "GPU preprocess kernel pass failed");
            goto done;
        }
        if (!check_cuda(cudaStreamSynchronize(workspace.stream), &summary.issues, "preprocess", "cudaStreamSynchronize part")) goto done;
        if (!check_cuda(csv::release_part(&device_state, part_id), &summary.issues, "preprocess", "release_part")) goto done;
        ++summary.parts_processed;
        if (loaded_here && config.drop_host_parts) cs::drop_part(&matrix, part_id);
    }

    if (!cpre::build_gene_filter_mask(&workspace, (unsigned int) matrix.cols, config.gene_filter, nullptr)
        || !check_cuda(cudaStreamSynchronize(workspace.stream), &summary.issues, "preprocess", "cudaStreamSynchronize final")) {
        push_issue(&summary.issues, issue_severity::error, "preprocess", "failed to finalize gene keep mask");
        goto done;
    }

    host_keep_genes.assign((std::size_t) matrix.cols, 0u);
    host_gene_sum.assign((std::size_t) matrix.cols, 0.0f);
    if (!check_cuda(cudaMemcpy(host_keep_genes.data(),
                               workspace.d_keep_genes,
                               host_keep_genes.size() * sizeof(unsigned char),
                               cudaMemcpyDeviceToHost),
                    &summary.issues,
                    "preprocess",
                    "cudaMemcpy keep genes")) goto done;
    if (!check_cuda(cudaMemcpy(host_gene_sum.data(),
                               workspace.d_gene_sum,
                               host_gene_sum.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    &summary.issues,
                    "preprocess",
                    "cudaMemcpy gene sum")) goto done;
    if (!check_cuda(cudaMemcpy(&kept_cells,
                               workspace.d_active_rows,
                               sizeof(float),
                               cudaMemcpyDeviceToHost),
                    &summary.issues,
                    "preprocess",
                    "cudaMemcpy kept cells")) goto done;

    summary.device = config.device;
    summary.rows = matrix.rows;
    summary.cols = matrix.cols;
    summary.nnz = matrix.nnz;
    summary.kept_cells = kept_cells;
    for (unsigned char keep : host_keep_genes) summary.kept_genes += keep != 0;
    for (float value : host_gene_sum) summary.gene_sum_checksum += (double) value;
    summary.ok = true;

done:
    cpre::clear(&workspace);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return summary;
}

} // namespace cellerator::apps::workbench
