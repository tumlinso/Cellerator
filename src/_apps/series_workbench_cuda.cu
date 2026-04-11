#include "series_workbench.hh"

#include "../compute/preprocess/operators.cuh"
#include "../compute/preprocess/workspace.cuh"
#include "../ingest/common/metadata_table.cuh"
#include "../ingest/series/series_ingest.cuh"

#include "../../extern/CellShard/src/sharded/disk.cuh"
#include "../../extern/CellShard/src/sharded/distributed.cuh"
#include "../../extern/CellShard/src/sharded/sharded_device.cuh"
#include "../../extern/CellShard/src/sharded/sharded_host.cuh"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cellerator::apps::workbench {

namespace ccommon = ::cellerator::ingest::common;
namespace cpre = ::cellerator::compute::preprocess;
namespace cseries = ::cellerator::ingest::series;
namespace cs = ::cellshard;
namespace csd = ::cellshard::distributed;
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

inline cellshard::series_text_column_view as_text_view(const ccommon::text_column *column) {
    cellshard::series_text_column_view view;
    view.count = column != nullptr ? column->count : 0u;
    view.bytes = column != nullptr ? column->bytes : 0u;
    view.offsets = column != nullptr ? column->offsets : nullptr;
    view.data = column != nullptr ? column->data : nullptr;
    return view;
}

struct owned_metadata_table {
    ccommon::metadata_table table;

    owned_metadata_table() { ccommon::init(&table); }
    ~owned_metadata_table() { ccommon::clear(&table); }
    owned_metadata_table(const owned_metadata_table &) = delete;
    owned_metadata_table &operator=(const owned_metadata_table &) = delete;
};

struct browse_cache_owned {
    std::vector<std::uint32_t> selected_feature_indices;
    std::vector<float> gene_sum;
    std::vector<float> gene_detected;
    std::vector<float> gene_sq_sum;
    std::vector<float> dataset_feature_mean;
    std::vector<float> shard_feature_mean;
    std::vector<std::uint32_t> part_sample_row_offsets;
    std::vector<std::uint64_t> part_sample_global_rows;
    std::vector<float> part_sample_values;
};

struct gene_metric_partial {
    std::vector<float> gene_sum;
    std::vector<float> gene_detected;
    std::vector<float> gene_sq_sum;
    float active_rows = 0.0f;
    bool ok = false;
};

struct selected_feature_partial {
    std::vector<float> dataset_feature_sum;
    std::vector<float> shard_feature_sum;
    bool ok = false;
};

__global__ void accumulate_selected_feature_sums_kernel(csv::compressed_view src,
                                                        const unsigned int * __restrict__ selected,
                                                        unsigned int selected_count,
                                                        float * __restrict__ dst_a,
                                                        float * __restrict__ dst_b) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int nz = tid;

    while (nz < src.nnz) {
        const unsigned int gene = src.minorIdx[nz];
        const float value = __half2float(src.val[nz]);
        for (unsigned int k = 0; k < selected_count; ++k) {
            if (selected[k] != gene) continue;
            if (dst_a != nullptr) atomicAdd(dst_a + k, value);
            if (dst_b != nullptr) atomicAdd(dst_b + k, value);
            break;
        }
        nz += stride;
    }
}

__global__ void extract_sample_tile_kernel(csv::compressed_view src,
                                           const unsigned int * __restrict__ sample_rows,
                                           unsigned int sample_count,
                                           const unsigned int * __restrict__ selected,
                                           unsigned int selected_count,
                                           float * __restrict__ out) {
    const unsigned int sample_idx = (unsigned int) blockIdx.x;
    if (sample_idx >= sample_count) return;

    const unsigned int row = sample_rows[sample_idx];
    if (row >= src.rows) return;

    for (unsigned int k = threadIdx.x; k < selected_count; k += blockDim.x) {
        out[(std::size_t) sample_idx * selected_count + k] = 0.0f;
    }
    __syncthreads();

    const unsigned int begin = src.majorPtr[row];
    const unsigned int end = src.majorPtr[row + 1u];
    for (unsigned int nz = begin + (unsigned int) threadIdx.x; nz < end; nz += (unsigned int) blockDim.x) {
        const unsigned int gene = src.minorIdx[nz];
        const float value = __half2float(src.val[nz]);
        for (unsigned int k = 0; k < selected_count; ++k) {
            if (selected[k] != gene) continue;
            out[(std::size_t) sample_idx * selected_count + k] = value;
            break;
        }
    }
}

inline bool append_embedded_metadata_tables(const ingest_plan &plan,
                                            std::vector<issue> *issues) {
    std::vector<owned_metadata_table *> owned;
    std::vector<cellshard::series_metadata_table_view> views;
    std::vector<std::uint32_t> dataset_indices;
    std::vector<std::uint64_t> global_row_begin;
    std::vector<std::uint64_t> global_row_end;
    cellshard::series_embedded_metadata_view metadata_view{};
    bool ok = true;

    owned.reserve(plan.datasets.size());
    views.reserve(plan.datasets.size());
    dataset_indices.reserve(plan.datasets.size());
    global_row_begin.reserve(plan.datasets.size());
    global_row_end.reserve(plan.datasets.size());

    for (std::size_t i = 0; i < plan.datasets.size(); ++i) {
        const planned_dataset &dataset = plan.datasets[i];
        const source_entry &source = plan.sources[dataset.source_index];
        if (source.metadata_path.empty()) continue;

        owned_metadata_table *owned_table = new owned_metadata_table();
        if (!ccommon::load_tsv(source.metadata_path.c_str(), &owned_table->table, 1)) {
            push_issue(issues, issue_severity::warning, "metadata", "failed to embed metadata table for " + dataset.dataset_id);
            delete owned_table;
            continue;
        }
        if (owned_table->table.num_rows != dataset.rows) {
            push_issue(issues,
                       issue_severity::error,
                       "metadata",
                       "metadata row count does not match barcodes for " + dataset.dataset_id);
            delete owned_table;
            ok = false;
            continue;
        }

        dataset_indices.push_back((std::uint32_t) i);
        global_row_begin.push_back((std::uint64_t) dataset.global_row_begin);
        global_row_end.push_back((std::uint64_t) dataset.global_row_end);
        views.push_back(cellshard::series_metadata_table_view{
            owned_table->table.num_rows,
            owned_table->table.num_cols,
            as_text_view(&owned_table->table.column_names),
            as_text_view(&owned_table->table.field_values),
            owned_table->table.row_offsets
        });
        owned.push_back(owned_table);
    }

    if (!ok) {
        for (owned_metadata_table *table : owned) delete table;
        return false;
    }

    metadata_view.count = (std::uint32_t) views.size();
    metadata_view.dataset_indices = dataset_indices.empty() ? nullptr : dataset_indices.data();
    metadata_view.global_row_begin = global_row_begin.empty() ? nullptr : global_row_begin.data();
    metadata_view.global_row_end = global_row_end.empty() ? nullptr : global_row_end.data();
    metadata_view.tables = views.empty() ? nullptr : views.data();

    if (!cellshard::append_series_embedded_metadata_h5(plan.policy.output_path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to append embedded metadata to series.csh5");
        ok = false;
    }

    for (owned_metadata_table *table : owned) delete table;
    return ok;
}

inline std::vector<unsigned int> choose_sample_rows(unsigned int rows, unsigned int sample_rows) {
    std::vector<unsigned int> out(sample_rows, std::numeric_limits<unsigned int>::max());
    if (rows == 0 || sample_rows == 0) return out;
    if (rows <= sample_rows) {
        for (unsigned int i = 0; i < rows; ++i) out[i] = i;
        return out;
    }
    for (unsigned int i = 0; i < sample_rows; ++i) {
        const unsigned long num = (unsigned long) i * (unsigned long) rows;
        out[i] = (unsigned int) std::min<unsigned long>(rows - 1u, num / sample_rows);
    }
    return out;
}

inline bool build_gene_metric_partials(const std::string &path,
                                       const std::vector<int> &shard_owner,
                                       unsigned int worker_slot,
                                       int device_id,
                                       unsigned int cols,
                                       unsigned int max_rows,
                                       unsigned int max_nnz,
                                       gene_metric_partial *out,
                                       std::vector<issue> *issues) {
    cs::sharded<cs::sparse::compressed> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::compressed> device_state;
    cpre::device_workspace workspace;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);
    cpre::init(&workspace);

    if (!cs::load_header(path.c_str(), &matrix, &storage)
        || !csv::reserve(&device_state, matrix.num_parts)
        || !cpre::setup(&workspace, device_id, (cudaStream_t) 0)
        || !cpre::reserve(&workspace, max_rows, cols, max_nnz)
        || !cpre::zero_gene_metrics(&workspace, cols)) {
        push_issue(issues, issue_severity::error, "browse", "failed to set up multi-GPU gene metric worker");
        goto done;
    }

    for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        if (shard_id >= shard_owner.size() || shard_owner[shard_id] != (int) worker_slot) continue;
        const unsigned long part_begin = cs::first_part_in_shard(&matrix, shard_id);
        const unsigned long part_end = cs::last_part_in_shard(&matrix, shard_id);
        for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
            csv::compressed_view part_view;
            if (!cs::fetch_part(&matrix, &storage, part_id)
                || !check_cuda(csv::upload_part(&device_state, &matrix, part_id, device_id), issues, "browse", "upload_part gene metrics")
                || !cpre::bind_uploaded_part_view(&part_view, &matrix, device_state.parts + part_id, part_id)
                || !cpre::accumulate_gene_metrics(&part_view, &workspace, nullptr, nullptr)
                || !check_cuda(cudaStreamSynchronize(workspace.stream), issues, "browse", "cudaStreamSynchronize gene metrics")
                || !check_cuda(csv::release_part(&device_state, part_id), issues, "browse", "release_part gene metrics")) {
                goto done;
            }
            cs::drop_part(&matrix, part_id);
        }
    }

    out->gene_sum.assign(cols, 0.0f);
    out->gene_detected.assign(cols, 0.0f);
    out->gene_sq_sum.assign(cols, 0.0f);
    if (!check_cuda(cudaMemcpy(out->gene_sum.data(), workspace.d_gene_sum, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                    issues, "browse", "cudaMemcpy gene_sum")
        || !check_cuda(cudaMemcpy(out->gene_detected.data(), workspace.d_gene_detected, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy gene_detected")
        || !check_cuda(cudaMemcpy(out->gene_sq_sum.data(), workspace.d_gene_sq_sum, (std::size_t) cols * sizeof(float), cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy gene_sq_sum")
        || !check_cuda(cudaMemcpy(&out->active_rows, workspace.d_active_rows, sizeof(float), cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy active_rows")) {
        goto done;
    }
    out->ok = true;

done:
    cpre::clear(&workspace);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return out->ok;
}

inline bool build_selected_feature_partials(const std::string &path,
                                            const std::vector<int> &shard_owner,
                                            unsigned int worker_slot,
                                            int device_id,
                                            const std::vector<std::uint32_t> &part_dataset_indices,
                                            unsigned int dataset_count,
                                            unsigned int shard_count,
                                            const std::vector<std::uint32_t> &selected_features,
                                            unsigned int sample_rows_per_part,
                                            std::vector<float> *dataset_feature_sum,
                                            std::vector<float> *shard_feature_sum,
                                            std::vector<std::uint64_t> *part_sample_global_rows,
                                            std::vector<float> *part_sample_values,
                                            std::vector<issue> *issues) {
    cs::sharded<cs::sparse::compressed> matrix;
    cs::shard_storage storage;
    csv::sharded_device<cs::sparse::compressed> device_state;
    unsigned int *d_selected = nullptr;
    unsigned int *d_sample_rows = nullptr;
    float *d_dataset = nullptr;
    float *d_shards = nullptr;
    float *d_tile = nullptr;
    bool ok = false;

    cs::init(&matrix);
    cs::init(&storage);
    csv::init(&device_state);

    if (!cs::load_header(path.c_str(), &matrix, &storage)
        || !csv::reserve(&device_state, matrix.num_parts)
        || !check_cuda(cudaSetDevice(device_id), issues, "browse", "cudaSetDevice selected features")) goto done;

    if (!selected_features.empty()
        && !check_cuda(cudaMalloc((void **) &d_selected, selected_features.size() * sizeof(unsigned int)), issues, "browse", "cudaMalloc selected features")) goto done;
    if (!selected_features.empty()
        && !check_cuda(cudaMemcpy(d_selected,
                                  selected_features.data(),
                                  selected_features.size() * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice),
                       issues, "browse", "cudaMemcpy selected features")) goto done;
    if (sample_rows_per_part != 0u
        && !check_cuda(cudaMalloc((void **) &d_sample_rows, (std::size_t) sample_rows_per_part * sizeof(unsigned int)), issues, "browse", "cudaMalloc sample rows")) goto done;
    if (!selected_features.empty()
        && dataset_count != 0u
        && !check_cuda(cudaMalloc((void **) &d_dataset,
                                  (std::size_t) dataset_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc dataset feature sums")) goto done;
    if (!selected_features.empty()
        && shard_count != 0u
        && !check_cuda(cudaMalloc((void **) &d_shards,
                                  (std::size_t) shard_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc shard feature sums")) goto done;
    if (d_dataset != nullptr
        && !check_cuda(cudaMemset(d_dataset, 0, (std::size_t) dataset_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMemset dataset feature sums")) goto done;
    if (d_shards != nullptr
        && !check_cuda(cudaMemset(d_shards, 0, (std::size_t) shard_count * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMemset shard feature sums")) goto done;
    if (sample_rows_per_part != 0u && !selected_features.empty()
        && !check_cuda(cudaMalloc((void **) &d_tile,
                                  (std::size_t) sample_rows_per_part * selected_features.size() * sizeof(float)),
                       issues, "browse", "cudaMalloc sample tile")) goto done;

    dataset_feature_sum->assign((std::size_t) dataset_count * selected_features.size(), 0.0f);
    shard_feature_sum->assign((std::size_t) shard_count * selected_features.size(), 0.0f);

    for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        if (shard_id >= shard_owner.size() || shard_owner[shard_id] != (int) worker_slot) continue;
        const unsigned long part_begin = cs::first_part_in_shard(&matrix, shard_id);
        const unsigned long part_end = cs::last_part_in_shard(&matrix, shard_id);
        for (unsigned long part_id = part_begin; part_id < part_end; ++part_id) {
            csv::compressed_view part_view;
            const std::size_t tile_offset = (std::size_t) part_id * sample_rows_per_part * selected_features.size();
            const std::size_t row_offset = (std::size_t) part_id * sample_rows_per_part;
            const std::uint32_t dataset_id = part_id < part_dataset_indices.size() ? part_dataset_indices[part_id] : 0u;
            const unsigned int blocks = (unsigned int) std::min<unsigned long>(4096ul, (matrix.part_nnz[part_id] + 255ul) / 256ul);
            std::vector<unsigned int> sample_rows = choose_sample_rows((unsigned int) matrix.part_rows[part_id], sample_rows_per_part);

            if (!cs::fetch_part(&matrix, &storage, part_id)
                || !check_cuda(csv::upload_part(&device_state, &matrix, part_id, device_id), issues, "browse", "upload_part selected features")
                || !cpre::bind_uploaded_part_view(&part_view, &matrix, device_state.parts + part_id, part_id)) {
                goto done;
            }

            if (!selected_features.empty()) {
                accumulate_selected_feature_sums_kernel<<<std::max(1u, blocks), 256>>>(
                    part_view,
                    d_selected,
                    (unsigned int) selected_features.size(),
                    d_dataset != nullptr ? d_dataset + (std::size_t) dataset_id * selected_features.size() : nullptr,
                    d_shards != nullptr ? d_shards + (std::size_t) shard_id * selected_features.size() : nullptr
                );
                if (!check_cuda(cudaGetLastError(), issues, "browse", "accumulate_selected_feature_sums_kernel")) goto done;
            }

            if (sample_rows_per_part != 0u && !selected_features.empty()) {
                if (!check_cuda(cudaMemcpy(d_sample_rows,
                                           sample_rows.data(),
                                           (std::size_t) sample_rows_per_part * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice),
                                issues, "browse", "cudaMemcpy sample rows")) goto done;
                if (!check_cuda(cudaMemset(d_tile,
                                           0,
                                           (std::size_t) sample_rows_per_part * selected_features.size() * sizeof(float)),
                                issues, "browse", "cudaMemset sample tile")) goto done;
                extract_sample_tile_kernel<<<sample_rows_per_part, 128>>>(
                    part_view,
                    d_sample_rows,
                    sample_rows_per_part,
                    d_selected,
                    (unsigned int) selected_features.size(),
                    d_tile
                );
                if (!check_cuda(cudaGetLastError(), issues, "browse", "extract_sample_tile_kernel")) goto done;
                if (!check_cuda(cudaMemcpy(part_sample_values->data() + tile_offset,
                                           d_tile,
                                           (std::size_t) sample_rows_per_part * selected_features.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                issues, "browse", "cudaMemcpy sample tile")) goto done;
            }

            if (!check_cuda(cudaDeviceSynchronize(), issues, "browse", "cudaDeviceSynchronize selected features")
                || !check_cuda(csv::release_part(&device_state, part_id), issues, "browse", "release_part selected features")) goto done;
            for (unsigned int row = 0; row < sample_rows_per_part; ++row) {
                const unsigned int local_row = sample_rows[row];
                (*part_sample_global_rows)[row_offset + row] =
                    local_row < matrix.part_rows[part_id]
                        ? (std::uint64_t) (matrix.part_offsets[part_id] + local_row)
                        : std::numeric_limits<std::uint64_t>::max();
            }
            cs::drop_part(&matrix, part_id);
        }
    }

    if (d_dataset != nullptr
        && !check_cuda(cudaMemcpy(dataset_feature_sum->data(),
                                  d_dataset,
                                  dataset_feature_sum->size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy dataset_feature_sum")) goto done;
    if (d_shards != nullptr
        && !check_cuda(cudaMemcpy(shard_feature_sum->data(),
                                  d_shards,
                                  shard_feature_sum->size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       issues, "browse", "cudaMemcpy shard_feature_sum")) goto done;

    ok = true;

done:
    if (d_tile != nullptr) cudaFree(d_tile);
    if (d_shards != nullptr) cudaFree(d_shards);
    if (d_dataset != nullptr) cudaFree(d_dataset);
    if (d_sample_rows != nullptr) cudaFree(d_sample_rows);
    if (d_selected != nullptr) cudaFree(d_selected);
    csv::clear(&device_state);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
}

inline bool build_browse_cache_multigpu(const std::string &path,
                                        const ingest_plan &plan,
                                        std::vector<issue> *issues) {
    cs::sharded<cs::sparse::compressed> matrix;
    cs::shard_storage storage;
    csd::local_context ctx;
    csd::shard_map shard_map;
    browse_cache_owned owned;
    bool ok = false;
    unsigned int max_rows = 0;
    unsigned int max_nnz = 0;

    cs::init(&matrix);
    cs::init(&storage);
    csd::init(&ctx);
    csd::init(&shard_map);

    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(issues, issue_severity::error, "browse", "failed to reload series header for browse cache build");
        goto done;
    }
    max_rows = max_part_rows(matrix);
    max_nnz = max_part_nnz(matrix);

    if (!check_cuda(csd::discover_local(&ctx, 1, cudaStreamNonBlocking), issues, "browse", "discover_local")
        || !check_cuda(csd::enable_peer_access(&ctx), issues, "browse", "enable_peer_access")) goto done;
    if (ctx.device_count < 4u) {
        push_issue(issues, issue_severity::error, "browse", "browse cache generation requires 4 visible GPUs");
        goto done;
    }
    if (!csd::assign_shards_by_bytes(&shard_map, &matrix, &ctx)) {
        push_issue(issues, issue_severity::error, "browse", "failed to assign shards across GPUs");
        goto done;
    }

    {
        std::vector<int> shard_owner(matrix.num_shards, -1);
        std::vector<gene_metric_partial> partials(ctx.device_count);
        std::vector<std::thread> workers;
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }

        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.emplace_back([&, slot]() {
                (void) build_gene_metric_partials(path,
                                                  shard_owner,
                                                  slot,
                                                  ctx.device_ids[slot],
                                                  (unsigned int) matrix.cols,
                                                  max_rows,
                                                  max_nnz,
                                                  partials.data() + slot,
                                                  issues);
            });
        }
        for (std::thread &worker : workers) worker.join();

        owned.gene_sum.assign((std::size_t) matrix.cols, 0.0f);
        owned.gene_detected.assign((std::size_t) matrix.cols, 0.0f);
        owned.gene_sq_sum.assign((std::size_t) matrix.cols, 0.0f);
        for (const gene_metric_partial &partial : partials) {
            if (!partial.ok) goto done;
            for (std::size_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
                owned.gene_sum[gene] += partial.gene_sum[gene];
                owned.gene_detected[gene] += partial.gene_detected[gene];
                owned.gene_sq_sum[gene] += partial.gene_sq_sum[gene];
            }
        }
    }

    {
        std::vector<std::pair<float, std::uint32_t>> ranked;
        ranked.reserve(owned.gene_sum.size());
        for (std::uint32_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
            ranked.emplace_back(owned.gene_sum[gene], gene);
        }
        std::partial_sort(ranked.begin(),
                          ranked.begin() + std::min<std::size_t>(plan.policy.browse_top_features, ranked.size()),
                          ranked.end(),
                          [](const auto &lhs, const auto &rhs) {
                              if (lhs.first != rhs.first) return lhs.first > rhs.first;
                              return lhs.second < rhs.second;
                          });
        const std::size_t count = std::min<std::size_t>(plan.policy.browse_top_features, ranked.size());
        owned.selected_feature_indices.resize(count, 0u);
        for (std::size_t i = 0; i < count; ++i) owned.selected_feature_indices[i] = ranked[i].second;
    }

    if (owned.selected_feature_indices.empty()) {
        push_issue(issues, issue_severity::warning, "browse", "browse cache skipped because no features were selected");
        ok = true;
        goto done;
    }

    owned.dataset_feature_mean.assign(plan.datasets.size() * owned.selected_feature_indices.size(), 0.0f);
    owned.shard_feature_mean.assign((std::size_t) matrix.num_shards * owned.selected_feature_indices.size(), 0.0f);
    owned.part_sample_row_offsets.resize((std::size_t) matrix.num_parts + 1u, 0u);
    for (unsigned long part_id = 0; part_id < matrix.num_parts; ++part_id) {
        owned.part_sample_row_offsets[part_id + 1u] =
            owned.part_sample_row_offsets[part_id] + plan.policy.browse_sample_rows_per_part;
    }
    owned.part_sample_global_rows.assign((std::size_t) matrix.num_parts * plan.policy.browse_sample_rows_per_part,
                                         std::numeric_limits<std::uint64_t>::max());
    owned.part_sample_values.assign((std::size_t) matrix.num_parts
                                        * plan.policy.browse_sample_rows_per_part
                                        * owned.selected_feature_indices.size(),
                                    0.0f);

    {
        std::vector<int> shard_owner(matrix.num_shards, -1);
        std::vector<std::uint32_t> part_dataset_indices(plan.parts.size(), 0u);
        std::vector<std::vector<float>> dataset_partials(ctx.device_count);
        std::vector<std::vector<float>> shard_partials(ctx.device_count);
        std::vector<std::thread> workers;
        std::vector<std::uint32_t> source_to_dataset(plan.sources.size(), std::numeric_limits<std::uint32_t>::max());
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }
        for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
            source_to_dataset[plan.datasets[dataset_index].source_index] = (std::uint32_t) dataset_index;
        }
        for (std::size_t part_index = 0; part_index < plan.parts.size(); ++part_index) {
            const std::size_t source_index = plan.parts[part_index].source_index;
            if (source_index < source_to_dataset.size() && source_to_dataset[source_index] != std::numeric_limits<std::uint32_t>::max()) {
                part_dataset_indices[part_index] = source_to_dataset[source_index];
            }
        }

        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.emplace_back([&, slot]() {
                (void) build_selected_feature_partials(path,
                                                       shard_owner,
                                                       slot,
                                                       ctx.device_ids[slot],
                                                       part_dataset_indices,
                                                       (unsigned int) plan.datasets.size(),
                                                       (unsigned int) matrix.num_shards,
                                                       owned.selected_feature_indices,
                                                       plan.policy.browse_sample_rows_per_part,
                                                       dataset_partials.data() + slot,
                                                       shard_partials.data() + slot,
                                                       &owned.part_sample_global_rows,
                                                       &owned.part_sample_values,
                                                       issues);
            });
        }
        for (std::thread &worker : workers) worker.join();

        for (const std::vector<float> &partial : dataset_partials) {
            if (partial.empty()) continue;
            for (std::size_t i = 0; i < owned.dataset_feature_mean.size(); ++i) owned.dataset_feature_mean[i] += partial[i];
        }
        for (const std::vector<float> &partial : shard_partials) {
            if (partial.empty()) continue;
            for (std::size_t i = 0; i < owned.shard_feature_mean.size(); ++i) owned.shard_feature_mean[i] += partial[i];
        }
    }

    for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
        const double denom = plan.datasets[dataset_index].rows != 0 ? (double) plan.datasets[dataset_index].rows : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] =
                (float) (owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] / denom);
        }
    }
    for (std::size_t shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        const double denom = cs::rows_in_shard(&matrix, (unsigned long) shard_id) != 0
            ? (double) cs::rows_in_shard(&matrix, (unsigned long) shard_id)
            : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] =
                (float) (owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] / denom);
        }
    }

    {
        cellshard::series_browse_cache_view view{};
        view.selected_feature_count = (std::uint32_t) owned.selected_feature_indices.size();
        view.selected_feature_indices = owned.selected_feature_indices.data();
        view.gene_sum = owned.gene_sum.data();
        view.gene_detected = owned.gene_detected.data();
        view.gene_sq_sum = owned.gene_sq_sum.data();
        view.dataset_count = (std::uint32_t) plan.datasets.size();
        view.dataset_feature_mean = owned.dataset_feature_mean.data();
        view.shard_count = (std::uint32_t) matrix.num_shards;
        view.shard_feature_mean = owned.shard_feature_mean.data();
        view.part_count = (std::uint32_t) matrix.num_parts;
        view.sample_rows_per_part = plan.policy.browse_sample_rows_per_part;
        view.part_sample_row_offsets = owned.part_sample_row_offsets.data();
        view.part_sample_global_rows = owned.part_sample_global_rows.data();
        view.part_sample_values = owned.part_sample_values.data();
        if (!cellshard::append_series_browse_cache_h5(path.c_str(), &view)) {
            push_issue(issues, issue_severity::error, "browse", "failed to append browse cache to series.csh5");
            goto done;
        }
    }

    ok = true;

done:
    csd::clear(&shard_map);
    csd::clear(&ctx);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
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

    if (plan.policy.embed_metadata) {
        report.events.push_back({"metadata", "embedding metadata tables"});
        if (!append_embedded_metadata_tables(plan, &report.issues)) {
            cseries::clear(&manifest);
            return report;
        }
    }

    if (plan.policy.build_browse_cache) {
        report.events.push_back({"browse", "building 4-GPU browse cache"});
        if (!build_browse_cache_multigpu(plan.policy.output_path, plan, &report.issues)) {
            cseries::clear(&manifest);
            return report;
        }
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
    const cpre::cell_filter_params cell_filter = {
        config.min_counts,
        config.min_genes,
        config.max_mito_fraction
    };
    const cpre::gene_filter_params gene_filter = {
        config.min_gene_sum,
        config.min_detected_cells,
        config.min_variance
    };

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
        if (!cpre::preprocess_part_inplace(&part_view, &workspace, cell_filter, config.target_sum, nullptr)) {
            push_issue(&summary.issues, issue_severity::error, "preprocess", "GPU preprocess kernel pass failed");
            goto done;
        }
        if (!check_cuda(cudaStreamSynchronize(workspace.stream), &summary.issues, "preprocess", "cudaStreamSynchronize part")) goto done;
        if (!check_cuda(csv::release_part(&device_state, part_id), &summary.issues, "preprocess", "release_part")) goto done;
        ++summary.parts_processed;
        if (loaded_here && config.drop_host_parts) cs::drop_part(&matrix, part_id);
    }

    if (!cpre::build_gene_filter_mask(&workspace, (unsigned int) matrix.cols, gene_filter, nullptr)
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
