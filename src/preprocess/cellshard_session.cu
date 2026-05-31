#include <Cellerator/preprocess/runtime.hh>

#include <Cellerator/compute/runtime.hh>
#include <CellShard/export/dataset_export.hh>
#include <CellShard/runtime/device/sharded_device.cuh>
#include <CellShard/runtime/host/sharded_host.cuh>
#include <CellShard/runtime/storage/disk.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace cpre = ::cellerator::preprocess;
namespace copt = ::cellerator::optimize;
namespace crt = ::cellerator::compute::runtime;
namespace cse = ::cellshard::exporting;
namespace csd = ::cellshard::device;

namespace cellerator::preprocess {
namespace {

constexpr const char *k_default_group_names[] = {"mt", "ribo", "hb"};

void set_status(status *out, int code, const char *message) {
    if (out == nullptr) return;
    out->code = code;
    std::snprintf(out->message, sizeof(out->message), "%s", message != nullptr ? message : "");
}

int fail(status *out, int code, const char *message) {
    set_status(out, code, message);
    return 0;
}

int cuda_fail(status *out, cudaError_t err, const char *label) {
    char message[192];
    std::snprintf(message, sizeof(message), "%s: %s", label, cudaGetErrorString(err));
    return fail(out, status_invalid_argument, message);
}

int cuda_ok(cudaError_t err, status *out, const char *label) {
    return err == cudaSuccess ? 1 : cuda_fail(out, err, label);
}

template<typename T>
T *alloc_host(std::uint64_t count) {
    if (count == 0u) return nullptr;
    if (count > (std::numeric_limits<std::size_t>::max() / sizeof(T))) return nullptr;
    return static_cast<T *>(std::calloc((std::size_t) count, sizeof(T)));
}

template<typename T>
int copy_device_to_host(T *dst, const T *src, std::uint64_t count, int device, status *out, const char *label) {
    if (count == 0u) return 1;
    if (dst == nullptr || src == nullptr) return fail(out, status_invalid_argument, label);
    if (!cuda_ok(cudaSetDevice(device), out, "cudaSetDevice(copy_device_to_host)")) return 0;
    return cuda_ok(cudaMemcpy(dst, src, (std::size_t) count * sizeof(T), cudaMemcpyDeviceToHost), out, label);
}

int sync_fleet(preprocess_fleet_workspace *fleet, status *out) {
    if (fleet == nullptr) return 0;
    for (std::uint32_t i = 0u; i < fleet->slot_count; ++i) {
        const std::uint32_t slot = fleet->slots[i];
        const int device = crt::fleet_device_id(fleet->fleet, slot);
        if (!cuda_ok(cudaSetDevice(device), out, "cudaSetDevice(sync preprocess fleet)")) return 0;
        if (!cuda_ok(cudaStreamSynchronize(crt::fleet_stream(fleet->fleet, slot)), out, "cudaStreamSynchronize(preprocess fleet)")) {
            return 0;
        }
    }
    return 1;
}

int fleet_device_id(const preprocess_fleet_workspace *fleet, std::uint32_t index) {
    if (fleet == nullptr || index >= fleet->slot_count || fleet->slots == nullptr) return -1;
    return crt::fleet_device_id(fleet->fleet, fleet->slots[index]);
}

cudaStream_t fleet_stream(const preprocess_fleet_workspace *fleet, std::uint32_t index) {
    if (fleet == nullptr || index >= fleet->slot_count || fleet->slots == nullptr) return (cudaStream_t) 0;
    return crt::fleet_stream(fleet->fleet, fleet->slots[index]);
}

template<typename MatrixT>
struct loaded_dataset {
    ::cellshard::sharded<MatrixT> view;
    ::cellshard::shard_storage storage;

    loaded_dataset() {
        ::cellshard::init(&view);
        ::cellshard::init(&storage);
    }

    ~loaded_dataset() {
        ::cellshard::clear(&storage);
        ::cellshard::clear(&view);
    }
};

csd::blocked_ell_view device_view_from_record(const csd::partition_record<::cellerator::core::matrix::blocked_ell> &record,
                                              status *out) {
    csd::blocked_ell_view view{};
    if (record.view == nullptr) {
        set_status(out, status_invalid_argument, "missing uploaded blocked-ELL device descriptor");
        return view;
    }
    if (!cuda_ok(cudaMemcpy(&view, record.view, sizeof(view), cudaMemcpyDeviceToHost), out, "copy blocked-ELL device descriptor")) {
        return csd::blocked_ell_view{};
    }
    return view;
}

csd::sliced_ell_view device_view_from_record(const csd::partition_record<::cellerator::core::matrix::sliced_ell> &record,
                                             status *out) {
    csd::sliced_ell_view view{};
    if (record.view == nullptr) {
        set_status(out, status_invalid_argument, "missing uploaded Sliced-ELL device descriptor");
        return view;
    }
    if (!cuda_ok(cudaMemcpy(&view, record.view, sizeof(view), cudaMemcpyDeviceToHost), out, "copy Sliced-ELL device descriptor")) {
        return csd::sliced_ell_view{};
    }
    return view;
}

int run_fleet_preprocess(csd::blocked_ell_view *views,
                         preprocess_fleet_workspace *fleet,
                         const qc_group_config_view *groups,
                         const cell_qc_filter_params *filter,
                         float target_sum,
                         preprocess_fleet_result *out) {
    return preprocess_blocked_ell_qc_groups_fleet_inplace(views, fleet, groups, filter, target_sum, out);
}

int run_fleet_preprocess(csd::sliced_ell_view *views,
                         preprocess_fleet_workspace *fleet,
                         const qc_group_config_view *groups,
                         const cell_qc_filter_params *filter,
                         float target_sum,
                         preprocess_fleet_result *out) {
    return preprocess_sliced_ell_qc_groups_fleet_inplace(views, fleet, groups, filter, target_sum, out);
}

struct device_gene_accumulator {
    int device = -1;
    cudaStream_t stream = (cudaStream_t) 0;
    std::uint64_t cols = 0u;
    float *block = nullptr;
    float *sum = nullptr;
    float *sq_sum = nullptr;
    float *detected = nullptr;
};

void clear_device_accumulator(device_gene_accumulator *acc) {
    if (acc == nullptr) return;
    if (acc->device >= 0) (void) cudaSetDevice(acc->device);
    if (acc->block != nullptr) (void) cudaFree(acc->block);
    *acc = device_gene_accumulator{};
}

int setup_device_accumulator(device_gene_accumulator *acc, int device, cudaStream_t stream, std::uint64_t cols, status *out) {
    if (acc == nullptr) return fail(out, status_invalid_argument, "missing gene accumulator");
    *acc = device_gene_accumulator{};
    acc->device = device;
    acc->stream = stream;
    acc->cols = cols;
    if (cols == 0u) return 1;
    if (cols > ((std::numeric_limits<std::size_t>::max() / sizeof(float)) / 3u)) return fail(out, status_invalid_argument, "gene metric size overflows host size_t");
    if (!cuda_ok(cudaSetDevice(device), out, "cudaSetDevice(setup gene accumulator)")) return 0;
    if (!cuda_ok(cudaMalloc((void **) &acc->block, (std::size_t) 3u * cols * sizeof(float)), out, "cudaMalloc gene metric accumulator")) return 0;
    acc->sum = acc->block;
    acc->sq_sum = acc->block + cols;
    acc->detected = acc->block + 2u * cols;
    if (!cuda_ok(cudaMemsetAsync(acc->block, 0, (std::size_t) 3u * cols * sizeof(float), stream), out, "cudaMemsetAsync gene metric accumulator")) return 0;
    return cuda_ok(cudaStreamSynchronize(stream), out, "cudaStreamSynchronize(setup gene accumulator)");
}

int add_metric(float *dst, const float *src, std::uint64_t count, int device, cudaStream_t stream, status *out, const char *label) {
    if (count == 0u) return 1;
    if (dst == nullptr || src == nullptr) return fail(out, status_invalid_argument, label);
    try {
        crt::execution_context ctx;
        ctx.device = device;
        ctx.stream = stream;
        ctx.owns_stream = false;
        crt::dense_add_inplace_f32(ctx, dst, src, (std::size_t) count);
    } catch (...) {
        return fail(out, status_invalid_argument, label);
    }
    return 1;
}

int add_reduced_gene_metrics(device_gene_accumulator *acc,
                             const preprocess_fleet_result *wave,
                             std::uint64_t cols,
                             status *out) {
    if (acc == nullptr || wave == nullptr) return fail(out, status_invalid_argument, "missing reduced gene metrics");
    if (!cuda_ok(cudaSetDevice(acc->device), out, "cudaSetDevice(add reduced gene metrics)")) return 0;
    const gene_metric_packet_view src{(unsigned int) cols,
                                      wave->reduced_gene.sum,
                                      wave->reduced_gene.sq_sum,
                                      wave->reduced_gene.detected_cells,
                                      nullptr};
    const gene_metric_packet_view dst{(unsigned int) cols, acc->sum, acc->sq_sum, acc->detected, nullptr};
    if (gene_metric_packet_is_contiguous(&src, 0u) && gene_metric_packet_is_contiguous(&dst, 0u)) {
        return add_metric(acc->sum, wave->reduced_gene.sum, gene_metric_packet_float_count((unsigned int) cols, 0u), acc->device, acc->stream, out, "add reduced gene metric packet")
            && cuda_ok(cudaStreamSynchronize(acc->stream), out, "cudaStreamSynchronize(add reduced gene metrics)");
    }
    if (!add_metric(acc->sum, wave->reduced_gene.sum, cols, acc->device, acc->stream, out, "add reduced gene sum")) return 0;
    if (!add_metric(acc->sq_sum, wave->reduced_gene.sq_sum, cols, acc->device, acc->stream, out, "add reduced gene sq_sum")) return 0;
    if (!add_metric(acc->detected, wave->reduced_gene.detected_cells, cols, acc->device, acc->stream, out, "add reduced gene detected")) return 0;
    return cuda_ok(cudaStreamSynchronize(acc->stream), out, "cudaStreamSynchronize(add reduced gene metrics)");
}

template<typename MatrixT>
int copy_cell_metrics(const ::cellshard::sharded<MatrixT> *view,
                      unsigned long part_id,
                      const part_preprocess_result *part,
                      std::uint32_t group_count,
                      int device,
                      preprocess_cellshard_session_result *result,
                      status *out) {
    const std::uint64_t row_offset = (std::uint64_t) view->partition_offsets[part_id];
    const std::uint64_t part_rows = (std::uint64_t) view->partition_rows[part_id];
    const std::uint64_t group_values = part_rows * group_count;
    if (!copy_device_to_host(result->cell_total_counts + row_offset, part->cell.total_counts, part_rows, device, out, "copy cell total counts")) return 0;
    if (!copy_device_to_host(result->cell_max_counts + row_offset, part->cell.max_counts, part_rows, device, out, "copy cell max counts")) return 0;
    if (!copy_device_to_host(result->cell_detected_genes + row_offset, part->cell.detected_genes, part_rows, device, out, "copy cell detected genes")) return 0;
    if (!copy_device_to_host(result->cell_keep + row_offset, part->cell.keep_cells, part_rows, device, out, "copy cell keep mask")) return 0;
    if (group_values != 0u) {
        const std::uint64_t group_offset = row_offset * group_count;
        if (!copy_device_to_host(result->cell_group_counts + group_offset, part->cell.cell_group_counts, group_values, device, out, "copy cell group counts")) return 0;
        if (!copy_device_to_host(result->cell_group_pct + group_offset, part->cell.cell_group_pct, group_values, device, out, "copy cell group pct")) return 0;
        for (std::uint64_t row = 0u; row < part_rows; ++row) {
            result->cell_mito_counts[row_offset + row] = result->cell_group_counts[(row_offset + row) * group_count + qc_group_mt];
        }
    } else if (!copy_device_to_host(result->cell_mito_counts + row_offset, part->cell.mito_counts, part_rows, device, out, "copy cell mito counts")) {
        return 0;
    }
    return 1;
}

template<typename MatrixT, typename DeviceViewT>
int process_wave(loaded_dataset<MatrixT> *loaded,
                 csd::sharded_device<MatrixT> *device_state,
                 preprocess_fleet_workspace *fleet,
                 unsigned long part_begin,
                 unsigned int part_count,
                 const qc_group_config_view *groups,
                 const cell_qc_filter_params *filter,
                 float target_sum,
                 device_gene_accumulator *gene_acc,
                 preprocess_cellshard_session_result *result,
                 status *out) {
    std::unique_ptr<DeviceViewT[]> views(new (std::nothrow) DeviceViewT[part_count]);
    std::unique_ptr<unsigned long[]> parts(new (std::nothrow) unsigned long[part_count]);
    if (!views || !parts) return fail(out, status_invalid_argument, "failed to allocate preprocess wave metadata");

    for (unsigned int slot = 0u; slot < part_count; ++slot) {
        const unsigned long part_id = part_begin + slot;
        const int device = fleet_device_id(fleet, slot);
        const cudaStream_t stream = fleet_stream(fleet, slot);
        parts[slot] = part_id;
        if (!::cellshard::fetch_partition(&loaded->view, &loaded->storage, part_id)) {
            return fail(out, status_invalid_argument, "failed to fetch CellShard partition");
        }
        if (!cuda_ok(csd::upload_partition_async(device_state, &loaded->view, part_id, device, stream),
                     out,
                     "upload CellShard partition")) {
            return 0;
        }
        if (!cuda_ok(cudaStreamSynchronize(stream), out, "cudaStreamSynchronize(upload CellShard partition)")) return 0;
        views[slot] = device_view_from_record(device_state->parts[part_id], out);
        if (out != nullptr && out->code != status_ok) return 0;
    }

    preprocess_fleet_result wave{};
    if (!run_fleet_preprocess(views.get(), fleet, groups, filter, target_sum, &wave)) {
        return fail(out, status_invalid_argument, "native Cellerator fleet preprocessing failed");
    }
    if (!sync_fleet(fleet, out)) return 0;

    for (unsigned int slot = 0u; slot < part_count; ++slot) {
        const int device = fleet_device_id(fleet, slot);
        if (!copy_cell_metrics(&loaded->view, parts[slot], wave.slot_results + slot, groups->group_count, device, result, out)) return 0;
        if (!cuda_ok(csd::release_partition(device_state, parts[slot]), out, "release CellShard GPU partition")) return 0;
        ++result->partitions_processed;
    }

    return add_reduced_gene_metrics(gene_acc, &wave, result->cols, out);
}

void assign_group_names(const preprocess_cellshard_session_options *options,
                        preprocess_cellshard_session_result *result) {
    for (std::uint32_t i = 0u; i < result->group_count && i < CELLERATOR_PREPROCESS_MAX_QC_GROUPS; ++i) {
        if (options->group_names != nullptr && options->group_names[i] != nullptr) {
            result->group_names[i] = options->group_names[i];
        } else if (i < 3u) {
            result->group_names[i] = k_default_group_names[i];
        } else {
            result->group_names[i] = "custom";
        }
    }
}

int compile_feature_masks(const cse::dataset_summary &summary,
                          const preprocess_cellshard_session_options *options,
                          preprocess_cellshard_session_result *result,
                          status *out) {
    const std::uint64_t cols = summary.cols;
    if (cols == 0u) return 1;
    if (options->feature_group_masks != nullptr) {
        std::memcpy(result->feature_group_masks, options->feature_group_masks, (std::size_t) cols * sizeof(std::uint32_t));
        return 1;
    }

    std::unique_ptr<const char *[]> ids(new (std::nothrow) const char *[(std::size_t) cols]);
    std::unique_ptr<const char *[]> names(new (std::nothrow) const char *[(std::size_t) cols]);
    std::unique_ptr<const char *[]> types(new (std::nothrow) const char *[(std::size_t) cols]);
    std::unique_ptr<const char *[]> modalities(new (std::nothrow) const char *[(std::size_t) cols]);
    if (!ids || !names || !types || !modalities) return fail(out, status_invalid_argument, "failed to allocate feature metadata pointers");
    for (std::uint64_t i = 0u; i < cols; ++i) {
        ids[(std::size_t) i] = i < summary.var_ids.size() ? summary.var_ids[(std::size_t) i].c_str() : "";
        names[(std::size_t) i] = i < summary.var_names.size() ? summary.var_names[(std::size_t) i].c_str() : ids[(std::size_t) i];
        types[(std::size_t) i] = i < summary.var_types.size() ? summary.var_types[(std::size_t) i].c_str() : "";
        modalities[(std::size_t) i] = "rna";
    }
    qc_feature_annotation_view features{ids.get(), names.get(), types.get(), modalities.get(), (std::uint32_t) cols};
    if (!compile_default_qc_feature_group_masks(&features, nullptr, result->feature_group_masks)) {
        return fail(out, status_invalid_argument, "failed to compile default QC feature group masks");
    }
    return 1;
}

int allocate_result_buffers(const cse::dataset_summary &summary,
                            const preprocess_cellshard_session_options *options,
                            preprocess_cellshard_session_result *result,
                            status *out) {
    const std::uint64_t rows = summary.rows;
    const std::uint64_t cols = summary.cols;
    const std::uint32_t group_count = options->group_count != 0u ? options->group_count : 3u;
    if (group_count > CELLERATOR_PREPROCESS_MAX_QC_GROUPS) return fail(out, status_invalid_argument, "too many QC groups");
    if (rows > (std::uint64_t) std::numeric_limits<std::size_t>::max()) return fail(out, status_invalid_argument, "row count overflows host size_t");
    if (cols > (std::uint64_t) std::numeric_limits<std::size_t>::max()) return fail(out, status_invalid_argument, "column count overflows host size_t");
    if (rows != 0u && group_count > (std::numeric_limits<std::uint64_t>::max() / rows)) return fail(out, status_invalid_argument, "cell group metric count overflows");

    result->rows = rows;
    result->cols = cols;
    result->nnz = summary.nnz;
    result->group_count = group_count;
    assign_group_names(options, result);

    result->feature_group_masks = alloc_host<std::uint32_t>(cols);
    result->cell_keep = alloc_host<std::uint8_t>(rows);
    result->cell_total_counts = alloc_host<float>(rows);
    result->cell_mito_counts = alloc_host<float>(rows);
    result->cell_max_counts = alloc_host<float>(rows);
    result->cell_detected_genes = alloc_host<std::uint32_t>(rows);
    result->cell_group_counts = alloc_host<float>(rows * group_count);
    result->cell_group_pct = alloc_host<float>(rows * group_count);
    result->gene_keep = alloc_host<std::uint8_t>(cols);
    result->gene_sum = alloc_host<float>(cols);
    result->gene_sq_sum = alloc_host<float>(cols);
    result->gene_detected_cells = alloc_host<float>(cols);

    if ((cols != 0u && (result->feature_group_masks == nullptr || result->gene_keep == nullptr
                        || result->gene_sum == nullptr || result->gene_sq_sum == nullptr || result->gene_detected_cells == nullptr))
        || (rows != 0u && (result->cell_keep == nullptr || result->cell_total_counts == nullptr
                           || result->cell_mito_counts == nullptr || result->cell_max_counts == nullptr
                           || result->cell_detected_genes == nullptr || result->cell_group_counts == nullptr
                           || result->cell_group_pct == nullptr))) {
        return fail(out, status_invalid_argument, "failed to allocate preprocess session result buffers");
    }
    return compile_feature_masks(summary, options, result, out);
}

preprocess_reduction_mode configured_reduction_mode(const preprocess_fleet_workspace *fleet) {
    if (fleet == nullptr || fleet->slot_count <= 1u) return preprocess_reduction_single_device;
#if CELLERATOR_DIST_HAS_NCCL
    if (fleet->ranked_nccl.ready != 0u || ::cellerator::dist::local_nccl_ready(&fleet->fleet.local)) return preprocess_reduction_nccl;
#endif
    return preprocess_reduction_peer_copy;
}

template<typename MatrixT, typename DeviceViewT>
int run_layout(const cse::dataset_summary &summary,
               native_sparse_layout layout,
               const preprocess_cellshard_session_options *options,
               preprocess_cellshard_session_result *result,
               status *out) {
    loaded_dataset<MatrixT> loaded;
    csd::sharded_device<MatrixT> device_state;
    preprocess_fleet_workspace main_fleet;
    device_gene_accumulator gene_acc;
    int ok = 0;

    csd::init(&device_state);
    init(&main_fleet);
    if (!::cellshard::load_header(options->input_path, &loaded.view, &loaded.storage)) {
        fail(out, status_invalid_argument, "failed to load CellShard dataset header");
        goto done;
    }
    if (!csd::reserve(&device_state, loaded.view.num_partitions)) {
        fail(out, status_invalid_argument, "failed to reserve CellShard GPU partition residency records");
        goto done;
    }

    {
        preprocess_fleet_config config{options->device_ids,
                                       options->device_count,
                                       options->enable_peer_access,
                                       options->stream_flags,
                                       options->ranked_nccl};
        if (!setup_fleet(&main_fleet, &config)) {
            fail(out, status_invalid_argument, "failed to set up Cellerator preprocess GPU fleet");
            goto done;
        }
    }

    result->layout = layout;
    result->device_count = main_fleet.slot_count;
    result->reduction_mode = configured_reduction_mode(&main_fleet);
    result->execution_plan = preprocess_execution_fused;
    if (options->optimizer == nullptr || options->optimizer->mode == copt::optimizer_mode::disabled) {
        copt::mark_disabled(&result->optimizer_result);
    } else {
        result->optimizer_result.provider = copt::optimizer_provider::preprocess;
        result->optimizer_result.selected_plan = preprocess_execution_fused;
        result->optimizer_result.stop_reason = copt::optimizer_stop_reason::budget_skipped;
        copt::set_message(&result->optimizer_result, "C++ fleet session keeps fused default; use plan-aware compute primitives for explicit calibration");
    }

    if (!setup_device_accumulator(&gene_acc, fleet_device_id(&main_fleet, 0u), fleet_stream(&main_fleet, 0u), result->cols, out)) goto done;

    {
        const char *group_names[CELLERATOR_PREPROCESS_MAX_QC_GROUPS] = {};
        for (std::uint32_t i = 0u; i < result->group_count; ++i) group_names[i] = result->group_names[i];
        qc_group_config_view groups{result->group_count, group_names, result->feature_group_masks, nullptr};
        cell_qc_filter_params cell_filter{options->min_counts,
                                          options->min_features,
                                          options->max_group_fraction,
                                          options->fraction_group_index};

        for (unsigned long shard = 0u; shard < loaded.view.num_shards; ++shard) {
            const unsigned long shard_begin = ::cellshard::first_partition_in_shard(&loaded.view, shard);
            const unsigned long shard_end = ::cellshard::last_partition_in_shard(&loaded.view, shard);
            if (shard_begin >= shard_end) continue;
            ++result->shards_visited;
            for (unsigned long part = shard_begin; part < shard_end;) {
                const unsigned int active = (unsigned int) std::min<unsigned long>(main_fleet.slot_count, shard_end - part);
                if (active == main_fleet.slot_count) {
                    if (!process_wave<MatrixT, DeviceViewT>(&loaded,
                                                            &device_state,
                                                            &main_fleet,
                                                            part,
                                                            active,
                                                            &groups,
                                                            &cell_filter,
                                                            options->target_sum,
                                                            &gene_acc,
                                                            result,
                                                            out)) {
                        goto done;
                    }
                } else {
                    std::unique_ptr<int[]> device_ids(new (std::nothrow) int[active]);
                    preprocess_fleet_workspace sub_fleet;
                    init(&sub_fleet);
                    if (!device_ids) {
                        clear(&sub_fleet);
                        fail(out, status_invalid_argument, "failed to allocate partial fleet device ids");
                        goto done;
                    }
                    for (unsigned int i = 0u; i < active; ++i) device_ids[i] = fleet_device_id(&main_fleet, i);
                    {
                        preprocess_fleet_config config{device_ids.get(),
                                                       active,
                                                       options->enable_peer_access,
                                                       options->stream_flags,
                                                       nullptr};
                        if (!setup_fleet(&sub_fleet, &config)) {
                            clear(&sub_fleet);
                            fail(out, status_invalid_argument, "failed to set up partial preprocess GPU fleet");
                            goto done;
                        }
                    }
                    if (!process_wave<MatrixT, DeviceViewT>(&loaded,
                                                            &device_state,
                                                            &sub_fleet,
                                                            part,
                                                            active,
                                                            &groups,
                                                            &cell_filter,
                                                            options->target_sum,
                                                            &gene_acc,
                                                            result,
                                                            out)) {
                        clear(&sub_fleet);
                        goto done;
                    }
                    clear(&sub_fleet);
                }
                part += active;
            }
        }
    }

    if (!copy_device_to_host(result->gene_sum, gene_acc.sum, result->cols, gene_acc.device, out, "copy final gene sum")) goto done;
    if (!copy_device_to_host(result->gene_sq_sum, gene_acc.sq_sum, result->cols, gene_acc.device, out, "copy final gene sq_sum")) goto done;
    if (!copy_device_to_host(result->gene_detected_cells, gene_acc.detected, result->cols, gene_acc.device, out, "copy final gene detected")) goto done;

    for (std::uint64_t row = 0u; row < result->rows; ++row) result->kept_cells += result->cell_keep[(std::size_t) row] != 0u ? 1u : 0u;
    for (std::uint64_t col = 0u; col < result->cols; ++col) result->gene_sum_checksum += result->gene_sum[(std::size_t) col];
    {
        gene_filter_params gene_filter{options->min_gene_sum, options->min_detected_cells, options->min_variance};
        if (!finalize_gene_keep_mask_host(result->gene_sum,
                                          result->gene_sq_sum,
                                          result->gene_detected_cells,
                                          (std::uint32_t) result->cols,
                                          (float) result->kept_cells,
                                          &gene_filter,
                                          result->gene_keep,
                                          &result->kept_genes)) {
            fail(out, status_invalid_argument, "failed to finalize gene keep mask");
            goto done;
        }
    }

    ok = 1;

done:
    csd::clear(&device_state);
    clear_device_accumulator(&gene_acc);
    clear(&main_fleet);
    return ok;
}

} // namespace

void clear(preprocess_cellshard_session_result *result) {
    if (result == nullptr) return;
    std::free(result->feature_group_masks);
    std::free(result->cell_keep);
    std::free(result->cell_total_counts);
    std::free(result->cell_mito_counts);
    std::free(result->cell_max_counts);
    std::free(result->cell_detected_genes);
    std::free(result->cell_group_counts);
    std::free(result->cell_group_pct);
    std::free(result->gene_keep);
    std::free(result->gene_sum);
    std::free(result->gene_sq_sum);
    std::free(result->gene_detected_cells);
    *result = preprocess_cellshard_session_result{};
}

int preprocess_cellshard_session_all_gpus(const preprocess_cellshard_session_options *options,
                                          preprocess_cellshard_session_result *result,
                                          status *out) {
    if (result == nullptr) return fail(out, status_invalid_argument, "missing preprocess session result");
    clear(result);
    clear_status(out);
    if (options == nullptr || options->input_path == nullptr || options->input_path[0] == '\0') {
        return fail(out, status_invalid_argument, "missing CellShard input path");
    }

    preprocess_state_view state{};
    state.assay = options->assay;
    state.matrix_orientation = options->matrix_orientation;
    state.matrix_state = options->matrix_state;
    state.feature_namespace = options->feature_namespace;
    state.preprocess_available = 0u;
    state.raw_counts_available = 1u;
    state.processed_matrix_available = 0u;
    if (!validate_raw_count_state(&state, out)) return 0;

    cse::dataset_summary summary;
    std::string error;
    if (!cse::load_dataset_summary(options->input_path, &summary, &error)) {
        return fail(out, status_invalid_argument, error.empty() ? "failed to load CellShard dataset summary" : error.c_str());
    }
    if (summary.rows > (std::uint64_t) std::numeric_limits<std::uint32_t>::max()
        || summary.cols > (std::uint64_t) std::numeric_limits<std::uint32_t>::max()) {
        return fail(out, status_invalid_argument, "dataset rows/cols exceed current 32-bit preprocess kernels");
    }
    if (summary.num_partitions == 0u) return fail(out, status_invalid_argument, "CellShard dataset has no partitions");
    if (!allocate_result_buffers(summary, options, result, out)) {
        clear(result);
        return 0;
    }

    int ok = 0;
    if (summary.matrix_format == "blocked_ell") {
        ok = run_layout<::cellerator::core::matrix::blocked_ell, csd::blocked_ell_view>(
            summary, native_sparse_blocked_ell, options, result, out);
    } else if (summary.matrix_format == "sliced_ell") {
        ok = run_layout<::cellerator::core::matrix::sliced_ell, csd::sliced_ell_view>(
            summary, native_sparse_sliced_ell, options, result, out);
    } else {
        fail(out, status_unsupported_layout, "C++ CellShard preprocessing supports blocked_ell and sliced_ell .csh5 datasets");
    }

    if (!ok) {
        clear(result);
        return 0;
    }
    if (result->reduction_mode == preprocess_reduction_peer_copy) {
        set_status(out, status_ok, "preprocess complete; NCCL unavailable, used CUDA peer-copy reduction fallback");
    } else {
        set_status(out, status_ok, "preprocess complete");
    }
    return 1;
}

} // namespace cellerator::preprocess
