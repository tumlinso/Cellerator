#include <Cellerator/compute/preprocess/preprocess.cuh>

#include <CellShard/runtime/mask_groups.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>

#include <cuda_fp16.h>

namespace cellerator::compute::preprocess {

namespace cs_runtime = ::cellshard::runtime;

namespace {

std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int cuda_ok(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "Cellerator preprocess CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

void bind_cell_metrics(preprocess_workspace *workspace, unsigned int rows, unsigned int group_count, cell_metrics_view *out) {
    if (out == nullptr) return;
    out->rows = rows;
    out->group_count = group_count;
    out->total_counts = workspace->total_counts;
    out->mito_counts = group_count != 0u ? workspace->cell_group_counts : workspace->mito_counts;
    out->max_counts = workspace->max_counts;
    out->detected_genes = workspace->detected_genes;
    out->keep_cells = workspace->keep_cells;
    out->cell_group_counts = workspace->cell_group_counts;
    out->cell_group_pct = workspace->cell_group_pct;
}

void bind_gene_metrics(preprocess_workspace *workspace, unsigned int cols, gene_metrics_view *out) {
    if (out == nullptr) return;
    out->cols = cols;
    out->sum = workspace->gene_sum;
    out->sq_sum = workspace->gene_sq_sum;
    out->detected_cells = workspace->gene_detected;
    out->keep_genes = workspace->keep_genes;
    out->feature_group_masks = workspace->feature_group_masks;
    out->gene_flags = workspace->gene_flags;
}

int ensure_runtime_mask_workspace(preprocess_workspace *workspace) {
    if (workspace == nullptr) return 0;
    if (workspace->mask_groups.device >= 0) return 1;
    if (workspace->device < 0) return 0;
    return cs_runtime::setup(&workspace->mask_groups, workspace->device, workspace->stream);
}

void alias_runtime_group_outputs(preprocess_workspace *workspace,
                                 const cs_runtime::sparse_group_reduce_result *runtime) {
    if (workspace == nullptr || runtime == nullptr) return;
    workspace->total_counts = runtime->row_totals;
    workspace->max_counts = runtime->max_values;
    workspace->detected_genes = runtime->detected_features;
    workspace->keep_cells = runtime->row_keep;
    workspace->cell_group_counts = runtime->group_counts;
    workspace->cell_group_pct = runtime->group_percentages;
}

cs_runtime::group_mask_config_view translate_groups(const qc_group_config_view *groups,
                                                    const preprocess_workspace *workspace) {
    cs_runtime::group_mask_config_view out{};
    if (groups == nullptr) return out;
    out.group_count = groups->group_count;
    out.group_names = groups->group_names;
    out.feature_group_masks = workspace != nullptr ? workspace->mask_groups.feature_group_masks : nullptr;
    return out;
}

cs_runtime::sparse_group_filter_params translate_filter(const cell_qc_filter_params *filter) {
    cs_runtime::sparse_group_filter_params out{};
    if (filter == nullptr) return out;
    out.min_total = filter->min_counts;
    out.min_detected_features = filter->min_features;
    out.max_group_fraction = filter->max_group_fraction;
    out.fraction_group_index = filter->fraction_group_index;
    return out;
}

__global__ void normalize_log1p_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        unsigned int ell_col = lane;

        if (keep) {
            while (ell_col < src.ell_cols) {
                const unsigned long offset = (unsigned long) row * src.ell_cols + ell_col;
                const float value = __half2float(src.val[offset]);
                src.val[offset] = value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
                ell_col += 32u;
            }
        }
        row += warp_stride;
    }
}

__global__ void accumulate_gene_metrics_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned long tid = (unsigned long) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = (unsigned long) (gridDim.x * blockDim.x);
    const unsigned long total = (unsigned long) src.rows * (unsigned long) src.ell_cols;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned long linear = tid;

    while (linear < total) {
        const unsigned int row = (unsigned int) (linear / src.ell_cols);
        const unsigned int ell_col = (unsigned int) (linear % src.ell_cols);
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
        const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
        const unsigned int block_col = ell_width_blocks != 0u
            ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
            : cellshard::sparse::blocked_ell_invalid_col;
        const unsigned int col = block_col != cellshard::sparse::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if ((keep_cells == nullptr || keep_cells[row] != 0u) && col < src.cols) {
            const float value = __half2float(src.val[linear]);
            if (value != 0.0f) {
                atomicAdd(gene_sum + col, value);
                atomicAdd(gene_detected + col, 1.0f);
                atomicAdd(gene_sq_sum + col, value * value);
            }
        }
        linear += stride;
    }
}

__global__ void normalize_log1p_sliced_ell_kernel(
    cs_device::sliced_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        unsigned int slice = 0u, row_begin = 0u, width = 0u;
        unsigned long slot_base = 0ul;
        if (src.slice_count != 0u) {
            if (src.slice_rows == 32u) {
                slice = row >> 5;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else if (src.slice_rows != 0u) {
                slice = row / src.slice_rows;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else {
                while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
            }
            row_begin = src.slice_row_offsets[slice];
            width = src.slice_widths[slice];
            slot_base = (unsigned long) src.slice_slot_offsets[slice]
                + (unsigned long) (row - row_begin) * (unsigned long) width;
        }

        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        if (keep) {
            for (unsigned int slot = lane; slot < width; slot += 32u) {
                const unsigned int col = src.col_idx[slot_base + slot];
                const float value = __half2float(src.val[slot_base + slot]);
                if (col < src.cols && value != 0.0f) {
                    src.val[slot_base + slot] = __float2half(log1pf(value * scale));
                }
            }
        }
        row += warp_stride;
    }
}

__global__ void normalize_log1p_compressed_kernel(
    cs_device::compressed_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        if (keep) {
            const unsigned int end = src.majorPtr[row + 1u];
            for (unsigned int idx = src.majorPtr[row] + lane; idx < end; idx += 32u) {
                const float value = __half2float(src.val[idx]);
                src.val[idx] = value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
            }
        }
        row += warp_stride;
    }
}

__global__ void accumulate_gene_metrics_sliced_ell_kernel(
    cs_device::sliced_ell_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        unsigned int slice = 0u, row_begin = 0u, width = 0u;
        unsigned long slot_base = 0ul;
        if (src.slice_count != 0u) {
            if (src.slice_rows == 32u) {
                slice = row >> 5;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else if (src.slice_rows != 0u) {
                slice = row / src.slice_rows;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else {
                while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
            }
            row_begin = src.slice_row_offsets[slice];
            width = src.slice_widths[slice];
            slot_base = (unsigned long) src.slice_slot_offsets[slice]
                + (unsigned long) (row - row_begin) * (unsigned long) width;
        }

        if (keep_cells == nullptr || keep_cells[row] != 0u) {
            for (unsigned int slot = lane; slot < width; slot += 32u) {
                const unsigned int col = src.col_idx[slot_base + slot];
                const float value = __half2float(src.val[slot_base + slot]);
                if (col < src.cols && value != 0.0f) {
                    atomicAdd(gene_sum + col, value);
                    atomicAdd(gene_detected + col, 1.0f);
                    atomicAdd(gene_sq_sum + col, value * value);
                }
            }
        }
        row += warp_stride;
    }
}

__global__ void accumulate_gene_metrics_compressed_kernel(
    cs_device::compressed_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned long tid = (unsigned long) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = (unsigned long) (gridDim.x * blockDim.x);
    unsigned long row = (unsigned long) blockIdx.y;

    while (row < src.rows) {
        if (keep_cells == nullptr || keep_cells[row] != 0u) {
            const unsigned int begin = src.majorPtr[row];
            const unsigned int end = src.majorPtr[row + 1u];
            for (unsigned long idx = begin + tid; idx < end; idx += stride) {
                const unsigned int col = src.minorIdx[idx];
                const float value = __half2float(src.val[idx]);
                if (col < src.cols && value != 0.0f) {
                    atomicAdd(gene_sum + col, value);
                    atomicAdd(gene_detected + col, 1.0f);
                    atomicAdd(gene_sq_sum + col, value * value);
                }
            }
        }
        row += (unsigned long) gridDim.y;
    }
}

__global__ void count_active_rows_kernel(unsigned int rows,
                                         const unsigned char *__restrict__ keep_cells,
                                         float *__restrict__ active_rows) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int row = tid;
    float local = 0.0f;
    while (row < rows) {
        local += keep_cells == nullptr || keep_cells[row] != 0u ? 1.0f : 0.0f;
        row += stride;
    }
    if (local != 0.0f) atomicAdd(active_rows, local);
}

__global__ void build_gene_filter_mask_kernel(unsigned int cols,
                                              const float *__restrict__ active_rows,
                                              gene_filter_params filter,
                                              const float *__restrict__ sum,
                                              const float *__restrict__ sq_sum,
                                              const float *__restrict__ detected_cells,
                                              unsigned char *__restrict__ keep) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    const float active = active_rows != nullptr ? active_rows[0] : 0.0f;
    const float inv_cells = active > 0.0f ? 1.0f / active : 0.0f;
    unsigned int gene = tid;

    while (gene < cols) {
        const float mean = sum[gene] * inv_cells;
        const float var = fmaxf(sq_sum[gene] * inv_cells - mean * mean, 0.0f);
        keep[gene] = (unsigned char) (sum[gene] >= filter.min_sum
                                      && detected_cells[gene] >= filter.min_detected_cells
                                      && var >= filter.min_variance);
        gene += stride;
    }
}

void bind_fleet_result(preprocess_fleet_workspace *fleet, unsigned int leader_index, unsigned int cols, preprocess_fleet_result *out) {
    if (out == nullptr) return;
    out->slot_count = fleet != nullptr ? fleet->slot_count : 0u;
    out->leader_index = leader_index;
    out->slot_results = fleet != nullptr ? fleet->results : nullptr;
    if (fleet != nullptr && fleet->devices != nullptr && leader_index < fleet->slot_count) {
        bind_gene_metrics(fleet->devices + leader_index, cols, &out->reduced_gene);
    } else {
        out->reduced_gene = gene_metrics_view{};
    }
}

int selected_device_id(const preprocess_fleet_workspace *fleet, unsigned int index) {
    if (fleet == nullptr || index >= fleet->slot_count || fleet->slots == nullptr) return -1;
    return cs_compute_runtime::fleet_device_id(fleet->fleet, fleet->slots[index]);
}

int reduce_sum_to_leader_f32(preprocess_fleet_workspace *fleet,
                             float *const *partials,
                             std::size_t count,
                             unsigned int leader_index,
                             float *leader_out) {
    if (fleet == nullptr || partials == nullptr || leader_index >= fleet->slot_count || (leader_out == nullptr && count != 0u)) return 0;
    if (fleet->slot_count == 0u || count == 0u) return 1;
    cs_compute_runtime::reduce_sum_to_leader_f32_options options{};
#if CELLERATOR_DIST_HAS_NCCL
    options.ranked_nccl = fleet->ranked_nccl.ready != 0u ? &fleet->ranked_nccl : nullptr;
#endif
    try {
        cs_compute_runtime::reduce_sum_to_leader_f32(&fleet->fleet,
                                                     fleet->slots,
                                                     fleet->slot_count,
                                                     (const float *const *) partials,
                                                     count,
                                                     fleet->slots[leader_index],
                                                     leader_out,
                                                     &options);
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "Cellerator preprocess reduction error: %s\n", exc.what());
        return 0;
    }
    return 1;
}

int reduce_gene_metrics_to_leader(preprocess_fleet_workspace *fleet, unsigned int cols, unsigned int leader_index) {
    if (fleet == nullptr || cols == 0u) return fleet != nullptr;
    std::unique_ptr<float *[]> partials(new (std::nothrow) float *[fleet->slot_count]);
    if (!partials) return 0;
    int contiguous = 1;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        preprocess_workspace *workspace = fleet->devices + i;
        const gene_metric_packet_view packet{cols, workspace->gene_sum, workspace->gene_sq_sum, workspace->gene_detected, workspace->active_rows};
        contiguous = contiguous && gene_metric_packet_is_contiguous(&packet, 1u);
    }
    if (contiguous != 0) {
        for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_sum;
        return reduce_sum_to_leader_f32(fleet,
                                        partials.get(),
                                        gene_metric_packet_float_count(cols, 1u),
                                        leader_index,
                                        fleet->devices[leader_index].gene_sum);
    }
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_sum;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), cols, leader_index, fleet->devices[leader_index].gene_sum)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_sq_sum;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), cols, leader_index, fleet->devices[leader_index].gene_sq_sum)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_detected;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), cols, leader_index, fleet->devices[leader_index].gene_detected)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].active_rows;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), 1u, leader_index, fleet->devices[leader_index].active_rows)) return 0;
    return 1;
}

int prepare_qc_metric_buffers(preprocess_workspace *workspace,
                              unsigned int rows,
                              unsigned int cols,
                              unsigned int values,
                              const qc_group_config_view *groups,
                              unsigned int *requested_groups) {
    if (workspace == nullptr || requested_groups == nullptr) return 0;
    *requested_groups = groups != nullptr ? groups->group_count : 0u;
    if (*requested_groups > CELLERATOR_PREPROCESS_MAX_QC_GROUPS) return 0;
    if (!reserve_qc_groups(workspace, rows, cols, values, *requested_groups)) return 0;
    if (groups != nullptr && groups->feature_group_masks != nullptr
        && groups->feature_group_masks != workspace->feature_group_masks
        && !upload_feature_group_masks(workspace, cols, groups->feature_group_masks)) {
        return 0;
    }
    if (groups != nullptr && groups->explicit_feature_group_masks != nullptr) {
        if (!upload_feature_group_masks(workspace, cols, groups->explicit_feature_group_masks)) return 0;
    } else if (groups == nullptr || groups->feature_group_masks == nullptr) {
        if (!upload_feature_group_masks(workspace, cols, nullptr)) return 0;
    }
    if (!cuda_ok(cudaMemsetAsync(workspace->keep_cells, 0, (std::size_t) rows, workspace->stream),
                 "cudaMemsetAsync keep cells")) return 0;
    if (*requested_groups != 0u) {
        const std::size_t group_values = (std::size_t) rows * *requested_groups;
        if (!cuda_ok(cudaMemsetAsync(workspace->cell_group_counts,
                                     0,
                                     group_values * sizeof(float),
                                     workspace->stream),
                     "cudaMemsetAsync cell group counts")) return 0;
        if (!cuda_ok(cudaMemsetAsync(workspace->cell_group_pct,
                                     0,
                                     group_values * sizeof(float),
                                     workspace->stream),
                     "cudaMemsetAsync cell group pct")) return 0;
    }
    return 1;
}

int update_active_rows(preprocess_workspace *workspace, unsigned int rows, const unsigned char *device_keep_cells) {
    if (workspace == nullptr || workspace->active_rows == nullptr) return 0;
    if (!cuda_ok(cudaMemsetAsync(workspace->active_rows, 0, sizeof(float), workspace->stream),
                 "cudaMemsetAsync active rows")) return 0;
    unsigned int blocks = (rows + 255u) >> 8;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    count_active_rows_kernel<<<blocks, 256, 0, workspace->stream>>>(rows, device_keep_cells, workspace->active_rows);
    return cuda_ok(cudaGetLastError(), "count_active_rows_kernel");
}

} // namespace

void init(preprocess_workspace *workspace) {
    if (workspace == nullptr) return;
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->device = -1;
    cs_runtime::init(&workspace->mask_groups);
}

void init(preprocess_fleet_workspace *fleet) {
    if (fleet == nullptr) return;
    std::memset(fleet, 0, sizeof(*fleet));
    cs_compute_runtime::init(&fleet->fleet);
#if CELLERATOR_DIST_HAS_NCCL
    cs_dist::init(&fleet->ranked_nccl);
#endif
}

void clear(preprocess_workspace *workspace) {
    if (workspace == nullptr) return;
    if (workspace->device >= 0) (void) cudaSetDevice(workspace->device);
    cs_runtime::clear(&workspace->mask_groups);
    if (workspace->owns_stream != 0 && workspace->stream != (cudaStream_t) 0) (void) cudaStreamDestroy(workspace->stream);
    if (workspace->gene_block != nullptr) (void) cudaFree(workspace->gene_block);
    if (workspace->cell_block != nullptr) (void) cudaFree(workspace->cell_block);
    init(workspace);
}

void clear(preprocess_fleet_workspace *fleet) {
    if (fleet == nullptr) return;
    if (fleet->devices != nullptr) {
        for (unsigned int i = 0u; i < fleet->slot_count; ++i) clear(fleet->devices + i);
    }
#if CELLERATOR_DIST_HAS_NCCL
    cs_dist::clear(&fleet->ranked_nccl);
#endif
    std::free(fleet->results);
    std::free(fleet->devices);
    std::free(fleet->slots);
    cs_compute_runtime::clear(&fleet->fleet);
    init(fleet);
}

int setup(preprocess_workspace *workspace, int device, cudaStream_t stream) {
    if (workspace == nullptr) return 0;
    clear(workspace);
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice preprocess setup")) return 0;
    workspace->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&workspace->stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags preprocess")) return 0;
        workspace->owns_stream = 1;
    } else {
        workspace->stream = stream;
        workspace->owns_stream = 0;
    }
    if (!cs_runtime::setup(&workspace->mask_groups, device, workspace->stream)) return 0;
    return 1;
}

int setup_fleet(preprocess_fleet_workspace *fleet, const preprocess_fleet_config *config) {
    if (fleet == nullptr) return 0;
    clear(fleet);
    init(fleet);

    const unsigned int stream_flags = config != nullptr ? config->stream_flags : cudaStreamNonBlocking;
    const unsigned int enable_peer = config == nullptr || config->enable_peer_access != 0u;
    if (config != nullptr && config->device_count != 0u && config->device_ids == nullptr) return 0;
    try {
        cs_compute_runtime::discover_fleet(&fleet->fleet, true, stream_flags, enable_peer != 0u);
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "Cellerator preprocess fleet setup error: %s\n", exc.what());
        return 0;
    }
    if (fleet->fleet.local.device_count == 0u) return 0;

    const unsigned int requested = config != nullptr ? config->device_count : 0u;
    const unsigned int selected_count = requested != 0u ? requested : fleet->fleet.local.device_count;
    fleet->slots = (unsigned int *) std::calloc((std::size_t) selected_count, sizeof(unsigned int));
    fleet->devices = (preprocess_workspace *) std::calloc((std::size_t) selected_count, sizeof(preprocess_workspace));
    fleet->results = (part_preprocess_result *) std::calloc((std::size_t) selected_count, sizeof(part_preprocess_result));
    if (fleet->slots == nullptr || fleet->devices == nullptr || fleet->results == nullptr) {
        clear(fleet);
        return 0;
    }
    fleet->slot_count = selected_count;
    for (unsigned int i = 0u; i < selected_count; ++i) init(fleet->devices + i);

    if (requested == 0u) {
        unsigned int default_count = cs_compute_runtime::default_mode_fleet_slots(fleet->fleet, fleet->slots, selected_count);
        if (default_count == 0u) default_count = cs_compute_runtime::default_generic_fleet_slots(fleet->fleet, fleet->slots, selected_count);
        fleet->slot_count = default_count;
    }

    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        const int requested_device = requested != 0u ? config->device_ids[i] : cs_compute_runtime::fleet_device_id(fleet->fleet, fleet->slots[i]);
        int found = -1;
        for (unsigned int slot = 0u; slot < fleet->fleet.local.device_count; ++slot) {
            if (fleet->fleet.local.device_ids[slot] == requested_device) {
                found = (int) slot;
                break;
            }
        }
        if (found < 0) {
            clear(fleet);
            return 0;
        }
        fleet->slots[i] = (unsigned int) found;
        if (!setup(fleet->devices + i, requested_device, cs_compute_runtime::fleet_stream(fleet->fleet, (unsigned int) found))) {
            clear(fleet);
            return 0;
        }
    }

#if CELLERATOR_DIST_HAS_NCCL
    if (config != nullptr && config->ranked_nccl != nullptr && config->ranked_nccl->unique_id != nullptr) {
        if (config->ranked_nccl->local_world_ranks == nullptr
            || config->ranked_nccl->world_size <= 0
            || config->ranked_nccl->unique_id_bytes != sizeof(ncclUniqueId)) {
            clear(fleet);
            return 0;
        }
        std::unique_ptr<int[]> device_ids(new (std::nothrow) int[selected_count]);
        if (!device_ids) {
            clear(fleet);
            return 0;
        }
        for (unsigned int i = 0u; i < selected_count; ++i) device_ids[i] = selected_device_id(fleet, i);
        ncclUniqueId unique_id;
        std::memcpy(&unique_id, config->ranked_nccl->unique_id, sizeof(unique_id));
        if (cs_dist::init_ranked_nccl_communicator(&fleet->ranked_nccl,
                                                   device_ids.get(),
                                                   fleet->slots,
                                                   selected_count,
                                                   config->ranked_nccl->local_world_ranks,
                                                   config->ranked_nccl->world_size,
                                                   &unique_id) != ncclSuccess) {
            clear(fleet);
            return 0;
        }
    }
#else
    if (config != nullptr && config->ranked_nccl != nullptr && config->ranked_nccl->unique_id != nullptr) {
        clear(fleet);
        return 0;
    }
#endif

    return 1;
}

int reserve_qc_groups(preprocess_workspace *workspace,
                      unsigned int rows,
                      unsigned int cols,
                      unsigned int values,
                      unsigned int group_count) {
    if (workspace == nullptr) return 0;
    if (group_count > CELLERATOR_PREPROCESS_MAX_QC_GROUPS) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess reserve")) return 0;

    if (rows > workspace->rows_capacity || group_count > workspace->group_capacity) {
        std::size_t bytes = 0u;
        char *base = nullptr;
        const unsigned int alloc_rows = rows > workspace->rows_capacity ? rows : workspace->rows_capacity;
        const unsigned int alloc_groups = group_count > workspace->group_capacity ? group_count : workspace->group_capacity;
        if (workspace->cell_block != nullptr) (void) cudaFree(workspace->cell_block);
        workspace->cell_block = nullptr;

        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned int));
        bytes += (std::size_t) alloc_rows * sizeof(unsigned int);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) alloc_rows * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);

        if (bytes != 0u && !cuda_ok(cudaMalloc(&workspace->cell_block, bytes), "cudaMalloc preprocess cell block")) return 0;
        base = (char *) workspace->cell_block;
        bytes = 0u;
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->total_counts = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->mito_counts = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->max_counts = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned int));
        workspace->detected_genes = (unsigned int *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(unsigned int);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->keep_cells = (unsigned char *) (base + bytes);
        bytes += (std::size_t) alloc_rows * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->cell_group_counts = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->cell_group_pct = (float *) (base + bytes);
        bytes += (std::size_t) alloc_rows * alloc_groups * sizeof(float);
        workspace->rows_capacity = alloc_rows;
        workspace->group_capacity = alloc_groups;
    }

    if (cols > workspace->cols_capacity) {
        std::size_t bytes = 0u;
        char *base = nullptr;
        if (workspace->gene_block != nullptr) (void) cudaFree(workspace->gene_block);
        workspace->gene_block = nullptr;

        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) cols * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) cols * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(std::uint32_t));
        bytes += (std::size_t) cols * sizeof(std::uint32_t);

        if (bytes != 0u && !cuda_ok(cudaMalloc(&workspace->gene_block, bytes), "cudaMalloc preprocess gene block")) return 0;
        base = (char *) workspace->gene_block;
        bytes = 0u;
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->gene_sum = (float *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->gene_sq_sum = (float *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->gene_detected = (float *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->active_rows = (float *) (base + bytes);
        bytes += sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->keep_genes = (unsigned char *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->gene_flags = (unsigned char *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(std::uint32_t));
        workspace->feature_group_masks = (std::uint32_t *) (base + bytes);
        workspace->cols_capacity = cols;
    }

    if (values > workspace->values_capacity) workspace->values_capacity = values;
    return 1;
}

int reserve(preprocess_workspace *workspace, unsigned int rows, unsigned int cols, unsigned int values) {
    return reserve_qc_groups(workspace, rows, cols, values, workspace != nullptr ? workspace->group_capacity : 0u);
}

int upload_feature_group_masks(preprocess_workspace *workspace,
                               unsigned int cols,
                               const std::uint32_t *host_masks) {
    if (workspace == nullptr) return 0;
    if (!reserve_qc_groups(workspace, workspace->rows_capacity, cols, workspace->values_capacity, workspace->group_capacity)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    if (cols == 0u) return 1;
    if (host_masks == nullptr) {
        if (!cuda_ok(cudaMemsetAsync(workspace->feature_group_masks,
                                     0,
                                     (std::size_t) cols * sizeof(std::uint32_t),
                                     workspace->stream),
                     "cudaMemsetAsync feature group masks")) return 0;
        return cs_runtime::upload_feature_group_masks(&workspace->mask_groups, cols, nullptr);
    }
    if (!cuda_ok(cudaMemcpyAsync(workspace->feature_group_masks,
                                 host_masks,
                                 (std::size_t) cols * sizeof(std::uint32_t),
                                 cudaMemcpyHostToDevice,
                                 workspace->stream),
                 "cudaMemcpyAsync feature group masks")) return 0;
    return cs_runtime::upload_feature_group_masks(&workspace->mask_groups, cols, host_masks);
}

int upload_gene_flags(preprocess_workspace *workspace,
                      unsigned int cols,
                      const unsigned char *host_flags) {
    if (workspace == nullptr) return 0;
    if (!reserve_qc_groups(workspace, workspace->rows_capacity, cols, workspace->values_capacity, 1u)) return 0;
    if (cols == 0u) return 1;
    if (host_flags == nullptr) {
        if (!cuda_ok(cudaMemsetAsync(workspace->gene_flags, 0, (std::size_t) cols, workspace->stream),
                     "cudaMemsetAsync gene flags")) return 0;
        return cuda_ok(cudaMemsetAsync(workspace->feature_group_masks,
                                       0,
                                       (std::size_t) cols * sizeof(std::uint32_t),
                                       workspace->stream),
                       "cudaMemsetAsync legacy feature group masks");
    }
    std::unique_ptr<std::uint32_t[]> masks(new (std::nothrow) std::uint32_t[cols]);
    if (!masks) return 0;
    for (unsigned int col = 0u; col < cols; ++col) {
        masks[col] = (host_flags[col] & gene_flag_mito) != 0u ? qc_group_bit(qc_group_mt) : 0u;
    }
    if (!cuda_ok(cudaMemcpyAsync(workspace->gene_flags,
                                 host_flags,
                                 (std::size_t) cols,
                                 cudaMemcpyHostToDevice,
                                 workspace->stream),
                 "cudaMemcpyAsync gene flags")) return 0;
    return upload_feature_group_masks(workspace, cols, masks.get());
}

int zero_gene_metrics(preprocess_workspace *workspace, unsigned int cols) {
    if (workspace == nullptr) return 0;
    if (!reserve(workspace, workspace->rows_capacity, cols, workspace->values_capacity)) return 0;
    if (workspace->gene_sq_sum == workspace->gene_sum + cols
        && workspace->gene_detected == workspace->gene_sum + 2u * cols
        && workspace->active_rows == workspace->gene_sum + 3u * cols) {
        if (!cuda_ok(cudaMemsetAsync(workspace->gene_sum,
                                     0,
                                     ((std::size_t) 3u * cols + 1u) * sizeof(float),
                                     workspace->stream),
                     "cudaMemsetAsync contiguous gene metrics")) return 0;
    } else {
        if (!cuda_ok(cudaMemsetAsync(workspace->gene_sum, 0, (std::size_t) cols * sizeof(float), workspace->stream),
                     "cudaMemsetAsync gene sum")) return 0;
        if (!cuda_ok(cudaMemsetAsync(workspace->gene_sq_sum, 0, (std::size_t) cols * sizeof(float), workspace->stream),
                     "cudaMemsetAsync gene sq sum")) return 0;
        if (!cuda_ok(cudaMemsetAsync(workspace->gene_detected, 0, (std::size_t) cols * sizeof(float), workspace->stream),
                     "cudaMemsetAsync gene detected")) return 0;
        if (workspace->active_rows != nullptr
            && !cuda_ok(cudaMemsetAsync(workspace->active_rows, 0, sizeof(float), workspace->stream),
                        "cudaMemsetAsync active rows")) return 0;
    }
    return cuda_ok(cudaMemsetAsync(workspace->keep_genes, 0, (std::size_t) cols, workspace->stream),
                   "cudaMemsetAsync keep genes");
}

int compute_cell_metrics(const cs_device::blocked_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace->feature_group_masks;
    cell_qc_filter_params generic_filter{filter->min_counts, filter->min_genes, filter->max_mito_fraction, qc_group_mt};
    return compute_qc_metrics(src, workspace, &groups, &generic_filter, out);
}

int compute_cell_metrics(const cs_device::sliced_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace->feature_group_masks;
    cell_qc_filter_params generic_filter{filter->min_counts, filter->min_genes, filter->max_mito_fraction, qc_group_mt};
    return compute_qc_metrics(src, workspace, &groups, &generic_filter, out);
}

int compute_cell_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                             preprocess_workspace *workspace,
                                             const cell_filter_params *filter,
                                             cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace->feature_group_masks;
    cell_qc_filter_params generic_filter{filter->min_counts, filter->min_genes, filter->max_mito_fraction, qc_group_mt};
    return compute_qc_metrics_compressed_fallback(src, workspace, &groups, &generic_filter, out);
}

int compute_qc_metrics(const cs_device::blocked_ell_view *src,
                       preprocess_workspace *workspace,
                       const qc_group_config_view *groups,
                       const cell_qc_filter_params *filter,
                       cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    unsigned int requested_groups = 0u;
    if (!prepare_qc_metric_buffers(workspace, src->rows, src->cols, src->rows * src->ell_cols, groups, &requested_groups)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    cs_runtime::group_mask_config_view runtime_groups = translate_groups(groups, workspace);
    runtime_groups.group_count = requested_groups;
    cs_runtime::sparse_group_filter_params runtime_filter = translate_filter(filter);
    cs_runtime::sparse_group_reduce_result runtime_result{};
    if (!cs_runtime::compute_sparse_group_reduce(src,
                                                 &workspace->mask_groups,
                                                 &runtime_groups,
                                                 nullptr,
                                                 &runtime_filter,
                                                 &runtime_result)) return 0;
    alias_runtime_group_outputs(workspace, &runtime_result);
    bind_cell_metrics(workspace, src->rows, runtime_result.group_count, out);
    return 1;
}

int compute_qc_metrics(const cs_device::sliced_ell_view *src,
                       preprocess_workspace *workspace,
                       const qc_group_config_view *groups,
                       const cell_qc_filter_params *filter,
                       cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    unsigned int requested_groups = 0u;
    if (!prepare_qc_metric_buffers(workspace, src->rows, src->cols, src->nnz, groups, &requested_groups)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    cs_runtime::group_mask_config_view runtime_groups = translate_groups(groups, workspace);
    runtime_groups.group_count = requested_groups;
    cs_runtime::sparse_group_filter_params runtime_filter = translate_filter(filter);
    cs_runtime::sparse_group_reduce_result runtime_result{};
    if (!cs_runtime::compute_sparse_group_reduce(src,
                                                 &workspace->mask_groups,
                                                 &runtime_groups,
                                                 nullptr,
                                                 &runtime_filter,
                                                 &runtime_result)) return 0;
    alias_runtime_group_outputs(workspace, &runtime_result);
    bind_cell_metrics(workspace, src->rows, runtime_result.group_count, out);
    return 1;
}

int compute_qc_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                           preprocess_workspace *workspace,
                                           const qc_group_config_view *groups,
                                           const cell_qc_filter_params *filter,
                                           cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    unsigned int requested_groups = 0u;
    if (!prepare_qc_metric_buffers(workspace, src->rows, src->cols, src->nnz, groups, &requested_groups)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    cs_runtime::group_mask_config_view runtime_groups = translate_groups(groups, workspace);
    runtime_groups.group_count = requested_groups;
    cs_runtime::sparse_group_filter_params runtime_filter = translate_filter(filter);
    cs_runtime::sparse_group_reduce_result runtime_result{};
    if (!cs_runtime::compute_sparse_group_reduce_compressed_fallback(src,
                                                                    &workspace->mask_groups,
                                                                    &runtime_groups,
                                                                    nullptr,
                                                                    &runtime_filter,
                                                                    &runtime_result)) return 0;
    alias_runtime_group_outputs(workspace, &runtime_result);
    bind_cell_metrics(workspace, src->rows, runtime_result.group_count, out);
    return 1;
}

int compute_qc_metrics_fleet(const cs_device::blocked_ell_view *src_by_slot,
                             preprocess_fleet_workspace *fleet,
                             const qc_group_config_view *groups,
                             const cell_qc_filter_params *filter,
                             preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
        if (!compute_qc_metrics(src_by_slot + i, fleet->devices + i, groups, filter, &fleet->results[i].cell)) return 0;
        fleet->results[i].gene = gene_metrics_view{};
    }
    bind_fleet_result(fleet, 0u, 0u, out);
    return 1;
}

int compute_qc_metrics_fleet(const cs_device::sliced_ell_view *src_by_slot,
                             preprocess_fleet_workspace *fleet,
                             const qc_group_config_view *groups,
                             const cell_qc_filter_params *filter,
                             preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
        if (!compute_qc_metrics(src_by_slot + i, fleet->devices + i, groups, filter, &fleet->results[i].cell)) return 0;
        fleet->results[i].gene = gene_metrics_view{};
    }
    bind_fleet_result(fleet, 0u, 0u, out);
    return 1;
}

int normalize_log1p_inplace(cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess normalize")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_blocked_ell_kernel");
}

int normalize_log1p_inplace(cs_device::sliced_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess normalize sliced")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_sliced_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_sliced_ell_kernel");
}

int normalize_log1p_compressed_fallback_inplace(cs_device::compressed_view *src,
                                                preprocess_workspace *workspace,
                                                const float *device_total_counts,
                                                const unsigned char *device_keep_cells,
                                                float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess normalize compressed")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_compressed_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_compressed_kernel");
}

int accumulate_gene_metrics(const cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->rows * src->ell_cols)) return 0;
    if (!update_active_rows(workspace, src->rows, device_keep_cells)) return 0;
    unsigned int blocks = (unsigned int) ((((unsigned long) src->rows * (unsigned long) src->ell_cols) + 255ul) >> 8);
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    accumulate_gene_metrics_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_blocked_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int accumulate_gene_metrics(const cs_device::sliced_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->nnz)) return 0;
    if (!update_active_rows(workspace, src->rows, device_keep_cells)) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    accumulate_gene_metrics_sliced_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_sliced_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int accumulate_gene_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                                preprocess_workspace *workspace,
                                                const unsigned char *device_keep_cells,
                                                gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->nnz)) return 0;
    if (!update_active_rows(workspace, src->rows, device_keep_cells)) return 0;
    unsigned int blocks_x = (src->nnz + 255u) >> 8;
    if (blocks_x < 1u) blocks_x = 1u;
    if (blocks_x > 1024u) blocks_x = 1024u;
    unsigned int blocks_y = src->rows < 4096u ? src->rows : 4096u;
    if (blocks_y < 1u) blocks_y = 1u;
    accumulate_gene_metrics_compressed_kernel<<<dim3(blocks_x, blocks_y), 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_compressed_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int preprocess_blocked_ell_inplace(cs_device::blocked_ell_view *src,
                                   preprocess_workspace *workspace,
                                   const cell_filter_params *cell_filter,
                                   float target_sum,
                                   part_preprocess_result *out) {
    if (cell_filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace != nullptr ? workspace->feature_group_masks : nullptr;
    cell_qc_filter_params generic_filter{cell_filter->min_counts, cell_filter->min_genes, cell_filter->max_mito_fraction, qc_group_mt};
    return preprocess_blocked_ell_qc_groups_inplace(src, workspace, &groups, &generic_filter, target_sum, out);
}

int preprocess_sliced_ell_inplace(cs_device::sliced_ell_view *src,
                                  preprocess_workspace *workspace,
                                  const cell_filter_params *cell_filter,
                                  float target_sum,
                                  part_preprocess_result *out) {
    if (cell_filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace != nullptr ? workspace->feature_group_masks : nullptr;
    cell_qc_filter_params generic_filter{cell_filter->min_counts, cell_filter->min_genes, cell_filter->max_mito_fraction, qc_group_mt};
    return preprocess_sliced_ell_qc_groups_inplace(src, workspace, &groups, &generic_filter, target_sum, out);
}

int preprocess_compressed_fallback_inplace(cs_device::compressed_view *src,
                                           preprocess_workspace *workspace,
                                           const cell_filter_params *cell_filter,
                                           float target_sum,
                                           part_preprocess_result *out) {
    if (cell_filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace != nullptr ? workspace->feature_group_masks : nullptr;
    cell_qc_filter_params generic_filter{cell_filter->min_counts, cell_filter->min_genes, cell_filter->max_mito_fraction, qc_group_mt};
    return preprocess_compressed_fallback_qc_groups_inplace(src, workspace, &groups, &generic_filter, target_sum, out);
}

int preprocess_blocked_ell_qc_groups_inplace(cs_device::blocked_ell_view *src,
                                             preprocess_workspace *workspace,
                                             const qc_group_config_view *groups,
                                             const cell_qc_filter_params *cell_filter,
                                             float target_sum,
                                             part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_qc_metrics(src, workspace, groups, cell_filter, &cell)) return 0;
    if (!normalize_log1p_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (!accumulate_gene_metrics(src, workspace, cell.keep_cells, &gene)) return 0;
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int preprocess_sliced_ell_qc_groups_inplace(cs_device::sliced_ell_view *src,
                                            preprocess_workspace *workspace,
                                            const qc_group_config_view *groups,
                                            const cell_qc_filter_params *cell_filter,
                                            float target_sum,
                                            part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_qc_metrics(src, workspace, groups, cell_filter, &cell)) return 0;
    if (!normalize_log1p_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (!accumulate_gene_metrics(src, workspace, cell.keep_cells, &gene)) return 0;
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int preprocess_compressed_fallback_qc_groups_inplace(cs_device::compressed_view *src,
                                                     preprocess_workspace *workspace,
                                                     const qc_group_config_view *groups,
                                                     const cell_qc_filter_params *cell_filter,
                                                     float target_sum,
                                                     part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_qc_metrics_compressed_fallback(src, workspace, groups, cell_filter, &cell)) return 0;
    if (!normalize_log1p_compressed_fallback_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (!accumulate_gene_metrics_compressed_fallback(src, workspace, cell.keep_cells, &gene)) return 0;
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int preprocess_blocked_ell_qc_groups_fleet_inplace(cs_device::blocked_ell_view *src_by_slot,
                                                   preprocess_fleet_workspace *fleet,
                                                   const qc_group_config_view *groups,
                                                   const cell_qc_filter_params *cell_filter,
                                                   float target_sum,
                                                   preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || cell_filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
    }
    if (!compute_qc_metrics_fleet(src_by_slot, fleet, groups, cell_filter, nullptr)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (!normalize_log1p_inplace(src_by_slot + i,
                                     fleet->devices + i,
                                     fleet->results[i].cell.total_counts,
                                     fleet->results[i].cell.keep_cells,
                                     target_sum)) {
            return 0;
        }
    }
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (!zero_gene_metrics(fleet->devices + i, cols)) return 0;
        if (!accumulate_gene_metrics(src_by_slot + i, fleet->devices + i, fleet->results[i].cell.keep_cells, &fleet->results[i].gene)) return 0;
    }
    if (!reduce_gene_metrics_to_leader(fleet, cols, 0u)) return 0;
    bind_fleet_result(fleet, 0u, cols, out);
    return 1;
}

int preprocess_sliced_ell_qc_groups_fleet_inplace(cs_device::sliced_ell_view *src_by_slot,
                                                  preprocess_fleet_workspace *fleet,
                                                  const qc_group_config_view *groups,
                                                  const cell_qc_filter_params *cell_filter,
                                                  float target_sum,
                                                  preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || cell_filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
    }
    if (!compute_qc_metrics_fleet(src_by_slot, fleet, groups, cell_filter, nullptr)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (!normalize_log1p_inplace(src_by_slot + i,
                                     fleet->devices + i,
                                     fleet->results[i].cell.total_counts,
                                     fleet->results[i].cell.keep_cells,
                                     target_sum)) {
            return 0;
        }
    }
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (!zero_gene_metrics(fleet->devices + i, cols)) return 0;
        if (!accumulate_gene_metrics(src_by_slot + i, fleet->devices + i, fleet->results[i].cell.keep_cells, &fleet->results[i].gene)) return 0;
    }
    if (!reduce_gene_metrics_to_leader(fleet, cols, 0u)) return 0;
    bind_fleet_result(fleet, 0u, cols, out);
    return 1;
}

int build_gene_filter_mask(preprocess_workspace *workspace,
                           unsigned int cols,
                           const gene_filter_params *filter,
                           gene_metrics_view *out) {
    if (workspace == nullptr || filter == nullptr) return 0;
    if (!reserve(workspace, workspace->rows_capacity, cols, workspace->values_capacity)) return 0;
    unsigned int blocks = (cols + 255u) >> 8;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    build_gene_filter_mask_kernel<<<blocks, 256, 0, workspace->stream>>>(
        cols,
        workspace->active_rows,
        *filter,
        workspace->gene_sum,
        workspace->gene_sq_sum,
        workspace->gene_detected,
        workspace->keep_genes);
    if (!cuda_ok(cudaGetLastError(), "build_gene_filter_mask_kernel")) return 0;
    bind_gene_metrics(workspace, cols, out);
    return 1;
}

int finalize_gene_keep_mask_host(const float *gene_sum,
                                 const float *gene_sq_sum,
                                 const float *gene_detected,
                                 unsigned int cols,
                                 float kept_cells,
                                 const gene_filter_params *filter,
                                 unsigned char *keep_genes,
                                 unsigned int *kept_genes) {
    if (gene_sum == nullptr || gene_sq_sum == nullptr || gene_detected == nullptr || filter == nullptr || keep_genes == nullptr) return 0;
    const float inv_cells = kept_cells > 0.0f ? 1.0f / kept_cells : 0.0f;
    unsigned int kept = 0u;
    for (unsigned int gene = 0u; gene < cols; ++gene) {
        const float mean = gene_sum[gene] * inv_cells;
        float var = gene_sq_sum[gene] * inv_cells - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const unsigned char keep = (unsigned char) (gene_sum[gene] >= filter->min_sum
                                                    && gene_detected[gene] >= filter->min_detected_cells
                                                    && var >= filter->min_variance);
        keep_genes[gene] = keep;
        kept += keep != 0u ? 1u : 0u;
    }
    if (kept_genes != nullptr) *kept_genes = kept;
    return 1;
}

} // namespace cellerator::compute::preprocess
