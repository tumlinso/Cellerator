#include "preprocess_internal.cuh"

#include <cstring>
#include <memory>
#include <new>

namespace cellerator::compute::preprocess {

void init(preprocess_workspace *workspace) {
    if (workspace == nullptr) return;
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->device = -1;
    cs_runtime::init(&workspace->mask_groups);
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

} // namespace cellerator::compute::preprocess
