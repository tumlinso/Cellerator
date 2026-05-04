#ifndef CELLERATOR_ABI_H
#define CELLERATOR_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CELLERATOR_ABI_VERSION 1u
#define CELLERATOR_TENSOR_MAX_RANK 4u

typedef enum cellerator_status {
    CELLERATOR_STATUS_SUCCESS = 0,
    CELLERATOR_STATUS_INVALID_ARGUMENT = 1,
    CELLERATOR_STATUS_INVALID_SHAPE = 2,
    CELLERATOR_STATUS_UNSUPPORTED = 3,
    CELLERATOR_STATUS_CUDA_ERROR = 4,
    CELLERATOR_STATUS_INTERNAL_ERROR = 5
} cellerator_status;

typedef enum cellerator_dtype {
    CELLERATOR_DTYPE_INVALID = 0,
    CELLERATOR_DTYPE_F16 = 1,
    CELLERATOR_DTYPE_F32 = 2,
    CELLERATOR_DTYPE_U8 = 3,
    CELLERATOR_DTYPE_U32 = 4
} cellerator_dtype;

typedef enum cellerator_sparse_layout {
    CELLERATOR_LAYOUT_CSR = 1,
    CELLERATOR_LAYOUT_BLOCKED_ELL = 2,
    CELLERATOR_LAYOUT_SLICED_ELL = 3,
    CELLERATOR_LAYOUT_QUANTIZED_BLOCKED_ELL = 4
} cellerator_sparse_layout;

typedef enum cellerator_saved_state_kind {
    CELLERATOR_SAVED_STATE_NONE = 0,
    CELLERATOR_SAVED_STATE_SPARSE_PROJECTION = 1,
    CELLERATOR_SAVED_STATE_ROW_SCALE = 2,
    CELLERATOR_SAVED_STATE_RECONSTRUCTION_LOSS = 3
} cellerator_saved_state_kind;

typedef struct cellerator_context cellerator_context;
typedef struct cellerator_saved_state cellerator_saved_state;

typedef struct cellerator_execution_config {
    uint32_t version;
    size_t size;
    int32_t device;
    void *stream;
    void *scratch;
    size_t scratch_bytes;
} cellerator_execution_config;

typedef struct cellerator_tensor_desc {
    uint32_t version;
    size_t size;
    void *data;
    cellerator_dtype dtype;
    uint32_t rank;
    uint64_t shape[CELLERATOR_TENSOR_MAX_RANK];
    int64_t stride[CELLERATOR_TENSOR_MAX_RANK];
} cellerator_tensor_desc;

typedef struct cellerator_sparse_layout_desc {
    uint32_t version;
    size_t size;
    cellerator_sparse_layout layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    uint32_t block_size;
    uint32_t ell_cols;
    uint32_t row_stride_bytes;
    uint32_t bits;
    uint32_t decode_policy;
    cellerator_dtype value_dtype;
    const uint32_t *major_ptr;
    const uint32_t *minor_idx;
    const uint32_t *block_col_idx;
    const void *values;
    const uint8_t *packed_values;
    const float *column_scales;
    const float *column_offsets;
    const float *row_offsets;
} cellerator_sparse_layout_desc;

typedef struct cellerator_feature_affine_reconstruction_config {
    uint32_t version;
    size_t size;
    uint32_t bits;
    float scale_floor;
    float reconstruction_weight;
    float range_weight;
    float min_dynamic_range;
} cellerator_feature_affine_reconstruction_config;

const char *cellerator_status_string(cellerator_status status);
const char *cellerator_last_error(void);

cellerator_status cellerator_context_create(
    const cellerator_execution_config *config,
    cellerator_context **out_context);

cellerator_status cellerator_context_synchronize(cellerator_context *context);

void cellerator_context_destroy(cellerator_context *context);

void cellerator_saved_state_destroy(cellerator_saved_state *saved_state);

cellerator_status cellerator_sparse_projection_forward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *rhs,
    cellerator_tensor_desc *out,
    cellerator_saved_state **saved_state);

cellerator_status cellerator_sparse_projection_backward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *rhs,
    const cellerator_tensor_desc *grad_out,
    cellerator_tensor_desc *grad_values,
    cellerator_tensor_desc *grad_rhs);

cellerator_status cellerator_csr_row_scale_forward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *row_scales,
    cellerator_tensor_desc *out_values,
    cellerator_saved_state **saved_state);

cellerator_status cellerator_csr_row_scale_backward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *grad_out_values,
    const cellerator_tensor_desc *row_scales,
    cellerator_tensor_desc *grad_values,
    cellerator_tensor_desc *grad_row_scales);

cellerator_status cellerator_sparse_reconstruction_loss_forward_backward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *log_scale,
    const cellerator_tensor_desc *offset,
    const cellerator_feature_affine_reconstruction_config *config,
    cellerator_tensor_desc *reconstruction_loss,
    cellerator_tensor_desc *range_loss,
    cellerator_tensor_desc *grad_log_scale,
    cellerator_tensor_desc *grad_offset);

#ifdef __cplusplus
}
#endif

#endif
