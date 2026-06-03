#include <Cellerator/abi.h>

#include "../compute/sparse/project/project.hh"
#include "../compute/sparse/ops/ops.hh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace runtime = ::cellerator::compute::runtime;
namespace sparse_ops = ::cellerator::compute::sparse::ops;
namespace sparse_project = ::cellerator::compute::sparse::project;

struct cellerator_context {
    runtime::execution_context execution{};
    runtime::scratch_arena scratch{};
    runtime::cusparse_cache sparse_cache{};
    runtime::cublas_cache dense_cache{};
    bool owns_scratch = true;
};

struct cellerator_saved_state {
    uint32_t version = CELLERATOR_ABI_VERSION;
    cellerator_saved_state_kind kind = CELLERATOR_SAVED_STATE_NONE;
    cellerator_sparse_layout layout = CELLERATOR_LAYOUT_CSR;
    uint32_t rows = 0u;
    uint32_t cols = 0u;
    uint32_t nnz = 0u;
    uint32_t out_cols = 0u;
};

namespace {

thread_local std::string last_error_;

void set_last_error_(const char *message) {
    last_error_ = message == nullptr ? "" : message;
}

void clear_last_error_() {
    last_error_.clear();
}

cellerator_status fail_(cellerator_status status, const char *message) {
    set_last_error_(message);
    return status;
}

cellerator_status map_exception_() {
    try {
        throw;
    } catch (const std::invalid_argument &err) {
        set_last_error_(err.what());
        return CELLERATOR_STATUS_INVALID_ARGUMENT;
    } catch (const std::out_of_range &err) {
        set_last_error_(err.what());
        return CELLERATOR_STATUS_INVALID_SHAPE;
    } catch (const std::runtime_error &err) {
        set_last_error_(err.what());
        return CELLERATOR_STATUS_CUDA_ERROR;
    } catch (const std::exception &err) {
        set_last_error_(err.what());
        return CELLERATOR_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_last_error_("unknown Cellerator ABI error");
        return CELLERATOR_STATUS_INTERNAL_ERROR;
    }
}

bool valid_versioned_(uint32_t version, size_t size, size_t minimum_size) {
    return version == CELLERATOR_ABI_VERSION && size >= minimum_size;
}

cellerator_status validate_context_(const cellerator_context *context) {
    return context == nullptr
        ? fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "Cellerator ABI requires a context")
        : CELLERATOR_STATUS_SUCCESS;
}

cellerator_status validate_layout_(const cellerator_sparse_layout_desc *layout) {
    if (layout == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "layout descriptor is null");
    if (!valid_versioned_(layout->version, layout->size, sizeof(cellerator_sparse_layout_desc))) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "layout descriptor version or size mismatch");
    }
    if (layout->rows == 0u || layout->cols == 0u) {
        return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "layout rows and columns must be nonzero");
    }
    return CELLERATOR_STATUS_SUCCESS;
}

cellerator_status validate_dense_tensor_(
    const cellerator_tensor_desc *tensor,
    cellerator_dtype dtype,
    uint32_t rank,
    uint64_t rows,
    uint64_t cols,
    const char *name) {
    if (tensor == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "tensor descriptor is null");
    if (!valid_versioned_(tensor->version, tensor->size, sizeof(cellerator_tensor_desc))) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "tensor descriptor version or size mismatch");
    }
    if (tensor->data == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, name);
    if (tensor->dtype != dtype) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "tensor dtype mismatch");
    if (tensor->rank != rank) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "tensor rank mismatch");
    if (rank >= 1u && tensor->shape[0] != rows) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "tensor row dimension mismatch");
    if (rank >= 2u && tensor->shape[1] != cols) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "tensor column dimension mismatch");
    return CELLERATOR_STATUS_SUCCESS;
}

cellerator_status validate_vector_tensor_(
    const cellerator_tensor_desc *tensor,
    cellerator_dtype dtype,
    uint64_t count,
    const char *name) {
    return validate_dense_tensor_(tensor, dtype, 1u, count, 0u, name);
}

cellerator_status validate_scalar_tensor_(
    const cellerator_tensor_desc *tensor,
    cellerator_dtype dtype,
    const char *name) {
    if (tensor == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "tensor descriptor is null");
    if (!valid_versioned_(tensor->version, tensor->size, sizeof(cellerator_tensor_desc))) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "tensor descriptor version or size mismatch");
    }
    if (tensor->data == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, name);
    if (tensor->dtype != dtype) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "scalar dtype mismatch");
    if (tensor->rank > 1u) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "scalar rank mismatch");
    if (tensor->rank == 1u && tensor->shape[0] != 1u) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "scalar shape mismatch");
    return CELLERATOR_STATUS_SUCCESS;
}

bool has_contiguous_matrix_storage_(const cellerator_tensor_desc *tensor) {
    if (tensor->rank != 2u) return false;
    return tensor->stride[1] == 1 || tensor->stride[1] == 0;
}

int64_t matrix_ld_(const cellerator_tensor_desc *tensor) {
    return tensor->stride[0] == 0 ? static_cast<int64_t>(tensor->shape[1]) : tensor->stride[0];
}

cellerator_status validate_csr_f16_(const cellerator_sparse_layout_desc *layout) {
    if (layout->layout != CELLERATOR_LAYOUT_CSR) {
        return fail_(CELLERATOR_STATUS_UNSUPPORTED, "operator currently requires CSR layout");
    }
    if (layout->value_dtype != CELLERATOR_DTYPE_F16) {
        return fail_(CELLERATOR_STATUS_UNSUPPORTED, "operator currently requires fp16 sparse values");
    }
    if (layout->major_ptr == nullptr || layout->minor_idx == nullptr || layout->values == nullptr) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "CSR layout requires row offsets, column indices, and values");
    }
    return CELLERATOR_STATUS_SUCCESS;
}

cellerator_status validate_blocked_ell_f16_(const cellerator_sparse_layout_desc *layout) {
    if (layout->value_dtype != CELLERATOR_DTYPE_F16) {
        return fail_(CELLERATOR_STATUS_UNSUPPORTED, "Blocked-ELL operator currently requires fp16 values");
    }
    if (layout->block_size == 0u || layout->ell_cols == 0u) {
        return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "Blocked-ELL requires block_size and ell_cols");
    }
    if (layout->block_col_idx == nullptr || layout->values == nullptr) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "Blocked-ELL requires block column indices and values");
    }
    return CELLERATOR_STATUS_SUCCESS;
}

cellerator_status validate_reconstruction_config_(const cellerator_feature_affine_reconstruction_config *config) {
    if (config == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "reconstruction config is null");
    if (!valid_versioned_(config->version, config->size, sizeof(cellerator_feature_affine_reconstruction_config))) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "reconstruction config version or size mismatch");
    }
    if (config->bits != 1u && config->bits != 2u && config->bits != 4u && config->bits != 8u) {
        return fail_(CELLERATOR_STATUS_UNSUPPORTED, "reconstruction config bits must be 1, 2, 4, or 8");
    }
    if (config->scale_floor <= 0.0f) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "reconstruction config scale_floor must be positive");
    }
    return CELLERATOR_STATUS_SUCCESS;
}

cellerator_saved_state *make_saved_state_(
    cellerator_saved_state_kind kind,
    const cellerator_sparse_layout_desc *layout,
    uint32_t out_cols) {
    cellerator_saved_state *saved = new cellerator_saved_state();
    saved->kind = kind;
    saved->layout = layout->layout;
    saved->rows = layout->rows;
    saved->cols = layout->cols;
    saved->nnz = layout->nnz;
    saved->out_cols = out_cols;
    return saved;
}

} // namespace

extern "C" const char *cellerator_status_string(cellerator_status status) {
    switch (status) {
        case CELLERATOR_STATUS_SUCCESS: return "success";
        case CELLERATOR_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case CELLERATOR_STATUS_INVALID_SHAPE: return "invalid shape";
        case CELLERATOR_STATUS_UNSUPPORTED: return "unsupported";
        case CELLERATOR_STATUS_CUDA_ERROR: return "cuda error";
        case CELLERATOR_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

extern "C" const char *cellerator_last_error(void) {
    return last_error_.empty() ? "" : last_error_.c_str();
}

extern "C" cellerator_status cellerator_context_create(
    const cellerator_execution_config *config,
    cellerator_context **out_context) {
    clear_last_error_();
    if (out_context == nullptr) return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "out_context is null");
    *out_context = nullptr;
    if (config != nullptr && !valid_versioned_(config->version, config->size, sizeof(cellerator_execution_config))) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "execution config version or size mismatch");
    }

    try {
        cellerator_context *context = new cellerator_context();
        runtime::init(&context->execution, config == nullptr ? -1 : config->device, config == nullptr ? nullptr : reinterpret_cast<cudaStream_t>(config->stream));
        runtime::init(&context->scratch);
        if (config != nullptr && config->scratch != nullptr) {
            context->scratch.data = config->scratch;
            context->scratch.bytes = config->scratch_bytes;
            context->owns_scratch = false;
        }
        runtime::init(&context->sparse_cache);
        runtime::init(&context->dense_cache);
        *out_context = context;
        return CELLERATOR_STATUS_SUCCESS;
    } catch (...) {
        return map_exception_();
    }
}

extern "C" cellerator_status cellerator_context_synchronize(cellerator_context *context) {
    clear_last_error_();
    if (const cellerator_status status = validate_context_(context); status != CELLERATOR_STATUS_SUCCESS) return status;
    try {
        runtime::cuda_require(cudaSetDevice(context->execution.device), "cudaSetDevice(cellerator_context_synchronize)");
        runtime::cuda_require(cudaStreamSynchronize(context->execution.stream), "cudaStreamSynchronize(cellerator_context_synchronize)");
        return CELLERATOR_STATUS_SUCCESS;
    } catch (...) {
        return map_exception_();
    }
}

extern "C" void cellerator_context_destroy(cellerator_context *context) {
    if (context == nullptr) return;
    runtime::clear(&context->dense_cache);
    runtime::clear(&context->sparse_cache);
    if (context->owns_scratch) {
        runtime::clear(&context->scratch);
    } else {
        context->scratch.data = nullptr;
        context->scratch.bytes = 0u;
    }
    runtime::clear(&context->execution);
    delete context;
}

extern "C" void cellerator_saved_state_destroy(cellerator_saved_state *saved_state) {
    delete saved_state;
}

extern "C" cellerator_status cellerator_sparse_projection_forward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *rhs,
    cellerator_tensor_desc *out,
    cellerator_saved_state **saved_state) {
    clear_last_error_();
    if (saved_state != nullptr) *saved_state = nullptr;
    if (const cellerator_status status = validate_context_(context); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_layout_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_dense_tensor_(rhs, CELLERATOR_DTYPE_F32, 2u, layout->cols, rhs == nullptr ? 0u : rhs->shape[1], "rhs data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (!has_contiguous_matrix_storage_(rhs)) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "rhs must be row-major or explicit leading-dimension matrix");
    const uint64_t out_cols = rhs->shape[1];
    if (out_cols > static_cast<uint64_t>(INT32_MAX)) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "projection output column count is too large");
    if (const cellerator_status status = validate_dense_tensor_(out, CELLERATOR_DTYPE_F32, 2u, layout->rows, out_cols, "out data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (!has_contiguous_matrix_storage_(out)) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "out must be row-major or explicit leading-dimension matrix");

    try {
        if (layout->layout == CELLERATOR_LAYOUT_CSR) {
            if (const cellerator_status status = validate_csr_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
            sparse_project::csr_spmm_fwd_f16_f32(
                context->execution,
                layout->major_ptr,
                layout->minor_idx,
                static_cast<const __half *>(layout->values),
                layout->rows,
                layout->cols,
                static_cast<const float *>(rhs->data),
                matrix_ld_(rhs),
                static_cast<int64_t>(out_cols),
                static_cast<float *>(out->data),
                matrix_ld_(out));
        } else if (layout->layout == CELLERATOR_LAYOUT_BLOCKED_ELL) {
            if (const cellerator_status status = validate_blocked_ell_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
            sparse_project::blocked_ell_spmm_fwd_f16_f32(
                context->execution,
                layout->block_col_idx,
                static_cast<const __half *>(layout->values),
                layout->rows,
                layout->cols,
                layout->block_size,
                layout->ell_cols,
                static_cast<const float *>(rhs->data),
                matrix_ld_(rhs),
                static_cast<int64_t>(out_cols),
                static_cast<float *>(out->data),
                matrix_ld_(out));
        } else {
            return fail_(CELLERATOR_STATUS_UNSUPPORTED, "projection forward supports CSR and Blocked-ELL fp16 layouts");
        }
        if (saved_state != nullptr) *saved_state = make_saved_state_(CELLERATOR_SAVED_STATE_SPARSE_PROJECTION, layout, static_cast<uint32_t>(out_cols));
        return CELLERATOR_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return fail_(CELLERATOR_STATUS_INTERNAL_ERROR, "saved-state allocation failed");
    } catch (...) {
        return map_exception_();
    }
}

extern "C" cellerator_status cellerator_sparse_projection_backward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *rhs,
    const cellerator_tensor_desc *grad_out,
    cellerator_tensor_desc *grad_values,
    cellerator_tensor_desc *grad_rhs) {
    clear_last_error_();
    if (const cellerator_status status = validate_context_(context); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_layout_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_csr_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_dense_tensor_(rhs, CELLERATOR_DTYPE_F32, 2u, layout->cols, rhs == nullptr ? 0u : rhs->shape[1], "rhs data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    const uint64_t out_cols = rhs->shape[1];
    if (const cellerator_status status = validate_dense_tensor_(grad_out, CELLERATOR_DTYPE_F32, 2u, layout->rows, out_cols, "grad_out data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (!has_contiguous_matrix_storage_(rhs) || !has_contiguous_matrix_storage_(grad_out)) {
        return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "projection backward requires row-major or explicit leading-dimension matrices");
    }
    if (grad_values == nullptr && grad_rhs == nullptr) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "projection backward requires at least one gradient output");
    }
    if (grad_values != nullptr) {
        if (const cellerator_status status = validate_vector_tensor_(grad_values, CELLERATOR_DTYPE_F32, layout->nnz, "grad_values data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    }
    if (grad_rhs != nullptr) {
        if (const cellerator_status status = validate_dense_tensor_(grad_rhs, CELLERATOR_DTYPE_F32, 2u, layout->cols, out_cols, "grad_rhs data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
        if (!has_contiguous_matrix_storage_(grad_rhs)) return fail_(CELLERATOR_STATUS_INVALID_SHAPE, "grad_rhs must be row-major or explicit leading-dimension matrix");
    }

    try {
        if (grad_values != nullptr) {
            sparse_ops::base::csr_spmm_bwd_values_f16_f32(
                context->execution,
                layout->major_ptr,
                layout->minor_idx,
                static_cast<const float *>(grad_out->data),
                static_cast<const float *>(rhs->data),
                layout->rows,
                matrix_ld_(rhs),
                static_cast<int64_t>(out_cols),
                static_cast<float *>(grad_values->data));
        }
        if (grad_rhs != nullptr) {
            sparse_ops::base::csr_spmm_bwd_rhs_f16_f32(
                context->execution,
                layout->major_ptr,
                layout->minor_idx,
                static_cast<const __half *>(layout->values),
                static_cast<const float *>(grad_out->data),
                layout->rows,
                layout->cols,
                matrix_ld_(grad_out),
                static_cast<int64_t>(out_cols),
                static_cast<float *>(grad_rhs->data),
                matrix_ld_(grad_rhs));
        }
        return CELLERATOR_STATUS_SUCCESS;
    } catch (...) {
        return map_exception_();
    }
}

extern "C" cellerator_status cellerator_csr_row_scale_forward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *row_scales,
    cellerator_tensor_desc *out_values,
    cellerator_saved_state **saved_state) {
    clear_last_error_();
    if (saved_state != nullptr) *saved_state = nullptr;
    if (const cellerator_status status = validate_context_(context); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_layout_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_csr_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(row_scales, CELLERATOR_DTYPE_F32, layout->rows, "row_scales data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(out_values, CELLERATOR_DTYPE_F32, layout->nnz, "out_values data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;

    try {
        sparse_ops::base::csr_row_scale_fwd_f16_f32(
            context->execution,
            layout->major_ptr,
            static_cast<const __half *>(layout->values),
            static_cast<const float *>(row_scales->data),
            layout->rows,
            static_cast<float *>(out_values->data));
        if (saved_state != nullptr) *saved_state = make_saved_state_(CELLERATOR_SAVED_STATE_ROW_SCALE, layout, 0u);
        return CELLERATOR_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return fail_(CELLERATOR_STATUS_INTERNAL_ERROR, "saved-state allocation failed");
    } catch (...) {
        return map_exception_();
    }
}

extern "C" cellerator_status cellerator_csr_row_scale_backward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *grad_out_values,
    const cellerator_tensor_desc *row_scales,
    cellerator_tensor_desc *grad_values,
    cellerator_tensor_desc *grad_row_scales) {
    clear_last_error_();
    if (const cellerator_status status = validate_context_(context); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_layout_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_csr_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(grad_out_values, CELLERATOR_DTYPE_F32, layout->nnz, "grad_out_values data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (grad_values == nullptr && grad_row_scales == nullptr) {
        return fail_(CELLERATOR_STATUS_INVALID_ARGUMENT, "row-scale backward requires at least one gradient output");
    }
    if (grad_values != nullptr) {
        if (const cellerator_status status = validate_vector_tensor_(row_scales, CELLERATOR_DTYPE_F32, layout->rows, "row_scales data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
        if (const cellerator_status status = validate_vector_tensor_(grad_values, CELLERATOR_DTYPE_F32, layout->nnz, "grad_values data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    }
    if (grad_row_scales != nullptr) {
        if (const cellerator_status status = validate_vector_tensor_(grad_row_scales, CELLERATOR_DTYPE_F32, layout->rows, "grad_row_scales data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    }

    try {
        if (grad_values != nullptr) {
            sparse_ops::base::csr_row_scale_bwd_values_f16_f32(
                context->execution,
                layout->major_ptr,
                static_cast<const float *>(grad_out_values->data),
                static_cast<const float *>(row_scales->data),
                layout->rows,
                static_cast<float *>(grad_values->data));
        }
        if (grad_row_scales != nullptr) {
            sparse_ops::base::csr_row_scale_bwd_scales_f16_f32(
                context->execution,
                layout->major_ptr,
                static_cast<const __half *>(layout->values),
                static_cast<const float *>(grad_out_values->data),
                layout->rows,
                static_cast<float *>(grad_row_scales->data));
        }
        return CELLERATOR_STATUS_SUCCESS;
    } catch (...) {
        return map_exception_();
    }
}

extern "C" cellerator_status cellerator_sparse_reconstruction_loss_forward_backward(
    cellerator_context *context,
    const cellerator_sparse_layout_desc *layout,
    const cellerator_tensor_desc *log_scale,
    const cellerator_tensor_desc *offset,
    const cellerator_feature_affine_reconstruction_config *config,
    cellerator_tensor_desc *reconstruction_loss,
    cellerator_tensor_desc *range_loss,
    cellerator_tensor_desc *grad_log_scale,
    cellerator_tensor_desc *grad_offset) {
    clear_last_error_();
    if (const cellerator_status status = validate_context_(context); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_layout_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_reconstruction_config_(config); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(log_scale, CELLERATOR_DTYPE_F32, layout->cols, "log_scale data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(offset, CELLERATOR_DTYPE_F32, layout->cols, "offset data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_scalar_tensor_(reconstruction_loss, CELLERATOR_DTYPE_F32, "reconstruction_loss data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_scalar_tensor_(range_loss, CELLERATOR_DTYPE_F32, "range_loss data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(grad_log_scale, CELLERATOR_DTYPE_F32, layout->cols, "grad_log_scale data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;
    if (const cellerator_status status = validate_vector_tensor_(grad_offset, CELLERATOR_DTYPE_F32, layout->cols, "grad_offset data is null"); status != CELLERATOR_STATUS_SUCCESS) return status;

    sparse_ops::feature_affine_quantize_config native_config{};
    native_config.bits = config->bits;
    native_config.scale_floor = config->scale_floor;
    native_config.reconstruction_weight = config->reconstruction_weight;
    native_config.range_weight = config->range_weight;
    native_config.min_dynamic_range = config->min_dynamic_range;

    try {
        if (layout->layout == CELLERATOR_LAYOUT_CSR) {
            if (const cellerator_status status = validate_csr_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
            sparse_ops::base::csr_feature_affine_quantize_fwd_bwd_f16_f32(
                context->execution,
                layout->major_ptr,
                layout->minor_idx,
                static_cast<const __half *>(layout->values),
                layout->rows,
                layout->cols,
                static_cast<const float *>(log_scale->data),
                static_cast<const float *>(offset->data),
                native_config,
                static_cast<float *>(reconstruction_loss->data),
                static_cast<float *>(range_loss->data),
                static_cast<float *>(grad_log_scale->data),
                static_cast<float *>(grad_offset->data));
        } else if (layout->layout == CELLERATOR_LAYOUT_BLOCKED_ELL) {
            if (const cellerator_status status = validate_blocked_ell_f16_(layout); status != CELLERATOR_STATUS_SUCCESS) return status;
            sparse_ops::base::blocked_ell_feature_affine_quantize_fwd_bwd_f16_f32(
                context->execution,
                layout->block_col_idx,
                static_cast<const __half *>(layout->values),
                layout->rows,
                layout->cols,
                layout->block_size,
                layout->ell_cols,
                static_cast<const float *>(log_scale->data),
                static_cast<const float *>(offset->data),
                native_config,
                static_cast<float *>(reconstruction_loss->data),
                static_cast<float *>(range_loss->data),
                static_cast<float *>(grad_log_scale->data),
                static_cast<float *>(grad_offset->data));
        } else {
            return fail_(CELLERATOR_STATUS_UNSUPPORTED, "reconstruction loss supports CSR and Blocked-ELL fp16 layouts");
        }
        return CELLERATOR_STATUS_SUCCESS;
    } catch (...) {
        return map_exception_();
    }
}
