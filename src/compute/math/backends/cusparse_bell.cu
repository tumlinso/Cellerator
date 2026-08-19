/* CP-MATH-07: cuSPARSE Blocked-ELL on sm_70. Descriptors, scratch slices, and
 * CP-MATH-04 value-order normalization are prepared once. BELL8 uses 16-row
 * vendor descriptors; column-major chunks stage tightly pitched outputs.
 * Run submits vendor SpMM, device copies, and the generic epilogue only.
 * Focused evidence: math_cusparse_bell_test.cu.
 */

#include "cusparse_bell.hh"

#include <Cellerator/runtime/libraries.cuh>
#include <Cellerator/types.cuh>

#include <cuda_fp16.h>
#include <cusparse.h>

#include <cstring>
#include <limits>
#include <new>

namespace cellerator::compute::math {
using cusparse_bell_detail::align_workspace_bytes;
using cusparse_bell_detail::destroy_state;
using cusparse_bell_detail::fail;
using cusparse_bell_detail::normalized_value_bytes;
using cusparse_bell_detail::prepared_state;
using cusparse_bell_detail::same_candidate;
using cusparse_bell_detail::scalar_f32;

backend_status CusparseBellBackend::prepare(
    PreparedExecution *prepared) noexcept {
    if (prepared == nullptr || prepared->backend != this
        || !prepared->device.initialized) {
        return fail(backend_status_code::backend_mismatch,
            "BELL prepare requires its initialized PreparedExecution");
    }
    const backend_capability capability =
        query(prepared->request.operation, prepared->device.capabilities);
    if (!capability) return detail::capability_failure(capability);

    const auto *bound = static_cast<const physical_bell_view *>(
        prepared->request.bindings.sparse_matrix);
    if (bound == nullptr || !same_candidate(view_, *bound)
        || prepared->request.bindings.sparse_matrix_identity
            != view_.candidate_identity
        || prepared->request.bindings.dense_rhs == nullptr
        || prepared->request.bindings.output == nullptr) {
        return fail(backend_status_code::invalid_argument,
            "BELL bindings do not match the backend candidate",
            capability_code::supported,
            request_validation_code::invalid_identity);
    }

    auto *state = new (std::nothrow) prepared_state;
    if (state == nullptr) {
        return fail(backend_status_code::runtime_failure,
            "BELL prepared descriptor state allocation failed");
    }
    prepared->backend_state = state;

    const u32 chunk_rows = bound->block_size == 8u ? 16u
        : bound->padded_row_count;
    state->chunk_count = (bound->padded_row_count + chunk_rows - 1u)
        / chunk_rows;
    state->chunks = new (std::nothrow)
        prepared_state::chunk[state->chunk_count];
    if (state->chunks == nullptr) {
        return fail(backend_status_code::runtime_failure,
            "BELL prepared chunk descriptor allocation failed");
    }

    if (!scalar_f32(prepared->request.operation.alpha, &state->alpha)
        || !scalar_f32(prepared->request.operation.beta, &state->beta)) {
        return fail(backend_status_code::invalid_argument,
            "BELL alpha and beta must be canonical f32 scalars",
            capability_code::unsupported_type,
            request_validation_code::invalid_scalar);
    }

    try {
        state->handle = acquire_cusparse(&prepared->device);
        const spmm_request &request = prepared->request.operation;
        const cusparseOrder_t rhs_order =
            request.dense_rhs_layout == dense_layout_kind::row_major
            ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
        const cusparseOrder_t output_order =
            request.output_layout == dense_layout_kind::row_major
            ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
        cusparseStatus_t status = cusparseCreateDnMat(
            &state->dense_rhs,
            static_cast<std::int64_t>(bound->padded_feature_count),
            static_cast<std::int64_t>(request.n),
            static_cast<std::int64_t>(request.dense_rhs_leading_dimension),
            const_cast<void *>(prepared->request.bindings.dense_rhs),
            CUDA_R_16F,
            rhs_order);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            return fail(backend_status_code::backend_failure,
                "cusparseCreateDnMat(BELL RHS) failed");
        }
        const std::size_t blocks_per_row =
            bound->ell_columns / bound->block_size;
        const auto *source_values = static_cast<const __half *>(bound->values);
        auto *output_base = static_cast<unsigned char *>(
            prepared->request.bindings.output);
        std::size_t external_bytes = 0u;
        std::size_t output_staging_bytes = 0u;
        for (std::size_t index = 0u; index < state->chunk_count; ++index) {
            const u32 row_begin = static_cast<u32>(index) * chunk_rows;
            const u32 rows_remaining = bound->padded_row_count - row_begin;
            const u32 rows_here = rows_remaining < chunk_rows
                ? rows_remaining : chunk_rows;
            state->chunks[index].row_begin = row_begin;
            state->chunks[index].rows = rows_here;
            const std::size_t block_row_begin = row_begin / bound->block_size;
            const auto *columns = bound->column_indices
                + block_row_begin * blocks_per_row;
            const auto *values = source_values
                + static_cast<std::size_t>(row_begin) * bound->ell_columns;
            status = cusparseCreateBlockedEll(
                &state->chunks[index].sparse,
                static_cast<std::int64_t>(rows_here),
                static_cast<std::int64_t>(bound->padded_feature_count),
                static_cast<std::int64_t>(bound->block_size),
                static_cast<std::int64_t>(bound->ell_columns),
                const_cast<std::int32_t *>(columns),
                const_cast<__half *>(values),
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_16F);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                return fail(backend_status_code::backend_failure,
                    "cusparseCreateBlockedEll(prepared chunk) failed");
            }
            const std::size_t output_element =
                request.output_layout == dense_layout_kind::row_major
                ? static_cast<std::size_t>(row_begin)
                    * request.output_leading_dimension
                : row_begin;
            const std::int64_t output_leading_dimension =
                state->chunk_count > 1u
                    && request.output_layout == dense_layout_kind::column_major
                ? static_cast<std::int64_t>(rows_here)
                : static_cast<std::int64_t>(
                    request.output_leading_dimension);
            status = cusparseCreateDnMat(
                &state->chunks[index].output,
                static_cast<std::int64_t>(rows_here),
                static_cast<std::int64_t>(request.n),
                output_leading_dimension,
                output_base + output_element * sizeof(float),
                CUDA_R_32F,
                output_order);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                return fail(backend_status_code::backend_failure,
                    "cusparseCreateDnMat(BELL output chunk) failed");
            }
            std::size_t chunk_bytes = 0u;
            status = cusparseSpMM_bufferSize(
                state->handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &state->alpha,
                state->chunks[index].sparse,
                state->dense_rhs,
                &state->beta,
                state->chunks[index].output,
                CUDA_R_32F,
                CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
                &chunk_bytes);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                return fail(backend_status_code::backend_failure,
                    "cusparseSpMM_bufferSize(BELL chunk) failed");
            }
            if (!align_workspace_bytes(
                    chunk_bytes, &state->chunks[index].workspace_span)
                || state->chunks[index].workspace_span
                    > std::numeric_limits<std::size_t>::max()
                        - external_bytes) {
                return fail(backend_status_code::runtime_failure,
                    "BELL chunk workspace size overflows");
            }
            external_bytes += state->chunks[index].workspace_span;
            if (state->chunk_count > 1u
                && request.output_layout == dense_layout_kind::column_major) {
                const std::size_t stage_bytes = static_cast<std::size_t>(
                    rows_here) * request.n * sizeof(float);
                if (!align_workspace_bytes(
                        stage_bytes, &state->chunks[index].output_span)
                    || state->chunks[index].output_span
                        > std::numeric_limits<std::size_t>::max()
                            - output_staging_bytes) {
                    return fail(backend_status_code::runtime_failure,
                        "BELL staged output size overflows");
                }
                output_staging_bytes += state->chunks[index].output_span;
            }
        }

        std::size_t normalization_bytes = 0u, normalization_span = 0u;
        if (!normalized_value_bytes(*bound, &normalization_bytes)
            || !align_workspace_bytes(normalization_bytes, &normalization_span)
            || output_staging_bytes
                > std::numeric_limits<std::size_t>::max() - normalization_span
            || external_bytes > std::numeric_limits<std::size_t>::max()
                - normalization_span - output_staging_bytes) {
            return fail(backend_status_code::runtime_failure,
                "BELL prepared workspace size overflows");
        }
        const std::size_t total_workspace = normalization_span
            + output_staging_bytes + external_bytes;

        const workspace_policy &policy = request.workspace;
        if ((policy.kind == workspace_policy_kind::no_additional_workspace
                && total_workspace != 0u)
            || (policy.kind == workspace_policy_kind::caller_limit
                && total_workspace > policy.byte_limit)) {
            return fail(backend_status_code::capability_rejected,
                "cuSPARSE BELL workspace exceeds request policy",
                capability_code::workspace_policy_rejected);
        }
        auto *workspace_base = static_cast<unsigned char *>(request_workspace(
            &prepared->device, total_workspace));
        auto *output_stage = output_staging_bytes == 0u
            ? nullptr : workspace_base + normalization_span;
        auto *chunk_workspace = external_bytes == 0u ? nullptr
            : workspace_base + normalization_span + output_staging_bytes;
        for (std::size_t index = 0u; index < state->chunk_count; ++index) {
            if (state->chunks[index].output_span != 0u) {
                state->chunks[index].output_stage =
                    reinterpret_cast<float *>(output_stage);
                output_stage += state->chunks[index].output_span;
                const cusparseStatus_t set_status = cusparseDnMatSetValues(
                    state->chunks[index].output,
                    state->chunks[index].output_stage);
                if (set_status != CUSPARSE_STATUS_SUCCESS) {
                    return fail(backend_status_code::backend_failure,
                        "cusparseDnMatSetValues(staged BELL output) failed");
                }
            }
            state->chunks[index].workspace =
                state->chunks[index].workspace_span == 0u
                ? nullptr : chunk_workspace;
            if (state->chunks[index].workspace_span != 0u) {
                chunk_workspace += state->chunks[index].workspace_span;
            }
        }
        prepared->plan.workspace_bytes = total_workspace;

        if (normalization_bytes != 0u) {
            auto *normalized = reinterpret_cast<__half *>(workspace_base);
            const std::size_t block_rows =
                bound->padded_row_count / bound->block_size;
            for (std::size_t block_row = 0u; block_row < block_rows; ++block_row) {
                for (u32 lane = 0u; lane < bound->block_size; ++lane) {
                    for (std::size_t slot = 0u; slot < blocks_per_row; ++slot) {
                        const std::size_t source =
                            ((block_row * blocks_per_row + slot)
                                * bound->block_size + lane)
                                * bound->block_size;
                        const std::size_t target =
                            ((block_row * bound->block_size + lane)
                                * blocks_per_row + slot)
                                * bound->block_size;
                        const cudaError_t error = cudaMemcpyAsync(
                            normalized + target,
                            source_values + source,
                            bound->block_size * sizeof(__half),
                            cudaMemcpyDeviceToDevice,
                            prepared->device.execution.stream);
                        if (error != cudaSuccess) {
                            return {backend_status_code::runtime_failure,
                                capability_code::supported,
                                request_validation_code::ok,
                                error,
                                "BELL prepared value normalization copy failed"};
                        }
                    }
                }
            }
            for (std::size_t index = 0u; index < state->chunk_count; ++index) {
                const u32 row_begin = static_cast<u32>(index) * chunk_rows;
                status = cusparseSpMatSetValues(
                    state->chunks[index].sparse,
                    normalized + static_cast<std::size_t>(row_begin)
                        * bound->ell_columns);
                if (status != CUSPARSE_STATUS_SUCCESS) {
                    return fail(backend_status_code::backend_failure,
                        "cusparseSpMatSetValues(normalized BELL) failed");
                }
            }
        }
        return {};
    } catch (...) {
        return fail(backend_status_code::runtime_failure,
            "CUDA or cuSPARSE BELL preparation failed");
    }
}

backend_status CusparseBellBackend::run(PreparedExecution *prepared) noexcept {
    if (prepared == nullptr || prepared->backend != this
        || prepared->backend_state == nullptr) {
        return fail(backend_status_code::backend_mismatch,
            "BELL run requires matching prepared descriptor state");
    }
    auto *state = static_cast<prepared_state *>(prepared->backend_state);
    for (std::size_t index = 0u; index < state->chunk_count; ++index) {
        const cusparseStatus_t status = cusparseSpMM(
            state->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &state->alpha,
            state->chunks[index].sparse,
            state->dense_rhs,
            &state->beta,
            state->chunks[index].output,
            CUDA_R_32F,
            CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
            state->chunks[index].workspace);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            return fail(backend_status_code::backend_failure,
                "cusparseSpMM(BELL chunk) failed");
        }
        if (state->chunks[index].output_stage != nullptr) {
            auto *output = static_cast<float *>(
                prepared->request.bindings.output)
                + state->chunks[index].row_begin;
            const std::size_t width = state->chunks[index].rows * sizeof(float);
            const cudaError_t error = cudaMemcpy2DAsync(
                output,
                prepared->request.operation.output_leading_dimension
                    * sizeof(float),
                state->chunks[index].output_stage,
                width,
                width,
                prepared->request.operation.n,
                cudaMemcpyDeviceToDevice,
                prepared->device.execution.stream);
            if (error != cudaSuccess) {
                return {backend_status_code::runtime_failure,
                    capability_code::supported,
                    request_validation_code::ok,
                    error,
                    "BELL staged output copy failed"};
            }
        }
    }
    return launch_generic_unfused_epilogue(
        &prepared->device,
        prepared->request.operation,
        prepared->request.bindings);
}

void CusparseBellBackend::release(PreparedExecution *prepared) noexcept {
    if (prepared == nullptr) return;
    destroy_state(static_cast<prepared_state *>(prepared->backend_state));
    prepared->backend_state = nullptr;
}

} // namespace cellerator::compute::math
