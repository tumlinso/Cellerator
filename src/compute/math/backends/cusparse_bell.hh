#pragma once

#include <Cellerator/compute/math/backend.hh>
#include <Cellerator/compute/math/physical_bell.hh>

#include <cuda_fp16.h>

#include <cstring>
#include <limits>

namespace cellerator::compute::math {

namespace cusparse_bell_detail {

inline constexpr u64 backend_tag = 0x4355535042454c4cull;
inline constexpr u64 algorithm_identity = 0x42454c4c414c4731ull;
inline constexpr u64 kernel_identity = 0x4631364633320000ull;

struct prepared_state {
    cusparseHandle_t handle = nullptr;
    cusparseDnMatDescr_t dense_rhs = nullptr;
    struct chunk {
        cusparseSpMatDescr_t sparse = nullptr;
        cusparseDnMatDescr_t output = nullptr;
        void *workspace = nullptr;
        std::size_t workspace_span = 0u;
        float *output_stage = nullptr;
        std::size_t output_span = 0u;
        u32 row_begin = 0u;
        u32 rows = 0u;
    } *chunks = nullptr;
    std::size_t chunk_count = 0u;
    float alpha = 0.0f;
    float beta = 0.0f;
};

inline backend_capability reject(
    capability_code code,
    const char *message,
    request_validation_code validation = request_validation_code::ok) noexcept {
    return {code, validation, message};
}

inline backend_status fail(
    backend_status_code code,
    const char *message,
    capability_code capability = capability_code::supported,
    request_validation_code validation = request_validation_code::ok) noexcept {
    return {code, capability, validation, cudaSuccess, message};
}

inline bool legal_block_size(u32 value) noexcept {
    return value == 8u || value == 16u || value == 32u;
}

inline bool same_candidate(
    const physical_bell_view &lhs,
    const physical_bell_view &rhs) noexcept {
    return lhs.schema_version == rhs.schema_version
        && lhs.block_size == rhs.block_size
        && lhs.row_count == rhs.row_count
        && lhs.feature_count == rhs.feature_count
        && lhs.padded_row_count == rhs.padded_row_count
        && lhs.padded_feature_count == rhs.padded_feature_count
        && lhs.ell_columns == rhs.ell_columns
        && lhs.value_size_bytes == rhs.value_size_bytes
        && lhs.feature_block_geometry_identity
            == rhs.feature_block_geometry_identity
        && lhs.ordering_identity == rhs.ordering_identity
        && lhs.row_domain_identity == rhs.row_domain_identity
        && lhs.candidate_identity == rhs.candidate_identity;
}

inline bool scalar_f32(const scalar_value &value, float *out) noexcept {
    if (out == nullptr
        || value.type_code != static_cast<u32>(real::value_f32)
        || value.reserved != 0u || (value.bits >> 32u) != 0u) {
        return false;
    }
    std::memcpy(out, &value.bits, sizeof(*out));
    return true;
}

inline void destroy_state(prepared_state *state) noexcept {
    if (state == nullptr) return;
    for (std::size_t index = 0u; index < state->chunk_count; ++index) {
        if (state->chunks[index].output != nullptr) {
            cusparseDestroyDnMat(state->chunks[index].output);
        }
        if (state->chunks[index].sparse != nullptr) {
            cusparseDestroySpMat(state->chunks[index].sparse);
        }
    }
    delete[] state->chunks;
    if (state->dense_rhs != nullptr) cusparseDestroyDnMat(state->dense_rhs);
    delete state;
}

inline bool normalized_value_bytes(
    const physical_bell_view &view,
    std::size_t *bytes) noexcept {
    if (bytes == nullptr) return false;
    *bytes = 0u;
    if (view.ell_columns == view.block_size) return true;
    const std::size_t rows = view.padded_row_count;
    if (rows != 0u && view.ell_columns
            > std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    const std::size_t elements = rows * view.ell_columns;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(__half)) {
        return false;
    }
    *bytes = elements * sizeof(__half);
    return true;
}

inline bool align_workspace_bytes(
    std::size_t bytes,
    std::size_t *aligned) noexcept {
    constexpr std::size_t alignment = 256u;
    if (aligned == nullptr
        || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1u)) {
        return false;
    }
    *aligned = (bytes + alignment - 1u) & ~(alignment - 1u);
    return true;
}

inline u64 candidate_backend_identity(
    const physical_bell_view &view) noexcept {
    u64 hash = backend_tag;
    hash = detail::mix_fingerprint(hash, &view.schema_version,
        sizeof(view.schema_version));
    hash = detail::mix_fingerprint(hash, &view.block_size,
        sizeof(view.block_size));
    hash = detail::mix_fingerprint(hash, &view.candidate_identity,
        sizeof(view.candidate_identity));
    return hash == 0u ? backend_tag : hash;
}

} // namespace cusparse_bell_detail

// One instance represents one materialized BELL8/16/32 candidate. The view and
// its device buffers are borrowed and must outlive every prepared execution.
// Dense RHS and output bindings include the view's padded feature/row domains;
// the logical dimensions in spmm_request remain row_count/feature_count.
class CusparseBellBackend final : public SpMMBackend {
public:
    explicit CusparseBellBackend(const physical_bell_view &view) noexcept;

    CusparseBellBackend(const CusparseBellBackend &) = delete;
    CusparseBellBackend &operator=(const CusparseBellBackend &) = delete;
    CusparseBellBackend(CusparseBellBackend &&) = delete;
    CusparseBellBackend &operator=(CusparseBellBackend &&) = delete;

    u64 identity() const noexcept override;
    const char *name() const noexcept override;
    backend_capability query(
        const spmm_request &request,
        const DeviceCapabilities &device) const noexcept override;
    backend_status prepare(PreparedExecution *prepared) noexcept override;
    backend_status run(PreparedExecution *prepared) noexcept override;
    void release(PreparedExecution *prepared) noexcept override;

    const physical_bell_view &view() const noexcept { return view_; }

private:
    physical_bell_view view_{};
    u64 identity_ = 0u;
};

inline CusparseBellBackend::CusparseBellBackend(
    const physical_bell_view &view) noexcept
    : view_(view), identity_(cusparse_bell_detail::candidate_backend_identity(view)) {}

inline u64 CusparseBellBackend::identity() const noexcept {
    return identity_;
}

inline const char *CusparseBellBackend::name() const noexcept {
    return "cusparse-blocked-ell-f16-f16-f32";
}

inline backend_capability CusparseBellBackend::query(
    const spmm_request &request,
    const DeviceCapabilities &device) const noexcept {
    const request_validation_result validation = validate_spmm_request(request);
    if (!validation) {
        return cusparse_bell_detail::reject(
            capability_code::invalid_request, validation.message,
            validation.code);
    }
    if (device.compute_capability_major < 7 || !device.tensor_core_capable) {
        return cusparse_bell_detail::reject(capability_code::unsupported_device,
            "cuSPARSE Blocked-ELL f16 requires Tensor Core capability >= 7.0");
    }
    const u32 b = view_.block_size;
    if (view_.schema_version != physical_bell_schema_version
        || !cusparse_bell_detail::legal_block_size(b)
        || view_.candidate_identity == 0u
        || view_.row_count != request.m
        || view_.feature_count != request.k
        || view_.padded_row_count < view_.row_count
        || view_.padded_feature_count < view_.feature_count
        || view_.padded_row_count % b != 0u
        || view_.padded_feature_count % b != 0u
        || view_.ell_columns == 0u || view_.ell_columns % b != 0u
        || view_.ell_columns > view_.padded_feature_count
        || view_.value_size_bytes != sizeof(__half)
        || view_.column_indices == nullptr || view_.values == nullptr) {
        return cusparse_bell_detail::reject(capability_code::unsupported_layout,
            "backend requires one legal materialized BELL8/16/32 f16 view");
    }
    if (request.sparse_storage_type_code != static_cast<u32>(real::value_f16)
        || request.dense_storage_type_code != static_cast<u32>(real::value_f16)
        || request.output_storage_type_code != static_cast<u32>(real::value_f32)
        || request.compute_type_code != static_cast<u32>(real::value_f32)
        || request.accumulation_type_code != static_cast<u32>(real::value_f32)) {
        return cusparse_bell_detail::reject(capability_code::unsupported_type,
            "V100 Blocked-ELL backend requires f16 A/B and f32 C/compute");
    }
    if (request.transpose_sparse != transpose_kind::none
        || request.transpose_dense != transpose_kind::none) {
        return cusparse_bell_detail::reject(
            capability_code::unsupported_transpose,
            "cuSPARSE Blocked-ELL backend supports non-transposed A and B");
    }
    if (request.determinism == determinism_requirement::deterministic) {
        return cusparse_bell_detail::reject(
            capability_code::unsupported_determinism,
            "cuSPARSE does not guarantee deterministic Blocked-ELL SpMM");
    }
    if (!scalar_is_zero(request.beta)) {
        return cusparse_bell_detail::reject(capability_code::backend_unavailable,
            "V100 cuSPARSE Blocked-ELL candidate requires beta=0");
    }
    if (request.sparse_feature_order.kind != feature_order_kind::packed
        || request.dense_feature_order.kind != feature_order_kind::packed
        || request.sparse_feature_order.packing_geometry_identity
            != view_.feature_block_geometry_identity
        || request.dense_feature_order.packing_geometry_identity
            != view_.feature_block_geometry_identity) {
        return cusparse_bell_detail::reject(capability_code::unsupported_layout,
            "BELL and dense operands must share the candidate packing geometry");
    }

    std::size_t normalization_bytes = 0u, normalization_span = 0u;
    if (!cusparse_bell_detail::normalized_value_bytes(
            view_, &normalization_bytes)
        || !cusparse_bell_detail::align_workspace_bytes(
            normalization_bytes, &normalization_span)) {
        return cusparse_bell_detail::reject(
            capability_code::workspace_policy_rejected,
            "BELL prepared workspace size overflows");
    }
    backend_capability result =
        query_generic_unfused_epilogue_capability(request);
    if (!result) return result;
    result.physical_view_schema_version = physical_bell_schema_version;
    result.algorithm_identity = cusparse_bell_detail::algorithm_identity;
    result.kernel_variant_identity = cusparse_bell_detail::kernel_identity | b;
    result.tuning_identity = view_.candidate_identity;
    result.workspace_bytes = normalization_span;
    result.preprocessing = normalization_bytes == 0u
        ? preprocessing_kind::none : preprocessing_kind::backend_preprocess;
    return result;
}

} // namespace cellerator::compute::math
