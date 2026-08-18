#include <Cellerator/compute/math/operation.hh>

#include <cstring>
#include <limits>

namespace cellerator::compute::math {

namespace {

bool supported_type_code(u32 code) noexcept {
    return code == static_cast<u32>(real::value_f16)
        || code == static_cast<u32>(real::value_f32)
        || code == static_cast<u32>(real::value_f64)
        || code == static_cast<u32>(real::value_bf16)
        || code == static_cast<u32>(real::value_fp8_e4m3)
        || code == static_cast<u32>(real::value_fp8_e5m2);
}

bool supported_transpose(transpose_kind value) noexcept {
    return value == transpose_kind::none || value == transpose_kind::transpose;
}

bool supported_layout(dense_layout_kind value) noexcept {
    return value == dense_layout_kind::row_major
        || value == dense_layout_kind::column_major;
}

bool supported_determinism(determinism_requirement value) noexcept {
    return value == determinism_requirement::allow_nondeterministic
        || value == determinism_requirement::deterministic;
}

bool supported_workspace_policy(workspace_policy_kind value) noexcept {
    return value == workspace_policy_kind::reusable_pool
        || value == workspace_policy_kind::caller_limit
        || value == workspace_policy_kind::no_additional_workspace;
}

bool supported_reuse(expected_reuse_kind value) noexcept {
    return value == expected_reuse_kind::single_run
        || value == expected_reuse_kind::bounded
        || value == expected_reuse_kind::persistent;
}

bool supported_epilogue(epilogue_kind value) noexcept {
    return value == epilogue_kind::none || value == epilogue_kind::bias
        || value == epilogue_kind::relu
        || value == epilogue_kind::gelu_exact_erf
        || value == epilogue_kind::gelu_tanh_approximate
        || value == epilogue_kind::bias_relu
        || value == epilogue_kind::bias_gelu_exact_erf
        || value == epilogue_kind::bias_gelu_tanh_approximate;
}

request_validation_result error(
    request_validation_code code,
    const char *message) noexcept {
    return {code, message};
}

bool bias_epilogue(epilogue_kind kind) noexcept {
    return kind == epilogue_kind::bias || kind == epilogue_kind::bias_relu
        || kind == epilogue_kind::bias_gelu_exact_erf
        || kind == epilogue_kind::bias_gelu_tanh_approximate;
}

bool canonical_scalar(const scalar_value &value) noexcept {
    if (value.reserved != 0u) {
        return false;
    }
    if (value.type_code == static_cast<u32>(real::value_f32)) {
        return (value.bits >> 32u) == 0u;
    }
    return value.type_code == static_cast<u32>(real::value_f64);
}

} // namespace

scalar_value make_scalar(float value) noexcept {
    scalar_value result;
    result.type_code = static_cast<u32>(real::value_f32);
    static_assert(sizeof(value) <= sizeof(result.bits), "scalar payload is too small");
    std::memcpy(&result.bits, &value, sizeof(value));
    return result;
}

scalar_value make_scalar(double value) noexcept {
    scalar_value result;
    result.type_code = static_cast<u32>(real::value_f64);
    static_assert(sizeof(value) == sizeof(result.bits), "scalar payload size changed");
    std::memcpy(&result.bits, &value, sizeof(value));
    return result;
}

bool scalar_is_zero(const scalar_value &value) noexcept {
    if (value.type_code == static_cast<u32>(real::value_f32)) {
        float decoded = 0.0f;
        std::memcpy(&decoded, &value.bits, sizeof(decoded));
        return decoded == 0.0f;
    }
    if (value.type_code == static_cast<u32>(real::value_f64)) {
        double decoded = 0.0;
        std::memcpy(&decoded, &value.bits, sizeof(decoded));
        return decoded == 0.0;
    }
    return false;
}

bool same_feature_order(
    const feature_order_identity &lhs,
    const feature_order_identity &rhs) noexcept {
    return lhs.schema_version == rhs.schema_version && lhs.kind == rhs.kind
        && lhs.feature_count == rhs.feature_count
        && lhs.feature_axis_identity_version == rhs.feature_axis_identity_version
        && lhs.feature_axis_identity == rhs.feature_axis_identity
        && lhs.packing_geometry_identity == rhs.packing_geometry_identity;
}

request_validation_result validate_spmm_request(const spmm_request &request) noexcept {
    if (request.schema_version != operation_contract_schema_version
        || request.operation != operation_kind::spmm) {
        return error(request_validation_code::unsupported_version,
            "unsupported SpMM operation contract");
    }
    if (request.sparse_nnz != 0u && (request.m == 0u || request.k == 0u)) {
        return error(request_validation_code::invalid_shape,
            "nonzero sparse_nnz requires nonzero M and K");
    }
    const bool sparse_shape_overflow = request.m != 0u
        && request.k > std::numeric_limits<u64>::max() / request.m;
    const bool output_shape_overflow = request.m != 0u
        && request.n > std::numeric_limits<u64>::max() / request.m;
    const bool dense_shape_overflow = request.k != 0u
        && request.n > std::numeric_limits<u64>::max() / request.k;
    if (sparse_shape_overflow || dense_shape_overflow || output_shape_overflow
        || (!sparse_shape_overflow && request.sparse_nnz > request.m * request.k)) {
        return error(request_validation_code::invalid_shape,
            "SpMM shape or sparse_nnz exceeds the addressable logical matrix");
    }
    if (!supported_transpose(request.transpose_sparse)
        || !supported_transpose(request.transpose_dense)) {
        return error(request_validation_code::invalid_layout,
            "SpMM request contains an unknown transpose mode");
    }
    if (!supported_layout(request.dense_rhs_layout)
        || !supported_layout(request.output_layout)) {
        return error(request_validation_code::invalid_layout,
            "unknown SpMM dense layout");
    }
    const u64 rhs_rows = request.transpose_dense == transpose_kind::none
        ? request.k : request.n;
    const u64 rhs_columns = request.transpose_dense == transpose_kind::none
        ? request.n : request.k;
    const u64 minimum_rhs_leading_dimension =
        request.dense_rhs_layout == dense_layout_kind::row_major
        ? rhs_columns : rhs_rows;
    const u64 minimum_output_leading_dimension =
        request.output_layout == dense_layout_kind::row_major
        ? request.n : request.m;
    if ((rhs_rows != 0u && rhs_columns != 0u
            && request.dense_rhs_leading_dimension
                < minimum_rhs_leading_dimension)
        || (request.m != 0u && request.n != 0u
            && request.output_leading_dimension
                < minimum_output_leading_dimension)) {
        return error(request_validation_code::invalid_layout,
            "dense leading dimension is smaller than its physical matrix");
    }
    if (!supported_type_code(request.sparse_storage_type_code)
        || !supported_type_code(request.dense_storage_type_code)
        || !supported_type_code(request.output_storage_type_code)
        || !supported_type_code(request.compute_type_code)
        || !supported_type_code(request.accumulation_type_code)) {
        return error(request_validation_code::invalid_type,
            "SpMM request contains an unknown numeric type code");
    }
    if (request.alpha.type_code != request.compute_type_code
        || request.beta.type_code != request.compute_type_code
        || !canonical_scalar(request.alpha) || !canonical_scalar(request.beta)) {
        return error(request_validation_code::invalid_scalar,
            "alpha and beta must use canonical request-compute f32 or f64 values");
    }
    if (!supported_determinism(request.determinism)) {
        return error(request_validation_code::invalid_determinism,
            "SpMM request contains an unknown determinism requirement");
    }
    if (!supported_workspace_policy(request.workspace.kind)
        || request.workspace.reserved != 0u
        || (request.workspace.kind == workspace_policy_kind::caller_limit
            && request.workspace.byte_limit == 0u)
        || (request.workspace.kind != workspace_policy_kind::caller_limit
            && request.workspace.byte_limit != 0u)) {
        return error(request_validation_code::invalid_workspace_policy,
            "workspace kind and byte limit disagree");
    }
    if (!supported_reuse(request.reuse.kind) || request.reuse.reserved != 0u
        || (request.reuse.kind == expected_reuse_kind::single_run
            && request.reuse.expected_run_count != 1u)
        || (request.reuse.kind == expected_reuse_kind::bounded
            && request.reuse.expected_run_count == 0u)
        || (request.reuse.kind == expected_reuse_kind::persistent
            && request.reuse.expected_run_count != 0u)) {
        return error(request_validation_code::invalid_reuse,
            "reuse kind and expected run count disagree");
    }
    if (request.sparse_structure.schema_version
            != sparse_structure_identity_schema_version
        || request.sparse_structure.identity_version == 0u
        || request.sparse_structure.value == 0u) {
        return error(request_validation_code::invalid_identity,
            "SpMM request requires a stable sparse structure identity");
    }
    const auto valid_order = [&](const feature_order_identity &order) {
        return order.schema_version == feature_order_identity_schema_version
            && order.feature_count == request.k
            && order.feature_axis_identity != 0u
            && order.feature_axis_identity_version != 0u
            && ((order.kind == feature_order_kind::canonical
                    && order.packing_geometry_identity == 0u)
                || (order.kind == feature_order_kind::packed
                    && order.packing_geometry_identity != 0u));
    };
    if (!valid_order(request.sparse_feature_order)
        || !valid_order(request.dense_feature_order)) {
        return error(request_validation_code::invalid_feature_order,
            "SpMM feature-order identity is incomplete or disagrees with K");
    }
    if (!same_feature_order(
            request.sparse_feature_order, request.dense_feature_order)) {
        return error(request_validation_code::feature_order_mismatch,
            "sparse and dense operands use different feature orders");
    }
    if (!supported_epilogue(request.epilogue.kind)) {
        return error(request_validation_code::invalid_epilogue,
            "SpMM request contains an unknown epilogue");
    }
    if (bias_epilogue(request.epilogue.kind)) {
        if (request.epilogue.bias_element_count != request.n
            || request.epilogue.bias_type_code != request.output_storage_type_code) {
            return error(request_validation_code::missing_bias,
                "bias epilogue requires N output-typed elements");
        }
    } else if (request.epilogue.bias_element_count != 0u
        || request.epilogue.bias_type_code != 0u) {
        return error(request_validation_code::unexpected_bias,
            "non-bias epilogue must not describe bias storage");
    }
    return {};
}

request_validation_result validate_math_request(const math_request &request) noexcept {
    const request_validation_result metadata = validate_spmm_request(request.operation);
    if (!metadata) {
        return metadata;
    }
    const trivial_operation_kind work = classify_trivial_operation(request.operation);
    if (work == trivial_operation_kind::no_output) {
        return {};
    }
    if (request.bindings.output == nullptr) {
        return error(request_validation_code::missing_binding,
            "nonempty SpMM output requires an output binding");
    }
    if (bias_epilogue(request.operation.epilogue.kind)) {
        if (request.bindings.bias == nullptr) {
            return error(request_validation_code::missing_bias,
                "bias epilogue requires a bias binding");
        }
    } else if (request.bindings.bias != nullptr) {
        return error(request_validation_code::unexpected_bias,
            "non-bias epilogue received a bias binding");
    }
    if (work == trivial_operation_kind::none
        && (request.bindings.sparse_matrix == nullptr
            || request.bindings.dense_rhs == nullptr)) {
        return error(request_validation_code::missing_binding,
            "nontrivial SpMM requires sparse and dense input bindings");
    }
    return {};
}

trivial_operation_kind classify_trivial_operation(
    const spmm_request &request) noexcept {
    if (request.m == 0u || request.n == 0u) {
        return trivial_operation_kind::no_output;
    }
    if (request.k == 0u || request.sparse_nnz == 0u
        || scalar_is_zero(request.alpha)) {
        return trivial_operation_kind::epilogue_only;
    }
    return trivial_operation_kind::none;
}

} // namespace cellerator::compute::math
