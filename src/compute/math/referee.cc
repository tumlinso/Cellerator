#include <Cellerator/compute/math/referee.hh>
#include <Cellerator/types.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cellerator::compute::math {

namespace {

referee_status fail(referee_status_code code, const char *message) noexcept {
    return {code, message};
}

bool supported_reference_type(u32 type_code) noexcept {
    return type_code == static_cast<u32>(real::value_f16)
        || type_code == static_cast<u32>(real::value_f32)
        || type_code == static_cast<u32>(real::value_f64);
}

bool read_value(const void *values, u32 type_code, u64 index, double *out) noexcept {
    if (values == nullptr || out == nullptr) return false;
    if (type_code == static_cast<u32>(real::value_f16)) {
        *out = static_cast<double>(__half2float(
            static_cast<const __half *>(values)[index]));
        return true;
    }
    if (type_code == static_cast<u32>(real::value_f32)) {
        *out = static_cast<double>(static_cast<const float *>(values)[index]);
        return true;
    }
    if (type_code == static_cast<u32>(real::value_f64)) {
        *out = static_cast<const double *>(values)[index];
        return true;
    }
    return false;
}

bool decode_scalar(const scalar_value &scalar, double *out) noexcept {
    if (out == nullptr) return false;
    if (scalar.type_code == static_cast<u32>(real::value_f32)) {
        float value = 0.0f;
        std::memcpy(&value, &scalar.bits, sizeof(value));
        *out = value;
        return true;
    }
    if (scalar.type_code == static_cast<u32>(real::value_f64)) {
        std::memcpy(out, &scalar.bits, sizeof(*out));
        return true;
    }
    return false;
}

u64 dense_index(const logical_dense_view &view, u64 row, u64 column) noexcept {
    return view.layout == dense_layout_kind::row_major
        ? row * view.leading_dimension + column
        : column * view.leading_dimension + row;
}

bool valid_dense_shape(
    const logical_dense_view &view,
    u64 rows,
    u64 columns) noexcept {
    if (view.rows != rows || view.columns != columns) return false;
    if (rows == 0u || columns == 0u) return true;
    const u64 minimum = view.layout == dense_layout_kind::row_major
        ? columns : rows;
    return view.values != nullptr && view.leading_dimension >= minimum
        && (view.layout == dense_layout_kind::row_major
            || view.layout == dense_layout_kind::column_major);
}

bool bias_epilogue(epilogue_kind kind) noexcept {
    return kind == epilogue_kind::bias || kind == epilogue_kind::bias_relu
        || kind == epilogue_kind::bias_gelu_exact_erf
        || kind == epilogue_kind::bias_gelu_tanh_approximate;
}

double apply_epilogue(double value, double bias, epilogue_kind kind) noexcept {
    if (bias_epilogue(kind)) value += bias;
    if (kind == epilogue_kind::relu || kind == epilogue_kind::bias_relu) {
        return std::fmax(value, 0.0);
    }
    if (kind == epilogue_kind::gelu_exact_erf
        || kind == epilogue_kind::bias_gelu_exact_erf) {
        return 0.5 * value * (1.0 + std::erf(value * 0.7071067811865475244));
    }
    if (kind == epilogue_kind::gelu_tanh_approximate
        || kind == epilogue_kind::bias_gelu_tanh_approximate) {
        return 0.5 * value
            * (1.0 + std::tanh(0.7978845608028653559
                * (value + 0.044715 * value * value * value)));
    }
    return value;
}

} // namespace

referee_status build_spmm_reference(
    const spmm_request &request,
    const logical_csr_view &sparse,
    const logical_dense_view &dense_rhs,
    const logical_dense_view &initial_output,
    const void *bias,
    double *reference,
    u64 reference_capacity) noexcept {
    const request_validation_result validation = validate_spmm_request(request);
    if (!validation) {
        return fail(referee_status_code::invalid_argument, validation.message);
    }
    if (request.m != 0u
        && request.n > std::numeric_limits<u64>::max() / request.m) {
        return fail(referee_status_code::overflow, "reference element count overflows");
    }
    const u64 output_count = request.m * request.n;
    if (output_count > reference_capacity
        || (output_count != 0u && reference == nullptr)) {
        return fail(referee_status_code::insufficient_capacity,
            "reference output capacity is too small");
    }
    if (!supported_reference_type(request.sparse_storage_type_code)
        || !supported_reference_type(request.dense_storage_type_code)
        || !supported_reference_type(request.output_storage_type_code)) {
        return fail(referee_status_code::unsupported_type,
            "reference supports f16, f32, and f64 storage");
    }

    const u64 sparse_rows = request.transpose_sparse == transpose_kind::none
        ? request.m : request.k;
    const u64 sparse_columns = request.transpose_sparse == transpose_kind::none
        ? request.k : request.m;
    if (sparse.rows != sparse_rows || sparse.columns != sparse_columns
        || sparse.nnz != request.sparse_nnz
        || sparse.value_type_code != request.sparse_storage_type_code
        || (sparse_rows != 0u && sparse.row_offsets == nullptr)
        || (sparse.nnz != 0u
            && (sparse.column_indices == nullptr || sparse.values == nullptr))) {
        return fail(referee_status_code::invalid_shape,
            "logical CSR view disagrees with the SpMM request");
    }
    if (sparse_rows != 0u
        && (sparse.row_offsets[0] != 0u
            || sparse.row_offsets[sparse_rows] != sparse.nnz)) {
        return fail(referee_status_code::invalid_shape,
            "logical CSR row offsets do not span nnz");
    }

    const u64 rhs_rows = request.transpose_dense == transpose_kind::none
        ? request.k : request.n;
    const u64 rhs_columns = request.transpose_dense == transpose_kind::none
        ? request.n : request.k;
    if (!valid_dense_shape(dense_rhs, rhs_rows, rhs_columns)
        || dense_rhs.value_type_code != request.dense_storage_type_code
        || dense_rhs.layout != request.dense_rhs_layout
        || dense_rhs.leading_dimension != request.dense_rhs_leading_dimension) {
        return fail(referee_status_code::invalid_shape,
            "logical dense RHS disagrees with the SpMM request");
    }

    double alpha = 0.0, beta = 0.0;
    if (!decode_scalar(request.alpha, &alpha) || !decode_scalar(request.beta, &beta)) {
        return fail(referee_status_code::unsupported_type,
            "reference scalar type is unsupported");
    }
    if (beta != 0.0
        && (!valid_dense_shape(initial_output, request.m, request.n)
            || initial_output.value_type_code != request.output_storage_type_code
            || initial_output.layout != request.output_layout
            || initial_output.leading_dimension != request.output_leading_dimension)) {
        return fail(referee_status_code::invalid_shape,
            "initial output is required when beta is nonzero");
    }
    if (bias_epilogue(request.epilogue.kind) && bias == nullptr) {
        return fail(referee_status_code::invalid_argument,
            "bias epilogue requires bias values");
    }

    for (u64 row = 0u; row < request.m; ++row) {
        for (u64 column = 0u; column < request.n; ++column) {
            double initial = 0.0;
            if (beta != 0.0
                && !read_value(initial_output.values,
                    initial_output.value_type_code,
                    dense_index(initial_output, row, column), &initial)) {
                return fail(referee_status_code::unsupported_type,
                    "initial output type is unsupported");
            }
            reference[row * request.n + column] = beta * initial;
        }
    }

    for (u64 physical_row = 0u; physical_row < sparse_rows; ++physical_row) {
        const u64 begin = sparse.row_offsets[physical_row];
        const u64 end = sparse.row_offsets[physical_row + 1u];
        if (begin > end || end > sparse.nnz) {
            return fail(referee_status_code::invalid_shape,
                "logical CSR row offsets are not monotonic");
        }
        for (u64 entry = begin; entry < end; ++entry) {
            const u64 physical_column = sparse.column_indices[entry];
            if (physical_column >= sparse_columns) {
                return fail(referee_status_code::invalid_shape,
                    "logical CSR column index is out of bounds");
            }
            const u64 output_row = request.transpose_sparse == transpose_kind::none
                ? physical_row : physical_column;
            const u64 reduction = request.transpose_sparse == transpose_kind::none
                ? physical_column : physical_row;
            double sparse_value = 0.0;
            if (!read_value(sparse.values, sparse.value_type_code,
                    entry, &sparse_value)) {
                return fail(referee_status_code::unsupported_type,
                    "sparse reference value type is unsupported");
            }
            for (u64 output_column = 0u;
                 output_column < request.n; ++output_column) {
                const u64 rhs_row = request.transpose_dense == transpose_kind::none
                    ? reduction : output_column;
                const u64 rhs_column = request.transpose_dense == transpose_kind::none
                    ? output_column : reduction;
                double dense_value = 0.0;
                if (!read_value(dense_rhs.values, dense_rhs.value_type_code,
                        dense_index(dense_rhs, rhs_row, rhs_column), &dense_value)) {
                    return fail(referee_status_code::unsupported_type,
                        "dense reference value type is unsupported");
                }
                reference[output_row * request.n + output_column]
                    += alpha * sparse_value * dense_value;
            }
        }
    }

    for (u64 row = 0u; row < request.m; ++row) {
        for (u64 column = 0u; column < request.n; ++column) {
            double bias_value = 0.0;
            if (bias_epilogue(request.epilogue.kind)
                && !read_value(bias, request.epilogue.bias_type_code,
                    column, &bias_value)) {
                return fail(referee_status_code::unsupported_type,
                    "bias reference value type is unsupported");
            }
            double &value = reference[row * request.n + column];
            value = apply_epilogue(value, bias_value, request.epilogue.kind);
            if (!std::isfinite(value)) {
                return fail(referee_status_code::non_finite_reference,
                    "reference result contains a non-finite value");
            }
        }
    }
    return {};
}

} // namespace cellerator::compute::math
