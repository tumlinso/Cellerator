#pragma once

#include <Cellerator/compat/cp_math_v1/operation.hh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace cellerator::compute::math {

inline constexpr u32 referee_schema_version = 1u;

enum class referee_status_code : u32 {
    ok = 0u,
    invalid_argument = 1u,
    invalid_shape = 2u,
    unsupported_type = 3u,
    insufficient_capacity = 4u,
    non_finite_reference = 5u,
    non_finite_candidate = 6u,
    overflow = 7u,
    cuda_failure = 8u,
    io_failure = 9u
};

struct referee_status {
    referee_status_code code = referee_status_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == referee_status_code::ok;
    }
};

// Host-resident logical views used only by correctness reference work. They
// borrow storage, allocate nothing, and never participate in backend planning.
struct logical_csr_view {
    u64 rows = 0u;
    u64 columns = 0u;
    u64 nnz = 0u;
    const u64 *row_offsets = nullptr;
    const u32 *column_indices = nullptr;
    const void *values = nullptr;
    u32 value_type_code = 0u;
};

struct logical_dense_view {
    const void *values = nullptr;
    u64 rows = 0u;
    u64 columns = 0u;
    u64 leading_dimension = 0u;
    dense_layout_kind layout = dense_layout_kind::row_major;
    u32 value_type_code = 0u;
};

struct numerical_tolerance {
    double absolute = 0.0;
    double relative = 0.0;
    double relative_floor = 1.0e-30;
};

struct numerical_comparison {
    u64 element_count = 0u;
    u64 mismatch_count = 0u;
    u64 non_finite_reference_count = 0u;
    u64 non_finite_candidate_count = 0u;
    u64 worst_index = 0u;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double mean_absolute_error = 0.0;
    double root_mean_square_error = 0.0;
    bool within_tolerance = true;
};

struct determinism_digest {
    u64 low = 0u;
    u64 high = 0u;
};

namespace referee_detail {

inline u64 dense_offset(
    const logical_dense_view &view,
    u64 row,
    u64 column) noexcept {
    return view.layout == dense_layout_kind::row_major
        ? row * view.leading_dimension + column
        : column * view.leading_dimension + row;
}

inline bool valid_dense(const logical_dense_view &view) noexcept {
    if (view.rows == 0u || view.columns == 0u) return true;
    const u64 minimum = view.layout == dense_layout_kind::row_major
        ? view.columns : view.rows;
    return view.values != nullptr && view.leading_dimension >= minimum
        && (view.layout == dense_layout_kind::row_major
            || view.layout == dense_layout_kind::column_major);
}

inline bool read_dense(
    const logical_dense_view &view,
    u64 row,
    u64 column,
    double *out) noexcept {
    if (out == nullptr || !valid_dense(view)) return false;
    const u64 index = dense_offset(view, row, column);
    if (view.value_type_code == static_cast<u32>(real::value_f16)) {
        *out = __half2float(static_cast<const __half *>(view.values)[index]);
        return true;
    }
    if (view.value_type_code == static_cast<u32>(real::value_f32)) {
        *out = static_cast<const float *>(view.values)[index];
        return true;
    }
    if (view.value_type_code == static_cast<u32>(real::value_f64)) {
        *out = static_cast<const double *>(view.values)[index];
        return true;
    }
    return false;
}

inline void digest_mix(u64 value, determinism_digest *digest) noexcept {
    digest->low ^= value;
    digest->low *= 1099511628211ull;
    digest->high ^= value + 0x9e3779b97f4a7c15ull
        + (digest->high << 6u) + (digest->high >> 2u);
}

} // namespace referee_detail

referee_status build_spmm_reference(
    const spmm_request &request,
    const logical_csr_view &sparse,
    const logical_dense_view &dense_rhs,
    const logical_dense_view &initial_output,
    const void *bias,
    double *reference_row_major,
    u64 reference_capacity) noexcept;

inline referee_status compare_spmm_reference(
    const double *reference_row_major,
    u64 reference_count,
    const logical_dense_view &candidate,
    const numerical_tolerance &tolerance,
    numerical_comparison *comparison) noexcept {
    if (candidate.rows != 0u
        && candidate.columns > std::numeric_limits<u64>::max() / candidate.rows) {
        return {referee_status_code::overflow,
            "candidate element count overflows"};
    }
    if (comparison == nullptr || (reference_count != 0u
            && reference_row_major == nullptr)
        || tolerance.absolute < 0.0 || tolerance.relative < 0.0
        || tolerance.relative_floor <= 0.0
        || candidate.rows * candidate.columns != reference_count
        || !referee_detail::valid_dense(candidate)) {
        return {referee_status_code::invalid_argument,
            "invalid numerical comparison arguments"};
    }
    numerical_comparison out;
    out.element_count = reference_count;
    long double absolute_sum = 0.0L, squared_sum = 0.0L;
    for (u64 row = 0u; row < candidate.rows; ++row) {
        for (u64 column = 0u; column < candidate.columns; ++column) {
            const u64 logical = row * candidate.columns + column;
            double actual = 0.0;
            if (!referee_detail::read_dense(candidate, row, column, &actual)) {
                return {referee_status_code::unsupported_type,
                    "candidate value type is unsupported"};
            }
            const double expected = reference_row_major[logical];
            if (!std::isfinite(expected)) {
                ++out.non_finite_reference_count;
                out.within_tolerance = false;
                continue;
            }
            if (!std::isfinite(actual)) {
                ++out.non_finite_candidate_count;
                out.within_tolerance = false;
                continue;
            }
            const double absolute = std::fabs(actual - expected);
            const double relative = absolute
                / std::fmax(std::fabs(expected), tolerance.relative_floor);
            absolute_sum += absolute;
            squared_sum += static_cast<long double>(absolute) * absolute;
            if (absolute > out.max_absolute_error) {
                out.max_absolute_error = absolute;
                out.worst_index = logical;
            }
            out.max_relative_error = std::fmax(out.max_relative_error, relative);
            if (absolute > tolerance.absolute
                    + tolerance.relative * std::fabs(expected)) {
                ++out.mismatch_count;
                out.within_tolerance = false;
            }
        }
    }
    if (reference_count != 0u) {
        out.mean_absolute_error = static_cast<double>(absolute_sum / reference_count);
        out.root_mean_square_error =
            std::sqrt(static_cast<double>(squared_sum / reference_count));
    }
    *comparison = out;
    if (out.non_finite_reference_count != 0u) {
        return {referee_status_code::non_finite_reference,
            "reference contains non-finite values"};
    }
    if (out.non_finite_candidate_count != 0u) {
        return {referee_status_code::non_finite_candidate,
            "candidate contains non-finite values"};
    }
    return {};
}

inline referee_status digest_logical_dense(
    const logical_dense_view &view,
    determinism_digest *digest) noexcept {
    if (digest == nullptr || !referee_detail::valid_dense(view)) {
        return {referee_status_code::invalid_argument,
            "invalid determinism digest view"};
    }
    determinism_digest out{1469598103934665603ull, 0x6a09e667f3bcc909ull};
    referee_detail::digest_mix(view.rows, &out);
    referee_detail::digest_mix(view.columns, &out);
    referee_detail::digest_mix(view.value_type_code, &out);
    for (u64 row = 0u; row < view.rows; ++row) {
        for (u64 column = 0u; column < view.columns; ++column) {
            double value = 0.0;
            if (!referee_detail::read_dense(view, row, column, &value)) {
                return {referee_status_code::unsupported_type,
                    "digest value type is unsupported"};
            }
            u64 bits = 0u;
            std::memcpy(&bits, &value, sizeof(bits));
            referee_detail::digest_mix(bits, &out);
        }
    }
    *digest = out;
    return {};
}

constexpr bool same_digest(
    const determinism_digest &lhs,
    const determinism_digest &rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

static_assert(std::is_trivially_copyable<numerical_comparison>::value,
    "numerical comparison must remain report-serializable");

} // namespace cellerator::compute::math
