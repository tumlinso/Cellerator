#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cellerator::ce_geo::validation {

enum class numerical_status : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    non_finite_reference = 2u,
    non_finite_candidate = 3u
};

enum class operand_precision : std::uint8_t {
    fp32 = 1u,
    fp64 = 2u
};

enum class referee_precision : std::uint8_t {
    operand = 1u,
    fp32 = 2u,
    fp64 = 3u
};

struct logical_spmm_problem {
    std::uint32_t rows = 0u;
    std::uint32_t reduction = 0u;
    std::uint32_t columns = 0u;
    std::uint64_t nnz = 0u;
    const std::uint64_t *row_offsets = nullptr;
    const std::uint32_t *column_indices = nullptr;
    const double *sparse_values = nullptr;
    const double *dense_values = nullptr;
    const double *initial_output = nullptr;
    double alpha = 1.0;
    double beta = 0.0;
    operand_precision operands = operand_precision::fp32;
};

struct tolerance_policy {
    double absolute = 0.0;
    double relative = 0.0;
    double relative_floor = 1.0e-30;
    double maximum_relative_l2 = 0.0;
    bool reject_non_finite = true;
};

struct numerical_report {
    std::uint64_t element_count = 0u;
    std::uint64_t mixed_tolerance_mismatches = 0u;
    std::uint64_t non_finite_reference_count = 0u;
    std::uint64_t non_finite_candidate_count = 0u;
    std::uint64_t worst_index = 0u;
    std::uint32_t worst_degree = 0u;
    std::uint32_t worst_depth = 0u;
    double alpha = 0.0;
    double beta = 0.0;
    double max_absolute_error = 0.0;
    double relative_l2_error = 0.0;
    double relative_frobenius_error = 0.0;
    double degree_normalized_max_error = 0.0;
    double depth_normalized_max_error = 0.0;
    double absolute_error_p95 = 0.0;
    double absolute_error_p99 = 0.0;
    bool within_tolerance = false;
};

inline double operand_round(double value, operand_precision precision) noexcept {
    return precision == operand_precision::fp32
        ? static_cast<double>(static_cast<float>(value))
        : value;
}

inline bool valid_problem(const logical_spmm_problem &problem) noexcept {
    if (problem.rows == 0u || problem.reduction == 0u
        || problem.columns == 0u || problem.row_offsets == nullptr
        || (problem.nnz != 0u && (problem.column_indices == nullptr
            || problem.sparse_values == nullptr))
        || problem.dense_values == nullptr
        || (problem.beta != 0.0 && problem.initial_output == nullptr)
        || (problem.operands != operand_precision::fp32
            && problem.operands != operand_precision::fp64))
        return false;
    if (problem.row_offsets[0] != 0u
        || problem.row_offsets[problem.rows] != problem.nnz)
        return false;
    for (std::uint32_t row = 0u; row < problem.rows; ++row) {
        if (problem.row_offsets[row] > problem.row_offsets[row + 1u]
            || problem.row_offsets[row + 1u] > problem.nnz)
            return false;
    }
    for (std::uint64_t edge = 0u; edge < problem.nnz; ++edge)
        if (problem.column_indices[edge] >= problem.reduction) return false;
    return true;
}

inline numerical_status build_logical_spmm_reference(
    const logical_spmm_problem &problem,
    referee_precision precision,
    double *output,
    std::uint64_t output_capacity) noexcept {
    const std::uint64_t output_count =
        static_cast<std::uint64_t>(problem.rows) * problem.columns;
    if (!valid_problem(problem) || output == nullptr
        || output_capacity < output_count
        || (precision != referee_precision::operand
            && precision != referee_precision::fp32
            && precision != referee_precision::fp64))
        return numerical_status::invalid_argument;

    const referee_precision effective = precision == referee_precision::operand
        ? (problem.operands == operand_precision::fp32
            ? referee_precision::fp32 : referee_precision::fp64)
        : precision;
    const double alpha = operand_round(problem.alpha, problem.operands);
    const double beta = operand_round(problem.beta, problem.operands);
    for (std::uint32_t row = 0u; row < problem.rows; ++row) {
        for (std::uint32_t column = 0u; column < problem.columns; ++column) {
            const std::uint64_t output_index =
                static_cast<std::uint64_t>(row) * problem.columns + column;
            const double initial = problem.initial_output == nullptr
                ? 0.0
                : operand_round(problem.initial_output[output_index],
                    problem.operands);
            if (effective == referee_precision::fp32) {
                float sum = static_cast<float>(beta)
                    * static_cast<float>(initial);
                for (std::uint64_t edge = problem.row_offsets[row];
                     edge < problem.row_offsets[row + 1u]; ++edge) {
                    const std::uint32_t reduction =
                        problem.column_indices[edge];
                    const float sparse = static_cast<float>(operand_round(
                        problem.sparse_values[edge], problem.operands));
                    const float dense = static_cast<float>(operand_round(
                        problem.dense_values[static_cast<std::uint64_t>(
                            reduction) * problem.columns + column],
                        problem.operands));
                    sum += static_cast<float>(alpha) * sparse * dense;
                }
                output[output_index] = sum;
            } else {
                double sum = beta * initial;
                for (std::uint64_t edge = problem.row_offsets[row];
                     edge < problem.row_offsets[row + 1u]; ++edge) {
                    const std::uint32_t reduction =
                        problem.column_indices[edge];
                    sum += alpha * operand_round(problem.sparse_values[edge],
                        problem.operands) * operand_round(
                        problem.dense_values[static_cast<std::uint64_t>(
                            reduction) * problem.columns + column],
                        problem.operands);
                }
                output[output_index] = sum;
            }
            if (!std::isfinite(output[output_index]))
                return numerical_status::non_finite_reference;
        }
    }
    return numerical_status::success;
}

inline double nearest_rank_quantile(
    const std::vector<double> &sorted, double quantile) noexcept {
    if (sorted.empty()) return 0.0;
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(sorted.size())));
    return sorted[std::max<std::size_t>(1u, rank) - 1u];
}

inline numerical_status compare_numerical_results(
    const double *reference,
    const double *candidate,
    std::uint64_t count,
    const std::uint32_t *degrees,
    const std::uint32_t *depths,
    double alpha,
    double beta,
    const tolerance_policy &tolerance,
    numerical_report *report) {
    if (report == nullptr || (count != 0u
            && (reference == nullptr || candidate == nullptr
                || degrees == nullptr || depths == nullptr))
        || tolerance.absolute < 0.0 || tolerance.relative < 0.0
        || tolerance.relative_floor <= 0.0
        || tolerance.maximum_relative_l2 < 0.0)
        return numerical_status::invalid_argument;

    numerical_report result{};
    result.element_count = count;
    result.alpha = alpha;
    result.beta = beta;
    std::vector<double> absolute_errors;
    absolute_errors.reserve(static_cast<std::size_t>(count));
    long double error_squared = 0.0L;
    long double reference_squared = 0.0L;
    for (std::uint64_t index = 0u; index < count; ++index) {
        const bool finite_reference = std::isfinite(reference[index]);
        const bool finite_candidate = std::isfinite(candidate[index]);
        if (!finite_reference)
            ++result.non_finite_reference_count;
        if (!finite_candidate)
            ++result.non_finite_candidate_count;
        if (!finite_reference || !finite_candidate) continue;
        const double absolute = std::fabs(candidate[index] - reference[index]);
        absolute_errors.push_back(absolute);
        error_squared += static_cast<long double>(absolute) * absolute;
        reference_squared += static_cast<long double>(reference[index])
            * reference[index];
        if (absolute > result.max_absolute_error) {
            result.max_absolute_error = absolute;
            result.worst_index = index;
            result.worst_degree = degrees[index];
            result.worst_depth = depths[index];
        }
        result.degree_normalized_max_error = std::max(
            result.degree_normalized_max_error,
            absolute / std::max(1u, degrees[index]));
        result.depth_normalized_max_error = std::max(
            result.depth_normalized_max_error,
            absolute / std::max(1u, depths[index]));
        if (absolute > tolerance.absolute
                + tolerance.relative * std::fabs(reference[index]))
            ++result.mixed_tolerance_mismatches;
    }
    const long double denominator = std::max(reference_squared,
        static_cast<long double>(tolerance.relative_floor)
            * tolerance.relative_floor);
    result.relative_l2_error = std::sqrt(
        static_cast<double>(error_squared / denominator));
    result.relative_frobenius_error = result.relative_l2_error;
    std::sort(absolute_errors.begin(), absolute_errors.end());
    result.absolute_error_p95 = nearest_rank_quantile(absolute_errors, 0.95);
    result.absolute_error_p99 = nearest_rank_quantile(absolute_errors, 0.99);
    const bool finite = result.non_finite_reference_count == 0u
        && result.non_finite_candidate_count == 0u;
    result.within_tolerance = result.mixed_tolerance_mismatches == 0u
        && result.relative_l2_error <= tolerance.maximum_relative_l2
        && (finite || !tolerance.reject_non_finite);
    *report = result;
    if (tolerance.reject_non_finite) {
        if (result.non_finite_reference_count != 0u)
            return numerical_status::non_finite_reference;
        if (result.non_finite_candidate_count != 0u)
            return numerical_status::non_finite_candidate;
    }
    return numerical_status::success;
}

} // namespace cellerator::ce_geo::validation
