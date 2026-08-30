#include "numerical_referee.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace referee = cellerator::ce_geo::validation;

namespace {

void require_near(double actual, double expected, double tolerance) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void test_reference_precisions_and_alpha_beta() {
    const std::uint64_t row_offsets[]{0u, 3u, 5u};
    const std::uint32_t columns[]{0u, 1u, 2u, 0u, 2u};
    const double sparse[]{1.0 / 3.0, -2.0, 4.0, 0.25, -0.75};
    const double dense[]{1.0, -2.0, 0.5, 3.0, -4.0, 0.125};
    const double initial[]{2.0, -1.0, 0.5, 4.0};
    referee::logical_spmm_problem problem{};
    problem.rows = 2u;
    problem.reduction = 3u;
    problem.columns = 2u;
    problem.nnz = 5u;
    problem.row_offsets = row_offsets;
    problem.column_indices = columns;
    problem.sparse_values = sparse;
    problem.dense_values = dense;
    problem.initial_output = initial;
    problem.alpha = 1.25;
    problem.beta = -0.5;

    double fp32[4]{};
    double fp64[4]{};
    double operand[4]{};
    problem.operands = referee::operand_precision::fp32;
    assert(referee::build_logical_spmm_reference(problem,
               referee::referee_precision::fp32, fp32, 4u)
        == referee::numerical_status::success);
    assert(referee::build_logical_spmm_reference(problem,
               referee::referee_precision::operand, operand, 4u)
        == referee::numerical_status::success);
    for (std::uint32_t index = 0u; index < 4u; ++index)
        assert(fp32[index] == operand[index]);

    problem.operands = referee::operand_precision::fp64;
    assert(referee::build_logical_spmm_reference(problem,
               referee::referee_precision::fp64, fp64, 4u)
        == referee::numerical_status::success);
    assert(referee::build_logical_spmm_reference(problem,
               referee::referee_precision::operand, operand, 4u)
        == referee::numerical_status::success);
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        long double expected = -0.5L * initial[index];
        const std::uint32_t row = index / 2u;
        const std::uint32_t output_column = index % 2u;
        for (std::uint64_t edge = row_offsets[row];
             edge < row_offsets[row + 1u]; ++edge)
            expected += 1.25L * sparse[edge]
                * dense[columns[edge] * 2u + output_column];
        require_near(fp64[index], static_cast<double>(expected), 1.0e-15);
        assert(fp64[index] == operand[index]);
    }
    assert(std::any_of(fp32, fp32 + 4u, [fp64, index = std::size_t{0u}]
        (double value) mutable { return value != fp64[index++]; }));
}

void test_metrics_tails_and_finite_policy() {
    const double reference[]{0.0, 10.0, -10.0, 100.0};
    const double candidate[]{0.001, 10.1, -10.2, 101.0};
    const std::uint32_t degrees[]{1u, 2u, 4u, 8u};
    const std::uint32_t depths[]{2u, 4u, 5u, 10u};
    const referee::tolerance_policy tolerance{0.01, 0.01, 1.0e-30, 0.02, true};
    referee::numerical_report report{};
    assert(referee::compare_numerical_results(reference, candidate, 4u,
               degrees, depths, 1.25, -0.5, tolerance, &report)
        == referee::numerical_status::success);
    assert(report.element_count == 4u);
    assert(report.mixed_tolerance_mismatches == 1u);
    assert(!report.within_tolerance);
    assert(report.alpha == 1.25 && report.beta == -0.5);
    require_near(report.max_absolute_error, 1.0, 1.0e-15);
    assert(report.worst_index == 3u && report.worst_degree == 8u
        && report.worst_depth == 10u);
    require_near(report.degree_normalized_max_error, 0.125, 1.0e-15);
    require_near(report.depth_normalized_max_error, 0.1, 1.0e-15);
    require_near(report.absolute_error_p95, 1.0, 1.0e-15);
    require_near(report.absolute_error_p99, 1.0, 1.0e-15);
    require_near(report.relative_l2_error,
        std::sqrt((0.001 * 0.001 + 0.1 * 0.1 + 0.2 * 0.2 + 1.0)
            / (0.0 + 100.0 + 100.0 + 10000.0)), 1.0e-15);
    assert(report.relative_l2_error == report.relative_frobenius_error);

    double non_finite_candidate[]{0.0,
        std::numeric_limits<double>::quiet_NaN()};
    const double finite_reference[]{0.0, 1.0};
    const std::uint32_t metadata[]{1u, 1u};
    assert(referee::compare_numerical_results(finite_reference,
               non_finite_candidate, 2u, metadata, metadata, 1.0, 0.0,
               tolerance, &report)
        == referee::numerical_status::non_finite_candidate);
    assert(report.non_finite_candidate_count == 1u
        && !report.within_tolerance);

    const double non_finite_reference[]{
        std::numeric_limits<double>::infinity(), 1.0};
    assert(referee::compare_numerical_results(non_finite_reference,
               finite_reference, 2u, metadata, metadata, 1.0, 0.0,
               tolerance, &report)
        == referee::numerical_status::non_finite_reference);
    assert(report.non_finite_reference_count == 1u
        && !report.within_tolerance);

    const double both_non_finite_reference[]{
        std::numeric_limits<double>::infinity()};
    const double both_non_finite_candidate[]{
        std::numeric_limits<double>::quiet_NaN()};
    assert(referee::compare_numerical_results(both_non_finite_reference,
               both_non_finite_candidate, 1u, metadata, metadata, 1.0, 0.0,
               tolerance, &report)
        == referee::numerical_status::non_finite_reference);
    assert(report.non_finite_reference_count == 1u
        && report.non_finite_candidate_count == 1u);
}

void test_randomized_mixed_tolerance() {
    constexpr std::uint32_t count = 1024u;
    std::mt19937 random(0x104u);
    std::uniform_real_distribution<double> values(-100.0, 100.0);
    std::vector<double> reference(count);
    std::vector<double> candidate(count);
    std::vector<std::uint32_t> degrees(count);
    std::vector<std::uint32_t> depths(count);
    for (std::uint32_t index = 0u; index < count; ++index) {
        reference[index] = values(random);
        const double bound = 1.0e-6 + 1.0e-5 * std::fabs(reference[index]);
        candidate[index] = reference[index] + (index % 2u == 0u
            ? 0.5 * bound : -0.5 * bound);
        degrees[index] = 1u + random() % 4096u;
        depths[index] = 1u + random() % 64u;
    }
    const referee::tolerance_policy tolerance{
        1.0e-6, 1.0e-5, 1.0e-30, 1.0e-5, true};
    referee::numerical_report report{};
    assert(referee::compare_numerical_results(reference.data(),
               candidate.data(), count, degrees.data(), depths.data(),
               0.75, 0.25, tolerance, &report)
        == referee::numerical_status::success);
    assert(report.within_tolerance);
    assert(report.mixed_tolerance_mismatches == 0u);

    candidate[count / 2u] = reference[count / 2u] + 4.0;
    assert(referee::compare_numerical_results(reference.data(),
               candidate.data(), count, degrees.data(), depths.data(),
               0.75, 0.25, tolerance, &report)
        == referee::numerical_status::success);
    assert(!report.within_tolerance
        && report.mixed_tolerance_mismatches == 1u);
}

} // namespace

int main() {
    test_reference_precisions_and_alpha_beta();
    test_metrics_tails_and_finite_policy();
    test_randomized_mixed_tolerance();
    std::cout << "numerical_referee_test passed modes=operand,fp32,fp64"
              << " metrics=abs,l2,frobenius,mixed,degree,depth,p95,p99"
              << " finite_policy=reject alpha_beta=reported\n";
    return 0;
}
