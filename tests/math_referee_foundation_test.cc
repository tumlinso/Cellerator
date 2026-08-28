#include <Cellerator/compat/cp_math_v1/referee.hh>
#include <Cellerator/types.cuh>

#include <bench/math/benchmark_support.hh>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

namespace cm = cellerator::compute::math;
namespace cmb = cellerator::compute::math::bench;
namespace cr = cellerator::real;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpMathRefereeFoundationTest: " << message << '\n';
        std::exit(1);
    }
}

cm::feature_order_identity canonical_order(cm::u32 features) {
    cm::feature_order_identity order;
    order.feature_count = features;
    order.feature_axis_identity_version = 1u;
    order.feature_axis_identity = 0x08aull;
    return order;
}

cm::spmm_request request_fixture() {
    cm::spmm_request request;
    request.m = 2u;
    request.k = 3u;
    request.n = 2u;
    request.sparse_nnz = 4u;
    request.sparse_structure.identity_version = 1u;
    request.sparse_structure.value = 0x08a08aull;
    request.dense_rhs_leading_dimension = 2u;
    request.output_leading_dimension = 2u;
    request.sparse_storage_type_code = cr::value_f32;
    request.dense_storage_type_code = cr::value_f32;
    request.output_storage_type_code = cr::value_f32;
    request.compute_type_code = cr::value_f32;
    request.accumulation_type_code = cr::value_f32;
    request.alpha = cm::make_scalar(2.0f);
    request.beta = cm::make_scalar(0.5f);
    request.epilogue.kind = cm::epilogue_kind::bias_relu;
    request.epilogue.bias_type_code = cr::value_f32;
    request.epilogue.bias_element_count = 2u;
    request.sparse_feature_order = canonical_order(3u);
    request.dense_feature_order = request.sparse_feature_order;
    return request;
}

void test_logical_reference_and_transpose() {
    const cm::u64 offsets[]{0u, 2u, 4u};
    const cm::u32 columns[]{0u, 2u, 1u, 2u};
    const float sparse_values[]{1.0f, 2.0f, 3.0f, 4.0f};
    const float rhs[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float initial[]{2.0f, 2.0f, 2.0f, 2.0f};
    const float bias[]{-30.0f, 1.0f};
    const cm::logical_csr_view sparse{
        2u, 3u, 4u, offsets, columns, sparse_values, cr::value_f32};
    const cm::logical_dense_view dense{
        rhs, 3u, 2u, 2u, cm::dense_layout_kind::row_major, cr::value_f32};
    const cm::logical_dense_view prior{
        initial, 2u, 2u, 2u, cm::dense_layout_kind::row_major, cr::value_f32};
    double reference[4]{};
    cm::spmm_request request = request_fixture();
    require(static_cast<bool>(cm::build_spmm_reference(
        request, sparse, dense, prior, bias, reference, 4u)),
        "logical SpMM reference failed");
    const double expected[]{0.0, 30.0, 29.0, 74.0};
    for (unsigned i = 0u; i < 4u; ++i) {
        require(std::fabs(reference[i] - expected[i]) < 1.0e-12,
            "logical SpMM reference mismatch");
    }

    const cm::u64 transposed_offsets[]{0u, 1u, 2u, 4u};
    const cm::u32 transposed_columns[]{0u, 1u, 0u, 1u};
    const float transposed_values[]{1.0f, 3.0f, 2.0f, 4.0f};
    const float transposed_rhs[]{1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
    const cm::logical_csr_view sparse_transposed{
        3u, 2u, 4u, transposed_offsets, transposed_columns,
        transposed_values, cr::value_f32};
    const cm::logical_dense_view dense_transposed{
        transposed_rhs, 2u, 3u, 3u,
        cm::dense_layout_kind::row_major, cr::value_f32};
    request.transpose_sparse = cm::transpose_kind::transpose;
    request.transpose_dense = cm::transpose_kind::transpose;
    request.dense_rhs_leading_dimension = 3u;
    double transposed_reference[4]{};
    require(static_cast<bool>(cm::build_spmm_reference(
        request, sparse_transposed, dense_transposed, prior, bias,
        transposed_reference, 4u)), "transposed logical reference failed");
    for (unsigned i = 0u; i < 4u; ++i) {
        require(std::fabs(transposed_reference[i] - expected[i]) < 1.0e-12,
            "transposed logical reference mismatch");
    }
}

void test_numerics_and_determinism() {
    const double reference[]{0.0, 30.0, 29.0, 74.0};
    float candidate[]{0.0f, 30.0f, 29.0f, 74.0f};
    const cm::logical_dense_view view{
        candidate, 2u, 2u, 2u, cm::dense_layout_kind::row_major, cr::value_f32};
    cm::numerical_comparison comparison;
    require(static_cast<bool>(cm::compare_spmm_reference(
        reference, 4u, view, {1.0e-6, 1.0e-6, 1.0e-12}, &comparison)),
        "exact numerical comparison failed");
    require(comparison.within_tolerance && comparison.mismatch_count == 0u,
        "exact candidate did not satisfy tolerance");

    cm::determinism_digest first, second;
    require(static_cast<bool>(cm::digest_logical_dense(view, &first)),
        "first digest failed");
    candidate[2] += 0.25f;
    require(static_cast<bool>(cm::digest_logical_dense(view, &second)),
        "second digest failed");
    require(!cm::same_digest(first, second),
        "determinism digest ignored a logical value change");
    require(static_cast<bool>(cm::compare_spmm_reference(
        reference, 4u, view, {1.0e-6, 1.0e-6, 1.0e-12}, &comparison)),
        "mismatch comparison failed");
    require(!comparison.within_tolerance && comparison.mismatch_count == 1u
            && comparison.worst_index == 2u,
        "numerical mismatch was not localized");

    candidate[1] = std::numeric_limits<float>::infinity();
    const cm::referee_status non_finite = cm::compare_spmm_reference(
        reference, 4u, view, {1.0e-6, 1.0e-6, 1.0e-12}, &comparison);
    require(non_finite.code == cm::referee_status_code::non_finite_candidate
            && comparison.non_finite_candidate_count == 1u,
        "non-finite candidate was not reported structurally");
}

void test_timing_memory_and_report() {
    const double samples[]{1.0, 2.0, 100.0, 3.0, 4.0};
    double scratch[5]{};
    cmb::timing_summary timing;
    require(static_cast<bool>(cmb::summarize_timing_samples(
        samples, 5u, scratch, 5u, &timing)), "timing summary failed");
    require(timing.median_ms == 3.0
            && timing.median_absolute_deviation_ms == 1.0
            && timing.minimum_ms == 1.0 && timing.maximum_ms == 100.0,
        "robust timing statistics are incorrect");

    cmb::memory_accounting memory;
    require(static_cast<bool>(cmb::account_benchmark_memory(
        100u, 150u, 20u, 30u, 40u, &memory)),
        "memory accounting failed");
    require(memory.total_runtime_bytes == 240u
            && std::fabs(memory.storage_expansion - 1.5) < 1.0e-12
            && std::fabs(memory.runtime_expansion - 2.4) < 1.0e-12,
        "memory expansion accounting is incorrect");
    require(cmb::account_benchmark_memory(
        1u, std::numeric_limits<cm::u64>::max(), 1u, 0u, 0u, &memory).code
            == cm::referee_status_code::overflow,
        "memory total overflow was accepted");

    cmb::benchmark_report report;
    report.operation = cm::make_operation_signature(request_fixture());
    report.backend_identity = 7u;
    report.algorithm_identity = 9u;
    report.warmup_iterations = 3u;
    report.measured_iterations = timing.sample_count;
    report.timing = timing;
    report.memory = memory;
    report.deterministic = true;
    std::FILE *file = std::tmpfile();
    require(file != nullptr, "tmpfile failed");
    require(static_cast<bool>(cmb::write_benchmark_report_json(
        file, "foundation", report)), "benchmark report write failed");
    std::rewind(file);
    char buffer[2048]{};
    const std::size_t read = std::fread(buffer, 1u, sizeof(buffer) - 1u, file);
    std::fclose(file);
    require(read != 0u && std::strstr(buffer, "\"schema_version\":1") != nullptr
            && std::strstr(buffer, "\"median_ms\":3") != nullptr
            && std::strstr(buffer, "\"deterministic\":true") != nullptr,
        "benchmark report schema is incomplete");
}

void test_cuda_event_timer() {
    cmb::cuda_event_timer timer;
    require(static_cast<bool>(timer.init()), "CUDA event timer init failed");
    require(static_cast<bool>(timer.begin()), "CUDA event timer begin failed");
    require(cudaMemset(nullptr, 0, 0) == cudaSuccess,
        "zero-byte CUDA operation failed");
    double elapsed_ms = -1.0;
    require(static_cast<bool>(timer.end(&elapsed_ms)), "CUDA event timer end failed");
    require(elapsed_ms >= 0.0 && std::isfinite(elapsed_ms),
        "CUDA event elapsed time is invalid");
}

} // namespace

int main() {
    test_logical_reference_and_transpose();
    test_numerics_and_determinism();
    test_timing_memory_and_report();
    test_cuda_event_timer();
    std::cout << "cpMathRefereeFoundationTest passed\n";
    return 0;
}
