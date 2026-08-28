#pragma once

#include "../benchmark_mutex.hh"

#include <Cellerator/compat/cp_math_v1/referee.hh>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>

namespace cellerator::compute::math::bench {

struct timing_summary {
    u64 sample_count = 0u;
    double minimum_ms = 0.0;
    double p05_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double maximum_ms = 0.0;
    double mean_ms = 0.0;
    double standard_deviation_ms = 0.0;
    double median_absolute_deviation_ms = 0.0;
};

struct memory_accounting {
    u64 logical_bytes = 0u;
    u64 storage_bytes = 0u;
    u64 persistent_bytes = 0u;
    u64 workspace_bytes = 0u;
    u64 output_bytes = 0u;
    u64 total_runtime_bytes = 0u;
    double storage_expansion = 0.0;
    double runtime_expansion = 0.0;
};

struct benchmark_report {
    u32 schema_version = referee_schema_version;
    operation_signature operation{};
    u64 backend_identity = 0u;
    u64 algorithm_identity = 0u;
    u64 warmup_iterations = 0u;
    u64 measured_iterations = 0u;
    numerical_comparison numerical{};
    timing_summary timing{};
    memory_accounting memory{};
    determinism_digest digest{};
    bool deterministic = false;
};

inline referee_status summarize_timing_samples(
    const double *samples,
    u64 sample_count,
    double *scratch,
    u64 scratch_capacity,
    timing_summary *summary) noexcept {
    if (summary == nullptr || sample_count == 0u || samples == nullptr
        || scratch == nullptr || scratch_capacity < sample_count) {
        return {referee_status_code::invalid_argument,
            "timing summary requires samples and caller-owned scratch"};
    }
    long double sum = 0.0L;
    for (u64 i = 0u; i < sample_count; ++i) {
        if (!std::isfinite(samples[i]) || samples[i] < 0.0) {
            return {referee_status_code::invalid_argument,
                "timing samples must be finite and nonnegative"};
        }
        scratch[i] = samples[i];
        sum += samples[i];
    }
    std::sort(scratch, scratch + sample_count);
    const auto percentile = [&](double fraction) {
        const double position = fraction * static_cast<double>(sample_count - 1u);
        const u64 lower = static_cast<u64>(position);
        const u64 upper = lower + 1u < sample_count ? lower + 1u : lower;
        const double weight = position - static_cast<double>(lower);
        return scratch[lower] * (1.0 - weight) + scratch[upper] * weight;
    };
    timing_summary out;
    out.sample_count = sample_count;
    out.minimum_ms = scratch[0];
    out.p05_ms = percentile(0.05);
    out.median_ms = percentile(0.5);
    out.p95_ms = percentile(0.95);
    out.maximum_ms = scratch[sample_count - 1u];
    out.mean_ms = static_cast<double>(sum / sample_count);
    long double squared = 0.0L;
    for (u64 i = 0u; i < sample_count; ++i) {
        const long double delta = samples[i] - out.mean_ms;
        squared += delta * delta;
        scratch[i] = std::fabs(samples[i] - out.median_ms);
    }
    out.standard_deviation_ms =
        std::sqrt(static_cast<double>(squared / sample_count));
    std::sort(scratch, scratch + sample_count);
    out.median_absolute_deviation_ms = percentile(0.5);
    *summary = out;
    return {};
}

inline referee_status account_benchmark_memory(
    u64 logical_bytes,
    u64 storage_bytes,
    u64 persistent_bytes,
    u64 workspace_bytes,
    u64 output_bytes,
    memory_accounting *accounting) noexcept {
    if (accounting == nullptr) {
        return {referee_status_code::invalid_argument,
            "memory accounting requires an output"};
    }
    u64 total = storage_bytes;
    const auto add = [&](u64 value) {
        if (value > std::numeric_limits<u64>::max() - total) return false;
        total += value;
        return true;
    };
    if (!add(persistent_bytes) || !add(workspace_bytes) || !add(output_bytes)) {
        return {referee_status_code::overflow,
            "runtime memory total overflows"};
    }
    memory_accounting out;
    out.logical_bytes = logical_bytes;
    out.storage_bytes = storage_bytes;
    out.persistent_bytes = persistent_bytes;
    out.workspace_bytes = workspace_bytes;
    out.output_bytes = output_bytes;
    out.total_runtime_bytes = total;
    if (logical_bytes != 0u) {
        out.storage_expansion = static_cast<double>(storage_bytes) / logical_bytes;
        out.runtime_expansion = static_cast<double>(total) / logical_bytes;
    }
    *accounting = out;
    return {};
}

class cuda_event_timer {
public:
    cuda_event_timer() noexcept = default;
    ~cuda_event_timer() { clear(); }
    cuda_event_timer(const cuda_event_timer &) = delete;
    cuda_event_timer &operator=(const cuda_event_timer &) = delete;

    referee_status init() noexcept {
        clear();
        cudaError_t error = cudaEventCreate(&start_);
        if (error != cudaSuccess) return cuda_error(error, "create start event failed");
        error = cudaEventCreate(&stop_);
        if (error != cudaSuccess) {
            clear();
            return cuda_error(error, "create stop event failed");
        }
        return {};
    }

    void clear() noexcept {
        if (stop_ != nullptr) (void) cudaEventDestroy(stop_);
        if (start_ != nullptr) (void) cudaEventDestroy(start_);
        start_ = nullptr;
        stop_ = nullptr;
    }

    referee_status begin(cudaStream_t stream = nullptr) noexcept {
        if (start_ == nullptr || stop_ == nullptr) {
            return {referee_status_code::invalid_argument,
                "CUDA event timer is not initialized"};
        }
        const cudaError_t error = cudaEventRecord(start_, stream);
        return error == cudaSuccess
            ? referee_status{}
            : cuda_error(error, "record start event failed");
    }

    referee_status end(
        double *elapsed_ms,
        cudaStream_t stream = nullptr) noexcept {
        if (elapsed_ms == nullptr || start_ == nullptr || stop_ == nullptr) {
            return {referee_status_code::invalid_argument,
                "CUDA event timer output is missing"};
        }
        cudaError_t error = cudaEventRecord(stop_, stream);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop_);
        float milliseconds = 0.0f;
        if (error == cudaSuccess) {
            error = cudaEventElapsedTime(&milliseconds, start_, stop_);
        }
        if (error != cudaSuccess) return cuda_error(error, "CUDA event timing failed");
        *elapsed_ms = milliseconds;
        return {};
    }

private:
    static referee_status cuda_error(cudaError_t, const char *message) noexcept {
        return {referee_status_code::cuda_failure, message};
    }

    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// Stable JSON-lines output for automation. Callers keep correctness and timing
// phases separate and acquire benchmark_mutex_guard before measured GPU work.
inline referee_status write_benchmark_report_json(
    std::FILE *file,
    const char *benchmark_name,
    const benchmark_report &report) noexcept {
    if (file == nullptr || benchmark_name == nullptr) {
        return {referee_status_code::invalid_argument,
            "benchmark report requires file and name"};
    }
    const int written = std::fprintf(file,
        "{\"schema_version\":%u,\"benchmark\":\"%s\","
        "\"operation_low\":%llu,\"operation_high\":%llu,"
        "\"backend_identity\":%llu,\"algorithm_identity\":%llu,"
        "\"warmup_iterations\":%llu,\"measured_iterations\":%llu,"
        "\"within_tolerance\":%s,\"mismatch_count\":%llu,"
        "\"max_absolute_error\":%.17g,\"max_relative_error\":%.17g,"
        "\"median_ms\":%.17g,\"p05_ms\":%.17g,\"p95_ms\":%.17g,"
        "\"mad_ms\":%.17g,\"logical_bytes\":%llu,"
        "\"storage_bytes\":%llu,\"workspace_bytes\":%llu,"
        "\"total_runtime_bytes\":%llu,\"storage_expansion\":%.17g,"
        "\"runtime_expansion\":%.17g,\"deterministic\":%s,"
        "\"digest_low\":%llu,\"digest_high\":%llu}\n",
        report.schema_version,
        benchmark_name,
        static_cast<unsigned long long>(report.operation.low),
        static_cast<unsigned long long>(report.operation.high),
        static_cast<unsigned long long>(report.backend_identity),
        static_cast<unsigned long long>(report.algorithm_identity),
        static_cast<unsigned long long>(report.warmup_iterations),
        static_cast<unsigned long long>(report.measured_iterations),
        report.numerical.within_tolerance ? "true" : "false",
        static_cast<unsigned long long>(report.numerical.mismatch_count),
        report.numerical.max_absolute_error,
        report.numerical.max_relative_error,
        report.timing.median_ms,
        report.timing.p05_ms,
        report.timing.p95_ms,
        report.timing.median_absolute_deviation_ms,
        static_cast<unsigned long long>(report.memory.logical_bytes),
        static_cast<unsigned long long>(report.memory.storage_bytes),
        static_cast<unsigned long long>(report.memory.workspace_bytes),
        static_cast<unsigned long long>(report.memory.total_runtime_bytes),
        report.memory.storage_expansion,
        report.memory.runtime_expansion,
        report.deterministic ? "true" : "false",
        static_cast<unsigned long long>(report.digest.low),
        static_cast<unsigned long long>(report.digest.high));
    return written < 0
        ? referee_status{referee_status_code::io_failure,
            "writing benchmark report failed"}
        : referee_status{};
}

static_assert(!std::is_copy_constructible<cuda_event_timer>::value,
    "CUDA events must remain unique-owner resources");
static_assert(std::is_trivially_copyable<benchmark_report>::value,
    "benchmark report must remain report-serializable");

} // namespace cellerator::compute::math::bench
