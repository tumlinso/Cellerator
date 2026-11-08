#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "sensitivity_bridge.hh"
#include "../detail/sensitivity.hh"

namespace velocity_cuda_bridge {
namespace {
namespace s = velocity::detail::scientific;
namespace q = velocity::detail::sensitivity;

__device__ bool present(DenseView v) { return v.values.data != nullptr; }
__device__ std::uint8_t load_flag(DenseView v, std::uint64_t i) {
    auto* p = static_cast<const std::uint8_t*>(v.values.data) + v.origin;
    return p[std::int64_t(i)*v.row_stride];
}
template<class T> __device__ void store(DenseView v, std::uint64_t i, T value) {
    if (!present(v)) return;
    auto* p = reinterpret_cast<T*>(static_cast<std::uint8_t*>(v.values.data) + v.origin);
    p[std::int64_t(i)*v.row_stride] = value;
}

__global__ void mask_rows(std::uint64_t rows, DenseView input, std::uint8_t* output) {
    auto i = std::uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < rows) output[i] = load_flag(input, i);
}

template<class T> __global__ void report_genes(std::uint64_t rows, std::uint64_t first,
    std::uint64_t count, const std::uint64_t* genes, DenseView gene_mask, Fields fields,
    Result results, Options opts, SensitivityOptions sensitivity, SensitivityOutput out,
    const std::uint8_t* mask) {
    auto j = std::uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (j >= count) return;
    auto k = first + j;
    auto* x = rows ? static_cast<const T*>(fields.ms) + j*rows : nullptr;
    auto* y = rows ? static_cast<const T*>(fields.mu) + j*rows : nullptr;
    auto* scratch = rows ? static_cast<T*>(fields.scores) + j*rows : nullptr;
    s::Fit<T> base{static_cast<const T*>(results.initial_gamma)[k],
        static_cast<const T*>(results.initial_offset)[k], static_cast<const T*>(results.r2)[k],
        s::FitStatus(results.initial_status[k]), bool(results.selected[k])};
    s::Options<T> fit{s::Trimming(opts.trimming), T(opts.percentile_low), T(opts.percentile_high),
        bool(opts.fit_intercept), bool(opts.positive_intercept), bool(opts.r2_adjusted),
        T(opts.ratio_min), T(opts.ratio_max), T(opts.min_r2), T(opts.min_ratio)};
    q::Options<T> diagnostic{T(sensitivity.band), T(sensitivity.prediction_limit), T(sensitivity.r2_limit)};
    const bool enabled = !present(gene_mask) || load_flag(gene_mask, genes[k]);
    auto r = q::report(x, y, rows, 1, base, fit, scratch, mask, enabled, diagnostic, s::HeapSort{});
    store(out.state, k, std::uint8_t(r.state));
    store(out.low, k, r.low); store(out.high, k, r.high); store(out.delta, k, r.delta);
    store(out.prediction_change, k, r.prediction_change); store(out.r2_change, k, r.r2_change);
    store(out.expanded_status, k, std::uint8_t(r.expanded_status));
    store(out.contracted_status, k, std::uint8_t(r.contracted_status));
    store(out.status_change, k, std::uint8_t(r.status_change));
    store(out.selection_change, k, std::uint8_t(r.selection_change));
}
}

int sensitivity(std::uint64_t rows, std::uint64_t first, std::uint64_t count,
    const std::uint64_t* genes, DenseView fit_mask, DenseView gene_mask, Fields fields,
    Result results, Options opts, SensitivityOptions options, SensitivityOutput output,
    bool fp32, void* stream) {
    if (!count) return int(cudaSuccess);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    const std::uint8_t* mask = nullptr;
    if (opts.trimming && fit_mask.values.data && rows) {
        mask_rows<<<unsigned((rows + 255)/256), 256, 0, cuda_stream>>>(rows, fit_mask, fields.mask);
        auto e = cudaGetLastError(); if (e != cudaSuccess) return int(e);
        mask = fields.mask;
    }
    auto blocks = unsigned((count + 127)/128);
    if (fp32) report_genes<float><<<blocks, 128, 0, cuda_stream>>>(rows, first, count, genes,
        gene_mask, fields, results, opts, options, output, mask);
    else report_genes<double><<<blocks, 128, 0, cuda_stream>>>(rows, first, count, genes,
        gene_mask, fields, results, opts, options, output, mask);
    auto e = cudaGetLastError(); if (e != cudaSuccess) return int(e);
    return int(cudaSuccess);
}
}
