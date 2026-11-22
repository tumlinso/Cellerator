#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>

#include <Cellerator/tensor/base/compressed.hh>
#ifdef CSR_BENCH_PRE_MOVE
#include <Cellerator/compute/operators/sparse/ops.hh>
#include <Cellerator/compute/candidate/sparse/project.hh>
#else
#include "src/ops/matrix/csr_multiply.cuh"
#include "src/ops/matrix/csr_cusparse.cuh"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
bool preprocess = false;

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
}

#ifndef CSR_BENCH_PRE_MOVE
void check(cusparseStatus_t status) {
    if (status == CUSPARSE_STATUS_SUCCESS) return;
    std::fprintf(stderr, "%s\n", cusparseGetErrorString(status));
    std::exit(1);
}

#endif

double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

template<class T> struct buffer {
    T *data = nullptr;
    size_t count;
    explicit buffer(size_t n) : count(n) {
        if (n) check(cudaMalloc(&data, n * sizeof(T)));
    }
    ~buffer() { if (data) check(cudaFree(data)); }
    size_t bytes() const { return count * sizeof(T); }
};

struct fixture {
    int rows = 0, cols = 0, width = 0;
    std::vector<int32_t> offsets, indices;
    std::vector<float> a, b, x;

    fixture(int m, int n, int w, bool skew) : rows(m), cols(n), width(w) {
        offsets.push_back(0);
        for (int r = 0; r < m; ++r) {
            const int degree = skew ? (r % 13 == 0 ? 128 : 4) : 24;
            std::vector<int> row;
            for (int j = 0; j < degree; ++j)
                row.push_back((uint64_t(r) * 1337 + uint64_t(j) * 17) % n);
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            for (int col : row) {
                indices.push_back(col);
                const float value = (int((col + uint64_t(r) * 5) % 31) - 15) / 32.0f;
                a.push_back(value);
                b.push_back(value * 0.5f + 0.0625f);
            }
            offsets.push_back(int32_t(indices.size()));
        }
        for (int i = 0; i < n * w; ++i)
            x.push_back((i % 37 - 18) / 32.0f);
    }

    fixture(const fixture& f, int) : rows(f.cols), cols(f.rows), width(f.width),
        offsets(rows + 1), indices(f.indices.size()), a(f.a.size()), b(f.b.size()) {
        for (int col : f.indices) ++offsets[col + 1];
        for (int r = 0; r < rows; ++r) offsets[r + 1] += offsets[r];
        auto slots = offsets;
        for (int r = 0; r < f.rows; ++r)
            for (int i = f.offsets[r]; i < f.offsets[r + 1]; ++i) {
                const int dst = slots[f.indices[i]]++;
                indices[dst] = r;
                a[dst] = f.a[i];
                b[dst] = f.b[i];
            }
        for (int i = 0; i < cols * width; ++i)
            x.push_back((i % 37 - 18) / 32.0f);
    }
};

template<class T> T convert(float x) { return x; }
template<> __half convert(float x) { return __float2half(x); }

template<class T>
std::vector<T> convert(const std::vector<float>& src) {
    std::vector<T> dst;
    dst.reserve(src.size());
    for (float x : src) dst.push_back(convert<T>(x));
    return dst;
}

template<class T>
void upload(buffer<T>& dst, const std::vector<T>& src, cudaStream_t s) {
    check(cudaMemcpyAsync(dst.data, src.data(), dst.bytes(), cudaMemcpyHostToDevice, s));
}

void verify(const fixture& f, const std::vector<float>& weights,
            const buffer<float>& y, cudaStream_t s) {
    std::vector<float> result(y.count);
    check(cudaMemcpyAsync(result.data(), y.data, y.bytes(), cudaMemcpyDeviceToHost, s));
    check(cudaStreamSynchronize(s));
    for (int r = 0; r < f.rows; r += std::max(1, f.rows / 101)) {
        for (int k = 0; k < f.width; ++k) {
            double sum = 0, abs_sum = 0;
            for (int i = f.offsets[r]; i < f.offsets[r + 1]; ++i) {
                const double product = double(weights[i]) * f.x[f.indices[i] * f.width + k];
                sum += product;
                abs_sum += std::abs(product);
            }
            const double error = std::abs(result[r * f.width + k] - sum);
            if (!(error < 2e-6 * std::max(1.0, abs_sum))) {
                std::fprintf(stderr, "oracle mismatch at row %d column %d: %.9g versus %.9g\n",
                             r, k, double(result[r * f.width + k]), sum);
                std::exit(1);
            }
        }
    }
}

template<class T>
void enqueue_native(const fixture& f, const buffer<int32_t>& offsets,
                    const buffer<int32_t>& indices, const buffer<T>& a,
                    const buffer<float>& x, buffer<float>& y, cudaStream_t s) {
    const auto* row = reinterpret_cast<const uint32_t*>(offsets.data);
    const auto* col = reinterpret_cast<const uint32_t*>(indices.data);
#ifdef CSR_BENCH_PRE_MOVE
    cellerator::runtime::execution_context ctx{};
    ctx.stream = s;
    ctx.device = 0;
    namespace project = cellerator::compute::sparse::project;
    if constexpr (std::is_same_v<T, __half>) {
        if (f.width == 1)
            cellerator::compute::sparse::ops::base::csr_spmv_fwd_f16_f32(
                ctx, row, col, a.data, f.rows, x.data, y.data);
        else
            project::csr_spmm_fwd_f16_f32(ctx, row, col, a.data, f.rows, f.cols,
                x.data, f.width, f.width, y.data, f.width);
    } else {
        project::csr_spmm_fwd_f32(ctx, row, col, a.data, f.rows, f.cols,
            x.data, f.width, f.width, y.data, f.width);
    }
#else
    namespace matrix = cellerator::ops::matrix;
    matrix::Csr<const T> csr{{uint64_t(f.rows), uint64_t(f.cols), a.count,
        cellerator::tensor::base::CompressedAxis::row, row, col}, a.data};
    if constexpr (std::is_same_v<T, __half>) {
        if (f.width == 1) {
            check(matrix::csr_spmv(csr, x.data, y.data, s));
            return;
        }
    }
    check(matrix::csr_spmm(csr, x.data, f.width, f.width, y.data, f.width, s));
#endif
}

#ifndef CSR_BENCH_PRE_MOVE
namespace vendor = cellerator::ops::matrix::cusparse;
#endif

template<class T>
void sample(const fixture& f, bool transpose, bool skew, int horizon,
            int repeat, bool use_vendor, double reorder_ms, cudaStream_t stream,
            cudaEvent_t begin, cudaEvent_t end) {
    const auto start = clock_type::now();
    const auto a = convert<T>(f.a), b = convert<T>(f.b);
    buffer<int32_t> offsets(f.offsets.size()), indices(f.indices.size());
    buffer<T> values(a.size()), version_a(a.size()), version_b(b.size());
    buffer<float> x(f.x.size()), y(size_t(f.rows) * f.width);
    upload(offsets, f.offsets, stream);
    upload(indices, f.indices, stream);
    upload(version_a, a, stream);
    upload(version_b, b, stream);
    upload(values, a, stream);
    upload(x, f.x, stream);
    size_t scratch_bytes = 0;
    bool use_preprocess = false;
#ifndef CSR_BENCH_PRE_MOVE
    cusparseHandle_t handle = nullptr;
    vendor::spmv_plan vector_plan;
    vendor::spmm_plan matrix_plan;
    if (use_vendor) {
        use_preprocess = preprocess && (f.width != 1 || vendor::spmv_preprocess_available);
        check(cusparseCreate(&handle));
        check(cusparseSetStream(handle, stream));
        const vendor::csr_view csr{{uint64_t(f.rows), uint64_t(f.cols), a.size(),
            cellerator::tensor::base::CompressedAxis::row, offsets.data, indices.data}, values.data};
        const auto type = std::is_same_v<T, __half> ? CUDA_R_16F : CUDA_R_32F;
        cusparseStatus_t status;
        if (f.width == 1) {
            status = vector_plan.provision(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, csr, type,
                {x.data, x.count}, {y.data, y.count}, CUSPARSE_SPMV_CSR_ALG1);
            scratch_bytes = vector_plan.workspace_bytes();
        } else {
            status = matrix_plan.provision(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, csr, type,
                {x.data, size_t(f.cols), size_t(f.width), size_t(f.width)},
                {y.data, size_t(f.rows), size_t(f.width), size_t(f.width)}, use_preprocess ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2);
            scratch_bytes = matrix_plan.workspace_bytes();
        }
        if (status == CUSPARSE_STATUS_NOT_SUPPORTED || status == CUSPARSE_STATUS_ARCH_MISMATCH) {
            std::printf("{\"backend\":\"vendor\",\"status\":\"unsupported\",\"storage\":\"%s\","
                        "\"rows\":%d,\"width\":%d}\n",
                std::is_same_v<T, __half> ? "f16" : "f32", f.rows, f.width);
            check(cusparseDestroy(handle));
            return;
        }
        check(status);
    }
    buffer<char> scratch(scratch_bytes);
    if (use_vendor)
        check(f.width == 1 ? vector_plan.bind_workspace(scratch.data, scratch_bytes, use_preprocess)
                           : matrix_plan.bind_workspace(scratch.data, scratch_bytes, use_preprocess));
#endif
    auto multiply = [&] {
#ifndef CSR_BENCH_PRE_MOVE
        if (use_vendor) {
            check(f.width == 1 ? vector_plan.enqueue() : matrix_plan.enqueue());
            return;
        }
#endif
        enqueue_native(f, offsets, indices, values, x, y, stream);
    };
    auto step = [&](int i) {
        const auto* src = i % 2 ? version_b.data : version_a.data;
        check(cudaMemcpyAsync(values.data, src, values.bytes(), cudaMemcpyDeviceToDevice, stream));
        multiply();
    };
    check(cudaStreamSynchronize(stream));
    const double setup_ms = milliseconds(start) + reorder_ms;
    check(cudaEventRecord(begin, stream));
    const auto step_start = clock_type::now();
    for (int i = 0; i < horizon; ++i) step(i);
    const double enqueue_ms = milliseconds(step_start);
    check(cudaEventRecord(end, stream));
    check(cudaStreamSynchronize(stream));
    const double step_ms = milliseconds(step_start);
    float gpu_ms = 0;
    check(cudaEventElapsedTime(&gpu_ms, begin, end));
    verify(f, horizon % 2 ? f.a : f.b, y, stream);
    step(horizon);
    verify(f, horizon % 2 ? f.b : f.a, y, stream);

    check(cudaEventRecord(begin, stream));
    const auto resident_start = clock_type::now();
    for (int i = 0; i < horizon; ++i) multiply();
    const double resident_enqueue_ms = milliseconds(resident_start);
    check(cudaEventRecord(end, stream));
    check(cudaStreamSynchronize(stream));
    float resident_gpu_ms = 0;
    check(cudaEventElapsedTime(&resident_gpu_ms, begin, end));

    const size_t peak = offsets.bytes() + indices.bytes() + values.bytes()
        + version_a.bytes() + version_b.bytes() + x.bytes() + y.bytes() + scratch_bytes;
#ifdef CSR_BENCH_PRE_MOVE
    const char* revision = "pre-move";
#else
    const char* revision = "current";
#endif
    std::printf("{\"revision\":\"%s\",\"backend\":\"%s\",\"op\":\"%s\","
                "\"storage\":\"%s\",\"orientation\":\"%s\",\"skew\":%s,"
                "\"rows\":%d,\"cols\":%d,\"nnz\":%zu,\"width\":%d,\"horizon\":%d,"
                "\"repeat\":%d,\"setup_ms\":%.9g,\"enqueue_ms\":%.9g,\"gpu_ms\":%.9g,"
                "\"step_ms\":%.9g,\"total_ms\":%.9g,"
                "\"resident_enqueue_ms\":%.9g,\"resident_gpu_ms\":%.9g,"
                "\"peak_explicit_device_bytes\":%zu,"
                "\"workspace_bytes\":%zu,\"preprocess\":%s,\"oracle\":\"pass\"}\n",
        revision, use_vendor ? "vendor" : "native",
        f.width != 1 ? "spmm" : (use_vendor || std::is_same_v<T, __half>) ? "spmv" : "spmm_n1",
        std::is_same_v<T, __half> ? "f16" : "f32", transpose ? "transpose" : "forward",
        skew ? "true" : "false", f.rows, f.cols, values.count, f.width, horizon, repeat,
        setup_ms, enqueue_ms, double(gpu_ms), step_ms, setup_ms + step_ms, resident_enqueue_ms, double(resident_gpu_ms),
        peak, scratch_bytes,
        use_preprocess ? "true" : "false");
#ifndef CSR_BENCH_PRE_MOVE
    if (use_vendor) {
        vector_plan.release();
        matrix_plan.release();
        check(cusparseDestroy(handle));
    }
#endif
}

template<class T>
void measure(const fixture& f, bool transpose, bool skew, int repeat, bool native,
             bool vendor_route, double reorder_ms, cudaStream_t stream,
             cudaEvent_t begin, cudaEvent_t end) {
    for (int horizon : {1, 8, 64}) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool use_vendor = (pass + repeat) % 2;
            if (use_vendor ? vendor_route : native)
                sample<T>(f, transpose, skew, horizon, repeat, use_vendor,
                          reorder_ms, stream, begin, end);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    bool small_only = false, native = true, vendor_route = true;
    int repeats = 3;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--small-only")) {
            small_only = true;
        } else if (!std::strcmp(argv[i], "--preprocess")) {
            preprocess = true;
        } else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) {
            const char* backend = argv[++i];
            native = !std::strcmp(backend, "both") || !std::strcmp(backend, "native");
            vendor_route = !std::strcmp(backend, "both") || !std::strcmp(backend, "vendor");
            if (!native && !vendor_route) return 2;
        } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
            char* end = nullptr;
            const long count = std::strtol(argv[++i], &end, 10);
            if (!end || *end || count < 1 || count > 20) return 2;
            repeats = int(count);
        } else {
            std::fprintf(stderr, "usage: %s [--backend both|native|vendor] [--small-only] [--preprocess] [--repeats N]\n", argv[0]);
            return 2;
        }
    }
#ifdef CSR_BENCH_PRE_MOVE
    if (!native) return 2;
    vendor_route = false;
#endif
    check(cudaSetDevice(0));
    cudaStream_t stream;
    cudaEvent_t begin, end;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    for (int m : {257, 32768}) {
        if (small_only && m != 257) break;
        for (int w : {1, 16, 64}) {
            for (bool skew : {false, true}) {
                fixture f(m, m - 14, w, skew);
                const auto start = clock_type::now();
                fixture t(f, 0);
                const double reorder_ms = milliseconds(start);
                for (int trial = 0; trial < repeats; ++trial) {
                    measure<__half>(f, false, skew, trial, native, vendor_route, 0, stream, begin, end);
                    measure<__half>(t, true, skew, trial, native, vendor_route, reorder_ms, stream, begin, end);
                    measure<float>(f, false, skew, trial, native, vendor_route, 0, stream, begin, end);
                    measure<float>(t, true, skew, trial, native, vendor_route, reorder_ms, stream, begin, end);
                }
            }
        }
    }
    check(cudaEventDestroy(end));
    check(cudaEventDestroy(begin));
    check(cudaStreamDestroy(stream));
}
