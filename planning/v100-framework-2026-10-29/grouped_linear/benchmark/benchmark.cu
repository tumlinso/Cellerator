#include <Cellerator/modeling/cuda/grouped_linear.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace cellerator::modeling::cuda;

namespace {
template <class T> T encode(float value) { return static_cast<T>(value); }
template <> __half encode(float value) { return __float2half_rn(value); }
template <class T> float decode(T value) { return static_cast<float>(value); }
template <> float decode(__half value) { return __half2float(value); }
template <class T>
std::vector<T> encode(const std::vector<float>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (const auto value : values) result.push_back(encode<T>(value));
  return result;
}
template <class T>
std::vector<float> decode(const std::vector<T>& values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (const auto value : values) result.push_back(decode(value));
  return result;
}

float fp16_ulp(float value) {
  const float magnitude = std::abs(value);
  if (magnitude < 0x1p-14f) return 0x1p-24f;
  return std::ldexp(1.0f, std::ilogb(magnitude) - 10);
}

float allowed_error(DType dtype, float expected, int dot_length,
                    float sum_abs_products) {
  // Bound two FP32 dot implementations by 2*gamma_(2n)*sum|products|;
  // FP16 also permits one output ULP for the final conversion.
  const double unit_roundoff = std::numeric_limits<float>::epsilon() / 2.0;
  const double k = 2.0 * dot_length;
  const double ku = k * unit_roundoff;
  const float accumulation_bound = static_cast<float>(
      (2.0 * ku / (1.0 - ku)) * sum_abs_products);
  return accumulation_bound +
      (dtype == DType::float16 ? fp16_ulp(expected) : 0.0f);
}

struct Config { int experts, rows_per_expert, inputs, outputs; };

struct PointerPlan {
  std::vector<const void*> a, b;
  std::vector<void*> c;
  DeviceBuffer d_a, d_b, d_c;
  PinnedBuffer h_a, h_b, h_c;
  std::vector<cublasOperation_t> trans_a{CUBLAS_OP_T}, trans_b{CUBLAS_OP_N};
  std::vector<int> m, n, k, lda, ldb, ldc, group_size;
  std::vector<float> alpha{1.0f}, beta{0.0f};

  PointerPlan(const Config& cfg, const std::vector<ConstDeviceView>& weights,
              ConstDeviceView input, DeviceView output, DType dtype,
              const Stream& stream)
      : a(cfg.experts), b(cfg.experts), c(cfg.experts),
        d_a(cfg.experts * sizeof(void*), input.device),
        d_b(cfg.experts * sizeof(void*), input.device),
        d_c(cfg.experts * sizeof(void*), input.device),
        h_a(cfg.experts * sizeof(void*)), h_b(cfg.experts * sizeof(void*)),
        h_c(cfg.experts * sizeof(void*)) {
    const auto type_size = item_size(dtype);
    for (int e = 0; e < cfg.experts; ++e) {
      const auto row = static_cast<std::size_t>(e) * cfg.rows_per_expert;
      a[e] = weights[e].data;
      b[e] = static_cast<const char*>(input.data) + row * cfg.inputs * type_size;
      c[e] = static_cast<char*>(output.data) + row * cfg.outputs * type_size;
    }
    m = {cfg.outputs};
    n = {cfg.rows_per_expert};
    k = {cfg.inputs};
    lda = {cfg.inputs};
    ldb = {cfg.inputs};
    ldc = {cfg.outputs};
    group_size = {cfg.experts};
    prepare(stream);
  }

  void prepare(const Stream& stream) {
    std::memcpy(h_a.data(), a.data(), a.size() * sizeof(void*));
    std::memcpy(h_b.data(), b.data(), b.size() * sizeof(void*));
    std::memcpy(h_c.data(), c.data(), c.size() * sizeof(void*));
    d_a.upload_async(h_a.data(), a.size() * sizeof(void*), stream);
    d_b.upload_async(h_b.data(), b.size() * sizeof(void*), stream);
    d_c.upload_async(h_c.data(), c.size() * sizeof(void*), stream);
  }

  void refresh(const Config& cfg, const std::vector<ConstDeviceView>& weights,
               ConstDeviceView input, DeviceView output, DType dtype,
               const Stream& stream) {
    // Weights are fixed by GroupedLinear for its lifetime, while activation
    // and output allocations are supplied by the caller per forward call.
    const auto type_size = item_size(dtype);
    for (int e = 0; e < cfg.experts; ++e) {
      const auto row = static_cast<std::size_t>(e) * cfg.rows_per_expert;
      b[e] = static_cast<const char*>(input.data) + row * cfg.inputs * type_size;
      c[e] = static_cast<char*>(output.data) + row * cfg.outputs * type_size;
    }
    std::memcpy(h_b.data(), b.data(), b.size() * sizeof(void*));
    std::memcpy(h_c.data(), c.data(), c.size() * sizeof(void*));
    d_b.upload_async(h_b.data(), b.size() * sizeof(void*), stream);
    d_c.upload_async(h_c.data(), c.size() * sizeof(void*), stream);
  }
};

void check_cublas(cublasStatus_t status, const char* where) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << where << " cuBLAS status=" << static_cast<int>(status) << '\n';
    std::abort();
  }
}

void grouped_call(cublasHandle_t handle, const PointerPlan& p, DType dtype) {
  const auto type = dtype == DType::float16 ? CUDA_R_16F : CUDA_R_32F;
  check_cublas(cublasGemmGroupedBatchedEx(
      handle, p.trans_a.data(), p.trans_b.data(), p.m.data(), p.n.data(),
      p.k.data(), p.alpha.data(), static_cast<const void* const*>(p.d_a.data()),
      type, p.lda.data(), static_cast<const void* const*>(p.d_b.data()), type,
      p.ldb.data(), p.beta.data(), static_cast<void* const*>(p.d_c.data()), type,
      p.ldc.data(), 1, p.group_size.data(), CUBLAS_COMPUTE_32F),
      "cublasGemmGroupedBatchedEx");
}

void batched_call(cublasHandle_t handle, const PointerPlan& p,
                  const Config& cfg, DType dtype) {
  const auto type = dtype == DType::float16 ? CUDA_R_16F : CUDA_R_32F;
  const float alpha = 1.0f, beta = 0.0f;
  check_cublas(cublasGemmBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, cfg.outputs, cfg.rows_per_expert,
      cfg.inputs, &alpha, static_cast<const void* const*>(p.d_a.data()), type,
      cfg.inputs, static_cast<const void* const*>(p.d_b.data()), type,
      cfg.inputs, &beta, static_cast<void* const*>(p.d_c.data()), type,
      cfg.outputs, cfg.experts, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmBatchedEx");
}

template <class Fn>
void measure(const char* name, Fn&& fn, const Stream& stream, int device,
             int warmups = 12, int repeats = 48, int samples = 9) {
  for (int i = 0; i < warmups; ++i) fn();
  stream.synchronize();
  Event begin(device, true), end(device, true);
  std::vector<float> device_ms, submit_ms;
  device_ms.reserve(samples);
  submit_ms.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    const auto wall_begin = std::chrono::steady_clock::now();
    begin.record(stream);
    for (int i = 0; i < repeats; ++i) fn();
    end.record(stream);
    end.synchronize();
    const auto wall_end = std::chrono::steady_clock::now();
    device_ms.push_back(begin.elapsed_ms(end) / repeats);
    submit_ms.push_back(std::chrono::duration<float, std::milli>(
        wall_end - wall_begin).count() / repeats);
  }
  auto median = [](std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  std::cout << "TIMING," << name << ",event_median_ms=" << median(device_ms)
            << ",submit_sync_median_ms=" << median(submit_ms)
            << ",event_samples_ms=";
  for (const auto value : device_ms) std::cout << value << ';';
  std::cout << std::endl;
}

template <class T>
void run_case(const Config& cfg, DType dtype, int device) {
  const std::size_t rows = static_cast<std::size_t>(cfg.experts) * cfg.rows_per_expert;
  std::mt19937 rng(0xC311E + cfg.experts * 17 + cfg.inputs);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> x(rows * cfg.inputs);
  for (auto& value : x) value = dist(rng);
  auto x_t = encode<T>(x);

  std::vector<DeviceBuffer> weight_buffers;
  std::vector<std::vector<T>> weight_host(cfg.experts);
  std::vector<ConstDeviceView> weights;
  std::vector<std::int64_t> offsets(cfg.experts + 1);
  weight_buffers.reserve(cfg.experts);
  weights.reserve(cfg.experts);
  for (int e = 0; e < cfg.experts; ++e) {
    std::vector<float> values(static_cast<std::size_t>(cfg.outputs) * cfg.inputs);
    for (auto& value : values) value = dist(rng);
    weight_host[e] = encode<T>(values);
    weight_buffers.emplace_back(weight_host[e].size() * sizeof(T), device);
    weights.push_back(weight_buffers.back().view());
    offsets[e] = static_cast<std::int64_t>(e) * cfg.rows_per_expert;
  }
  offsets[cfg.experts] = rows;

  DeviceBuffer input(rows * cfg.inputs * sizeof(T), device);
  DeviceBuffer reference_output(rows * cfg.outputs * sizeof(T), device);
  DeviceBuffer grouped_output(rows * cfg.outputs * sizeof(T), device);
  Stream stream(device);
  input.upload_async(x_t.data(), x_t.size() * sizeof(T), stream);
  for (int e = 0; e < cfg.experts; ++e)
    weight_buffers[e].upload_async(weight_host[e].data(),
                                   weight_host[e].size() * sizeof(T), stream);

  GroupedLinear incumbent(cfg.inputs, cfg.outputs, offsets, weights, dtype, device);
  auto pointers = PointerPlan(cfg, weights, input.view(), grouped_output.view(), dtype, stream);
  DeviceBuffer batch_pointer_c(rows * cfg.outputs * sizeof(T), device);
  // Use a separate pointer plan because the conventional batched path writes
  // to a different output allocation for the equality gate.
  auto batch_pointers = PointerPlan(cfg, weights, input.view(), batch_pointer_c.view(), dtype, stream);
  cublasHandle_t library = nullptr;
  check_cublas(cublasCreate(&library), "cublasCreate");
  check_cublas(cublasSetStream(library, stream.get()), "cublasSetStream");
  check_cublas(cublasSetMathMode(library, dtype == DType::float16
      ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

  const auto x_values = decode(x_t);
  std::vector<float> expected(rows * cfg.outputs, 0.0f);
  std::vector<float> expected_abs(rows * cfg.outputs, 0.0f);
  for (int e = 0; e < cfg.experts; ++e)
    for (int r = 0; r < cfg.rows_per_expert; ++r)
      for (int o = 0; o < cfg.outputs; ++o)
        for (int i = 0; i < cfg.inputs; ++i) {
          const float product =
              x_values[(static_cast<std::size_t>(e) * cfg.rows_per_expert + r) * cfg.inputs + i] *
              decode(weight_host[e][static_cast<std::size_t>(o) * cfg.inputs + i]);
          const auto index =
              (static_cast<std::size_t>(e) * cfg.rows_per_expert + r) * cfg.outputs + o;
          expected[index] += product;
          expected_abs[index] += std::abs(product);
        }
  for (auto& value : expected) value = decode(encode<T>(value));

  incumbent.forward(input.view(), reference_output.view(), stream);
  grouped_call(library, pointers, dtype);
  batched_call(library, batch_pointers, cfg, dtype);
  stream.synchronize();

  std::vector<T> got_reference(rows * cfg.outputs), got_grouped(rows * cfg.outputs),
                 got_batched(rows * cfg.outputs);
  reference_output.download_async(got_reference.data(), got_reference.size() * sizeof(T), stream);
  grouped_output.download_async(got_grouped.data(), got_grouped.size() * sizeof(T), stream);
  batch_pointer_c.download_async(got_batched.data(), got_batched.size() * sizeof(T), stream);
  stream.synchronize();
  auto reference = decode(got_reference);
  auto grouped = decode(got_grouped);
  auto batched = decode(got_batched);
  float maximum_reference_error = 0.0f, maximum_grouped_error = 0.0f,
        maximum_batched_error = 0.0f;
  for (std::size_t i = 0; i < rows * cfg.outputs; ++i) {
    const float reference_error = std::abs(reference[i] - expected[i]);
    const float grouped_error = std::abs(grouped[i] - expected[i]);
    const float batched_error = std::abs(batched[i] - expected[i]);
    maximum_reference_error = std::max(maximum_reference_error, reference_error);
    maximum_grouped_error = std::max(maximum_grouped_error, grouped_error);
    maximum_batched_error = std::max(maximum_batched_error, batched_error);
    const float tol = allowed_error(dtype, expected[i], cfg.inputs, expected_abs[i]);
    if (reference_error > tol || grouped_error > tol || batched_error > tol) {
      std::cerr << "oracle gate failed at " << i << ": ref=" << reference[i]
                << " grouped=" << grouped[i] << " batched=" << batched[i]
                << " expected=" << expected[i] << " tolerance=" << tol
                << " errors=" << reference_error << ',' << grouped_error << ','
                << batched_error << '\n';
      std::abort();
    }
  }
  std::cout << "PASS," << (dtype == DType::float16 ? "fp16" : "fp32") << ','
            << cfg.experts << ',' << cfg.rows_per_expert << ',' << cfg.inputs << ','
            << cfg.outputs << ",oracle_max_abs=" << maximum_reference_error
            << ",grouped_max_abs=" << maximum_grouped_error
            << ",batched_max_abs=" << maximum_batched_error << std::endl;

  measure("incumbent_one_call_per_expert", [&] {
    incumbent.forward(input.view(), reference_output.view(), stream);
  }, stream, device);
  measure("grouped_batched_resident_pointers", [&] {
    grouped_call(library, pointers, dtype);
  }, stream, device);
  measure("grouped_batched_pointer_refresh", [&] {
    pointers.refresh(cfg, weights, input.view(), grouped_output.view(), dtype, stream);
    // The persistent pinned host lists are refreshed again on each iteration;
    // wait until DMA has consumed them before rebuilding the next lists.
    stream.synchronize();
    grouped_call(library, pointers, dtype);
  }, stream, device);
  measure("cublas_batched_resident_pointers", [&] {
    batched_call(library, batch_pointers, cfg, dtype);
  }, stream, device);
  measure("cublas_batched_pointer_refresh", [&] {
    batch_pointers.refresh(cfg, weights, input.view(), batch_pointer_c.view(), dtype, stream);
    stream.synchronize();
    batched_call(library, batch_pointers, cfg, dtype);
  }, stream, device);
  check_cublas(cublasDestroy(library), "cublasDestroy");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "--gpu") {
    std::cerr << "usage: cellerator_grouped_linear_benchmark --gpu\n";
    return 2;
  }
  int device = 0, count = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device >= count) std::abort();
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "requires Tesla V100 sm_70, got " << properties.name << '\n';
    return 3;
  }
  std::cout << "DEVICE," << properties.name << ",cc=" << properties.major << '.'
            << properties.minor << ",visible_index=" << device << std::endl;
  const Config configs[] = {{2, 2, 64, 64}, {8, 2, 64, 64}, {32, 2, 64, 64},
                            {8, 16, 128, 128}, {16, 8, 256, 256}};
  for (const auto& cfg : configs) {
    run_case<__half>(cfg, DType::float16, device);
    run_case<float>(cfg, DType::float32, device);
  }
}
