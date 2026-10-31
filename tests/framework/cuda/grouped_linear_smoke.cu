#include <Cellerator/modeling/cuda/grouped_linear.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace cellerator::modeling::cuda;

template <class T> float as_float(T value) { return static_cast<float>(value); }
template <> float as_float(__half value) { return __half2float(value); }
template <class T> T from_float(float value) { return static_cast<T>(value); }
template <> __half from_float(float value) { return __float2half_rn(value); }

float fp16_ulp(float value) {
  const float magnitude = std::abs(value);
  if (magnitude < 0x1p-14f) return 0x1p-24f;
  return std::ldexp(1.0f, std::ilogb(magnitude) - 10);
}

float allowed_error(DType dtype, float expected, int dot_length,
                    float sum_abs_products) {
  // Two dot implementations can each incur a gamma_(2n) error; FP16 also
  // rounds the final FP32 accumulator to a half output value.
  const double unit_roundoff = std::numeric_limits<float>::epsilon() / 2.0;
  const double k = 2.0 * dot_length;
  const double ku = k * unit_roundoff;
  const float accumulation_bound = static_cast<float>(
      (2.0 * ku / (1.0 - ku)) * sum_abs_products);
  return accumulation_bound +
      (dtype == DType::float16 ? fp16_ulp(expected) : 0.0f);
}

struct GroupedGemmPlan {
  // cuBLAS reads dimensions/scales from host but matrix-pointer arrays from device.
  std::vector<cublasOperation_t> trans_a, trans_b;
  std::vector<int> m, n, k, lda, ldb, ldc, group_size;
  std::vector<float> alpha, beta;
  std::vector<const void*> a, b;
  std::vector<void*> c;
  DeviceBuffer device_a, device_b, device_c;
};

GroupedGemmPlan grouped_plan(const std::vector<std::int64_t>& offsets,
                             std::int64_t inputs, std::int64_t outputs,
                             const std::vector<ConstDeviceView>& weights,
                             ConstDeviceView x, DeviceView y, DType dtype,
                             const Stream& stream) {
  GroupedGemmPlan plan;
  const auto size = item_size(dtype);
  for (std::size_t e = 0; e + 1 < offsets.size(); ++e) {
    const auto rows = offsets[e + 1] - offsets[e];
    if (rows == 0) continue;
    plan.trans_a.push_back(CUBLAS_OP_T);
    plan.trans_b.push_back(CUBLAS_OP_N);
    plan.m.push_back(static_cast<int>(outputs));
    plan.n.push_back(static_cast<int>(rows));
    plan.k.push_back(static_cast<int>(inputs));
    plan.lda.push_back(static_cast<int>(inputs));
    plan.ldb.push_back(static_cast<int>(inputs));
    plan.ldc.push_back(static_cast<int>(outputs));
    plan.alpha.push_back(1.0f);
    plan.beta.push_back(0.0f);
    plan.a.push_back(weights[e].data);
    plan.b.push_back(static_cast<const char*>(x.data) +
                     static_cast<std::size_t>(offsets[e]) * inputs * size);
    plan.c.push_back(static_cast<char*>(y.data) +
                     static_cast<std::size_t>(offsets[e]) * outputs * size);
    plan.group_size.push_back(1);
  }
  const auto pointer_bytes = plan.a.size() * sizeof(const void*);
  if (pointer_bytes) {
    plan.device_a = DeviceBuffer(pointer_bytes, x.device);
    plan.device_b = DeviceBuffer(pointer_bytes, x.device);
    plan.device_c = DeviceBuffer(plan.c.size() * sizeof(void*), x.device);
    plan.device_a.upload_async(plan.a.data(), pointer_bytes, stream);
    plan.device_b.upload_async(plan.b.data(), pointer_bytes, stream);
    plan.device_c.upload_async(plan.c.data(), plan.c.size() * sizeof(void*), stream);
  }
  return plan;
}

void run_grouped_gemm(cublasHandle_t handle, const GroupedGemmPlan& plan,
                      DType dtype) {
  if (plan.m.empty()) return;
  const auto type = dtype == DType::float16 ? CUDA_R_16F : CUDA_R_32F;
  const auto status = cublasGemmGroupedBatchedEx(
      handle, plan.trans_a.data(), plan.trans_b.data(), plan.m.data(),
      plan.n.data(), plan.k.data(), plan.alpha.data(),
      static_cast<const void* const*>(plan.device_a.data()), type,
      plan.lda.data(), static_cast<const void* const*>(plan.device_b.data()), type,
      plan.ldb.data(), plan.beta.data(), static_cast<void* const*>(plan.device_c.data()),
      type, plan.ldc.data(), static_cast<int>(plan.m.size()),
      plan.group_size.data(), CUBLAS_COMPUTE_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasGemmGroupedBatchedEx failed: " << static_cast<int>(status) << '\n';
    std::abort();
  }
}

void compare(const char* label, const std::vector<float>& actual,
             const std::vector<float>& expected,
             const std::vector<float>& sum_abs_products,
             int dot_length, DType dtype) {
  assert(actual.size() == expected.size());
  assert(actual.size() == sum_abs_products.size());
  float max_error = 0.0f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(actual[i] - expected[i]);
    max_error = std::max(max_error, error);
    const float tolerance = allowed_error(dtype, expected[i], dot_length,
                                          sum_abs_products[i]);
    if (error > tolerance) {
      std::cerr << label << " mismatch at " << i << ": got " << actual[i]
                << ", expected " << expected[i] << ", absolute error " << error
                << ", allowed " << tolerance
                << '\n';
      std::abort();
    }
  }
  std::cout << label << " max_abs_error=" << max_error << '\n';
}

template <class T>
std::vector<T> encode(const std::vector<float>& input) {
  std::vector<T> result;
  result.reserve(input.size());
  for (const auto value : input) result.push_back(from_float<T>(value));
  return result;
}

template <class T>
std::vector<float> decode(const std::vector<T>& input) {
  std::vector<float> result;
  result.reserve(input.size());
  for (const auto value : input) result.push_back(as_float(value));
  return result;
}

template <class T>
void verify_case(const std::vector<std::int64_t>& offsets, int inputs,
                 int outputs, DType dtype, int device, const char* tag) {
  const auto experts = offsets.size() - 1;
  const auto rows = static_cast<std::size_t>(offsets.back());
  std::mt19937 rng(0xCE11E + static_cast<unsigned>(inputs * 31 + outputs));
  std::uniform_real_distribution<float> dist(-0.75f, 0.75f);
  std::vector<float> x(rows * inputs), dy(rows * outputs);
  for (auto& value : x) value = dist(rng);
  for (auto& value : dy) value = dist(rng);

  std::vector<std::vector<float>> weights(experts);
  std::vector<std::vector<T>> device_weights(experts);
  std::vector<ConstDeviceView> weight_views;
  std::vector<DeviceBuffer> weight_buffers;
  weight_buffers.reserve(experts);
  weight_views.reserve(experts);
  for (std::size_t e = 0; e < experts; ++e) {
    weights[e].resize(static_cast<std::size_t>(inputs) * outputs);
    for (auto& value : weights[e]) value = dist(rng);
    device_weights[e] = encode<T>(weights[e]);
    weight_buffers.emplace_back(device_weights[e].size() * sizeof(T), device);
    weight_views.push_back(weight_buffers.back().view());
  }

  const auto x_t = encode<T>(x), dy_t = encode<T>(dy);
  DeviceBuffer input_buffer(x_t.size() * sizeof(T), device);
  DeviceBuffer dy_buffer(dy_t.size() * sizeof(T), device);
  DeviceBuffer y_buffer(rows * outputs * sizeof(T), device);
  DeviceBuffer dx_buffer(rows * inputs * sizeof(T), device);
  Stream stream(device);
  input_buffer.upload_async(x_t.data(), x_t.size() * sizeof(T), stream);
  dy_buffer.upload_async(dy_t.data(), dy_t.size() * sizeof(T), stream);
  for (std::size_t e = 0; e < experts; ++e)
    weight_buffers[e].upload_async(device_weights[e].data(),
                                   device_weights[e].size() * sizeof(T), stream);

  GroupedLinear executor(inputs, outputs, offsets, weight_views, dtype, device);
  // The independent oracle consumes uploaded tensor values, accumulates in
  // FP32, and rounds once to the output dtype.
  std::vector<float> expected_y(rows * outputs, 0.0f);
  std::vector<float> expected_y_abs(rows * outputs, 0.0f);
  for (std::size_t e = 0; e < experts; ++e)
    for (auto r = offsets[e]; r < offsets[e + 1]; ++r)
      for (int o = 0; o < outputs; ++o) {
        float value = 0.0f;
        for (int i = 0; i < inputs; ++i) {
          const float product = as_float(x_t[static_cast<std::size_t>(r) * inputs + i]) *
              as_float(device_weights[e][static_cast<std::size_t>(o) * inputs + i]);
          value += product;
          expected_y_abs[static_cast<std::size_t>(r) * outputs + o] += std::abs(product);
        }
        expected_y[static_cast<std::size_t>(r) * outputs + o] =
            as_float(from_float<T>(value));
      }

  std::vector<T> y_seed(rows * outputs, from_float<T>(9.0f));
  y_buffer.upload_async(y_seed.data(), y_seed.size() * sizeof(T), stream);
  executor.forward(input_buffer.view(), y_buffer.view(), stream);
  stream.synchronize();
  std::vector<T> actual_y(y_seed.size());
  y_buffer.download_async(actual_y.data(), actual_y.size() * sizeof(T), stream);
  stream.synchronize();
  compare((std::string(tag) + ".forward").c_str(), decode(actual_y),
          expected_y, expected_y_abs, inputs, dtype);

  // One-call cuBLAS grouped GEMM is a library alternative and provides a
  // separate device implementation of the same forward oracle.
  DeviceBuffer grouped_y_buffer(rows * outputs * sizeof(T), device);
  std::vector<T> grouped_seed(rows * outputs, from_float<T>(-7.0f));
  grouped_y_buffer.upload_async(grouped_seed.data(), grouped_seed.size() * sizeof(T), stream);
  auto plan = grouped_plan(offsets, inputs, outputs, weight_views,
                           input_buffer.view(), grouped_y_buffer.view(), dtype, stream);
  cublasHandle_t grouped_handle = nullptr;
  assert(cublasCreate(&grouped_handle) == CUBLAS_STATUS_SUCCESS);
  assert(cublasSetStream(grouped_handle, stream.get()) == CUBLAS_STATUS_SUCCESS);
  assert(cublasSetMathMode(grouped_handle, dtype == DType::float16
      ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS);
  run_grouped_gemm(grouped_handle, plan, dtype);
  stream.synchronize();
  std::vector<T> grouped_y(grouped_seed.size());
  grouped_y_buffer.download_async(grouped_y.data(), grouped_y.size() * sizeof(T), stream);
  stream.synchronize();
  compare((std::string(tag) + ".grouped_cublas_forward").c_str(),
          decode(grouped_y), expected_y, expected_y_abs, inputs, dtype);
  assert(cublasDestroy(grouped_handle) == CUBLAS_STATUS_SUCCESS);

  std::vector<float> expected_dx_raw(rows * inputs, 0.0f);
  std::vector<float> expected_dx(rows * inputs, 0.0f);
  std::vector<float> expected_dx_abs(rows * inputs, 0.0f);
  std::vector<std::vector<float>> expected_dw_raw(experts), expected_dw(experts),
      expected_dw_abs(experts);
  for (std::size_t e = 0; e < experts; ++e) {
    expected_dw_raw[e].assign(static_cast<std::size_t>(outputs) * inputs, 0.0f);
    expected_dw[e].assign(static_cast<std::size_t>(outputs) * inputs, 0.0f);
    expected_dw_abs[e].assign(static_cast<std::size_t>(outputs) * inputs, 0.0f);
    for (auto r = offsets[e]; r < offsets[e + 1]; ++r) {
      for (int i = 0; i < inputs; ++i)
        for (int o = 0; o < outputs; ++o) {
          const float product =
              as_float(dy_t[static_cast<std::size_t>(r) * outputs + o]) *
              as_float(device_weights[e][static_cast<std::size_t>(o) * inputs + i]);
          expected_dx_raw[static_cast<std::size_t>(r) * inputs + i] += product;
          expected_dx_abs[static_cast<std::size_t>(r) * inputs + i] += std::abs(product);
        }
      for (int o = 0; o < outputs; ++o)
        for (int i = 0; i < inputs; ++i) {
          const float product =
              as_float(dy_t[static_cast<std::size_t>(r) * outputs + o]) *
              as_float(x_t[static_cast<std::size_t>(r) * inputs + i]);
          const auto index = static_cast<std::size_t>(o) * inputs + i;
          expected_dw_raw[e][index] += product;
          expected_dw_abs[e][index] += std::abs(product);
        }
    }
    for (std::size_t i = 0; i < expected_dw[e].size(); ++i)
      expected_dw[e][i] = as_float(from_float<T>(expected_dw_raw[e][i]));
  }
  for (std::size_t i = 0; i < expected_dx.size(); ++i)
    expected_dx[i] = as_float(from_float<T>(expected_dx_raw[i]));

  std::vector<T> dx_seed(rows * inputs, from_float<T>(0.25f));
  dx_buffer.upload_async(dx_seed.data(), dx_seed.size() * sizeof(T), stream);
  std::vector<DeviceBuffer> dw_buffers;
  std::vector<std::vector<T>> dw_host(experts,
      std::vector<T>(static_cast<std::size_t>(outputs) * inputs, from_float<T>(-0.5f)));
  std::vector<DeviceView> dw_views;
  dw_buffers.reserve(experts);
  dw_views.reserve(experts);
  for (std::size_t e = 0; e < experts; ++e) {
    dw_buffers.emplace_back(dw_host[e].size() * sizeof(T), device);
    dw_views.push_back(dw_buffers.back().view());
    dw_buffers.back().upload_async(dw_host[e].data(), dw_host[e].size() * sizeof(T), stream);
  }

  executor.backward(input_buffer.view(), dy_buffer.view(), dx_buffer.view(), dw_views,
                    GradientMode::overwrite, GradientMode::overwrite, stream);
  stream.synchronize();
  std::vector<T> actual_dx(dx_seed.size());
  dx_buffer.download_async(actual_dx.data(), actual_dx.size() * sizeof(T), stream);
  std::vector<std::vector<T>> actual_dw(experts);
  for (std::size_t e = 0; e < experts; ++e) {
    actual_dw[e].resize(dw_host[e].size());
    dw_buffers[e].download_async(actual_dw[e].data(), actual_dw[e].size() * sizeof(T), stream);
  }
  stream.synchronize();
  compare((std::string(tag) + ".dx_overwrite").c_str(), decode(actual_dx), expected_dx,
          expected_dx_abs, outputs, dtype);
  for (std::size_t e = 0; e < experts; ++e)
    compare((std::string(tag) + ".dw_overwrite_" + std::to_string(e)).c_str(),
            decode(actual_dw[e]), expected_dw[e], expected_dw_abs[e],
            static_cast<int>(offsets[e + 1] - offsets[e]), dtype);

  // Isolate accumulate semantics: seed gradients afresh so this checks
  // seed + one backward pass, rather than two passes after overwrite.
  dx_buffer.upload_async(dx_seed.data(), dx_seed.size() * sizeof(T), stream);
  for (std::size_t e = 0; e < experts; ++e)
    dw_buffers[e].upload_async(dw_host[e].data(), dw_host[e].size() * sizeof(T), stream);
  executor.backward(input_buffer.view(), dy_buffer.view(), dx_buffer.view(), dw_views,
                    GradientMode::accumulate, GradientMode::accumulate, stream);
  stream.synchronize();
  dx_buffer.download_async(actual_dx.data(), actual_dx.size() * sizeof(T), stream);
  for (std::size_t e = 0; e < experts; ++e)
    dw_buffers[e].download_async(actual_dw[e].data(), actual_dw[e].size() * sizeof(T), stream);
  stream.synchronize();
  std::vector<float> expected_dx_acc(expected_dx.size());
  auto expected_dx_acc_abs = expected_dx_abs;
  for (std::size_t i = 0; i < expected_dx_acc.size(); ++i)
    expected_dx_acc[i] = as_float(from_float<T>(as_float(dx_seed[i]) + expected_dx_raw[i]));
  for (std::size_t i = 0; i < expected_dx_acc_abs.size(); ++i)
    expected_dx_acc_abs[i] += std::abs(as_float(dx_seed[i]));
  compare((std::string(tag) + ".dx_accumulate").c_str(), decode(actual_dx),
          expected_dx_acc, expected_dx_acc_abs, outputs + 1, dtype);
  for (std::size_t e = 0; e < experts; ++e) {
    std::vector<float> expected(expected_dw[e].size());
    auto expected_abs = expected_dw_abs[e];
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = as_float(from_float<T>(as_float(dw_host[e][i]) + expected_dw_raw[e][i]));
    for (std::size_t i = 0; i < expected_abs.size(); ++i)
      expected_abs[i] += std::abs(as_float(dw_host[e][i]));
    compare((std::string(tag) + ".dw_accumulate_" + std::to_string(e)).c_str(),
            decode(actual_dw[e]), expected, expected_abs,
            static_cast<int>(offsets[e + 1] - offsets[e]) + 1, dtype);
  }
  std::cout << "PASS " << tag << " dtype="
            << (dtype == DType::float16 ? "fp16" : "fp32")
            << " experts=" << experts << " rows=" << rows
            << " in=" << inputs << " out=" << outputs << '\n';
}

template <class T>
void benchmark_case(int experts, int rows_per_expert, int inputs,
                    int outputs, DType dtype, int device) {
  const std::size_t rows = static_cast<std::size_t>(experts) * rows_per_expert;
  std::vector<std::int64_t> offsets(experts + 1);
  for (int e = 0; e <= experts; ++e) offsets[e] = e * rows_per_expert;
  std::mt19937 rng(0xB3A7C);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<float> x(rows * inputs);
  for (auto& value : x) value = dist(rng);
  auto x_t = encode<T>(x);
  std::vector<DeviceBuffer> weights;
  std::vector<std::vector<T>> weight_data(experts);
  std::vector<ConstDeviceView> weight_views;
  weights.reserve(experts);
  weight_views.reserve(experts);
  for (int e = 0; e < experts; ++e) {
    std::vector<float> w(static_cast<std::size_t>(inputs) * outputs);
    for (auto& value : w) value = dist(rng);
    weight_data[e] = encode<T>(w);
    weights.emplace_back(weight_data[e].size() * sizeof(T), device);
    weight_views.push_back(weights.back().view());
  }
  DeviceBuffer input(rows * inputs * sizeof(T), device);
  DeviceBuffer output(rows * outputs * sizeof(T), device);
  Stream stream(device);
  input.upload_async(x_t.data(), x_t.size() * sizeof(T), stream);
  for (int e = 0; e < experts; ++e)
    weights[e].upload_async(weight_data[e].data(), weight_data[e].size() * sizeof(T), stream);
  GroupedLinear executor(inputs, outputs, offsets, weight_views, dtype, device);
  for (int i = 0; i < 12; ++i) executor.forward(input.view(), output.view(), stream);
  stream.synchronize();

  constexpr int repeats = 48;
  constexpr int samples = 7;
  std::vector<float> gpu_ms, wall_ms;
  Event begin(device, true), end(device, true);
  for (int sample = 0; sample < samples; ++sample) {
    const auto wall_begin = std::chrono::steady_clock::now();
    begin.record(stream);
    for (int i = 0; i < repeats; ++i) executor.forward(input.view(), output.view(), stream);
    end.record(stream);
    end.synchronize();
    const auto wall_end = std::chrono::steady_clock::now();
    gpu_ms.push_back(begin.elapsed_ms(end) / repeats);
    wall_ms.push_back(std::chrono::duration<float, std::milli>(wall_end - wall_begin).count() / repeats);
  }
  auto median = [](std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  std::cout << "BENCH grouped_linear," << (dtype == DType::float16 ? "fp16" : "fp32")
            << ',' << experts << ',' << rows_per_expert << ',' << inputs << ',' << outputs
            << ",gpu_event_median_ms=" << median(gpu_ms)
            << ",submit_sync_median_ms=" << median(wall_ms)
            << ",event_samples=";
  for (auto value : gpu_ms) std::cout << value << ';';
  std::cout << '\n';
}

void gpu_tests(int device) {
  const int cases[][3] = {{7, 5, 0}, {8, 16, 1}, {3, 9, 2}};
  for (const auto& test : cases) {
    const auto input_features = test[0], output_features = test[1];
    const std::vector<std::int64_t> offsets = test[2] == 0
        ? std::vector<std::int64_t>{0, 3, 3, 5, 9}
        : test[2] == 1 ? std::vector<std::int64_t>{0, 1, 9, 17}
                       : std::vector<std::int64_t>{0, 2, 7, 8};
    verify_case<float>(offsets, input_features, output_features, DType::float32, device,
                       ("ragged_" + std::to_string(test[2])).c_str());
    verify_case<__half>(offsets, input_features, output_features, DType::float16, device,
                        ("ragged_" + std::to_string(test[2])).c_str());
  }
}

void benchmark(int device) {
  std::cout << "BENCH schema=grouped_linear_v1; dtype,experts,rows_per_expert,input,output,device_event,submission_and_sync\n";
  const int configs[][4] = {{2, 2, 64, 64}, {8, 2, 64, 64},
                            {32, 2, 64, 64}, {8, 16, 128, 128},
                            {16, 8, 256, 256}};
  for (const auto& config : configs) {
    benchmark_case<__half>(config[0], config[1], config[2], config[3], DType::float16, device);
    benchmark_case<float>(config[0], config[1], config[2], config[3], DType::float32, device);
  }
}

void host_reference_smoke() {
  const std::vector<float> x{1, 1, 2, 0}, w{1, 2, 3, 4};
  assert(std::abs(x[0] * w[0] + x[1] * w[1] - 3.0f) < 1e-6f);
  assert(std::abs(x[0] * w[2] + x[1] * w[3] - 7.0f) < 1e-6f);
}
}  // namespace

int main(int argc, char** argv) {
  using namespace cellerator::modeling::cuda;
  if (argc < 2 || (std::string(argv[1]) != "--gpu" &&
                   std::string(argv[1]) != "--benchmark")) {
    host_reference_smoke();
    std::cout << "host-only grouped-linear reference passed; use --gpu or --benchmark under scheduler admission\n";
    return 0;
  }
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "get visible CUDA device count");
  if (device_count <= 0) throw std::runtime_error("no CUDA device is visible");
  int device = 0;
  check(cudaGetDevice(&device), "get scheduler-selected CUDA device");
  if (device >= device_count) throw std::runtime_error("selected CUDA device is not visible");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device), "get selected CUDA properties");
  std::cout << "DEVICE name=" << properties.name << " cc=" << properties.major << '.'
            << properties.minor << " visible_index=" << device << '\n';
  if (properties.major != 7 || properties.minor != 0)
    throw std::runtime_error("grouped-linear GPU checks require a V100 (sm_70)");
  if (std::string(argv[1]) == "--gpu") gpu_tests(device);
  else benchmark(device);
  return 0;
}
