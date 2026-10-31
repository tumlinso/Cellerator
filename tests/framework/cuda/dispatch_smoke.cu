#include <Cellerator/modeling/cuda/plans.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {
using namespace cellerator::modeling::cuda;

float as_float(float value) { return value; }
float as_float(__half value) { return __half2float(value); }
float as_storage(float value, float*) { return value; }
float as_storage(float value, __half*) { return __half2float(__float2half_rn(value)); }

template <class T>
T stored(float value) {
  if constexpr (std::is_same_v<T, float>) return value;
  else return __float2half_rn(value);
}

template <class T>
void close(float actual, float expected, float tolerance, const char* stage,
           std::int64_t features, std::size_t index) {
  const float limit = tolerance * (1.0f + std::abs(expected));
  if (!std::isfinite(actual) || std::abs(actual - expected) > limit) {
    std::cerr << "dispatch oracle mismatch: stage=" << stage
              << " dtype=" << (std::is_same_v<T, float> ? "fp32" : "fp16")
              << " features=" << features << " index=" << index
              << " actual=" << actual << " expected=" << expected
              << " allowed_error=" << limit << '\n';
    std::abort();
  }
}

template <class T>
void check_dispatch_case(std::int64_t features, const Stream& stream) {
  // Logical selection order is [origin 1, origin 0, origin 1, origin 3,
  // origin 0]. Origin 2 is idle. Per-origin reductions use d1,d4 and d0,d2.
  constexpr std::int64_t origins = 4;
  const std::vector<std::int64_t> selection_origin{1, 0, 1, 3, 0};
  const std::vector<float> weights{0.5f, -0.25f, 0.0f, 2.0f, -1.0f};
  const std::vector<std::int64_t> origin_offsets{0, 2, 4, 4, 5};
  const std::vector<std::int64_t> origin_dispatch{1, 4, 0, 2, 3};
  const std::vector<std::int64_t> expert_offsets{0, 2, 5};
  DispatchPlan plan(origins, selection_origin, weights, origin_offsets,
                    origin_dispatch, expert_offsets, stream);

  const auto input_count = static_cast<std::size_t>(origins * features);
  const auto selection_count = selection_origin.size();
  std::vector<T> input(input_count), expert_output(selection_count * features),
      grad_origin(input_count), grad_gathered(selection_count * features);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = stored<T>(static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f);
  for (std::size_t i = 0; i < expert_output.size(); ++i)
    expert_output[i] = stored<T>(static_cast<float>(static_cast<int>(i % 9) - 3) * 0.1875f);
  for (std::size_t i = 0; i < grad_origin.size(); ++i)
    grad_origin[i] = stored<T>(static_cast<float>(static_cast<int>(i % 7) - 2) * 0.25f);
  for (std::size_t i = 0; i < grad_gathered.size(); ++i)
    grad_gathered[i] = stored<T>(static_cast<float>(static_cast<int>(i % 11) - 4) * 0.125f);

  DeviceBuffer device_input(input.size() * sizeof(T), stream.device());
  DeviceBuffer device_gathered(selection_count * static_cast<std::size_t>(features) * sizeof(T), stream.device());
  DeviceBuffer device_expert(selection_count * static_cast<std::size_t>(features) * sizeof(T), stream.device());
  DeviceBuffer device_origin(origins * static_cast<std::size_t>(features) * sizeof(T), stream.device());
  DeviceBuffer device_grad_origin(grad_origin.size() * sizeof(T), stream.device());
  DeviceBuffer device_grad_gathered(grad_gathered.size() * sizeof(T), stream.device());
  DeviceBuffer device_grad_expert(selection_count * static_cast<std::size_t>(features) * sizeof(T), stream.device());
  DeviceBuffer device_grad_input(input.size() * sizeof(T), stream.device());
  DeviceBuffer device_grad_weight(selection_count * sizeof(float), stream.device());

  if (!input.empty()) device_input.upload_async(input.data(), device_input.bytes(), stream);
  if (!expert_output.empty()) device_expert.upload_async(expert_output.data(), device_expert.bytes(), stream);
  if (!grad_origin.empty()) device_grad_origin.upload_async(grad_origin.data(), device_grad_origin.bytes(), stream);
  if (!grad_gathered.empty()) device_grad_gathered.upload_async(grad_gathered.data(), device_grad_gathered.bytes(), stream);
  stream.synchronize();

  const auto dtype = std::is_same_v<T, float> ? DType::float32 : DType::float16;
  plan.gather(device_input.view(), device_gathered.view(), features, dtype, stream);
  plan.combine(device_expert.view(), device_origin.view(), features, dtype, stream);
  plan.gather_backward(device_grad_gathered.view(), device_grad_input.view(), features, dtype, stream);
  plan.combine_backward(device_grad_origin.view(), device_expert.view(),
                        device_grad_expert.view(), device_grad_weight.view(),
                        features, dtype, stream);

  std::vector<T> gathered(selection_count * static_cast<std::size_t>(features));
  std::vector<T> combined(input.size()), grad_input(input.size()),
      grad_expert(selection_count * static_cast<std::size_t>(features));
  std::vector<float> grad_weight(selection_count, -99.0f);
  if (!gathered.empty()) device_gathered.download_async(gathered.data(), device_gathered.bytes(), stream);
  if (!combined.empty()) device_origin.download_async(combined.data(), device_origin.bytes(), stream);
  if (!grad_input.empty()) device_grad_input.download_async(grad_input.data(), device_grad_input.bytes(), stream);
  if (!grad_expert.empty()) device_grad_expert.download_async(grad_expert.data(), device_grad_expert.bytes(), stream);
  device_grad_weight.download_async(grad_weight.data(), device_grad_weight.bytes(), stream);
  stream.synchronize();

  const float tolerance = std::is_same_v<T, float> ? 2.0e-5f : 2.0e-3f;
  for (std::size_t d = 0; d < selection_count; ++d)
    for (std::int64_t f = 0; f < features; ++f)
      close<T>(as_float(gathered[d * features + f]),
               as_float(input[static_cast<std::size_t>(selection_origin[d] * features + f)]),
               tolerance, "gather", features, d * features + f);

  for (std::int64_t origin = 0; origin < origins; ++origin) {
    for (std::int64_t f = 0; f < features; ++f) {
      float expected = 0.0f, expected_grad = 0.0f;
      for (auto p = origin_offsets[origin]; p < origin_offsets[origin + 1]; ++p) {
        const auto d = static_cast<std::size_t>(origin_dispatch[p]);
        expected += weights[d] * as_float(expert_output[d * features + f]);
        expected_grad += as_float(grad_gathered[d * features + f]);
      }
      const auto index = static_cast<std::size_t>(origin * features + f);
      close<T>(as_float(combined[index]), as_storage(expected, static_cast<T*>(nullptr)),
               tolerance, "combine", features, index);
      close<T>(as_float(grad_input[index]), as_storage(expected_grad, static_cast<T*>(nullptr)),
               tolerance, "gather_backward", features, index);
    }
  }

  for (std::size_t d = 0; d < selection_count; ++d) {
    float expected_weight_grad = 0.0f;
    for (std::int64_t f = 0; f < features; ++f) {
      const auto origin = static_cast<std::size_t>(selection_origin[d] * features + f);
      const auto expert = d * static_cast<std::size_t>(features) + static_cast<std::size_t>(f);
      const float expected_expert_grad = weights[d] * as_float(grad_origin[origin]);
      close<T>(as_float(grad_expert[expert]),
               as_storage(expected_expert_grad, static_cast<T*>(nullptr)),
               tolerance, "combine_backward.grad_expert", features, expert);
      expected_weight_grad += as_float(grad_origin[origin]) * as_float(expert_output[expert]);
    }
    close<float>(grad_weight[d], expected_weight_grad, 3.0e-5f,
                 "combine_backward.grad_weight", features, d);
  }

  // Directional derivatives independently check both reverse outputs. FP16
  // gradients are checked against their rounded analytic reference above.
  if constexpr (std::is_same_v<T, float>) {
    float lhs = 0.0f, rhs = 0.0f;
    std::vector<float> delta_weights(selection_count);
    for (std::size_t d = 0; d < selection_count; ++d) {
      delta_weights[d] = static_cast<float>(static_cast<int>((d + 1) % 3) - 1) * 0.0625f;
      rhs += delta_weights[d] * grad_weight[d];
    }
    for (std::int64_t origin = 0; origin < origins; ++origin) {
      for (std::int64_t f = 0; f < features; ++f) {
        float delta_output = 0.0f;
        for (auto p = origin_offsets[origin]; p < origin_offsets[origin + 1]; ++p) {
          const auto d = static_cast<std::size_t>(origin_dispatch[p]);
          const auto expert = d * static_cast<std::size_t>(features) + static_cast<std::size_t>(f);
          const float delta_expert = static_cast<float>(static_cast<int>((expert + 2) % 5) - 2) * 0.03125f;
          delta_output += weights[d] * delta_expert + delta_weights[d] * as_float(expert_output[expert]);
          rhs += delta_expert * as_float(grad_expert[expert]);
        }
        const auto origin_index = static_cast<std::size_t>(origin * features + f);
        lhs += delta_output * as_float(grad_origin[origin_index]);
      }
    }
    close<float>(lhs, rhs, 5.0e-5f, "combine_backward.directional_derivative",
                 features, 0);
  }
}

void check_empty_feature_weight_gradient(const Stream& stream) {
  const std::vector<std::int64_t> selection_origin{0, 1, 0};
  DispatchPlan plan(2, selection_origin, {1.0f, -0.5f, 2.0f},
                    {0, 2, 3}, {0, 2, 1}, {0, 3}, stream);
  DeviceBuffer grad_weight(3 * sizeof(float), stream.device());
  const std::vector<float> sentinel{7.0f, 7.0f, 7.0f};
  grad_weight.upload_async(sentinel.data(), grad_weight.bytes(), stream);
  plan.combine_backward({}, {}, {}, grad_weight.view(), 0, DType::float32, stream);
  std::vector<float> actual(3);
  grad_weight.download_async(actual.data(), grad_weight.bytes(), stream);
  stream.synchronize();
  for (std::size_t i = 0; i < actual.size(); ++i)
    close<float>(actual[i], 0.0f, 0.0f,
                 "combine_backward.empty_feature_grad_weight", 0, i);
}

void check_empty_dispatches(const Stream& stream) {
  constexpr std::int64_t origins = 3;
  constexpr std::int64_t features = 33;
  DispatchPlan plan(origins, {}, {}, {0, 0, 0, 0}, {}, {}, stream);
  DeviceBuffer input(origins * features * sizeof(float), stream.device());
  DeviceBuffer gathered(0, stream.device());
  DeviceBuffer output(origins * features * sizeof(float), stream.device());
  DeviceBuffer grad_origin(origins * features * sizeof(float), stream.device());
  DeviceBuffer grad_input(origins * features * sizeof(float), stream.device());
  const std::vector<float> sentinel(origins * features, 9.0f);
  const std::vector<float> zero_gradient(origins * features, 0.0f);
  output.upload_async(sentinel.data(), output.bytes(), stream);
  grad_origin.upload_async(zero_gradient.data(), grad_origin.bytes(), stream);
  grad_input.upload_async(sentinel.data(), grad_input.bytes(), stream);
  plan.gather(input.view(), gathered.view(), features, DType::float32, stream);
  plan.combine({}, output.view(), features, DType::float32, stream);
  plan.gather_backward({}, grad_input.view(), features, DType::float32, stream);
  plan.combine_backward(grad_origin.view(), {}, {}, {}, features, DType::float32, stream);
  std::vector<float> actual(sentinel.size()), actual_grad(sentinel.size());
  output.download_async(actual.data(), output.bytes(), stream);
  grad_input.download_async(actual_grad.data(), grad_input.bytes(), stream);
  stream.synchronize();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    close<float>(actual[i], 0.0f, 0.0f, "combine.empty_routes", features, i);
    close<float>(actual_grad[i], 0.0f, 0.0f,
                 "gather_backward.empty_routes", features, i);
  }
}
}  // namespace

int main() {
  using namespace cellerator::modeling::cuda;
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  assert(device_count > 0);
  Stream stream(0);
  for (std::int64_t features : {1, 3, 31, 32, 33, 129}) {
    check_dispatch_case<float>(features, stream);
    check_dispatch_case<__half>(features, stream);
  }
  check_empty_feature_weight_gradient(stream);
  check_empty_dispatches(stream);
  std::cout << "dispatch device oracle passed on device 0\n";
}
