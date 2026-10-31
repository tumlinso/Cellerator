// Trusted correctness/timing driver for the dispatch weight-gradient geometry
// experiment. All inputs are nonzero and derived from the declared fixture.
#include <Cellerator/modeling/cuda/plans.hh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace cellerator::modeling::cuda;
constexpr std::int64_t kOrigins = 64;
constexpr std::int64_t kSelections = 256;
constexpr std::uint32_t kSeed = 73119;
constexpr int kWarmups = 100;
constexpr int kSamples = 7;
constexpr int kIterationsPerSample = 300;

float fixture_value(std::uint64_t ordinal, std::uint32_t salt) {
  const auto magnitude = static_cast<int>((ordinal * 37 + salt * 19 + kSeed) % 101) + 1;
  const auto sign = ((ordinal + salt + kSeed) & 1u) == 0 ? 1.0f : -1.0f;
  return sign * static_cast<float>(magnitude) * 0.0005f;
}

[[noreturn]] void fail_value(const char* field, std::int64_t features,
                             std::size_t index, float actual, double expected,
                             double tolerance) {
  std::ostringstream message;
  message << std::setprecision(10) << "dispatch fixture mismatch: field=" << field
          << " features=" << features << " index=" << index
          << " actual=" << actual << " expected=" << expected
          << " tolerance=" << tolerance;
  throw std::runtime_error(message.str());
}

void close_value(const char* field, std::int64_t features, std::size_t index,
                 float actual, double expected, double relative_tolerance) {
  const double tolerance = relative_tolerance * (1.0 + std::abs(expected));
  if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > tolerance)
    fail_value(field, features, index, actual, expected, tolerance);
}

struct Fixture {
  explicit Fixture(std::int64_t feature_count)
      : features(feature_count), stream(0),
        selection_origin(kSelections), weights(kSelections),
        origin_offsets(kOrigins + 1), grad_origin(kOrigins * features),
        expert_output(kSelections * features),
        device_grad_origin(grad_origin.size() * sizeof(float), stream.device()),
        device_expert_output(expert_output.size() * sizeof(float), stream.device()),
        device_grad_expert(expert_output.size() * sizeof(float), stream.device()),
        device_grad_weight(kSelections * sizeof(float), stream.device()) {
    for (std::int64_t origin = 0; origin < kOrigins; ++origin) {
      origin_offsets[origin] = origin * (kSelections / kOrigins);
      for (std::int64_t k = 0; k < kSelections / kOrigins; ++k) {
        const auto dispatch = k * kOrigins + origin;
        selection_origin[dispatch] = origin;
        weights[dispatch] = (k & 1) != 0 ? -0.125f : 0.25f;
      }
    }
    origin_offsets[kOrigins] = kSelections;
    for (std::int64_t origin = 0; origin < kOrigins; ++origin)
      for (std::int64_t feature = 0; feature < features; ++feature)
        grad_origin[static_cast<std::size_t>(origin * features + feature)] =
            fixture_value(static_cast<std::uint64_t>(origin * features + feature), 3);
    for (std::int64_t dispatch = 0; dispatch < kSelections; ++dispatch)
      for (std::int64_t feature = 0; feature < features; ++feature)
        expert_output[static_cast<std::size_t>(dispatch * features + feature)] =
            fixture_value(static_cast<std::uint64_t>(dispatch * features + feature), 11);

    plan = std::make_unique<DispatchPlan>(
        kOrigins, selection_origin, weights, origin_offsets,
        make_dispatches(), std::vector<std::int64_t>{0, kSelections}, stream);
    device_grad_origin.upload_async(grad_origin.data(), device_grad_origin.bytes(), stream);
    device_expert_output.upload_async(expert_output.data(), device_expert_output.bytes(), stream);
    stream.synchronize();
  }

  std::vector<std::int64_t> make_dispatches() const {
    std::vector<std::int64_t> dispatches(kSelections);
    for (std::int64_t origin = 0; origin < kOrigins; ++origin)
      for (std::int64_t k = 0; k < kSelections / kOrigins; ++k)
        dispatches[static_cast<std::size_t>(origin * (kSelections / kOrigins) + k)] =
            k * kOrigins + origin;
    return dispatches;
  }

  void run_backward() {
    plan->combine_backward(device_grad_origin.view(), device_expert_output.view(),
                           device_grad_expert.view(), device_grad_weight.view(),
                           features, DType::float32, stream);
  }

  void check_oracle() {
    run_backward();
    stream.synchronize();
    std::vector<float> actual_expert(expert_output.size());
    std::vector<float> actual_weight(kSelections);
    device_grad_expert.download_async(actual_expert.data(), device_grad_expert.bytes(), stream);
    device_grad_weight.download_async(actual_weight.data(), device_grad_weight.bytes(), stream);
    stream.synchronize();

    for (std::int64_t dispatch = 0; dispatch < kSelections; ++dispatch) {
      const auto origin = selection_origin[dispatch];
      for (std::int64_t feature = 0; feature < features; ++feature) {
        const auto index = static_cast<std::size_t>(dispatch * features + feature);
        const auto origin_index = static_cast<std::size_t>(origin * features + feature);
        const double expected = static_cast<double>(weights[dispatch]) * grad_origin[origin_index];
        close_value("grad_expert", features, index, actual_expert[index], expected, 2.0e-6);
      }
      double expected_weight = 0.0;
      for (std::int64_t feature = 0; feature < features; ++feature) {
        const auto origin_index = static_cast<std::size_t>(origin * features + feature);
        const auto expert_index = static_cast<std::size_t>(dispatch * features + feature);
        expected_weight += static_cast<double>(grad_origin[origin_index]) *
                           expert_output[expert_index];
      }
      close_value("grad_weight", features, static_cast<std::size_t>(dispatch),
                  actual_weight[dispatch], expected_weight, 2.0e-4);
    }
  }

  double timed_sample_us() {
    Event start(stream.device(), true), stop(stream.device(), true);
    start.record(stream);
    for (int iteration = 0; iteration < kIterationsPerSample; ++iteration)
      run_backward();
    stop.record(stream);
    stop.synchronize();
    return static_cast<double>(start.elapsed_ms(stop)) * 1000.0 /
           static_cast<double>(kIterationsPerSample);
  }

  const std::int64_t features;
  Stream stream;
  std::vector<std::int64_t> selection_origin;
  std::vector<float> weights;
  std::vector<std::int64_t> origin_offsets;
  std::vector<float> grad_origin;
  std::vector<float> expert_output;
  std::unique_ptr<DispatchPlan> plan;
  DeviceBuffer device_grad_origin;
  DeviceBuffer device_expert_output;
  DeviceBuffer device_grad_expert;
  DeviceBuffer device_grad_weight;
};

void validate_all() {
  for (const auto features : {1, 32, 256, 2048}) {
    Fixture fixture(features);
    fixture.check_oracle();
  }
  std::cout << "dispatch geometry nonzero derivative oracle passed\n";
}

void benchmark(std::int64_t features) {
  Fixture fixture(features);
  fixture.check_oracle();
  for (int warmup = 0; warmup < kWarmups; ++warmup) fixture.run_backward();
  fixture.stream.synchronize();
  std::cout << std::setprecision(9) << "features=" << features << " samples_us=";
  for (int sample = 0; sample < kSamples; ++sample) {
    if (sample != 0) std::cout << ',';
    std::cout << fixture.timed_sample_us();
  }
  std::cout << " warmup=" << kWarmups << " iterations_per_sample="
            << kIterationsPerSample << " dtype=fp32 origins=" << kOrigins
            << " selections=" << kSelections << " seed=" << kSeed << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  using namespace cellerator::modeling::cuda;
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count != 1)
    throw std::runtime_error("driver requires exactly one scheduler-visible CUDA device");

  if (argc == 2 && std::string(argv[1]) == "--validate") {
    validate_all();
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--benchmark") {
    const auto features = std::strtoll(argv[2], nullptr, 10);
    if (features != 1 && features != 32 && features != 256 && features != 2048)
      throw std::invalid_argument("unsupported feature count");
    benchmark(features);
    return 0;
  }
  std::cerr << "usage: dispatch_geometry_driver --validate | --benchmark FEATURES\n";
  return 2;
}
