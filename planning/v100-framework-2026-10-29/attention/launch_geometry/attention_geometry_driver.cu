// Trusted exact-fixture oracle and timing driver for the attention row-grid probe.
#include <Cellerator/modeling/cuda/attention.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using cellerator::modeling::cuda::SparseAttentionPlan;
using cellerator::modeling::cuda::SparseAttentionWorkspace;

constexpr int kQueries = 512;
constexpr int kKeys = 96;
constexpr int kDim = 32;
constexpr int kValueDim = 32;
constexpr int kSamples = 7;
constexpr int kWarmupsPerSample = 4;
constexpr int kIterationsPerSample = 25;
constexpr float kScale = 0.1767766953f;

void check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expr) check((expr), #expr)

template<class T>
struct DeviceBuffer {
  explicit DeviceBuffer(std::size_t elements) : count(elements) {
    if (count) CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  }
  ~DeviceBuffer() { if (pointer) (void)cudaFree(pointer); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void upload(const std::vector<T>& values) {
    if (values.size() != count) throw std::invalid_argument("upload size mismatch");
    if (count) CUDA_CHECK(cudaMemcpy(pointer, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> values(count);
    if (count) CUDA_CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
  T* pointer = nullptr;
  std::size_t count = 0;
};

enum class Route { staged, fused };

struct Fixture {
  bool mixed_degree;
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> keys;
  std::vector<float> q, k, v;
  std::vector<double> expected;

  explicit Fixture(bool mixed) : mixed_degree(mixed), offsets{0} {
    for (int row = 0; row < kQueries; ++row) {
      const int degree = mixed_degree && (row % 16 == 0) ? 65 : 8;
      for (int edge = 0; edge < degree; ++edge)
        keys.push_back((row * 11 + edge * 7) % kKeys);
      offsets.push_back(static_cast<std::int64_t>(keys.size()));
    }
    q.resize(kQueries * kDim);
    k.resize(kKeys * kDim);
    v.resize(kKeys * kValueDim);
    for (std::size_t i = 0; i < q.size(); ++i)
      q[i] = static_cast<float>(patterned(i, 23, 0.025));
    for (std::size_t i = 0; i < k.size(); ++i)
      k[i] = static_cast<float>(patterned(i + 3, 19, 0.029));
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = static_cast<float>(patterned(i + 5, 17, 0.031));
    expected = oracle();
  }

  static double patterned(std::size_t index, int modulus, double magnitude) {
    return (static_cast<double>((index * 17 + 13) % static_cast<std::size_t>(modulus)) -
            modulus / 2.0) * magnitude;
  }

  std::vector<double> oracle() const {
    std::vector<double> output(static_cast<std::size_t>(kQueries) * kValueDim, 0.0);
    for (int row = 0; row < kQueries; ++row) {
      const auto begin = offsets[static_cast<std::size_t>(row)];
      const auto end = offsets[static_cast<std::size_t>(row + 1)];
      if (begin == end) continue;
      std::vector<double> logits(static_cast<std::size_t>(end - begin));
      double maximum = -std::numeric_limits<double>::infinity();
      for (auto edge = begin; edge < end; ++edge) {
        const auto key = keys[static_cast<std::size_t>(edge)];
        double dot = 0.0;
        for (int d = 0; d < kDim; ++d)
          dot += static_cast<double>(q[static_cast<std::size_t>(row * kDim + d)]) *
                 k[static_cast<std::size_t>(key * kDim + d)];
        const double score = static_cast<double>(kScale) * dot;
        logits[static_cast<std::size_t>(edge - begin)] = score;
        maximum = std::max(maximum, score);
      }
      double denominator = 0.0;
      for (auto& score : logits) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (auto edge = begin; edge < end; ++edge) {
        const double probability = logits[static_cast<std::size_t>(edge - begin)] / denominator;
        const auto key = keys[static_cast<std::size_t>(edge)];
        for (int d = 0; d < kValueDim; ++d)
          output[static_cast<std::size_t>(row * kValueDim + d)] += probability *
              v[static_cast<std::size_t>(key * kValueDim + d)];
      }
    }
    return output;
  }
};

struct DeviceFixture {
  explicit DeviceFixture(const Fixture& fixture)
      : q(fixture.q.size()), k(fixture.k.size()), v(fixture.v.size()),
        staged_output(static_cast<std::size_t>(kQueries) * kValueDim),
        fused_output(static_cast<std::size_t>(kQueries) * kValueDim),
        offsets(fixture.offsets.size()), keys(fixture.keys.size()),
        logits(fixture.keys.size()), probabilities(fixture.keys.size()) {
    q.upload(fixture.q); k.upload(fixture.k); v.upload(fixture.v);
    offsets.upload(fixture.offsets); keys.upload(fixture.keys);
    plan = {offsets.pointer, keys.pointer, kQueries, kKeys,
            static_cast<std::int64_t>(fixture.keys.size()), kDim, kValueDim};
    workspace = {logits.pointer, logits.count, probabilities.pointer,
                 probabilities.count, nullptr, 0, nullptr, 0};
  }

  template<cudaError_t (*Entry)(const float*, std::size_t, const float*, std::size_t,
                                const float*, std::size_t, float*, std::size_t,
                                SparseAttentionPlan, float, SparseAttentionWorkspace,
                                cudaStream_t)>
  void invoke(Route route) {
    auto* output = route == Route::staged ? staged_output.pointer : fused_output.pointer;
    CUDA_CHECK(Entry(q.pointer, q.count, k.pointer, k.count, v.pointer, v.count,
                     output, staged_output.count, plan, kScale, workspace, nullptr));
  }

  DeviceBuffer<float> q, k, v, staged_output, fused_output;
  DeviceBuffer<std::int64_t> offsets, keys;
  DeviceBuffer<float> logits, probabilities;
  SparseAttentionPlan plan{};
  SparseAttentionWorkspace workspace{};
};

template<cudaError_t (*Staged)(const float*, std::size_t, const float*, std::size_t,
                               const float*, std::size_t, float*, std::size_t,
                               SparseAttentionPlan, float, SparseAttentionWorkspace,
                               cudaStream_t),
         cudaError_t (*Fused)(const float*, std::size_t, const float*, std::size_t,
                              const float*, std::size_t, float*, std::size_t,
                              SparseAttentionPlan, float, SparseAttentionWorkspace,
                              cudaStream_t)>
void check_oracle(const Fixture& fixture, DeviceFixture& device) {
  device.template invoke<Staged>(Route::staged);
  device.template invoke<Fused>(Route::fused);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto staged = device.staged_output.download();
  const auto fused = device.fused_output.download();
  for (std::size_t i = 0; i < fixture.expected.size(); ++i) {
    const auto close = [&](float actual, const char* route) {
      const double expected = fixture.expected[i];
      const double allowance = 3.0e-5 + 3.0e-5 * std::abs(expected);
      if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > allowance) {
        std::fprintf(stderr,
            "%s oracle mismatch: degree=%s output[%zu] actual=%.10g expected=%.10g tolerance=%.4g\n",
            route, fixture.mixed_degree ? "8+65" : "8", i, actual, expected, allowance);
        throw std::runtime_error("attention geometry correctness check failed");
      }
    };
    close(staged[i], "staged");
    close(fused[i], "mixed-fused");
  }
}

template<cudaError_t (*Entry)(const float*, std::size_t, const float*, std::size_t,
                               const float*, std::size_t, float*, std::size_t,
                               SparseAttentionPlan, float, SparseAttentionWorkspace,
                               cudaStream_t)>
double timed_sample(DeviceFixture& device, Route route) {
  for (int warmup = 0; warmup < kWarmupsPerSample; ++warmup)
    device.template invoke<Entry>(route);
  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < kIterationsPerSample; ++iteration)
    device.template invoke<Entry>(route);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(milliseconds) * 1000.0 / kIterationsPerSample;
}

bool parse_case(const std::string& id, bool& mixed, Route& route) {
  if (id == "all8-staged") { mixed = false; route = Route::staged; return true; }
  if (id == "all8-fused") { mixed = false; route = Route::fused; return true; }
  if (id == "mixed8-65-staged") { mixed = true; route = Route::staged; return true; }
  if (id == "mixed8-65-fused") { mixed = true; route = Route::fused; return true; }
  return false;
}

template<cudaError_t (*Staged)(const float*, std::size_t, const float*, std::size_t,
                               const float*, std::size_t, float*, std::size_t,
                               SparseAttentionPlan, float, SparseAttentionWorkspace,
                               cudaStream_t),
         cudaError_t (*Fused)(const float*, std::size_t, const float*, std::size_t,
                              const float*, std::size_t, float*, std::size_t,
                              SparseAttentionPlan, float, SparseAttentionWorkspace,
                              cudaStream_t)>
void validate_all() {
  for (const bool mixed : {false, true}) {
    Fixture fixture(mixed);
    DeviceFixture device(fixture);
    check_oracle<Staged, Fused>(fixture, device);
  }
  std::puts("attention geometry independent FP64 forward oracle passed for staged and mixed-fused routes");
}

template<cudaError_t (*Staged)(const float*, std::size_t, const float*, std::size_t,
                               const float*, std::size_t, float*, std::size_t,
                               SparseAttentionPlan, float, SparseAttentionWorkspace,
                               cudaStream_t),
         cudaError_t (*Fused)(const float*, std::size_t, const float*, std::size_t,
                              const float*, std::size_t, float*, std::size_t,
                              SparseAttentionPlan, float, SparseAttentionWorkspace,
                              cudaStream_t)>
void benchmark(const std::string& id) {
  bool mixed = false;
  Route route = Route::staged;
  if (!parse_case(id, mixed, route)) throw std::invalid_argument("unsupported case id");
  Fixture fixture(mixed);
  DeviceFixture device(fixture);
  check_oracle<Staged, Fused>(fixture, device);
  std::vector<double> samples;
  samples.reserve(kSamples);
  if (route == Route::staged) {
    for (int i = 0; i < kSamples; ++i) samples.push_back(timed_sample<Staged>(device, route));
  } else {
    for (int i = 0; i < kSamples; ++i) samples.push_back(timed_sample<Fused>(device, route));
  }
  const auto& output = route == Route::staged ? device.staged_output : device.fused_output;
  const auto actual = output.download();
  for (std::size_t i = 0; i < fixture.expected.size(); ++i) {
    const double allowance = 3.0e-5 + 3.0e-5 * std::abs(fixture.expected[i]);
    if (!std::isfinite(actual[i]) || std::abs(static_cast<double>(actual[i]) - fixture.expected[i]) > allowance)
      throw std::runtime_error("timed route output changed after repeated execution");
  }
  std::printf("case=%s samples_us=", id.c_str());
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::printf("%s%.9g", i ? "," : "", samples[i]);
  std::printf(" warmup_per_sample=%d iterations_per_sample=%d dtype=fp32 Q=%d N=%d D=%d V=%d edges=%zu\n",
      kWarmupsPerSample, kIterationsPerSample, kQueries, kKeys, kDim, kValueDim,
      fixture.keys.size());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices != 1) throw std::runtime_error("driver requires exactly one scheduler-visible CUDA device");
    if (argc == 2 && std::string(argv[1]) == "--validate") {
      validate_all<cellerator::modeling::cuda::sparse_attention_fp32,
                   cellerator::modeling::cuda::sparse_attention_fused_small_fp32>();
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--benchmark") {
      benchmark<cellerator::modeling::cuda::sparse_attention_fp32,
                cellerator::modeling::cuda::sparse_attention_fused_small_fp32>(argv[2]);
      return 0;
    }
    std::fprintf(stderr, "usage: attention_geometry_driver --validate | --benchmark CASE_ID\n");
    return 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "attention geometry driver: %s\n", error.what());
    return 1;
  }
}
