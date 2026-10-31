#include <Cellerator/modeling/cuda/attention.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using cellerator::modeling::cuda::GradientWriteMode;
using cellerator::modeling::cuda::SparseAttentionPlan;
using cellerator::modeling::cuda::SparseAttentionWorkspace;

void check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, expression,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}
#define CUDA_CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

template<class T> struct Scalar;
template<> struct Scalar<float> {
  static float encode(double value) { return static_cast<float>(value); }
  static double decode(float value) { return value; }
  static cudaError_t staged(const float* q, size_t qc, const float* k, size_t kc,
                            const float* v, size_t vc, float* out, size_t oc,
                            SparseAttentionPlan plan, float scale,
                            SparseAttentionWorkspace workspace, cudaStream_t stream) {
    return cellerator::modeling::cuda::sparse_attention_fp32(
        q, qc, k, kc, v, vc, out, oc, plan, scale, workspace, stream);
  }
  static cudaError_t mixed(const float* q, size_t qc, const float* k, size_t kc,
                           const float* v, size_t vc, float* out, size_t oc,
                           SparseAttentionPlan plan, float scale,
                           SparseAttentionWorkspace workspace, cudaStream_t stream) {
    return cellerator::modeling::cuda::sparse_attention_fused_small_fp32(
        q, qc, k, kc, v, vc, out, oc, plan, scale, workspace, stream);
  }
  static cudaError_t backward(const float* q, size_t qc, const float* k, size_t kc,
                              const float* v, size_t vc, const float* go, size_t goc,
                              SparseAttentionPlan plan, float scale,
                              SparseAttentionWorkspace workspace, float* gq, size_t gqc,
                              float* gk, size_t gkc, float* gv, size_t gvc,
                              GradientWriteMode mode, cudaStream_t stream) {
    return cellerator::modeling::cuda::sparse_attention_backward_fp32(
        q, qc, k, kc, v, vc, go, goc, plan, scale, workspace, gq, gqc, gk, gkc,
        gv, gvc, mode, stream);
  }
};
template<> struct Scalar<__half> {
  static __half encode(double value) { return __float2half_rn(static_cast<float>(value)); }
  static double decode(__half value) { return __half2float(value); }
  static cudaError_t staged(const __half* q, size_t qc, const __half* k, size_t kc,
                            const __half* v, size_t vc, __half* out, size_t oc,
                            SparseAttentionPlan plan, float scale,
                            SparseAttentionWorkspace workspace, cudaStream_t stream) {
    return cellerator::modeling::cuda::sparse_attention_fp16(
        q, qc, k, kc, v, vc, out, oc, plan, scale, workspace, stream);
  }
  static cudaError_t mixed(const __half* q, size_t qc, const __half* k, size_t kc,
                           const __half* v, size_t vc, __half* out, size_t oc,
                           SparseAttentionPlan plan, float scale,
                           SparseAttentionWorkspace workspace, cudaStream_t stream) {
    return cellerator::modeling::cuda::sparse_attention_fused_small_fp16(
        q, qc, k, kc, v, vc, out, oc, plan, scale, workspace, stream);
  }
  static cudaError_t backward(const __half* q, size_t qc, const __half* k, size_t kc,
                              const __half* v, size_t vc, const __half* go, size_t goc,
                              SparseAttentionPlan plan, float scale,
                              SparseAttentionWorkspace workspace, float* gq, size_t gqc,
                              float* gk, size_t gkc, float* gv, size_t gvc,
                              GradientWriteMode mode, cudaStream_t stream) {
    return cellerator::modeling::cuda::sparse_attention_backward_fp16(
        q, qc, k, kc, v, vc, go, goc, plan, scale, workspace, gq, gqc, gk, gkc,
        gv, gvc, mode, stream);
  }
};

struct HostCase {
  int queries = 7;
  int keys = 13;
  int dim = 7;
  int value_dim = 5;
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> key_ids;
  std::vector<double> q, k, v, grad_output;
  double scale = 0.37;
};

double patterned(int seed, int modulus, double scale) {
  return (static_cast<double>((seed * 17 + 13) % modulus) - modulus / 2.0) * scale;
}

HostCase make_boundary_case(int value_dim = 5) {
  HostCase c;
  c.value_dim = value_dim;
  c.offsets = {0};
  for (const int degree : {0, 1, 2, 31, 32, 33, 65}) {
    for (int edge = 0; edge < degree; ++edge)
      c.key_ids.push_back((edge * 7 + static_cast<int>(c.offsets.size()) * 3) % c.keys);
    c.offsets.push_back(static_cast<std::int64_t>(c.key_ids.size()));
  }
  c.q.resize(c.queries * c.dim);
  c.k.resize(c.keys * c.dim);
  c.v.resize(c.keys * c.value_dim);
  c.grad_output.resize(c.queries * c.value_dim);
  for (size_t i = 0; i < c.q.size(); ++i) c.q[i] = patterned(static_cast<int>(i), 23, 0.071);
  for (size_t i = 0; i < c.k.size(); ++i) c.k[i] = patterned(static_cast<int>(i + 9), 29, 0.059);
  for (size_t i = 0; i < c.v.size(); ++i) c.v[i] = patterned(static_cast<int>(i + 5), 31, 0.083);
  for (size_t i = 0; i < c.grad_output.size(); ++i)
    c.grad_output[i] = patterned(static_cast<int>(i + 3), 19, 0.067);
  return c;
}

struct HostResult {
  std::vector<double> output, probabilities, grad_q, grad_k, grad_v;
};

HostResult reference(const HostCase& c, const std::vector<double>& q,
                     const std::vector<double>& k, const std::vector<double>& v,
                     const std::vector<double>& grad_output) {
  HostResult r;
  r.output.assign(c.queries * c.value_dim, 0.0);
  r.probabilities.assign(c.key_ids.size(), 0.0);
  r.grad_q.assign(c.queries * c.dim, 0.0);
  r.grad_k.assign(c.keys * c.dim, 0.0);
  r.grad_v.assign(c.keys * c.value_dim, 0.0);
  for (int row = 0; row < c.queries; ++row) {
    const auto begin = c.offsets[row], end = c.offsets[row + 1];
    if (begin == end) continue;
    std::vector<double> logits(static_cast<size_t>(end - begin));
    double maximum = -INFINITY;
    for (auto edge = begin; edge < end; ++edge) {
      const auto key = c.key_ids[edge];
      double dot = 0.0;
      for (int d = 0; d < c.dim; ++d)
        dot += q[row * c.dim + d] * k[key * c.dim + d];
      logits[edge - begin] = c.scale * dot;
      maximum = std::max(maximum, logits[edge - begin]);
    }
    double denominator = 0.0;
    for (auto edge = begin; edge < end; ++edge) {
      auto& p = r.probabilities[edge];
      p = std::exp(logits[edge - begin] - maximum);
      denominator += p;
    }
    for (auto edge = begin; edge < end; ++edge) {
      auto& p = r.probabilities[edge];
      p /= denominator;
      const auto key = c.key_ids[edge];
      for (int d = 0; d < c.value_dim; ++d)
        r.output[row * c.value_dim + d] += p * v[key * c.value_dim + d];
    }
  }
  for (int row = 0; row < c.queries; ++row) {
    for (auto edge = c.offsets[row]; edge < c.offsets[row + 1]; ++edge) {
      const auto key = c.key_ids[edge];
      double edge_dot = 0.0;
      for (int d = 0; d < c.value_dim; ++d)
        edge_dot += grad_output[row * c.value_dim + d] * v[key * c.value_dim + d];
      const double p = r.probabilities[edge];
      double correction = 0.0;
      for (auto other = c.offsets[row]; other < c.offsets[row + 1]; ++other) {
        const auto other_key = c.key_ids[other];
        double other_dot = 0.0;
        for (int d = 0; d < c.value_dim; ++d)
          other_dot += grad_output[row * c.value_dim + d] * v[other_key * c.value_dim + d];
        correction += r.probabilities[other] * other_dot;
      }
      const double dlogit = p * (edge_dot - correction);
      for (int d = 0; d < c.dim; ++d) {
        r.grad_q[row * c.dim + d] += c.scale * dlogit * k[key * c.dim + d];
        r.grad_k[key * c.dim + d] += c.scale * dlogit * q[row * c.dim + d];
      }
      for (int d = 0; d < c.value_dim; ++d)
        r.grad_v[key * c.value_dim + d] += p * grad_output[row * c.value_dim + d];
    }
  }
  return r;
}

double objective(const HostCase& c, const std::vector<double>& q,
                 const std::vector<double>& k, const std::vector<double>& v) {
  const auto result = reference(c, q, k, v, c.grad_output);
  return std::inner_product(result.output.begin(), result.output.end(),
                            c.grad_output.begin(), 0.0);
}

template<class T> struct DeviceBuffer {
  T* pointer = nullptr;
  size_t count = 0;
  explicit DeviceBuffer(size_t n) : count(n) {
    if (n != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), n * sizeof(T)));
  }
  ~DeviceBuffer() { if (pointer) (void)cudaFree(pointer); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void upload(const std::vector<T>& values) {
    assert(values.size() == count);
    if (count) CUDA_CHECK(cudaMemcpy(pointer, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> values(count);
    if (count) CUDA_CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};

template<class T> std::vector<T> encode(const std::vector<double>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (double value : values) result.push_back(Scalar<T>::encode(value));
  return result;
}

void close_enough(double actual, double expected, double atol, double rtol,
                  const char* name, size_t index) {
  const double error = std::abs(actual - expected);
  if (!(error <= atol + rtol * std::abs(expected))) {
    std::fprintf(stderr, "%s[%zu]: actual %.9g, expected %.9g, abs error %.3g\n",
                 name, index, actual, expected, error);
    std::exit(2);
  }
}

template<class T>
void run_device_case(bool fused, int value_dim = 5) {
  const auto c = make_boundary_case(value_dim);
  const auto qh = encode<T>(c.q), kh = encode<T>(c.k), vh = encode<T>(c.v);
  const auto goh = encode<T>(c.grad_output);
  HostCase represented = c;
  for (size_t i = 0; i < represented.q.size(); ++i) represented.q[i] = Scalar<T>::decode(qh[i]);
  for (size_t i = 0; i < represented.k.size(); ++i) represented.k[i] = Scalar<T>::decode(kh[i]);
  for (size_t i = 0; i < represented.v.size(); ++i) represented.v[i] = Scalar<T>::decode(vh[i]);
  for (size_t i = 0; i < represented.grad_output.size(); ++i)
    represented.grad_output[i] = Scalar<T>::decode(goh[i]);
  const auto expected = reference(represented, represented.q, represented.k,
                                  represented.v, represented.grad_output);

  DeviceBuffer<T> q(qh.size()), k(kh.size()), v(vh.size()), go(goh.size()), output(c.queries * c.value_dim);
  DeviceBuffer<std::int64_t> offsets(c.offsets.size()), keys(c.key_ids.size());
  DeviceBuffer<float> logits(c.key_ids.size()), probabilities(c.key_ids.size());
  DeviceBuffer<float> edge_dot(c.key_ids.size()), correction(c.queries);
  DeviceBuffer<float> grad_q(c.q.size()), grad_k(c.k.size()), grad_v(c.v.size());
  q.upload(qh); k.upload(kh); v.upload(vh); go.upload(goh);
  offsets.upload(c.offsets); keys.upload(c.key_ids);

  SparseAttentionPlan plan{offsets.pointer, keys.pointer, c.queries, c.keys,
                           static_cast<std::int64_t>(c.key_ids.size()), c.dim, c.value_dim};
  SparseAttentionWorkspace workspace{logits.pointer, logits.count, probabilities.pointer,
                                     probabilities.count, edge_dot.pointer, edge_dot.count,
                                     correction.pointer, correction.count};
  cudaStream_t stream = nullptr;
  auto forward = fused ? Scalar<T>::mixed : Scalar<T>::staged;
  CUDA_CHECK(forward(q.pointer, q.count, k.pointer, k.count, v.pointer, v.count,
                     output.pointer, output.count, plan, static_cast<float>(c.scale),
                     workspace, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const double output_atol = std::is_same_v<T, __half> ? 3e-3 : 3e-5;
  const double output_rtol = output_atol;
  const auto actual_output = output.download();
  for (size_t i = 0; i < actual_output.size(); ++i)
    close_enough(Scalar<T>::decode(actual_output[i]), expected.output[i],
                 output_atol, output_rtol, fused ? "fused output" : "staged output", i);
  const auto actual_probabilities = probabilities.download();
  for (size_t i = 0; i < actual_probabilities.size(); ++i)
    close_enough(actual_probabilities[i], expected.probabilities[i], 3e-5, 3e-5,
                 "probabilities", i);

  const float* grad_output_device = reinterpret_cast<const float*>(go.pointer);
  // Backward has storage-matched grad-output input, so the typed pointer remains T.
  (void)grad_output_device;
  auto run_backward = [&](GradientWriteMode mode) {
    return Scalar<T>::backward(q.pointer, q.count, k.pointer, k.count, v.pointer, v.count,
        go.pointer, go.count, plan, static_cast<float>(c.scale), workspace,
        grad_q.pointer, grad_q.count, grad_k.pointer, grad_k.count, grad_v.pointer,
        grad_v.count, mode, stream);
  };
  std::vector<float> initial_q(c.q.size(), 0.125f), initial_k(c.k.size(), -0.25f),
      initial_v(c.v.size(), 0.375f);
  grad_q.upload(initial_q); grad_k.upload(initial_k); grad_v.upload(initial_v);
  CUDA_CHECK(run_backward(GradientWriteMode::zero));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto actual_gq = grad_q.download(), actual_gk = grad_k.download(), actual_gv = grad_v.download();
  for (size_t i = 0; i < actual_gq.size(); ++i)
    close_enough(actual_gq[i], expected.grad_q[i], 5e-5, 5e-5, "grad q", i);
  for (size_t i = 0; i < actual_gk.size(); ++i)
    close_enough(actual_gk[i], expected.grad_k[i], 2e-4, 2e-4, "grad k", i);
  for (size_t i = 0; i < actual_gv.size(); ++i)
    close_enough(actual_gv[i], expected.grad_v[i], 1e-4, 1e-4, "grad v", i);

  grad_q.upload(initial_q); grad_k.upload(initial_k); grad_v.upload(initial_v);
  CUDA_CHECK(run_backward(GradientWriteMode::accumulate));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto accumulated_q = grad_q.download(), accumulated_k = grad_k.download(),
             accumulated_v = grad_v.download();
  for (size_t i = 0; i < accumulated_q.size(); ++i)
    close_enough(accumulated_q[i], initial_q[i] + expected.grad_q[i], 5e-5, 5e-5,
                 "accumulated grad q", i);
  for (size_t i = 0; i < accumulated_k.size(); ++i)
    close_enough(accumulated_k[i], initial_k[i] + expected.grad_k[i], 2e-4, 2e-4,
                 "accumulated grad k", i);
  for (size_t i = 0; i < accumulated_v.size(); ++i)
    close_enough(accumulated_v[i], initial_v[i] + expected.grad_v[i], 1e-4, 1e-4,
                 "accumulated grad v", i);
  std::printf("%s %s forward and VJP passed (V=%d; %lld edges; degrees 0/1/2/31/32/33/65)\n",
      std::is_same_v<T, __half> ? "FP16" : "FP32", fused ? "fused" : "staged",
      c.value_dim, static_cast<long long>(plan.edge_count));
}

void run_block_stride_case() {
  HostCase c;
  c.queries = 65536; c.keys = 1; c.dim = 1; c.value_dim = 1; c.scale = 1.0;
  c.offsets.resize(static_cast<size_t>(c.queries) + 1);
  c.key_ids.assign(c.queries, 0);
  for (int row = 0; row <= c.queries; ++row) c.offsets[row] = row;
  c.q.assign(c.queries, 0.25); c.k = {0.75}; c.v = {-0.5};
  c.grad_output.assign(c.queries, 0.125);
  const auto expected = reference(c, c.q, c.k, c.v, c.grad_output);
  DeviceBuffer<float> q(c.q.size()), k(c.k.size()), v(c.v.size()), out(c.queries);
  DeviceBuffer<std::int64_t> offsets(c.offsets.size()), keys(c.key_ids.size());
  DeviceBuffer<float> logits(c.key_ids.size()), probabilities(c.key_ids.size());
  q.upload(encode<float>(c.q)); k.upload(encode<float>(c.k)); v.upload(encode<float>(c.v));
  offsets.upload(c.offsets); keys.upload(c.key_ids);
  SparseAttentionPlan plan{offsets.pointer, keys.pointer, c.queries, c.keys,
      static_cast<std::int64_t>(c.key_ids.size()), c.dim, c.value_dim};
  SparseAttentionWorkspace workspace{logits.pointer, logits.count,
      probabilities.pointer, probabilities.count, nullptr, 0, nullptr, 0};
  CUDA_CHECK(Scalar<float>::mixed(q.pointer, q.count, k.pointer, k.count,
      v.pointer, v.count, out.pointer, out.count, plan, 1.0f, workspace, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto actual = out.download();
  for (size_t i = 0; i < actual.size(); ++i)
    close_enough(actual[i], expected.output[i], 1e-6, 1e-6, "block-stride output", i);
  std::puts("fused block-stride case passed (Q=65536; degree=1; V=1)");
}

HostCase make_timing_case(bool mixed_degree) {
  HostCase c;
  c.queries = 512; c.keys = 96; c.dim = 32; c.value_dim = 32; c.scale = 0.1767766953;
  c.offsets = {0};
  for (int row = 0; row < c.queries; ++row) {
    const int degree = mixed_degree && (row % 16 == 0) ? 65 : 8;
    for (int e = 0; e < degree; ++e) c.key_ids.push_back((row * 11 + e * 7) % c.keys);
    c.offsets.push_back(static_cast<std::int64_t>(c.key_ids.size()));
  }
  c.q.resize(c.queries * c.dim); c.k.resize(c.keys * c.dim);
  c.v.resize(c.keys * c.value_dim); c.grad_output.resize(c.queries * c.value_dim);
  for (size_t i = 0; i < c.q.size(); ++i) c.q[i] = patterned(i, 23, 0.025);
  for (size_t i = 0; i < c.k.size(); ++i) c.k[i] = patterned(i + 3, 19, 0.029);
  for (size_t i = 0; i < c.v.size(); ++i) c.v[i] = patterned(i + 5, 17, 0.031);
  return c;
}

struct EventPair {
  cudaEvent_t start = nullptr, stop = nullptr;
  EventPair() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
  ~EventPair() { if (start) (void)cudaEventDestroy(start); if (stop) (void)cudaEventDestroy(stop); }
};

template<class T, cudaError_t (*Entry)(const T*, size_t, const T*, size_t,
    const T*, size_t, T*, size_t, SparseAttentionPlan, float,
    SparseAttentionWorkspace, cudaStream_t)>
float time_route(const T* q, size_t qc, const T* k, size_t kc, const T* v,
                 size_t vc, T* output, size_t oc, SparseAttentionPlan plan,
                 SparseAttentionWorkspace workspace, int repetitions) {
  for (int warm = 0; warm < 4; ++warm)
    CUDA_CHECK(Entry(q, qc, k, kc, v, vc, output, oc, plan,
                     static_cast<float>(0.1767766953), workspace, nullptr));
  EventPair events;
  CUDA_CHECK(cudaEventRecord(events.start));
  for (int repeat = 0; repeat < repetitions; ++repeat)
    CUDA_CHECK(Entry(q, qc, k, kc, v, vc, output, oc, plan,
                     static_cast<float>(0.1767766953), workspace, nullptr));
  CUDA_CHECK(cudaEventRecord(events.stop));
  CUDA_CHECK(cudaEventSynchronize(events.stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, events.start, events.stop));
  return milliseconds * 1000.0f / repetitions;
}

std::pair<double, double> median_mad(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const double median = values[values.size() / 2];
  for (auto& value : values) value = std::abs(value - median);
  std::sort(values.begin(), values.end());
  return {median, values[values.size() / 2]};
}

void benchmark_case(bool mixed_degree) {
  auto c = make_timing_case(mixed_degree);
  const auto qh = encode<float>(c.q), kh = encode<float>(c.k), vh = encode<float>(c.v);
  DeviceBuffer<float> q(qh.size()), k(kh.size()), v(vh.size()), staged_output(c.queries * c.value_dim),
      mixed_output(c.queries * c.value_dim), logits(c.key_ids.size()), probabilities(c.key_ids.size());
  DeviceBuffer<std::int64_t> offsets(c.offsets.size()), keys(c.key_ids.size());
  q.upload(qh); k.upload(kh); v.upload(vh); offsets.upload(c.offsets); keys.upload(c.key_ids);
  SparseAttentionPlan plan{offsets.pointer, keys.pointer, c.queries, c.keys,
      static_cast<std::int64_t>(c.key_ids.size()), c.dim, c.value_dim};
  SparseAttentionWorkspace workspace{logits.pointer, logits.count, probabilities.pointer,
      probabilities.count, nullptr, 0, nullptr, 0};

  constexpr int samples = 19, repetitions = 25;
  std::vector<double> staged_times, mixed_times;
  for (int sample = 0; sample < samples; ++sample) {
    const auto staged = time_route<float, Scalar<float>::staged>(
        q.pointer, q.count, k.pointer, k.count, v.pointer, v.count,
        staged_output.pointer, staged_output.count, plan, workspace, repetitions);
    const auto mixed = time_route<float, Scalar<float>::mixed>(
        q.pointer, q.count, k.pointer, k.count, v.pointer, v.count,
        mixed_output.pointer, mixed_output.count, plan, workspace, repetitions);
    staged_times.push_back(staged); mixed_times.push_back(mixed);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto base = staged_output.download(), candidate = mixed_output.download();
  for (size_t i = 0; i < base.size(); ++i)
    close_enough(candidate[i], base[i], 3e-5, 3e-5, "timing candidate output", i);
  const auto [staged_median, staged_mad] = median_mad(staged_times);
  const auto [mixed_median, mixed_mad] = median_mad(mixed_times);
  std::printf("timing resident FP32 Q=%d N=%d D=%d V=%d edges=%zu degree=%s: "
      "staged %.3f +/- %.3f us, mixed-fused %.3f +/- %.3f us, ratio %.3fx\n",
      c.queries, c.keys, c.dim, c.value_dim, c.key_ids.size(),
      mixed_degree ? "8+65" : "8", staged_median, staged_mad, mixed_median,
      mixed_mad, staged_median / mixed_median);
}

void benchmark() {
  benchmark_case(false);
  benchmark_case(true);
}

void check_capacity_validation() {
  SparseAttentionPlan plan{};
  SparseAttentionWorkspace workspace{};
  plan.query_count = 1; plan.key_count = 1; plan.edge_count = 1;
  plan.query_key_dim = 1; plan.value_dim = 1;
  float dummy = 0.0f;
  assert(cellerator::modeling::cuda::sparse_attention_fp32(
      &dummy, 1, &dummy, 1, &dummy, 1, &dummy, 1, plan, 1.0f, workspace, nullptr)
      == cudaErrorInvalidValue);
  plan.edge_count = 0;
  plan.query_count = -1;
  assert(cellerator::modeling::cuda::sparse_attention_fp32(
      &dummy, 1, &dummy, 1, &dummy, 1, &dummy, 1, plan, 1.0f, workspace, nullptr)
      == cudaErrorInvalidValue);
}

void host_reference() {
  const auto c = make_boundary_case();
  const auto expected = reference(c, c.q, c.k, c.v, c.grad_output);
  assert(expected.output.size() == static_cast<size_t>(c.queries * c.value_dim));
  assert(expected.output[0] == 0.0);  // empty row
  for (const auto p : expected.probabilities) assert(std::isfinite(p));

  // Independently check the analytic VJP against central differences of the
  // scalar loss for active q, repeated-key k, and v coordinates.
  constexpr double epsilon = 1e-6;
  for (const auto index : {8u, 13u}) {
    auto plus = c.q, minus = c.q;
    plus[index] += epsilon; minus[index] -= epsilon;
    const auto finite_difference = (objective(c, plus, c.k, c.v) -
                                    objective(c, minus, c.k, c.v)) / (2 * epsilon);
    close_enough(finite_difference, expected.grad_q[index], 2e-8, 2e-7,
                 "oracle finite-difference dQ", index);
  }
  for (const auto index : {8u, 43u}) {
    auto plus = c.k, minus = c.k;
    plus[index] += epsilon; minus[index] -= epsilon;
    const auto finite_difference = (objective(c, c.q, plus, c.v) -
                                    objective(c, c.q, minus, c.v)) / (2 * epsilon);
    close_enough(finite_difference, expected.grad_k[index], 2e-8, 2e-7,
                 "oracle finite-difference dK", index);
  }
  for (const auto index : {2u, 17u}) {
    auto plus = c.v, minus = c.v;
    plus[index] += epsilon; minus[index] -= epsilon;
    const auto finite_difference = (objective(c, c.q, c.k, plus) -
                                    objective(c, c.q, c.k, minus)) / (2 * epsilon);
    close_enough(finite_difference, expected.grad_v[index], 2e-8, 2e-7,
                 "oracle finite-difference dV", index);
  }
  check_capacity_validation();
}
}  // namespace

int main(int argc, char** argv) {
  host_reference();
  if (argc == 1 || std::string(argv[1]) == "--host-only") {
    std::puts("attention host oracle and metadata checks passed; no GPU launched");
    return 0;
  }
  if (std::string(argv[1]) == "--benchmark") {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("attention benchmark on %s, compute capability %d.%d\n",
                properties.name, properties.major, properties.minor);
    benchmark();
    return 0;
  }
  if (std::string(argv[1]) != "--device") {
    std::fprintf(stderr, "usage: %s [--host-only|--device|--benchmark]\n", argv[0]);
    return 2;
  }
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("attention device run on %s, compute capability %d.%d\n",
              properties.name, properties.major, properties.minor);
  run_device_case<float>(false);
  run_device_case<float>(true);
  for (const int value_dim : {1, 31, 32, 33, 65})
    run_device_case<float>(true, value_dim);
  run_device_case<__half>(false);
  run_device_case<__half>(true);
  for (const int value_dim : {1, 31, 32, 33, 65})
    run_device_case<__half>(true, value_dim);
  run_block_stride_case();
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
