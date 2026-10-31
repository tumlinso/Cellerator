#include <Cellerator/modeling/cuda/packed_embedding.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace cuda = cellerator::modeling::cuda;

#define CUDA_CHECK(expression) do { \
  const auto status_ = (expression); \
  if (status_ != cudaSuccess) { \
    std::cerr << #expression << ": " << cudaGetErrorString(status_) << '\n'; \
    std::abort(); \
  } \
} while (false)

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t count) : count_(count) {
    if (count_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), count_ * sizeof(T)));
  }
  ~Buffer() { if (pointer_) cudaFree(pointer_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  void upload(const std::vector<T>& values, cudaStream_t stream) {
    assert(values.size() == count_);
    if (count_) CUDA_CHECK(cudaMemcpyAsync(pointer_, values.data(), count_ * sizeof(T),
                                            cudaMemcpyHostToDevice, stream));
  }
  std::vector<T> download(cudaStream_t stream) const {
    std::vector<T> values(count_);
    if (count_) CUDA_CHECK(cudaMemcpyAsync(values.data(), pointer_, count_ * sizeof(T),
                                            cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return values;
  }
  void zero(cudaStream_t stream) {
    if (count_) CUDA_CHECK(cudaMemsetAsync(pointer_, 0, count_ * sizeof(T), stream));
  }
 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

void close(float actual, float expected, float tolerance, const std::string& context) {
  if (!std::isfinite(actual) || std::abs(actual - expected) >
      tolerance * (1.0f + std::abs(expected))) {
    std::cerr << context << " actual=" << actual << " expected=" << expected << '\n';
    std::abort();
  }
}

float cast_to_table(float value, cuda::DType dtype) {
  return dtype == cuda::DType::float16
      ? __half2float(__float2half(value)) : value;
}

struct Fixture {
  std::uint32_t code_bits;
  std::uint32_t dimension;
  cuda::DType dtype;
  std::uint32_t row_count = 3;
  std::uint32_t state_features = 3;
  std::uint32_t segment_count = 4;
  std::vector<std::uint32_t> feature_map{1, 0, 1};
  std::vector<std::uint32_t> codebook_offsets{0};
  std::vector<std::uint32_t> codebook_sizes;
  std::vector<float> levels;
  std::vector<std::uint64_t> table_offsets{0U};
  std::vector<std::uint32_t> codes;
  std::vector<std::uint32_t> packed_words;
  std::vector<std::uint32_t> override_indices{1, 7};
  std::vector<float> override_values{0.25f, 0.75f};
  std::vector<std::uint32_t> occurrence_indices{0, 1, 1, 7, 8, 4, 2, 5};
  std::vector<float> weights{1.0f, -0.5f, 2.0f, 0.0f, 1.25f, -2.0f, 0.1f, 3.0f};
  std::vector<std::uint32_t> segment_offsets{0, 3, 3, 6, 8};
  std::vector<float> table;
  std::vector<float> output_gradient;
  std::vector<float> expected_output;
  std::vector<float> expected_table_gradient;
  std::vector<float> expected_weight_gradient;

  Fixture(std::uint32_t bits, std::uint32_t dim, cuda::DType value_type,
          std::uint32_t segments = 4U, std::uint32_t occurrence_count = 8U)
      : code_bits(bits), dimension(dim), dtype(value_type),
        segment_count(segments) {
    if (segments != 4U || occurrence_count != 8U) {
      occurrence_indices.clear();
      weights.clear();
      segment_offsets.resize(static_cast<std::size_t>(segments) + 1U);
      for (std::uint32_t segment = 0; segment <= segments; ++segment)
        segment_offsets[segment] =
            static_cast<std::uint64_t>(occurrence_count) * segment / segments;
      for (std::uint32_t i = 0; i < occurrence_count; ++i) {
        occurrence_indices.push_back(i % (row_count * state_features));
        weights.push_back((static_cast<int>(i % 17U) - 8) * 0.125f);
      }
    }
    const std::uint32_t capacity = 1U << code_bits;
    const std::uint32_t maximum_levels[] = {2U, 3U};
    const float all_levels[][3] = {{0.0f, 1.0f, 0.0f},
                                   {0.0f, 2.0f, 4.0f}};
    for (std::uint32_t f = 0; f < 2; ++f) {
      codebook_sizes.push_back(std::min(maximum_levels[f], capacity));
      for (std::uint32_t level = 0; level < codebook_sizes.back(); ++level)
        levels.push_back(all_levels[f][level]);
      codebook_offsets.push_back(codebook_offsets.back() + codebook_sizes.back());
      table_offsets.push_back(table_offsets.back() +
                              static_cast<std::uint64_t>(codebook_sizes[f]) * dimension);
    }
    const std::uint64_t table_size = table_offsets.back();
    table.resize(table_size);
    for (std::uint64_t i = 0; i < table_size; ++i)
      table[i] = cast_to_table((static_cast<int>(i % 17U) - 8) * 0.07f, dtype);

    const std::uint32_t per_word = 32U / code_bits;
    const std::uint32_t total = row_count * state_features;
    const std::uint32_t word_count = (total + per_word - 1U) / per_word;
    packed_words.assign(word_count, 0U);
    for (std::uint32_t logical = 0; logical < total; ++logical) {
      const auto feature = feature_map[logical % state_features];
      const auto count = codebook_sizes[feature];
      const auto code = logical % count;
      codes.push_back(code);
      packed_words[logical / per_word] |= code << ((logical % per_word) * code_bits);
    }
    output_gradient.resize(segment_count * dimension);
    for (std::uint32_t i = 0; i < output_gradient.size(); ++i)
      output_gradient[i] = cast_to_table((static_cast<int>(i % 9U) - 4) * 0.11f, dtype);
    reference();
  }

  void reference() {
    expected_output.assign(segment_count * dimension, 0.0f);
    expected_table_gradient.assign(table.size(), 0.0f);
    expected_weight_gradient.assign(weights.size(), 0.0f);
    for (std::uint32_t segment = 0; segment < segment_count; ++segment) {
      for (std::uint32_t o = segment_offsets[segment]; o < segment_offsets[segment + 1]; ++o) {
        const auto logical = occurrence_indices[o];
        const auto feature = feature_map[logical % state_features];
        const auto begin = codebook_offsets[feature];
        const auto code = codes[logical];
        std::uint32_t low = code, high = code;
        float mix = 0.0f;
        for (std::size_t i = 0; i < override_indices.size(); ++i) {
          if (override_indices[i] != logical) continue;
          const float value = override_values[i];
          const auto end = codebook_offsets[feature + 1U];
          const auto n = end - begin;
          if (value <= levels[begin]) low = high = 0;
          else if (value >= levels[end - 1U]) low = high = n - 1U;
          else {
            high = 0;
            while (levels[begin + high] < value) ++high;
            low = high - 1U;
            mix = (value - levels[begin + low]) /
                  (levels[begin + high] - levels[begin + low]);
          }
          break;
        }
        const auto base = table_offsets[feature];
        for (std::uint32_t d = 0; d < dimension; ++d) {
          const auto lo_index = base + static_cast<std::uint64_t>(low) * dimension + d;
          const auto hi_index = base + static_cast<std::uint64_t>(high) * dimension + d;
          const auto embedding = table[lo_index] * (1.0f - mix) + table[hi_index] * mix;
          expected_output[segment * dimension + d] += weights[o] * embedding;
          const auto dy = output_gradient[segment * dimension + d];
          expected_weight_gradient[o] += dy * embedding;
          expected_table_gradient[lo_index] += dy * weights[o] *
              (low == high ? 1.0f : 1.0f - mix);
          if (low != high)
            expected_table_gradient[hi_index] += dy * weights[o] * mix;
        }
      }
    }
    for (auto& value : expected_output) value = cast_to_table(value, dtype);
  }
};

template <typename TableElement>
void run_case(const Fixture& fixture, bool benchmark = false) {
  const auto case_label = std::string("packed_embedding bits=") +
      std::to_string(fixture.code_bits) + " dtype=" +
      (std::is_same_v<TableElement, __half> ? "fp16" : "fp32") +
      " dim=" + std::to_string(fixture.dimension) +
      " rows=" + std::to_string(fixture.row_count) +
      " state_features=" + std::to_string(fixture.state_features) +
      " segments=" + std::to_string(fixture.segment_count) +
      " occurrences=" + std::to_string(fixture.occurrence_indices.size());
  const auto stream = cuda::Stream(0);
  const std::uint32_t override_count = static_cast<std::uint32_t>(fixture.override_indices.size());
  const std::vector<std::uint32_t> host_override_count{override_count};
  Buffer<std::uint32_t> d_offsets(fixture.codebook_offsets.size()), d_features(fixture.feature_map.size());
  Buffer<float> d_levels(fixture.levels.size()), d_overrides(fixture.override_values.size());
  Buffer<std::uint32_t> d_override_indices(fixture.override_indices.size()), d_override_count(1);
  Buffer<std::uint32_t> d_words(fixture.packed_words.size());
  Buffer<std::uint64_t> d_table_offsets(fixture.table_offsets.size());
  const auto& table_offsets = fixture.table_offsets;
  Buffer<TableElement> d_table(fixture.table.size()), d_output_gradient(fixture.output_gradient.size());
  Buffer<std::uint32_t> d_occurrences(fixture.occurrence_indices.size()), d_segments(fixture.segment_offsets.size());
  Buffer<float> d_weights(fixture.weights.size());
  Buffer<TableElement> d_output(fixture.expected_output.size());
  Buffer<float> d_table_gradient(fixture.table.size()), d_weight_gradient(fixture.weights.size());
  Buffer<std::uint32_t> d_status(1);

  auto cast_table = [](float v) -> TableElement {
    if constexpr (std::is_same_v<TableElement, __half>) return __float2half(v);
    else return v;
  };
  std::vector<TableElement> table(fixture.table.size()), dy(fixture.output_gradient.size());
  for (std::size_t i = 0; i < table.size(); ++i) table[i] = cast_table(fixture.table[i]);
  for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = cast_table(fixture.output_gradient[i]);
  d_offsets.upload(fixture.codebook_offsets, stream.get()); d_features.upload(fixture.feature_map, stream.get());
  d_levels.upload(fixture.levels, stream.get()); d_overrides.upload(fixture.override_values, stream.get());
  d_override_indices.upload(fixture.override_indices, stream.get());
  d_override_count.upload(host_override_count, stream.get());
  d_words.upload(fixture.packed_words, stream.get()); d_table_offsets.upload(table_offsets, stream.get());
  d_table.upload(table, stream.get()); d_output_gradient.upload(dy, stream.get());
  d_occurrences.upload(fixture.occurrence_indices, stream.get()); d_segments.upload(fixture.segment_offsets, stream.get());
  d_weights.upload(fixture.weights, stream.get());
  d_table_gradient.zero(stream.get());

  cuda::QuantizationProfileView profile{
    d_offsets.get(), d_levels.get(), d_features.get(), 2, fixture.state_features,
    static_cast<std::uint32_t>(fixture.levels.size()), fixture.code_bits};
  cuda::ConstOverrideView overrides{d_override_indices.get(), d_overrides.get(),
                                      d_override_count.get(), override_count};
  cuda::PackedOccurrenceView occurrences{d_occurrences.get(), fixture.occurrence_indices.size(),
      d_weights.get(), fixture.weights.size(), d_segments.get(), fixture.segment_offsets.size(),
      static_cast<std::uint32_t>(fixture.occurrence_indices.size()), fixture.segment_count};
  cuda::PackedEmbeddingTableView table_view{d_table.get(), fixture.table.size(),
      d_table_offsets.get(), table_offsets.size(), fixture.dimension,
      std::is_same_v<TableElement, __half> ? cuda::DType::float16 : cuda::DType::float32};
  cuda::DeviceStatus status{d_status.get()};
  CUDA_CHECK(cuda::reset_status(status, stream.get()));
  CUDA_CHECK(cuda::validate_profile(profile, status, stream.get()));
  CUDA_CHECK(cuda::packed_embedding_bag_forward(
      {d_words.get(), fixture.packed_words.size()}, fixture.row_count, profile, overrides, true,
      occurrences, table_view, {d_output.get(), fixture.expected_output.size() * sizeof(TableElement), 0},
      status, stream.get()));
  const auto forward_status = d_status.download(stream.get());
  if (forward_status[0] != 0U) {
    std::cerr << case_label << " stage=forward status_bits=" << forward_status[0] << '\n';
    std::abort();
  }
  const float tolerance = std::is_same_v<TableElement, __half> ? 2e-3f : 2e-5f;
  const auto output = d_output.download(stream.get());
  for (std::size_t i = 0; i < output.size(); ++i)
    close(static_cast<float>(output[i]), fixture.expected_output[i], tolerance,
          case_label + " stage=forward index=" + std::to_string(i) +
          " segment=" + std::to_string(i / fixture.dimension) +
          " dimension=" + std::to_string(i % fixture.dimension));

  cuda::PackedEmbeddingGradientView gradients{
    d_table_gradient.get(), fixture.table.size(),
    d_weight_gradient.get(), fixture.weights.size()};
  CUDA_CHECK(cuda::packed_embedding_bag_backward(
      {d_words.get(), fixture.packed_words.size()}, fixture.row_count, profile, overrides, true,
      occurrences, table_view,
      {d_output_gradient.get(), fixture.output_gradient.size() * sizeof(TableElement), 0},
      gradients, status, stream.get()));
  const auto table_gradient = d_table_gradient.download(stream.get());
  const auto weight_gradient = d_weight_gradient.download(stream.get());
  const auto status_bits = d_status.download(stream.get());
  assert(status_bits[0] == 0U);
  for (std::size_t i = 0; i < table_gradient.size(); ++i)
    close(table_gradient[i], fixture.expected_table_gradient[i], tolerance * 2.0f,
          case_label + " stage=table_gradient index=" + std::to_string(i));
  for (std::size_t i = 0; i < weight_gradient.size(); ++i)
    close(weight_gradient[i], fixture.expected_weight_gradient[i], tolerance * 2.0f,
          case_label + " stage=weight_gradient occurrence=" + std::to_string(i));

  if (benchmark) {
    constexpr int kIterations = 30;
    cudaEvent_t begin = nullptr, end = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin, stream.get()));
    for (int i = 0; i < kIterations; ++i)
      CUDA_CHECK(cuda::packed_embedding_bag_forward(
          {d_words.get(), fixture.packed_words.size()}, fixture.row_count,
          profile, overrides, true, occurrences, table_view,
          {d_output.get(), fixture.expected_output.size() * sizeof(TableElement), 0},
          status, stream.get()));
    CUDA_CHECK(cudaEventRecord(end, stream.get()));
    CUDA_CHECK(cudaEventSynchronize(end));
    float forward_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&forward_ms, begin, end));

    CUDA_CHECK(cudaEventRecord(begin, stream.get()));
    for (int i = 0; i < kIterations; ++i)
      CUDA_CHECK(cuda::packed_embedding_bag_backward(
          {d_words.get(), fixture.packed_words.size()}, fixture.row_count,
          profile, overrides, true, occurrences, table_view,
          {d_output_gradient.get(), fixture.output_gradient.size() * sizeof(TableElement), 0},
          gradients, status, stream.get()));
    CUDA_CHECK(cudaEventRecord(end, stream.get()));
    CUDA_CHECK(cudaEventSynchronize(end));
    float backward_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&backward_ms, begin, end));
    std::cout << "packed_embedding bits=" << fixture.code_bits
              << " dtype=" << (std::is_same_v<TableElement, __half> ? "fp16" : "fp32")
              << " segments=" << fixture.segment_count
              << " occurrences=" << fixture.occurrence_indices.size()
              << " dim=" << fixture.dimension
              << " forward_ms=" << forward_ms / kIterations
              << " backward_ms=" << backward_ms / kIterations << '\n';
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
  }
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--host-only") {
    std::cout << "packed embedding GPU oracle smoke requires a scheduler-admitted V100\n";
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--reference-only") {
    for (const auto bits : {1U, 2U, 4U, 8U})
      for (const auto dimension : {7U, 33U})
        for (const auto dtype : {cuda::DType::float16, cuda::DType::float32})
          (void)Fixture(bits, dimension, dtype);
    std::cout << "packed embedding host oracle fixture passed\n";
    return 0;
  }
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  assert(device_count > 0);
  const bool performance = argc == 2 && std::string(argv[1]) == "--performance";
  for (const auto bits : {1U, 2U, 4U, 8U}) {
    for (const auto dimension : {7U, 33U}) {
      for (const auto dtype : {cuda::DType::float16, cuda::DType::float32}) {
        const Fixture fixture(bits, dimension, dtype);
        if (dtype == cuda::DType::float16) run_case<__half>(fixture);
        else run_case<float>(fixture);
      }
    }
  }
  if (performance) {
    const Fixture fixture(4U, 128U, cuda::DType::float32, 512U, 16384U);
    run_case<float>(fixture, true);
  }
  std::cout << "packed embedding oracle smoke passed\n";
}
