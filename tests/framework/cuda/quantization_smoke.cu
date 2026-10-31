#include <Cellerator/modeling/cuda/quantization.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif

namespace m = cellerator::modeling::cuda;

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T>
struct DeviceBuffer {
  T* data{};
  std::size_t size{};
  explicit DeviceBuffer(std::size_t count) : size(count) {
    if (count != 0) check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
  }
  ~DeviceBuffer() { if (data != nullptr) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void upload(const std::vector<T>& host, cudaStream_t stream) {
    assert(host.size() <= size);
    if (!host.empty()) check(cudaMemcpyAsync(data, host.data(), host.size() * sizeof(T),
                                             cudaMemcpyHostToDevice, stream));
  }
  std::vector<T> download(cudaStream_t stream) const {
    std::vector<T> host(size);
    if (size != 0) check(cudaMemcpyAsync(host.data(), data, size * sizeof(T),
                                         cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host;
  }
  std::vector<T> download_prefix(std::size_t count, cudaStream_t stream) const {
    assert(count <= size);
    std::vector<T> host(count);
    if (count != 0) check(cudaMemcpyAsync(host.data(), data, count * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host;
  }
};

struct Profile {
  std::vector<std::uint32_t> offsets{0, 4, 7, 8};
  std::vector<float> levels{0.0f, 0.25f, 0.5f, 1.0f,
                            -1.0f, 0.0f, 1.0f, 2.0f};
  // Physical feature order is profile feature 2, 0, 1.
  std::vector<std::uint32_t> feature_map{2, 0, 1};
};

std::uint32_t nearest(float value, const float* begin, std::uint32_t count) {
  std::uint32_t best = 0;
  float distance = std::abs(value - begin[0]);
  for (std::uint32_t i = 1; i < count; ++i) {
    const float candidate = std::abs(value - begin[i]);
    if (candidate < distance) {
      best = i;
      distance = candidate;
    }
  }
  return best;  // strict comparison makes ties prefer the lower codebook ordinal
}

std::vector<std::uint32_t> oracle_codes(const Profile& p, unsigned bits,
                                        const std::vector<float>& input) {
  const std::uint32_t mask = (1U << bits) - 1U;
  std::vector<std::uint32_t> result(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto physical = static_cast<std::uint32_t>(i % p.feature_map.size());
    const auto feature = p.feature_map[physical];
    const auto first = p.offsets[feature];
    const auto levels = p.offsets[feature + 1] - first;
    assert(levels <= (1U << bits));
    result[i] = nearest(input[i], p.levels.data() + first, levels) & mask;
  }
  return result;
}

std::uint32_t oracle_code(const Profile& p, unsigned bits,
                          std::uint32_t physical_feature, float value) {
  const auto feature = p.feature_map[physical_feature];
  const auto begin = p.offsets[feature];
  const auto count = p.offsets[feature + 1] - begin;
  assert(count <= (1U << bits));
  return nearest(value, p.levels.data() + begin, count);
}

std::vector<std::uint32_t> pack_oracle(const std::vector<std::uint32_t>& codes,
                                      unsigned bits) {
  const std::uint32_t per_word = 32U / bits;
  std::vector<std::uint32_t> words((codes.size() + per_word - 1U) / per_word, 0);
  for (std::size_t i = 0; i < codes.size(); ++i)
    words[i / per_word] |= codes[i] << ((i % per_word) * bits);
  return words;
}

std::vector<float> decode_oracle(const Profile& p, const std::vector<std::uint32_t>& codes,
                                 const std::vector<float>& input,
                                 const std::vector<std::uint32_t>& overrides) {
  std::vector<float> output(codes.size());
  for (std::size_t i = 0; i < codes.size(); ++i) {
    const auto feature = p.feature_map[i % p.feature_map.size()];
    output[i] = p.levels[p.offsets[feature] + codes[i]];
  }
  for (auto index : overrides) output[index] = input[index];
  return output;
}

template <typename T>
float as_float(T value) { return static_cast<float>(value); }

template <>
float as_float(__half value) { return __half2float(value); }

template <typename T>
T from_float(float value) { return static_cast<T>(value); }

template <>
__half from_float(float value) { return __float2half_rn(value); }

template <typename T>
void run_case(unsigned bits, bool short_profile = false) {
  constexpr std::uint32_t rows = 5;
  constexpr std::uint32_t features = 3;
  constexpr std::uint32_t count = rows * features;  // 15: partial word for every bit width
  Profile profile;
  if (short_profile) {
    // One-bit quantization requires at most two strictly ordered levels/feature.
    profile.offsets = {0, 2, 4, 5};
    profile.levels = {0.0f, 1.0f, -1.0f, 1.0f, 2.0f};
  }
  const std::vector<float> input32{
      0.1f, 0.125f, -0.8f, 0.2f, 0.4f, 0.7f, 0.9f, 0.0f,
      0.25f, 0.49f, 0.8f, 0.0f, 1.0f, 0.7f, -0.3f};
  std::vector<T> input(input32.size());
  std::transform(input32.begin(), input32.end(), input.begin(),
                 [](float value) { return from_float<T>(value); });
  std::vector<float> represented(input.size());
  std::transform(input.begin(), input.end(), represented.begin(), as_float<T>);

  const auto codes = oracle_codes(profile, bits, represented);
  const auto expected_words = pack_oracle(codes, bits);
  const std::vector<std::uint32_t> override_indices{1, 8, 14};
  std::vector<std::uint8_t> mask(count, 0);
  for (auto i : override_indices) mask[i] = 1;

  const auto words = expected_words.size();
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    DeviceBuffer<std::uint32_t> offsets(profile.offsets.size());
    DeviceBuffer<float> levels(profile.levels.size());
    DeviceBuffer<std::uint32_t> feature_map(profile.feature_map.size());
    DeviceBuffer<T> values(count), output(count), edit_values(5);
    DeviceBuffer<std::uint8_t> exact_mask(count);
    DeviceBuffer<std::uint32_t> packed(words), override_positions(count), override_count(1), status(1);
    offsets.upload(profile.offsets, stream);
    levels.upload(profile.levels, stream);
    feature_map.upload(profile.feature_map, stream);
    values.upload(input, stream);
    exact_mask.upload(mask, stream);
    const m::QuantizationProfileView view{
        offsets.data, levels.data, feature_map.data,
        static_cast<std::uint32_t>(profile.offsets.size() - 1), features,
        static_cast<std::uint32_t>(profile.levels.size()), bits};
    const m::DeviceStatus device_status{status.data};
    m::PackedStateView packed_view{packed.data, packed.size};
    m::OverrideView override_view{override_positions.data, nullptr, override_count.data, count};
    // The sidecar stores original FP32 values even when the input is FP16.
    DeviceBuffer<float> override_values(count);
    override_view.values = override_values.data;
    const m::ConstOverrideView const_override{
        override_positions.data, override_values.data, override_count.data, count};

    check(m::reset_status(device_status, stream));
    check(m::validate_profile(view, device_status, stream));
    check(m::reset_override_count(override_view, stream));
    if constexpr (std::is_same_v<T, float>)
      check(m::encode_state_fp32(values.data, rows, view, packed_view, exact_mask.data,
                                 override_view, device_status, stream));
    else
      check(m::encode_state_fp16(values.data, rows, view, packed_view, exact_mask.data,
                                 override_view, device_status, stream));
    assert(packed.download(stream) == expected_words);
    const auto written_overrides = override_count.download(stream)[0];
    assert(written_overrides == override_indices.size());
    // Only entries below the device-produced count are initialized. Reading
    // the rest of the reserved sidecar would be an invalid test-side read.
    const auto positions = override_positions.download_prefix(written_overrides, stream);
    const auto exact_values = override_values.download_prefix(written_overrides, stream);
    std::vector<std::uint32_t> found;
    for (std::uint32_t i = 0; i < override_indices.size(); ++i) {
      found.push_back(positions[i]);
      assert(exact_values[i] == represented[positions[i]]);
    }
    std::sort(found.begin(), found.end());
    assert(found == override_indices);

    if constexpr (std::is_same_v<T, float>)
      check(m::decode_state_fp32({packed.data, packed.size}, rows, view,
                                 const_override, output.data, device_status, stream));
    else
      check(m::decode_state_fp16({packed.data, packed.size}, rows, view,
                                 const_override, output.data, device_status, stream));
    auto expected_output = decode_oracle(profile, codes, represented, override_indices);
    const auto actual_output = output.download(stream);
    for (std::size_t i = 0; i < actual_output.size(); ++i) {
      const auto expected = from_float<T>(expected_output[i]);
      assert(as_float(actual_output[i]) == as_float(expected));
    }

    // Distinct adjacent indices share packed words; CAS must preserve all edits.
    const std::vector<std::uint32_t> updates{0, 1, 7, 8, 14};
    const std::vector<float> update_float{1.0f, 0.0f, -0.6f, 0.5f, 1.8f};
    std::vector<T> typed_updates(update_float.size());
    std::transform(update_float.begin(), update_float.end(), typed_updates.begin(),
                   [](float value) { return from_float<T>(value); });
    DeviceBuffer<std::uint32_t> update_indices(updates.size());
    update_indices.upload(updates, stream);
    edit_values.upload(typed_updates, stream);
    if constexpr (std::is_same_v<T, float>)
      check(m::update_codes_fp32(update_indices.data, edit_values.data,
                                 static_cast<std::uint32_t>(updates.size()), rows,
                                 view, packed_view, device_status, stream));
    else
      check(m::update_codes_fp16(update_indices.data, edit_values.data,
                                 static_cast<std::uint32_t>(updates.size()), rows,
                                 view, packed_view, device_status, stream));

    auto updated_codes = codes;
    for (std::size_t u = 0; u < updates.size(); ++u)
      updated_codes[updates[u]] = oracle_code(
          profile, bits, updates[u] % features, as_float(typed_updates[u]));
    const auto updated_output = decode_oracle(profile, updated_codes, represented, override_indices);
    if constexpr (std::is_same_v<T, float>)
      check(m::decode_state_fp32({packed.data, packed.size}, rows, view,
                                 const_override, output.data, device_status, stream));
    else
      check(m::decode_state_fp16({packed.data, packed.size}, rows, view,
                                 const_override, output.data, device_status, stream));
    const auto after = output.download(stream);
    for (std::size_t i = 0; i < after.size(); ++i)
      assert(as_float(after[i]) == as_float(from_float<T>(updated_output[i])));

    assert((status.download(stream)[0] & 7U) == 0U);
    std::cout << "quantization bit width " << bits << " dtype "
              << (std::is_same_v<T, float> ? "fp32" : "fp16")
              << (short_profile ? " short-profile" : " profile")
              << ": encode/decode/overrides/shared-word updates passed\n";
  }
  check(cudaStreamDestroy(stream));
}

void test_invalid_values_and_profiles() {
  Profile profile;
  constexpr unsigned bits = 4;
  constexpr std::uint32_t rows = 1;
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    DeviceBuffer<std::uint32_t> offsets(profile.offsets.size()), map(profile.feature_map.size());
    DeviceBuffer<float> levels(profile.levels.size()), input(3);
    DeviceBuffer<std::uint32_t> packed(1), status(1);
    offsets.upload(profile.offsets, stream);
    levels.upload(profile.levels, stream);
    map.upload(profile.feature_map, stream);
    const m::QuantizationProfileView view{offsets.data, levels.data, map.data, 3, 3, 8, bits};
    const m::PackedStateView output{packed.data, packed.size};
    const m::DeviceStatus device_status{status.data};
    std::vector<float> invalid{0.2f, std::numeric_limits<float>::quiet_NaN(), 1.0f};
    input.upload(invalid, stream);
    check(m::reset_status(device_status, stream));
    check(m::validate_profile(view, device_status, stream));
    check(m::encode_state_fp32(input.data, rows, view, output, nullptr, {}, device_status, stream));
    const auto bad_value_status = status.download(stream)[0];
    assert((bad_value_status & 4U) != 0U);

    // Invalid host arguments reject immediately without touching device memory.
    assert(m::encode_state_fp32(input.data, rows, view, {packed.data, 0}, nullptr,
                                {}, device_status, stream) == cudaErrorInvalidValue);
    assert(m::update_codes_fp32(nullptr, nullptr, 1, rows, view, output,
                                device_status, stream) == cudaErrorInvalidValue);
    assert(m::decode_state_fp32({}, rows, view, {}, nullptr,
                                device_status, stream) == cudaErrorInvalidValue);
  }
  check(cudaStreamDestroy(stream));
}

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count));
  if (count == 0) return 77;
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "expected a V100 (sm_70), got " << properties.name
              << " sm_" << properties.major << properties.minor << '\n';
    return 2;
  }
  for (unsigned bits : {1U, 2U, 4U, 8U}) {
    const bool short_profile = bits == 1U;
    run_case<float>(bits, short_profile);
    run_case<__half>(bits, short_profile);
  }
  test_invalid_values_and_profiles();
  std::cout << "V100 quantization correctness passed\n";
  return 0;
}
