#include <Cellerator/modeling/cuda/quantization.hh>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace m = cellerator::modeling::cuda;

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T>
struct Buffer {
  T* ptr{};
  explicit Buffer(std::size_t n) {
    if (n != 0) check(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
  }
  ~Buffer() { if (ptr != nullptr) cudaFree(ptr); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

float measure(cudaStream_t stream, const m::ConstPackedStateView& packed,
              std::uint32_t rows, const m::QuantizationProfileView& profile,
              m::ConstOverrideView sidecar, float* output,
              m::DeviceStatus status, int repetitions) {
  constexpr int samples = 9;
  std::array<float, samples> milliseconds{};
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  for (auto& elapsed : milliseconds) {
    check(cudaEventRecord(begin, stream));
    for (int i = 0; i < repetitions; ++i)
      check(m::decode_state_fp32(packed, rows, profile, sidecar, output, status, stream));
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float total_ms = 0.0f;
    check(cudaEventElapsedTime(&total_ms, begin, end));
    elapsed = total_ms * 1000.0f / repetitions;
  }
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  std::sort(milliseconds.begin(), milliseconds.end());
  return milliseconds[milliseconds.size() / 2];
}

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count));
  if (device_count == 0) return 77;
  cudaDeviceProp device{};
  check(cudaGetDeviceProperties(&device, 0));
  if (device.major != 7 || device.minor != 0) return 2;

  constexpr std::uint32_t bits = 4;
  constexpr std::uint32_t profile_features = 1;
  constexpr std::uint32_t state_features = 1;
  const std::vector<std::uint32_t> host_offsets{0, 16};
  std::vector<float> host_levels(16);
  for (std::uint32_t i = 0; i < host_levels.size(); ++i)
    host_levels[i] = static_cast<float>(i) / 15.0f;
  const std::vector<std::uint32_t> host_feature_map{0};
  Buffer<std::uint32_t> offsets(host_offsets.size()), feature_map(host_feature_map.size()), status(1);
  Buffer<float> levels(host_levels.size());
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaMemcpyAsync(offsets.ptr, host_offsets.data(), host_offsets.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(levels.ptr, host_levels.data(), host_levels.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(feature_map.ptr, host_feature_map.data(), sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemsetAsync(status.ptr, 0, sizeof(std::uint32_t), stream));
  const m::QuantizationProfileView profile{offsets.ptr, levels.ptr, feature_map.ptr,
      profile_features, state_features, static_cast<std::uint32_t>(host_levels.size()), bits};
  const m::DeviceStatus device_status{status.ptr};
  check(m::validate_profile(profile, device_status, stream));

  for (std::uint32_t count : {std::uint32_t{1024}, std::uint32_t{1048576},
                              std::uint32_t{4194304}}) {
    const std::uint32_t words = (count + 7U) / 8U;
    Buffer<std::uint32_t> packed_words(words), override_indices(count), override_count(1);
    Buffer<float> output(count), override_values(count);
    std::vector<std::uint32_t> host_packed(words, 0);
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint32_t code = i % 16U;
      host_packed[i / 8U] |= code << ((i % 8U) * bits);
    }
    std::vector<std::uint32_t> host_indices(count);
    std::vector<float> host_values(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      host_indices[i] = i;
      host_values[i] = host_levels[(i * 7U) % host_levels.size()];
    }
    check(cudaMemcpyAsync(packed_words.ptr, host_packed.data(), words * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(override_indices.ptr, host_indices.data(), count * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(override_values.ptr, host_values.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(override_count.ptr, 0, sizeof(std::uint32_t), stream));
    check(cudaStreamSynchronize(stream));
    const m::ConstPackedStateView packed{packed_words.ptr, words};
    const m::ConstOverrideView no_sidecar{};
    const m::ConstOverrideView empty_sidecar{override_indices.ptr, override_values.ptr,
                                              override_count.ptr, count};
    const m::ConstOverrideView sparse_sidecar{override_indices.ptr, override_values.ptr,
                                               override_count.ptr, count};
    const m::ConstOverrideView full_sidecar{override_indices.ptr, override_values.ptr,
                                             override_count.ptr, count};
    const auto repetitions = count < 4096U ? 2000 : (count < 2000000U ? 100 : 25);
    const float none = measure(stream, packed, count, profile, no_sidecar, output.ptr,
                               device_status, repetitions);
    const float empty = measure(stream, packed, count, profile, empty_sidecar, output.ptr,
                                device_status, repetitions);
    const std::array<float, 3> before{none, empty, none};
    std::cout << "count=" << count << " bits=4 decode_no_sidecar_us=" << none
              << " decode_empty_sidecar_us=" << empty
              << " override_capacity=" << count
              << " repetitions=" << repetitions << '\n';

    for (std::uint32_t density_count : {std::uint32_t{8}, count}) {
      check(cudaMemcpyAsync(override_count.ptr, &density_count, sizeof(density_count),
                            cudaMemcpyHostToDevice, stream));
      const float elapsed = measure(stream, packed, count, profile,
                                    density_count == 8 ? sparse_sidecar : full_sidecar,
                                    output.ptr, device_status, repetitions);
      std::cout << "count=" << count << " overrides=" << density_count
                << " decode_with_sidecar_us=" << elapsed << '\n';
    }
    (void)before;
  }
  check(cudaStreamDestroy(stream));
  return 0;
}
