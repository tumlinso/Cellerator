#include <Cellerator/modeling/cuda/quantization.hh>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {

constexpr std::uint32_t kBadProfile = 1U;
constexpr std::uint32_t kOverrideOverflow = 2U;
constexpr std::uint32_t kNonFiniteValue = 4U;
constexpr unsigned kThreads = 256;

struct ConstDeviceStatus {
  const std::uint32_t* bits;
};

__device__ inline bool valid_bits(std::uint32_t bits) {
  return bits == 1U || bits == 2U || bits == 4U || bits == 8U;
}

__device__ inline bool invalid(ConstDeviceStatus status) {
  return status.bits != nullptr && ((*status.bits & kBadProfile) != 0U);
}

__global__ void validate_profile_kernel(QuantizationProfileView p,
                                        DeviceStatus status) {
  const std::uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f == 0U) {
    if (p.codebook_offsets == nullptr || p.levels == nullptr ||
        p.feature_map == nullptr || p.profile_feature_count == 0U ||
        p.state_feature_count == 0U || p.level_count == 0U ||
        !valid_bits(p.code_bits) || p.codebook_offsets[0] != 0U ||
        p.codebook_offsets[p.profile_feature_count] != p.level_count) {
      atomicOr(status.bits, kBadProfile);
    }
  }

  if (f < p.state_feature_count && p.feature_map != nullptr &&
      p.profile_feature_count != 0U &&
      p.feature_map[f] >= p.profile_feature_count) {
    atomicOr(status.bits, kBadProfile);
  }
  if (f < p.profile_feature_count && p.codebook_offsets != nullptr &&
      p.levels != nullptr && valid_bits(p.code_bits)) {
    const std::uint32_t begin = p.codebook_offsets[f];
    const std::uint32_t end = p.codebook_offsets[f + 1U];
    const std::uint32_t capacity = 1U << p.code_bits;
    if (begin >= end || end > p.level_count || end - begin > capacity) {
      atomicOr(status.bits, kBadProfile);
      return;
    }
    for (std::uint32_t j = begin; j < end; ++j) {
      if (!isfinite(p.levels[j]) ||
          (j > begin && !(p.levels[j - 1U] < p.levels[j]))) {
        atomicOr(status.bits, kBadProfile);
        return;
      }
    }
  }
}

__device__ inline std::uint32_t nearest_level(float value,
                                               std::uint32_t begin,
                                               std::uint32_t end,
                                               const float* levels) {
  if (value <= levels[begin]) return 0U;
  if (value >= levels[end - 1U]) return end - begin - 1U;

  std::uint32_t lo = begin;
  std::uint32_t hi = end;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2U;
    if (levels[mid] < value) lo = mid + 1U;
    else hi = mid;
  }
  const std::uint32_t right = lo;
  const std::uint32_t left = right - 1U;
  const float left_error = value - levels[left];
  const float right_error = levels[right] - value;
  return (left_error <= right_error ? left : right) - begin;
}

__device__ inline std::uint32_t read_code(const std::uint32_t* words,
                                            std::uint64_t index,
                                            std::uint32_t bits) {
  const std::uint32_t per_word = 32U / bits;
  const std::uint32_t shift = static_cast<std::uint32_t>(index % per_word) * bits;
  return (words[index / per_word] >> shift) & ((1U << bits) - 1U);
}

template <typename Input>
__global__ void encode_kernel(const Input* values,
                              std::uint64_t count,
                              std::uint32_t rows,
                              QuantizationProfileView p,
                              std::uint32_t* packed,
                              const std::uint8_t* exact_mask,
                              OverrideView overrides,
                              DeviceStatus status) {
  const std::uint64_t word = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t per_word = 32U / p.code_bits;
  const std::uint64_t word_count = (count + per_word - 1U) / per_word;
  if (word >= word_count || invalid({status.bits})) return;

  const std::uint64_t first = word * per_word;
  std::uint32_t output = 0U;
  #pragma unroll
  for (std::uint32_t slot = 0U; slot < 32U; ++slot) {
    if (slot >= per_word) break;
    const std::uint64_t i = first + slot;
    if (i >= count) break;
    const std::uint32_t physical_feature =
        static_cast<std::uint32_t>(i % p.state_feature_count);
    const std::uint32_t feature = p.feature_map[physical_feature];
    const std::uint32_t begin = p.codebook_offsets[feature];
    const std::uint32_t end = p.codebook_offsets[feature + 1U];
    const float value = static_cast<float>(values[i]);
    // Keep a deterministic placeholder code for invalid input. In particular,
    // nearest_level assumes an ordered finite value and NaN would underflow
    // its left-neighbor index during the binary-search boundary case.
    std::uint32_t code = 0U;
    if (!isfinite(value)) {
      atomicOr(status.bits, kNonFiniteValue);
    } else {
      code = nearest_level(value, begin, end, p.levels);
    }
    output |= code << (slot * p.code_bits);

    if (exact_mask != nullptr && exact_mask[i] != 0U) {
      const std::uint32_t destination = atomicAdd(overrides.count, 1U);
      if (destination < overrides.capacity) {
        overrides.logical_indices[destination] = static_cast<std::uint32_t>(i);
        overrides.values[destination] = value;
      } else {
        atomicOr(status.bits, kOverrideOverflow);
      }
    }
  }
  packed[word] = output;  // each CTA lane owns exactly one complete word
  (void)rows;
}

template <typename Output>
__global__ void decode_kernel(const std::uint32_t* packed,
                              std::uint64_t count,
                              QuantizationProfileView p,
                              Output* output,
                              DeviceStatus status) {
  const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count || invalid({status.bits})) return;
  const std::uint32_t physical_feature =
      static_cast<std::uint32_t>(i % p.state_feature_count);
  const std::uint32_t feature = p.feature_map[physical_feature];
  const std::uint32_t begin = p.codebook_offsets[feature];
  const std::uint32_t end = p.codebook_offsets[feature + 1U];
  const std::uint32_t local_code = read_code(packed, i, p.code_bits);
  if (local_code >= end - begin) {
    atomicOr(status.bits, kBadProfile);
    return;
  }
  output[i] = static_cast<Output>(p.levels[begin + local_code]);
}

template <typename Output>
__global__ void apply_overrides(ConstOverrideView overrides,
                                std::uint64_t count,
                                Output* output,
                                DeviceStatus status) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (overrides.count == nullptr || *overrides.count > overrides.capacity) {
    atomicOr(status.bits, kOverrideOverflow);
    return;
  }
  if (i >= *overrides.count) return;
  const std::uint32_t index = overrides.logical_indices[i];
  if (index >= count) {
    atomicOr(status.bits, kBadProfile);
    return;
  }
  output[index] = static_cast<Output>(overrides.values[i]);
}

template <typename Input>
__global__ void update_kernel(const std::uint32_t* indices,
                              const Input* values,
                              std::uint32_t update_count,
                              std::uint64_t total_count,
                              QuantizationProfileView p,
                              std::uint32_t* packed,
                              DeviceStatus status) {
  const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= update_count || invalid({status.bits})) return;
  const std::uint32_t i = indices[u];
  if (i >= total_count) {
    atomicOr(status.bits, kBadProfile);
    return;
  }
  const std::uint32_t physical_feature = i % p.state_feature_count;
  const std::uint32_t feature = p.feature_map[physical_feature];
  const std::uint32_t begin = p.codebook_offsets[feature];
  const std::uint32_t end = p.codebook_offsets[feature + 1U];
  const float value = static_cast<float>(values[u]);
  if (!isfinite(value)) {
    atomicOr(status.bits, kNonFiniteValue);
    return;
  }
  const std::uint32_t code = nearest_level(value, begin, end, p.levels);
  const std::uint32_t per_word = 32U / p.code_bits;
  const std::uint32_t shift = (i % per_word) * p.code_bits;
  const std::uint32_t mask = ((1U << p.code_bits) - 1U) << shift;
  std::uint32_t* address = packed + i / per_word;
  std::uint32_t old = *address;
  std::uint32_t assumed;
  do {
    assumed = old;
    const std::uint32_t next = (assumed & ~mask) | (code << shift);
    old = atomicCAS(address, assumed, next);
  } while (old != assumed);
}

inline std::uint32_t blocks(std::uint64_t count, std::uint32_t per_block = kThreads) {
  return static_cast<std::uint32_t>((count + per_block - 1U) / per_block);
}

inline bool valid_host(const QuantizationProfileView& p) {
  return p.codebook_offsets != nullptr && p.levels != nullptr &&
         p.feature_map != nullptr && p.profile_feature_count != 0U &&
         p.state_feature_count != 0U && p.level_count != 0U &&
         (p.code_bits == 1U || p.code_bits == 2U ||
          p.code_bits == 4U || p.code_bits == 8U);
}

cudaError_t validate_common(std::uint32_t rows,
                            const QuantizationProfileView& p,
                            DeviceStatus status,
                            std::uint64_t* count) {
  if (!valid_host(p) || status.bits == nullptr || count == nullptr)
    return cudaErrorInvalidValue;
  *count = static_cast<std::uint64_t>(rows) * p.state_feature_count;
  if (*count > std::numeric_limits<std::uint32_t>::max())
    return cudaErrorInvalidValue;  // override indices use uint32
  return cudaSuccess;
}

template <typename Input>
cudaError_t encode(const Input* values, std::uint32_t rows,
                   QuantizationProfileView p, PackedStateView packed,
                   const std::uint8_t* exact_mask, OverrideView overrides,
                   DeviceStatus status, cudaStream_t stream) noexcept {
  std::uint64_t count = 0;
  auto error = validate_common(rows, p, status, &count);
  if (error != cudaSuccess || (count != 0U && (values == nullptr || packed.words == nullptr)))
    return error == cudaSuccess ? cudaErrorInvalidValue : error;
  const std::uint64_t code_words = (count + 32U / p.code_bits - 1U) / (32U / p.code_bits);
  if (code_words > packed.word_capacity) return cudaErrorInvalidValue;
  if (exact_mask != nullptr &&
      (overrides.count == nullptr || overrides.logical_indices == nullptr ||
       overrides.values == nullptr || overrides.capacity < count))
    return cudaErrorInvalidValue;
  if (count == 0U) return cudaSuccess;
  encode_kernel<<<blocks(code_words), kThreads, 0, stream>>>(
      values, count, rows, p, packed.words, exact_mask, overrides, status);
  return cudaGetLastError();
}

template <typename Output>
cudaError_t decode(ConstPackedStateView packed, std::uint32_t rows,
                   QuantizationProfileView p, ConstOverrideView overrides,
                   Output* output, DeviceStatus status,
                   cudaStream_t stream) noexcept {
  std::uint64_t count = 0;
  auto error = validate_common(rows, p, status, &count);
  if (error != cudaSuccess || (count != 0U && (packed.words == nullptr || output == nullptr)))
    return error == cudaSuccess ? cudaErrorInvalidValue : error;
  const std::uint64_t code_words = (count + 32U / p.code_bits - 1U) / (32U / p.code_bits);
  if (code_words > packed.word_capacity) return cudaErrorInvalidValue;
  if (overrides.count != nullptr &&
      (overrides.logical_indices == nullptr || overrides.values == nullptr))
    return cudaErrorInvalidValue;
  if (count != 0U) {
    decode_kernel<<<blocks(count), kThreads, 0, stream>>>(
        packed.words, count, p, output, status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  if (overrides.count != nullptr && overrides.capacity != 0U) {
    // Count is device resident: a grid-stride launch caps overhead while each
    // thread checks the actual count after prior same-stream decoding.
    apply_overrides<<<blocks(overrides.capacity), kThreads, 0, stream>>>(
        overrides, count, output, status);
    error = cudaGetLastError();
  }
  return error;
}

template <typename Input>
cudaError_t update(const std::uint32_t* indices, const Input* values,
                   std::uint32_t updates, std::uint32_t rows,
                   QuantizationProfileView p, PackedStateView packed,
                   DeviceStatus status, cudaStream_t stream) noexcept {
  std::uint64_t count = 0;
  auto error = validate_common(rows, p, status, &count);
  if (error != cudaSuccess ||
      (updates != 0U && (indices == nullptr || values == nullptr || packed.words == nullptr)))
    return error == cudaSuccess ? cudaErrorInvalidValue : error;
  const std::uint64_t code_words = (count + 32U / p.code_bits - 1U) / (32U / p.code_bits);
  if (code_words > packed.word_capacity) return cudaErrorInvalidValue;
  if (updates == 0U) return cudaSuccess;
  update_kernel<<<blocks(updates), kThreads, 0, stream>>>(
      indices, values, updates, count, p, packed.words, status);
  return cudaGetLastError();
}

}  // namespace

cudaError_t reset_status(DeviceStatus status, cudaStream_t stream) noexcept {
  if (status.bits == nullptr) return cudaErrorInvalidValue;
  return cudaMemsetAsync(status.bits, 0, sizeof(std::uint32_t), stream);
}

cudaError_t reset_override_count(OverrideView overrides,
                                 cudaStream_t stream) noexcept {
  if (overrides.count == nullptr) return cudaErrorInvalidValue;
  return cudaMemsetAsync(overrides.count, 0, sizeof(std::uint32_t), stream);
}

cudaError_t validate_profile(QuantizationProfileView p, DeviceStatus status,
                             cudaStream_t stream) noexcept {
  if (!valid_host(p) || status.bits == nullptr) return cudaErrorInvalidValue;
  const std::uint32_t count = p.profile_feature_count > p.state_feature_count
                                  ? p.profile_feature_count
                                  : p.state_feature_count;
  validate_profile_kernel<<<blocks(count), kThreads, 0, stream>>>(p, status);
  return cudaGetLastError();
}

cudaError_t encode_state_fp32(const float* values, std::uint32_t rows,
                              QuantizationProfileView p,
                              PackedStateView packed,
                              const std::uint8_t* exact_mask,
                              OverrideView overrides,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept {
  return encode(values, rows, p, packed, exact_mask, overrides, status, stream);
}
cudaError_t encode_state_fp16(const __half* values, std::uint32_t rows,
                              QuantizationProfileView p,
                              PackedStateView packed,
                              const std::uint8_t* exact_mask,
                              OverrideView overrides,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept {
  return encode(values, rows, p, packed, exact_mask, overrides, status, stream);
}
cudaError_t decode_state_fp32(ConstPackedStateView packed, std::uint32_t rows,
                              QuantizationProfileView p,
                              ConstOverrideView overrides,
                              float* output, DeviceStatus status,
                              cudaStream_t stream) noexcept {
  return decode(packed, rows, p, overrides, output, status, stream);
}
cudaError_t decode_state_fp16(ConstPackedStateView packed, std::uint32_t rows,
                              QuantizationProfileView p,
                              ConstOverrideView overrides,
                              __half* output, DeviceStatus status,
                              cudaStream_t stream) noexcept {
  return decode(packed, rows, p, overrides, output, status, stream);
}
cudaError_t update_codes_fp32(const std::uint32_t* indices,
                              const float* values,
                              std::uint32_t updates,
                              std::uint32_t rows,
                              QuantizationProfileView p,
                              PackedStateView packed,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept {
  return update(indices, values, updates, rows, p, packed, status, stream);
}
cudaError_t update_codes_fp16(const std::uint32_t* indices,
                              const __half* values,
                              std::uint32_t updates,
                              std::uint32_t rows,
                              QuantizationProfileView p,
                              PackedStateView packed,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept {
  return update(indices, values, updates, rows, p, packed, status, stream);
}

}  // namespace cellerator::modeling::cuda
