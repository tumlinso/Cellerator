#include <Cellerator/modeling/cuda/packed_embedding.hh>

#include <cuda_runtime.h>

#include <limits>

namespace cellerator::modeling::cuda {
namespace {
constexpr unsigned kThreads = 256;
constexpr std::uint32_t kBadInput = 1U;
constexpr std::uint32_t kOverrideOverflow = 2U;
constexpr std::uint32_t kNonFinite = 4U;

__device__ inline float read_value(const void* ptr, DType type, std::uint64_t i) {
    return type == DType::float16
               ? __half2float(static_cast<const __half*>(ptr)[i])
               : static_cast<const float*>(ptr)[i];
}
__device__ inline void write_value(void* ptr, DType type, std::uint64_t i, float value) {
    if (type == DType::float16)
        static_cast<__half*>(ptr)[i] = __float2half_rn(value);
    else
        static_cast<float*>(ptr)[i] = value;
}

__device__ inline std::uint32_t read_code(const std::uint32_t* words,
                                           std::uint64_t logical,
                                           std::uint32_t bits) {
    const std::uint32_t per_word = 32U / bits;
    return (words[logical / per_word] >> ((logical % per_word) * bits)) &
           ((1U << bits) - 1U);
}

// Resolve one active logical position to one codebook embedding row, or to a
// pair of rows for an exact value. The caller checks feature/table bounds
// before dereferencing the returned table offsets.
__device__ bool resolve(std::uint32_t logical,
                        std::uint32_t rows,
                        QuantizationProfileView p,
                        ConstPackedStateView packed,
                        ConstOverrideView overrides,
                        const std::uint64_t* table_offsets,
                        std::uint64_t table_capacity,
                        std::uint32_t dim,
                        DeviceStatus status,
                        std::uint32_t* feature_out,
                        std::uint32_t* lo_out,
                        std::uint32_t* hi_out,
                        float* mix_out,
                        std::uint64_t* table_base_out) {
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * p.state_feature_count;
    if (logical >= total || p.state_feature_count == 0U) {
        atomicOr(status.bits, kBadInput);
        return false;
    }
    const auto physical = logical % p.state_feature_count;
    const auto feature = p.feature_map[physical];
    if (feature >= p.profile_feature_count) {
        atomicOr(status.bits, kBadInput);
        return false;
    }
    const auto begin = p.codebook_offsets[feature];
    const auto end = p.codebook_offsets[feature + 1U];
    if (begin >= end || end > p.level_count) {
        atomicOr(status.bits, kBadInput);
        return false;
    }
    const auto nlevels = end - begin;
    const std::uint64_t expected = static_cast<std::uint64_t>(nlevels) * dim;
    const auto table_base = table_offsets[feature];
    const auto table_end = table_offsets[feature + 1U];
    if (table_base > table_end || table_end > table_capacity ||
        table_end - table_base != expected) {
        atomicOr(status.bits, kBadInput);
        return false;
    }

    std::uint32_t code = read_code(packed.words, logical, p.code_bits);
    std::uint32_t lo = code, hi = code;
    float mix = 0.0f;
    if (overrides.count != nullptr) {
        const auto count = *overrides.count;
        if (count > overrides.capacity) {
            atomicOr(status.bits, kOverrideOverflow);
            return false;
        }
        std::uint32_t left = 0U, right = count;
        while (left < right) {
            const auto mid = left + (right - left) / 2U;
            if (overrides.logical_indices[mid] < logical) left = mid + 1U;
            else right = mid;
        }
        if (left < count && overrides.logical_indices[left] == logical) {
            const float value = overrides.values[left];
            if (!isfinite(value)) {
                atomicOr(status.bits, kNonFinite);
                return false;
            }
            const float first = p.levels[begin];
            const float last = p.levels[end - 1U];
            if (value <= first) lo = hi = 0U;
            else if (value >= last) lo = hi = nlevels - 1U;
            else {
                std::uint32_t l = 0U, r = nlevels;
                while (l < r) {
                    const auto mid = l + (r - l) / 2U;
                    if (p.levels[begin + mid] < value) l = mid + 1U;
                    else r = mid;
                }
                hi = l;
                lo = hi - 1U;
                mix = (value - p.levels[begin + lo]) /
                      (p.levels[begin + hi] - p.levels[begin + lo]);
            }
        }
    }
    if (lo >= nlevels || hi >= nlevels) {
        atomicOr(status.bits, kBadInput);
        return false;
    }
    *feature_out = feature;
    *lo_out = lo;
    *hi_out = hi;
    *mix_out = mix;
    *table_base_out = table_base;
    return true;
}

__device__ bool locate_segment(const PackedOccurrenceView& occurrences,
                               std::uint32_t occurrence,
                               std::uint32_t* segment,
                               DeviceStatus status) {
    std::uint32_t left = 0U, right = occurrences.segment_count;
    while (left < right) {
        const auto mid = left + (right - left) / 2U;
        if (occurrences.segment_offsets[mid + 1U] <= occurrence) left = mid + 1U;
        else right = mid;
    }
    if (left >= occurrences.segment_count ||
        occurrence < occurrences.segment_offsets[left] ||
        occurrence >= occurrences.segment_offsets[left + 1U]) {
        atomicOr(status.bits, kBadInput);
        return false;
    }
    *segment = left;
    return true;
}

__global__ void validate_segments(const PackedOccurrenceView occurrences,
                                  DeviceStatus status) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0U && occurrences.segment_offsets[0] != 0U)
        atomicOr(status.bits, kBadInput);
    if (i == occurrences.segment_count &&
        occurrences.segment_offsets[i] != occurrences.occurrence_count)
        atomicOr(status.bits, kBadInput);
    if (i < occurrences.segment_count &&
        occurrences.segment_offsets[i] > occurrences.segment_offsets[i + 1U])
        atomicOr(status.bits, kBadInput);
}

__global__ void forward_kernel(ConstPackedStateView packed,
                               std::uint32_t rows,
                               QuantizationProfileView profile,
                               ConstOverrideView overrides,
                               PackedOccurrenceView occurrences,
                               PackedEmbeddingTableView table,
                               void* output,
                               std::uint64_t output_count,
                               DeviceStatus status) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    if ((*status.bits & kBadInput) != 0U) {
        write_value(output, table.dtype, index, 0.0f);
        return;
    }
    const std::uint32_t segment = static_cast<std::uint32_t>(index / table.embedding_dim);
    const std::uint32_t dim = static_cast<std::uint32_t>(index % table.embedding_dim);
    const auto first = occurrences.segment_offsets[segment];
    const auto last = occurrences.segment_offsets[segment + 1U];
    if (first > last || last > occurrences.occurrence_count) {
        atomicOr(status.bits, kBadInput);
        write_value(output, table.dtype, index, 0.0f);
        return;
    }
    float sum = 0.0f;
    for (std::uint32_t o = first; o < last; ++o) {
        std::uint32_t feature, lo, hi;
        float mix;
        std::uint64_t base;
        if (!resolve(occurrences.logical_indices[o], rows, profile, packed, overrides,
                     table.feature_offsets, table.value_capacity, table.embedding_dim,
                     status, &feature, &lo, &hi, &mix, &base)) continue;
        (void)feature;
        const std::uint64_t lo_index = base + static_cast<std::uint64_t>(lo) * table.embedding_dim + dim;
        const std::uint64_t hi_index = base + static_cast<std::uint64_t>(hi) * table.embedding_dim + dim;
        const float embedding = read_value(table.values, table.dtype, lo_index) * (1.0f - mix) +
                                read_value(table.values, table.dtype, hi_index) * mix;
        const float weight = occurrences.weights == nullptr ? 1.0f : occurrences.weights[o];
        if (!isfinite(weight)) {
            atomicOr(status.bits, kNonFinite);
            continue;
        }
        sum += weight * embedding;
    }
    write_value(output, table.dtype, index, sum);
}

__global__ void backward_kernel(ConstPackedStateView packed,
                                std::uint32_t rows,
                                QuantizationProfileView profile,
                                ConstOverrideView overrides,
                                PackedOccurrenceView occurrences,
                                PackedEmbeddingTableView table,
                                ConstDeviceView output_gradient,
                                PackedEmbeddingGradientView gradients,
                                DeviceStatus status) {
    const std::uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= occurrences.occurrence_count) return;
    if ((*status.bits & kBadInput) != 0U) return;
    std::uint32_t segment;
    if (!locate_segment(occurrences, o, &segment, status)) return;
    const auto logical = occurrences.logical_indices[o];
    std::uint32_t feature, lo, hi;
    float mix;
    std::uint64_t base;
    if (!resolve(logical, rows, profile, packed, overrides, table.feature_offsets,
                 table.value_capacity, table.embedding_dim, status,
                 &feature, &lo, &hi, &mix, &base)) return;
    (void)feature;
    const float occurrence_weight = occurrences.weights == nullptr ? 1.0f : occurrences.weights[o];
    if (!isfinite(occurrence_weight)) {
        atomicOr(status.bits, kNonFinite);
        return;
    }
    float weight_gradient = 0.0f;
    for (std::uint32_t d = 0; d < table.embedding_dim; ++d) {
        const auto y_index = static_cast<std::uint64_t>(segment) * table.embedding_dim + d;
        const float dy = read_value(output_gradient.data, table.dtype, y_index);
        const auto lo_index = base + static_cast<std::uint64_t>(lo) * table.embedding_dim + d;
        const auto hi_index = base + static_cast<std::uint64_t>(hi) * table.embedding_dim + d;
        const float emb_lo = read_value(table.values, table.dtype, lo_index);
        const float emb_hi = read_value(table.values, table.dtype, hi_index);
        const float embedding = emb_lo * (1.0f - mix) + emb_hi * mix;
        if (gradients.occurrence_weights != nullptr)
            weight_gradient += dy * embedding;
        if (lo == hi) {
            atomicAdd(&gradients.table[lo_index], dy * occurrence_weight);
        } else {
            atomicAdd(&gradients.table[lo_index], dy * occurrence_weight * (1.0f - mix));
            atomicAdd(&gradients.table[hi_index], dy * occurrence_weight * mix);
        }
    }
    if (gradients.occurrence_weights != nullptr)
        gradients.occurrence_weights[o] = weight_gradient;
}

// Experimental alternative used to compare occurrence-major strided work with
// one warp owning each sparse occurrence. The optional compile definition keeps
// the public API and caller contract identical for matched experiments.
#if defined(CELLERATOR_PACKED_EMBEDDING_WARP_BACKWARD)
__global__ void backward_warp_kernel(ConstPackedStateView packed,
                                     std::uint32_t rows,
                                     QuantizationProfileView profile,
                                     ConstOverrideView overrides,
                                     PackedOccurrenceView occurrences,
                                     PackedEmbeddingTableView table,
                                     ConstDeviceView output_gradient,
                                     PackedEmbeddingGradientView gradients,
                                     DeviceStatus status) {
    const std::uint32_t occurrence = blockIdx.x * 4U + threadIdx.y;
    if (occurrence >= occurrences.occurrence_count) return;
    if ((*status.bits & kBadInput) != 0U) return;

    constexpr unsigned kWarpMask = 0xffffffffU;
    const unsigned lane = threadIdx.x;
    std::uint32_t segment = 0U, feature = 0U, lo = 0U, hi = 0U;
    float mix = 0.0f, occurrence_weight = 1.0f;
    unsigned long long base = 0U;
    int valid = 1;
    if (lane == 0U) {
        std::uint64_t table_base = 0U;
        valid = locate_segment(occurrences, occurrence, &segment, status);
        if (valid) {
            valid = resolve(occurrences.logical_indices[occurrence], rows, profile,
                            packed, overrides, table.feature_offsets,
                            table.value_capacity, table.embedding_dim, status,
                            &feature, &lo, &hi, &mix, &table_base);
            base = static_cast<unsigned long long>(table_base);
        }
        if (valid && occurrences.weights != nullptr) {
            occurrence_weight = occurrences.weights[occurrence];
            if (!isfinite(occurrence_weight)) {
                atomicOr(status.bits, kNonFinite);
                valid = 0;
            }
        }
    }
    valid = __shfl_sync(kWarpMask, valid, 0);
    if (!valid) return;
    segment = __shfl_sync(kWarpMask, segment, 0);
    feature = __shfl_sync(kWarpMask, feature, 0);
    lo = __shfl_sync(kWarpMask, lo, 0);
    hi = __shfl_sync(kWarpMask, hi, 0);
    mix = __shfl_sync(kWarpMask, mix, 0);
    occurrence_weight = __shfl_sync(kWarpMask, occurrence_weight, 0);
    base = __shfl_sync(kWarpMask, base, 0);
    (void)feature;

    float weight_gradient = 0.0f;
    for (std::uint32_t d = lane; d < table.embedding_dim; d += 32U) {
        const auto y_index = static_cast<std::uint64_t>(segment) *
                             table.embedding_dim + d;
        const float dy = read_value(output_gradient.data, table.dtype, y_index);
        const auto lo_index = base + static_cast<unsigned long long>(lo) *
                                      table.embedding_dim + d;
        const auto hi_index = base + static_cast<unsigned long long>(hi) *
                                      table.embedding_dim + d;
        const float emb_lo = read_value(table.values, table.dtype, lo_index);
        const float emb_hi = read_value(table.values, table.dtype, hi_index);
        const float embedding = emb_lo * (1.0f - mix) + emb_hi * mix;
        weight_gradient += dy * embedding;
        if (lo == hi) {
            atomicAdd(&gradients.table[lo_index], dy * occurrence_weight);
        } else {
            atomicAdd(&gradients.table[lo_index],
                      dy * occurrence_weight * (1.0f - mix));
            atomicAdd(&gradients.table[hi_index],
                      dy * occurrence_weight * mix);
        }
    }
    for (unsigned delta = 16U; delta != 0U; delta >>= 1U)
        weight_gradient += __shfl_down_sync(kWarpMask, weight_gradient, delta);
    if (lane == 0U && gradients.occurrence_weights != nullptr)
        gradients.occurrence_weights[occurrence] = weight_gradient;
}
#endif

std::uint64_t elements(std::uint64_t a, std::uint64_t b) {
    if (b != 0U && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::invalid_argument("packed embedding shape overflow");
    return a * b;
}
std::uint32_t blocks(std::uint64_t n) {
    const auto b = (n + kThreads - 1U) / kThreads;
    if (b > 2147483647U) throw std::invalid_argument("packed embedding launch too large");
    return static_cast<std::uint32_t>(b);
}

cudaError_t validate(const ConstPackedStateView& packed, std::uint32_t rows,
                     const QuantizationProfileView& p,
                     const ConstOverrideView& overrides, bool sorted,
                     const PackedOccurrenceView& occurrences,
                     const PackedEmbeddingTableView& table,
                     DeviceStatus status, std::uint64_t* output_count) {
    if (status.bits == nullptr || packed.words == nullptr ||
        p.codebook_offsets == nullptr || p.levels == nullptr || p.feature_map == nullptr ||
        p.profile_feature_count == 0U || p.state_feature_count == 0U ||
        p.level_count == 0U || !(p.code_bits == 1U || p.code_bits == 2U ||
                                 p.code_bits == 4U || p.code_bits == 8U) ||
        table.values == nullptr || table.feature_offsets == nullptr ||
        table.embedding_dim == 0U ||
        !(table.dtype == DType::float16 || table.dtype == DType::float32) ||
        occurrences.segment_offsets == nullptr ||
        occurrences.segment_offset_capacity < static_cast<std::size_t>(occurrences.segment_count) + 1U ||
        (occurrences.occurrence_count != 0U &&
         (occurrences.logical_indices == nullptr ||
          occurrences.logical_index_capacity < occurrences.occurrence_count)) ||
        (occurrences.weights != nullptr && occurrences.weight_capacity < occurrences.occurrence_count))
        return cudaErrorInvalidValue;
    const std::uint64_t state_count = elements(rows, p.state_feature_count);
    if (state_count > std::numeric_limits<std::uint32_t>::max()) return cudaErrorInvalidValue;
    const std::uint64_t per_word = 32U / p.code_bits;
    const auto word_count = (state_count + per_word - 1U) / per_word;
    if (word_count > packed.word_capacity ||
        table.feature_offset_capacity < static_cast<std::size_t>(p.profile_feature_count) + 1U)
        return cudaErrorInvalidValue;
    if (overrides.count != nullptr) {
        if (!sorted || overrides.logical_indices == nullptr || overrides.values == nullptr)
            return cudaErrorInvalidValue;
    } else if (overrides.capacity != 0U || overrides.logical_indices != nullptr || overrides.values != nullptr) {
        return cudaErrorInvalidValue;
    }
    if (occurrences.occurrence_count && occurrences.segment_count == 0U) return cudaErrorInvalidValue;
    *output_count = elements(occurrences.segment_count, table.embedding_dim);
    if (*output_count > static_cast<std::uint64_t>(2147483647U) * kThreads)
        return cudaErrorInvalidValue;
    const auto out_bytes = elements(*output_count, item_size(table.dtype));
    if (out_bytes > std::numeric_limits<std::size_t>::max()) return cudaErrorInvalidValue;
    return cudaSuccess;
}

}  // namespace

cudaError_t packed_embedding_bag_forward(
    ConstPackedStateView packed, std::uint32_t rows,
    QuantizationProfileView profile, ConstOverrideView overrides,
    bool sorted, PackedOccurrenceView occurrences,
    PackedEmbeddingTableView table, DeviceView output,
    DeviceStatus status, cudaStream_t stream) noexcept {
    std::uint64_t count = 0;
    cudaError_t error;
    try { error = validate(packed, rows, profile, overrides, sorted, occurrences, table, status, &count); }
    catch (...) { return cudaErrorInvalidValue; }
    if (error != cudaSuccess) return error;
    std::uint64_t bytes;
    try { bytes = elements(count, item_size(table.dtype)); }
    catch (...) { return cudaErrorInvalidValue; }
    if (output.bytes < bytes || (bytes && output.data == nullptr)) return cudaErrorInvalidValue;
    validate_segments<<<blocks(static_cast<std::uint64_t>(occurrences.segment_count) + 1U),
                        kThreads, 0, stream>>>(occurrences, status);
    error = cudaGetLastError();
    if (error != cudaSuccess || count == 0U) return error;
    forward_kernel<<<blocks(count), kThreads, 0, stream>>>(packed, rows, profile, overrides,
        occurrences, table, output.data, count, status);
    return cudaGetLastError();
}

cudaError_t packed_embedding_bag_backward(
    ConstPackedStateView packed, std::uint32_t rows,
    QuantizationProfileView profile, ConstOverrideView overrides,
    bool sorted, PackedOccurrenceView occurrences,
    PackedEmbeddingTableView table, ConstDeviceView output_gradient,
    PackedEmbeddingGradientView gradients, DeviceStatus status,
    cudaStream_t stream) noexcept {
    std::uint64_t output_count = 0;
    cudaError_t error;
    try { error = validate(packed, rows, profile, overrides, sorted, occurrences, table, status, &output_count); }
    catch (...) { return cudaErrorInvalidValue; }
    if (error != cudaSuccess) return error;
    std::uint64_t grad_bytes, table_gradient_bytes;
    try {
        grad_bytes = elements(output_count, item_size(table.dtype));
        table_gradient_bytes = elements(table.value_capacity, sizeof(float));
    } catch (...) { return cudaErrorInvalidValue; }
    if ((grad_bytes && output_gradient.data == nullptr) || output_gradient.bytes < grad_bytes ||
        (table_gradient_bytes && gradients.table == nullptr) ||
        gradients.table_capacity < table.value_capacity ||
        (gradients.occurrence_weights != nullptr &&
         gradients.occurrence_weight_capacity < occurrences.occurrence_count))
        return cudaErrorInvalidValue;
    validate_segments<<<blocks(static_cast<std::uint64_t>(occurrences.segment_count) + 1U),
                        kThreads, 0, stream>>>(occurrences, status);
    error = cudaGetLastError();
    if (error != cudaSuccess || occurrences.occurrence_count == 0U) return error;
#if defined(CELLERATOR_PACKED_EMBEDDING_WARP_BACKWARD)
    const auto warp_blocks = (static_cast<std::uint64_t>(occurrences.occurrence_count) + 3U) / 4U;
    backward_warp_kernel<<<static_cast<std::uint32_t>(warp_blocks),
                           dim3(32U, 4U), 0, stream>>>(
        packed, rows, profile, overrides, occurrences, table, output_gradient,
        gradients, status);
#else
    backward_kernel<<<blocks(occurrences.occurrence_count), kThreads, 0, stream>>>(
        packed, rows, profile, overrides, occurrences, table, output_gradient,
        gradients, status);
#endif
    return cudaGetLastError();
}

}  // namespace cellerator::modeling::cuda
