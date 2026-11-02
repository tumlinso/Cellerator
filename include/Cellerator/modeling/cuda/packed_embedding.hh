#pragma once

#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/tensor/state_codes/view.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::modeling::cuda {

// Sparse active occurrences reference logical row-major state positions
// (row * state_feature_count + physical_feature). segment_offsets partitions
// occurrence order into ragged embedding bags; weights may be null (unit).
struct PackedOccurrenceView {
    const std::uint32_t* logical_indices = nullptr;
    std::size_t logical_index_capacity = 0;
    const float* weights = nullptr;  // optional [occurrence_count]
    std::size_t weight_capacity = 0;
    const std::uint32_t* segment_offsets = nullptr;  // [segment_count + 1]
    std::size_t segment_offset_capacity = 0;
    std::uint32_t occurrence_count = 0;
    std::uint32_t segment_count = 0;
};

// Ragged feature-specific lookup table. feature_offsets has one uint64 entry
// per profile feature plus a terminal offset; feature f stores
// codebook_size(f) consecutive vectors of embedding_dim values.
struct PackedEmbeddingTableView {
    const void* values = nullptr;
    std::size_t value_capacity = 0;  // elements, not bytes
    const std::uint64_t* feature_offsets = nullptr;
    std::size_t feature_offset_capacity = 0;
    std::uint32_t embedding_dim = 0;
    ::cellerator::memory::cuda::DType dtype =
        ::cellerator::memory::cuda::DType::float16;
};

struct PackedEmbeddingGradientView {
    float* table = nullptr;  // FP32 additive gradient, table-value_capacity elements
    std::size_t table_capacity = 0;
    float* occurrence_weights = nullptr;  // optional [occurrence_count], assigned
    std::size_t occurrence_weight_capacity = 0;
};

// Sparse weighted embedding-bag forward from packed state. The packed layout
// is the 1/2/4/8-bit little-slot uint32 state-code layout
// over row-major state. Exact overrides are supported by continuous
// interpolation between the adjacent codebook-level embeddings; the sidecar
// must be sorted by logical index. Passing an empty override view asserts
// bits-only semantics. All inputs are borrowed, launches are asynchronous,
// and no dense state decode or hidden allocation occurs.
cudaError_t packed_embedding_bag_forward(
    ::cellerator::tensor::state_codes::ConstPackedView packed,
    std::uint32_t row_count,
    ::cellerator::tensor::state_codes::ProfileView profile,
    ::cellerator::tensor::state_codes::ConstExactOverrideView exact_overrides,
    bool overrides_sorted_by_logical_index,
    PackedOccurrenceView occurrences, PackedEmbeddingTableView table,
    ::cellerator::memory::cuda::DeviceView output,
    ::cellerator::tensor::state_codes::DeviceStatus status,
    cudaStream_t stream) noexcept;

// Backward accumulates into an FP32 table-gradient buffer with atomic adds;
// zero it before the call to overwrite or seed it for accumulation. Optional
// occurrence-weight gradients are assigned per occurrence. Packed codes,
// feature mappings, selections, and exact values are treated as discrete/
// conditioning inputs and receive no gradients.
cudaError_t packed_embedding_bag_backward(
    ::cellerator::tensor::state_codes::ConstPackedView packed,
    std::uint32_t row_count,
    ::cellerator::tensor::state_codes::ProfileView profile,
    ::cellerator::tensor::state_codes::ConstExactOverrideView exact_overrides,
    bool overrides_sorted_by_logical_index,
    PackedOccurrenceView occurrences, PackedEmbeddingTableView table,
    ::cellerator::memory::cuda::ConstDeviceView output_gradient,
    PackedEmbeddingGradientView gradients,
    ::cellerator::tensor::state_codes::DeviceStatus status,
    cudaStream_t stream) noexcept;

}  // namespace cellerator::modeling::cuda
