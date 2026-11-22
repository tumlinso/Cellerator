#include "feature_major_tiles_internal.cuh"

namespace cellerator::ops::matrix {
namespace {

using tensor::FeatureMajorTileStructureView;
using tensor::FeatureMajorTilesView;
using layout::projection::FeatureMajorTilesIndexView;

constexpr uint32_t width = feature_major_tiles_dense_width;
constexpr uint32_t warp_rows = tensor::feature_major_tile_rows;
constexpr uint32_t warps_per_block = 4;
constexpr uint32_t vector_threads = 256;
constexpr uint32_t forward_records_per_block = 128;
constexpr uint32_t sampled_records_per_warp = 4;
constexpr uint32_t warp_mask = 0xffffffffu;

uint32_t block_count(uint32_t count, uint32_t work_per_block) noexcept {
    return count / work_per_block + (count % work_per_block != 0);
}

bool valid_structure(FeatureMajorTileStructureView x) noexcept {
    if (!x.tile_row_width || x.tile_row_width > warp_rows
        || x.tile_count != block_count(x.rows, x.tile_row_width)) return false;
    if (!x.nnz) return x.record_count == 0;
    return x.rows && x.features && x.record_count && x.record_count <= x.nnz
        && x.tile_offsets && x.feature_ids && x.row_masks && x.value_offsets;
}

__device__ uint32_t tile_value_position(uint32_t first, uint32_t mask, uint32_t lane) {
    return first + __popc(mask & ((1u << lane) - 1u));
}

__device__ uint32_t logical_row(const uint32_t *rows, uint32_t physical) {
    return rows ? rows[physical] : physical;
}

// First tile whose exclusive record end exceeds this record, including
// directories with runs of empty tiles.
__device__ uint32_t find_record_tile(FeatureMajorTileStructureView s, uint32_t record) {
    uint32_t begin = 0, end = s.tile_count;
    while (begin < end) {
        const uint32_t mid = begin + (end - begin) / 2;
        if (s.tile_offsets[mid + 1] <= record) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

template<uint32_t fixed_width>
__global__ void tile_forward(FeatureMajorTilesView<const __half> x,
    const uint32_t *logical_rows, const float *w, uint32_t requested_width, float *y) {
    const uint32_t dense_width = fixed_width ? fixed_width : requested_width;
    const uint32_t warp = threadIdx.x / warp_rows;
    const uint32_t lane = threadIdx.x % warp_rows;
    const uint32_t tile = blockIdx.x * warps_per_block + warp;
    if (tile >= x.structure.tile_count) return;

    float accum[width]{};
    __shared__ float dense_features[warps_per_block][width];
    float *features = dense_features[warp];
    const auto &s = x.structure;
    for (uint32_t record = s.tile_offsets[tile]; record < s.tile_offsets[tile + 1]; ++record) {
        if (lane < dense_width)
            features[lane] = w[size_t(s.feature_ids[record]) * dense_width + lane];
        __syncwarp();
        const uint32_t mask = s.row_masks[record];
        if (mask & (1u << lane)) {
            const uint32_t position = tile_value_position(s.value_offsets[record], mask, lane);
            const float value = __half2float(x.values[position]);
            #pragma unroll
            for (uint32_t column = 0; column < width; ++column)
                if (column < dense_width) accum[column] += value * features[column];
        }
        __syncwarp();
    }
    if (lane < s.tile_row_width) {
        const uint64_t physical = uint64_t(tile) * s.tile_row_width + lane;
        if (physical < s.rows) {
            float *out = y + size_t(logical_row(logical_rows, uint32_t(physical))) * dense_width;
            #pragma unroll
            for (uint32_t column = 0; column < width; ++column)
                if (column < dense_width) out[column] = accum[column];
        }
    }
}

// In batches with few tiles, split the records across CTAs. A CTA reduces
// its four warp contributions before writing each affected output once.
__global__ void forward_record_range(FeatureMajorTilesView<const __half> x,
    const uint32_t *logical_rows, const float *w, float *y) {
    const uint32_t warp = threadIdx.x / warp_rows;
    const uint32_t lane = threadIdx.x % warp_rows;
    const auto &s = x.structure;
    uint32_t begin = blockIdx.x * forward_records_per_block;
    const uint32_t available = s.record_count - begin;
    const uint32_t end = begin + min(available, forward_records_per_block);
    __shared__ uint32_t tile;
    __shared__ uint32_t active_rows[warps_per_block];
    __shared__ float dense_features[warps_per_block][width];
    // Padding keeps row-parallel stores and channel-parallel reads from
    // placing each group's row data in the same shared memory banks.
    __shared__ float partials[warps_per_block][warp_rows][width + 1];
    float *features = dense_features[warp];
    while (begin < end) {
        if (threadIdx.x == 0) tile = find_record_tile(s, begin);
        __syncthreads();
        const uint32_t tile_end = min(end, s.tile_offsets[tile + 1]);
        const uint32_t count = tile_end - begin;
        const uint32_t first_row = tile * warp_rows;
        bool present = false;
        float accum[width]{};
        for (uint32_t r = warp; r < count; r += warps_per_block) {
            const uint32_t record = begin + r;
            if (lane < width) features[lane] = w[size_t(s.feature_ids[record]) * width + lane];
            __syncwarp();
            const uint32_t mask = s.row_masks[record];
            if (mask & (1u << lane)) {
                present = true;
                const uint32_t p = tile_value_position(s.value_offsets[record], mask, lane);
                const float value = __half2float(x.values[p]);
                #pragma unroll
                for (uint32_t column = 0; column < width; ++column)
                    accum[column] += value * features[column];
            }
            __syncwarp();
        }
        const uint32_t own_rows = __ballot_sync(warp_mask, present);
        if (lane == 0) active_rows[warp] = own_rows;
        #pragma unroll
        for (uint32_t column = 0; column < width; ++column)
            partials[warp][lane][column] = accum[column];
        __syncthreads();

        uint32_t mask = 0;
        #pragma unroll
        for (uint32_t warp_id = 0; warp_id < warps_per_block; ++warp_id)
            mask |= active_rows[warp_id];
        for (uint32_t i = threadIdx.x; i < warp_rows * width; i += blockDim.x) {
            const uint32_t row = i / width, column = i % width;
            if (mask & (1u << row)) {
                float sum = 0;
                #pragma unroll
                for (uint32_t warp_id = 0; warp_id < warps_per_block; ++warp_id)
                    sum += partials[warp_id][row][column];
                atomicAdd(y + size_t(logical_row(logical_rows, first_row + row)) * width + column,
                    sum);
            }
        }
        begin = tile_end;
        if (begin < end) __syncthreads();
    }
}

__global__ void sampled_record_range(FeatureMajorTileStructureView s,
    const uint32_t *logical_rows, const float *g, const float *w, float *dx) {
    const uint32_t warp = blockIdx.x * warps_per_block + threadIdx.x / warp_rows;
    const uint32_t begin = warp * sampled_records_per_warp;
    if (begin >= s.record_count) return;
    const uint32_t end = begin + min(s.record_count - begin, sampled_records_per_warp);
    const uint32_t column = threadIdx.x % width;
    const uint32_t half_warp = threadIdx.x % warp_rows / width;
    uint32_t tile = find_record_tile(s, begin);
    for (uint32_t record = begin; record < end; ++record) {
        if (record == s.tile_offsets[tile + 1]) tile = find_record_tile(s, record);
        const float weight = w[size_t(s.feature_ids[record]) * width + column];
        uint32_t mask = s.row_masks[record];
        uint32_t position = s.value_offsets[record];
        while (mask) {
            const uint32_t remaining = mask & (mask - 1u);
            const uint32_t rows = half_warp ? remaining : mask;
            float dot = 0;
            if (rows) {
                const uint32_t row = tile * warp_rows
                    + uint32_t(__ffs(static_cast<int>(rows)) - 1);
                dot = weight * g[size_t(logical_row(logical_rows, row)) * width + column];
            }
            #pragma unroll
            for (uint32_t delta = width / 2; delta; delta /= 2)
                dot += __shfl_down_sync(warp_mask, dot, delta, width);
            if (column == 0 && rows) dx[position + half_warp] = dot;
            mask = remaining & (remaining - 1u);
            position += 2;
        }
    }
}

// Each half warp owns the 16 channels of an active row. Pairs of set
// bits are visited in increasing order and share each record's mask/base.
__device__ float transpose_range(FeatureMajorTilesView<const __half> x,
    FeatureMajorTilesIndexView index, const float *g, uint32_t begin, uint32_t end,
    uint32_t lane) {
    const uint32_t column = lane % width;
    const uint32_t half_warp = lane / width;
    float sum = 0;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t record = index.record_ids[i];
        const uint32_t first_row = index.tile_ids[i] * x.structure.tile_row_width;
        uint32_t mask = x.structure.row_masks[record];
        uint32_t position = x.structure.value_offsets[record];
        while (mask) {
            const uint32_t remaining = mask & (mask - 1u);
            const uint32_t rows = half_warp ? remaining : mask;
            float value = 0;
            if (column == 0 && rows) value = __half2float(x.values[position + half_warp]);
            value = __shfl_sync(warp_mask, value, 0, width);
            if (rows) {
                const uint32_t row = first_row + uint32_t(__ffs(static_cast<int>(rows)) - 1);
                sum += value * g[size_t(logical_row(index.logical_rows, row)) * width + column];
            }
            mask = remaining & (remaining - 1u);
            position += 2;
        }
    }
    return sum + __shfl_xor_sync(warp_mask, sum, width);
}

__global__ void transpose_partials(FeatureMajorTilesView<const __half> x,
    FeatureMajorTilesIndexView index, const float *g, float *partials) {
    const uint32_t lane = threadIdx.x % warp_rows;
    const uint32_t chunk = blockIdx.x * warps_per_block + threadIdx.x / warp_rows;
    if (chunk >= index.chunk_count) return;
    const auto range = index.chunks[chunk];
    const float sum = transpose_range(x, index, g, range.begin, range.end, lane);
    if (lane < width) partials[size_t(chunk) * width + lane] = sum;
}

__global__ void transpose_features(FeatureMajorTilesView<const __half> x,
    FeatureMajorTilesIndexView index, const float *g, const float *partials, float *dw) {
    const uint32_t lane = threadIdx.x % warp_rows;
    const uint32_t feature = blockIdx.x * warps_per_block + threadIdx.x / warp_rows;
    if (feature >= x.structure.features) return;
    if (index.chunk_count) {
        const uint32_t begin = index.feature_chunk_offsets[feature];
        const uint32_t end = index.feature_chunk_offsets[feature + 1];
        if (begin != end) {
            if (lane < width) {
                float sum = 0;
                for (uint32_t chunk = begin; chunk < end; ++chunk)
                    sum += partials[size_t(chunk) * width + lane];
                dw[size_t(feature) * width + lane] = sum;
            }
            return;
        }
    }
    const float sum = transpose_range(x, index, g, index.feature_offsets[feature],
        index.feature_offsets[feature + 1], lane);
    if (lane < width) dw[size_t(feature) * width + lane] = sum;
}

__global__ void gather_values(const __half *source, const uint32_t *packed_to_source,
    __half *packed, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) packed[i] = source[packed_to_source[i]];
}

__global__ void initialize_master(const float *source, const uint32_t *packed_to_source,
    float *master, __half *packed, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = source[packed_to_source[i]];
        master[i] = value;
        packed[i] = __float2half_rn(value);
    }
}

__global__ void cast_master(const float *master, __half *packed, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) packed[i] = __float2half_rn(master[i]);
}

__global__ void sparse_sgd(float *master, const float *dx, float learning_rate,
    __half *packed, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = master[i] - learning_rate * dx[i];
        master[i] = value;
        packed[i] = __float2half_rn(value);
    }
}

__global__ void export_gradient(const float *dx, const uint32_t *packed_to_source,
    float *gradient, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gradient[packed_to_source[i]] = dx[i];
}

} // namespace

namespace detail {

cudaError_t enqueue_feature_major_tiles_forward(FeatureMajorTilesView<const __half> x,
    const uint32_t *logical_rows, const float *w, uint32_t dense_width, float *y,
    cudaStream_t stream) noexcept {
    const auto &s = x.structure;
    if (!valid_structure(s) || !dense_width || dense_width > width)
        return cudaErrorInvalidValue;
    if (!s.rows) return cudaSuccess;
    if (!y) return cudaErrorInvalidValue;
    if (!s.nnz) return cudaMemsetAsync(y, 0, size_t(s.rows) * dense_width * sizeof(float), stream);
    if (!w || !x.values) return cudaErrorInvalidValue;

    const uint32_t blocks = block_count(s.tile_count, warps_per_block);
    const uint32_t threads = warps_per_block * warp_rows;
    if (dense_width == width)
        tile_forward<width><<<blocks, threads, 0, stream>>>(x, logical_rows, w, dense_width, y);
    else
        tile_forward<0><<<blocks, threads, 0, stream>>>(x, logical_rows, w, dense_width, y);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_feature_major_values(const __half *source, const uint32_t *packed_to_source,
    __half *packed, uint32_t count, cudaStream_t stream) noexcept {
    if (!count) return cudaSuccess;
    if (!source || !packed_to_source || !packed) return cudaErrorInvalidValue;
    gather_values<<<block_count(count, vector_threads), vector_threads, 0, stream>>>(
        source, packed_to_source, packed, count);
    return cudaPeekAtLastError();
}

} // namespace detail

cudaError_t feature_major_tiles_spmm(FeatureMajorTilesView<const __half> x,
    FeatureMajorTilesIndexView index, const float *w, float *y, cudaStream_t stream) noexcept {
    const auto &s = x.structure;
    if (s.tile_row_width != warp_rows) return cudaErrorInvalidValue;
    // Short tiles can use one warp in a batch that supplies enough tiles.
    const bool many_short_tiles = s.tile_count >= 512
        && s.record_count <= uint64_t(s.tile_count) * forward_records_per_block;
    if (s.record_count <= forward_records_per_block || many_short_tiles)
        return detail::enqueue_feature_major_tiles_forward(x, index.logical_rows, w, width, y, stream);
    if (!valid_structure(s) || !w || !x.values || !y) return cudaErrorInvalidValue;
    const auto zeroed = cudaMemsetAsync(y, 0, size_t(s.rows) * width * sizeof(float), stream);
    if (zeroed != cudaSuccess) return zeroed;
    forward_record_range<<<block_count(s.record_count, forward_records_per_block),
        warps_per_block * warp_rows, 0, stream>>>(x, index.logical_rows, w, y);
    return cudaPeekAtLastError();
}

size_t feature_major_tiles_transpose_partial_count(FeatureMajorTilesIndexView index) noexcept {
    return size_t(index.chunk_count) * width;
}

cudaError_t feature_major_tiles_spmm_transpose(FeatureMajorTilesView<const __half> x,
    FeatureMajorTilesIndexView index, const float *g, float *dw,
    float *partials, size_t partial_count, cudaStream_t stream) noexcept {
    const auto &s = x.structure;
    if (s.tile_row_width != warp_rows || !valid_structure(s)) return cudaErrorInvalidValue;
    if (!s.features) return cudaSuccess;
    if (!dw) return cudaErrorInvalidValue;
    if (!s.nnz) return cudaMemsetAsync(dw, 0, size_t(s.features) * width * sizeof(float), stream);
    if (!g || !x.values || !index.feature_offsets || !index.record_ids || !index.tile_ids
        || index.chunk_count > s.record_count
        || (index.chunk_count && (!index.feature_chunk_offsets || !index.chunks || !partials
            || partial_count < feature_major_tiles_transpose_partial_count(index))))
        return cudaErrorInvalidValue;

    const uint32_t threads = warps_per_block * warp_rows;
    if (index.chunk_count) {
        transpose_partials<<<block_count(index.chunk_count, warps_per_block), threads, 0, stream>>>(
            x, index, g, partials);
        const cudaError_t status = cudaPeekAtLastError();
        if (status != cudaSuccess) return status;
    }
    transpose_features<<<block_count(s.features, warps_per_block), threads, 0, stream>>>(
        x, index, g, partials, dw);
    return cudaPeekAtLastError();
}

cudaError_t feature_major_tiles_sampled_gradient(FeatureMajorTileStructureView structure,
    FeatureMajorTilesIndexView index, const float *g, const float *w, float *dx,
    cudaStream_t stream) noexcept {
    if (structure.tile_row_width != warp_rows || !valid_structure(structure))
        return cudaErrorInvalidValue;
    if (!structure.nnz) return cudaSuccess;
    if (!g || !w || !dx) return cudaErrorInvalidValue;
    sampled_record_range<<<block_count(structure.record_count, warps_per_block * sampled_records_per_warp),
        warps_per_block * warp_rows, 0, stream>>>(structure, index.logical_rows, g, w, dx);
    return cudaPeekAtLastError();
}

cudaError_t feature_major_tiles_refresh(FeatureMajorTilesView<__half> x,
    FeatureMajorTilesIndexView index, const __half *csr_values, cudaStream_t stream) noexcept {
    return detail::enqueue_feature_major_values(csr_values, index.source_value_positions,
        x.values, x.structure.nnz, stream);
}

cudaError_t feature_major_tiles_initialize_master(FeatureMajorTilesView<__half> x,
    FeatureMajorTilesIndexView index, const float *csr_values, float *packed_master,
    cudaStream_t stream) noexcept {
    const uint32_t count = x.structure.nnz;
    if (!count) return cudaSuccess;
    if (!index.source_value_positions || !csr_values || !packed_master || !x.values)
        return cudaErrorInvalidValue;
    initialize_master<<<block_count(count, vector_threads), vector_threads, 0, stream>>>(
        csr_values, index.source_value_positions, packed_master, x.values, count);
    return cudaPeekAtLastError();
}

cudaError_t feature_major_tiles_cast_master(FeatureMajorTilesView<__half> x,
    const float *packed_master, cudaStream_t stream) noexcept {
    const uint32_t count = x.structure.nnz;
    if (!count) return cudaSuccess;
    if (!packed_master || !x.values) return cudaErrorInvalidValue;
    cast_master<<<block_count(count, vector_threads), vector_threads, 0, stream>>>(
        packed_master, x.values, count);
    return cudaPeekAtLastError();
}

cudaError_t feature_major_tiles_sparse_sgd(FeatureMajorTilesView<__half> x,
    float *packed_master, const float *dx, float learning_rate, cudaStream_t stream) noexcept {
    const uint32_t count = x.structure.nnz;
    if (!count) return cudaSuccess;
    if (!packed_master || !dx || !x.values) return cudaErrorInvalidValue;
    sparse_sgd<<<block_count(count, vector_threads), vector_threads, 0, stream>>>(
        packed_master, dx, learning_rate, x.values, count);
    return cudaPeekAtLastError();
}

cudaError_t feature_major_tiles_export_gradient(FeatureMajorTileStructureView structure,
    FeatureMajorTilesIndexView index, const float *dx, float *csr_gradient,
    cudaStream_t stream) noexcept {
    const uint32_t count = structure.nnz;
    if (!count) return cudaSuccess;
    if (!index.source_value_positions || !dx || !csr_gradient) return cudaErrorInvalidValue;
    export_gradient<<<block_count(count, vector_threads), vector_threads, 0, stream>>>(
        dx, index.source_value_positions, csr_gradient, count);
    return cudaPeekAtLastError();
}

} // namespace cellerator::ops::matrix
