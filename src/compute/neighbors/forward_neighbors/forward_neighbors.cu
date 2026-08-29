#include <Cellerator/compute/neighbors/forward_neighbors.hh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cellerator::compute::neighbors::forward_neighbors {
namespace detail {

constexpr int kWarpThreads = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpThreads * kWarpsPerBlock;
constexpr std::uint32_t kInvalidBlockColumn = 0xffffffffu;

struct ProbeCandidate {
    float similarity = -INFINITY;
    std::int32_t list_offset = -1;
};

struct RoutedCandidate {
    float similarity = -INFINITY;
    float developmental_time = INFINITY;
    std::int64_t embryo_id = -1;
    std::int64_t cell_index = -1;
};

inline void cuda_require_(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

inline std::size_t checked_elements_(std::int64_t rows, int width, const char *label) {
    if (rows < 0 || width <= 0) throw std::invalid_argument(std::string(label) + " dimensions must be positive");
    const std::size_t row_count = static_cast<std::size_t>(rows);
    const std::size_t item_width = static_cast<std::size_t>(width);
    if (row_count > std::numeric_limits<std::size_t>::max() / item_width) {
        throw std::overflow_error(std::string(label) + " element count overflows size_t");
    }
    return row_count * item_width;
}

inline unsigned int blocks_for_rows_(std::int64_t rows) {
    const std::uint64_t blocks = (static_cast<std::uint64_t>(rows) + kWarpsPerBlock - 1u) / kWarpsPerBlock;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("forward-neighbor launch grid exceeds CUDA grid.x capacity");
    }
    return static_cast<unsigned int>(blocks);
}

inline void validate_query_(const ForwardNeighborQueryDeviceView &query) {
    if (query.rows < 0 || query.latent_dim <= 0) {
        throw std::invalid_argument("forward-neighbor query dimensions are invalid");
    }
    if (query.rows > 0 && (query.latent == nullptr || query.window_lower == nullptr || query.window_upper == nullptr)) {
        throw std::invalid_argument("forward-neighbor query is missing required device buffers");
    }
}

inline void validate_lists_(
    const ForwardNeighborAnnListDeviceView &lists,
    int latent_dim) {
    if (lists.count < 0 || lists.latent_dim != latent_dim) {
        throw std::invalid_argument("forward-neighbor ANN list dimensions do not match the query");
    }
    if (lists.count > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("forward-neighbor ANN list count exceeds compact offset capacity");
    }
    if (lists.count > 0 && (lists.centroids == nullptr || lists.row_begin == nullptr || lists.row_end == nullptr)) {
        throw std::invalid_argument("forward-neighbor ANN lists are missing required device buffers");
    }
}

inline void validate_config_(const ForwardNeighborAnnSearchConfig &config) {
    if (config.probe_count <= 0 || config.probe_count > kForwardNeighborMaxProbeCount) {
        throw std::invalid_argument("forward-neighbor probe count exceeds the bounded native contract");
    }
    if (config.top_k <= 0 || config.top_k > kForwardNeighborMaxTopK) {
        throw std::invalid_argument("forward-neighbor top-k exceeds the bounded native contract");
    }
}

inline void validate_policy_(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config) {
    if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only
        && query.rows > 0 && (query.embryo_ids == nullptr || (lists.count > 0 && lists.embryo_ids == nullptr))) {
        throw std::invalid_argument("same-embryo forward-neighbor search requires query and list embryo identities");
    }
}

inline void validate_workspace_(
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    std::int64_t query_rows,
    int probe_count) {
    const std::size_t required = checked_elements_(query_rows, probe_count, "forward-neighbor workspace");
    if (workspace.selected_list_capacity < required || (required > 0 && workspace.selected_list_offsets == nullptr)) {
        throw std::invalid_argument("forward-neighbor ANN workspace capacity is insufficient");
    }
}

inline void validate_result_(
    const ForwardNeighborResultDeviceView &result,
    std::int64_t query_rows,
    int top_k) {
    const std::size_t required = checked_elements_(query_rows, top_k, "forward-neighbor result");
    if (result.capacity < required) throw std::invalid_argument("forward-neighbor result capacity is insufficient");
    if (required > 0 && (result.cell_indices == nullptr || result.shard_indices == nullptr
            || result.developmental_time == nullptr || result.embryo_ids == nullptr || result.similarity == nullptr)) {
        throw std::invalid_argument("forward-neighbor result is missing required device buffers");
    }
}

template <typename Index>
inline void validate_index_common_(const Index &index, int latent_dim) {
    if (index.rows < 0 || index.latent_dim != latent_dim || index.shard_index < 0) {
        throw std::invalid_argument("forward-neighbor index identity or dimensions are invalid");
    }
    if (index.rows > 0 && (index.developmental_time == nullptr
            || index.embryo_ids == nullptr || index.cell_indices == nullptr)) {
        throw std::invalid_argument("forward-neighbor index is missing identity buffers");
    }
}

inline void validate_index_(const ForwardNeighborDenseIndexDeviceView &index, int latent_dim) {
    validate_index_common_(index, latent_dim);
    if (index.rows > 0 && index.latent == nullptr) {
        throw std::invalid_argument("dense forward-neighbor index is missing latent values");
    }
}

inline void validate_index_(const ForwardNeighborBlockedEllIndexDeviceView &index, int latent_dim) {
    validate_index_common_(index, latent_dim);
    if (index.block_size <= 0 || index.ell_cols <= 0 || index.ell_cols % index.block_size != 0) {
        throw std::invalid_argument("blocked-ELL forward-neighbor geometry is invalid");
    }
    if (index.rows > 0 && (index.block_col_indices == nullptr || index.values == nullptr)) {
        throw std::invalid_argument("blocked-ELL forward-neighbor index is missing projection buffers");
    }
}

inline void validate_index_(const ForwardNeighborSlicedEllIndexDeviceView &index, int latent_dim) {
    validate_index_common_(index, latent_dim);
    if (index.rows > 0 && (index.row_slot_offsets == nullptr || index.row_widths == nullptr
            || index.col_indices == nullptr || index.values == nullptr)) {
        throw std::invalid_argument("sliced-ELL forward-neighbor index is missing projection buffers");
    }
}

__device__ inline std::int64_t shfl_i64_(unsigned mask, std::int64_t value, int source_lane) {
    const std::uint64_t bits = static_cast<std::uint64_t>(value);
    const std::uint32_t lo = __shfl_sync(mask, static_cast<std::uint32_t>(bits), source_lane);
    const std::uint32_t hi = __shfl_sync(mask, static_cast<std::uint32_t>(bits >> 32), source_lane);
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(hi) << 32) | lo);
}

__device__ inline bool better_probe_(const ProbeCandidate &lhs, const ProbeCandidate &rhs) {
    if (!isfinite(lhs.similarity) || lhs.list_offset < 0) return false;
    if (!isfinite(rhs.similarity) || rhs.list_offset < 0) return true;
    if (lhs.similarity != rhs.similarity) return lhs.similarity > rhs.similarity;
    return lhs.list_offset < rhs.list_offset;
}

__device__ inline bool better_routed_(const RoutedCandidate &lhs, const RoutedCandidate &rhs) {
    if (!isfinite(lhs.similarity) || lhs.cell_index < 0) return false;
    if (!isfinite(rhs.similarity) || rhs.cell_index < 0) return true;
    if (lhs.similarity != rhs.similarity) return lhs.similarity > rhs.similarity;
    if (lhs.developmental_time != rhs.developmental_time) return lhs.developmental_time < rhs.developmental_time;
    if (lhs.embryo_id != rhs.embryo_id) return lhs.embryo_id < rhs.embryo_id;
    return lhs.cell_index < rhs.cell_index;
}

template <typename Candidate>
__device__ inline void initialize_candidates_(Candidate *items, int count) {
    for (int i = 0; i < count; ++i) items[i] = Candidate{};
}

__device__ inline void insert_probe_(ProbeCandidate candidate, ProbeCandidate *best, int count) {
    if (!better_probe_(candidate, best[count - 1])) return;
    int position = count - 1;
    while (position > 0 && better_probe_(candidate, best[position - 1])) {
        best[position] = best[position - 1];
        --position;
    }
    best[position] = candidate;
}

__device__ inline void insert_routed_(RoutedCandidate candidate, RoutedCandidate *best, int count) {
    if (!better_routed_(candidate, best[count - 1])) return;
    int position = count - 1;
    while (position > 0 && better_routed_(candidate, best[position - 1])) {
        best[position] = best[position - 1];
        --position;
    }
    best[position] = candidate;
}

__device__ inline ProbeCandidate shfl_probe_(unsigned mask, ProbeCandidate candidate, int source_lane) {
    return ProbeCandidate{
        __shfl_sync(mask, candidate.similarity, source_lane),
        __shfl_sync(mask, candidate.list_offset, source_lane)};
}

__device__ inline RoutedCandidate shfl_routed_(unsigned mask, RoutedCandidate candidate, int source_lane) {
    return RoutedCandidate{
        __shfl_sync(mask, candidate.similarity, source_lane),
        __shfl_sync(mask, candidate.developmental_time, source_lane),
        shfl_i64_(mask, candidate.embryo_id, source_lane),
        shfl_i64_(mask, candidate.cell_index, source_lane)};
}

__device__ inline float dot_half_float_(const __half *lhs, const float *rhs, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += __half2float(lhs[i]) * rhs[i];
    return sum;
}

__device__ inline float score_row_(
    const ForwardNeighborDenseIndexDeviceView &index,
    const __half *query,
    std::int64_t row) {
    const __half *rhs = index.latent + row * static_cast<std::int64_t>(index.latent_dim);
    float sum = 0.0f;
    int col = 0;
    for (; col + 1 < index.latent_dim; col += 2) {
        const float2 lhs2 = __half22float2(*reinterpret_cast<const __half2 *>(query + col));
        const float2 rhs2 = __half22float2(*reinterpret_cast<const __half2 *>(rhs + col));
        sum += lhs2.x * rhs2.x + lhs2.y * rhs2.y;
    }
    if (col < index.latent_dim) sum += __half2float(query[col]) * __half2float(rhs[col]);
    return sum;
}

__device__ inline float score_row_(
    const ForwardNeighborBlockedEllIndexDeviceView &index,
    const __half *query,
    std::int64_t row) {
    const int width = index.ell_cols / index.block_size;
    const std::int64_t row_block = row / index.block_size;
    float sum = 0.0f;
    for (int slot = 0; slot < width; ++slot) {
        const std::uint32_t block_column = index.block_col_indices[row_block * width + slot];
        if (block_column == kInvalidBlockColumn) continue;
        const std::int64_t base_column = static_cast<std::int64_t>(block_column) * index.block_size;
        const std::int64_t value_base = row * index.ell_cols + static_cast<std::int64_t>(slot) * index.block_size;
        for (int local = 0; local < index.block_size && base_column + local < index.latent_dim; ++local) {
            sum += __half2float(query[base_column + local]) * __half2float(index.values[value_base + local]);
        }
    }
    return sum;
}

__device__ inline float score_row_(
    const ForwardNeighborSlicedEllIndexDeviceView &index,
    const __half *query,
    std::int64_t row) {
    const std::size_t begin = static_cast<std::size_t>(index.row_slot_offsets[row]);
    const std::uint32_t width = index.row_widths[row];
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < width; ++slot) {
        const std::size_t offset = begin + slot;
        const std::uint32_t column = index.col_indices[offset];
        if (column < static_cast<std::uint32_t>(index.latent_dim)) {
            sum += __half2float(query[column]) * __half2float(index.values[offset]);
        }
    }
    return sum;
}

__global__ void initialize_result_kernel_(
    ForwardNeighborResultDeviceView result,
    std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    result.cell_indices[index] = -1;
    result.shard_indices[index] = -1;
    result.developmental_time[index] = INFINITY;
    result.embryo_ids[index] = -1;
    result.similarity[index] = -INFINITY;
}

template <int Capacity>
__global__ void select_lists_kernel_(
    ForwardNeighborQueryDeviceView query,
    ForwardNeighborAnnListDeviceView lists,
    ForwardNeighborAnnSearchConfig config,
    ForwardNeighborAnnWorkspaceDeviceView workspace) {
    static_assert(Capacity > 0 && Capacity <= kForwardNeighborMaxProbeCount);
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= query.rows) return;
    const unsigned mask = __activemask();
    __shared__ ProbeCandidate merged_storage[kWarpsPerBlock * Capacity];
    ProbeCandidate *merged = merged_storage + warp * Capacity;
    ProbeCandidate local[Capacity];
    initialize_candidates_(local, config.probe_count);
    const __half *query_row = query.latent + row * static_cast<std::int64_t>(query.latent_dim);
    const std::int64_t query_embryo = query.embryo_ids != nullptr ? query.embryo_ids[row] : -1;
    for (std::int64_t list = lane; list < lists.count; list += kWarpThreads) {
        if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only
            && query_embryo >= 0 && lists.embryo_ids != nullptr && lists.embryo_ids[list] != query_embryo) continue;
        insert_probe_(ProbeCandidate{
            dot_half_float_(query_row, lists.centroids + list * lists.latent_dim, lists.latent_dim),
            static_cast<std::int32_t>(list)}, local, config.probe_count);
    }
    if (lane == 0) initialize_candidates_(merged, config.probe_count);
    for (int source_lane = 0; source_lane < kWarpThreads; ++source_lane) {
        for (int slot = 0; slot < config.probe_count; ++slot) {
            const ProbeCandidate candidate = shfl_probe_(mask, local[slot], source_lane);
            if (lane == 0) insert_probe_(candidate, merged, config.probe_count);
        }
    }
    if (lane == 0) {
        const std::size_t base = static_cast<std::size_t>(row) * config.probe_count;
        for (int slot = 0; slot < config.probe_count; ++slot) {
            workspace.selected_list_offsets[base + slot] = merged[slot].list_offset;
        }
    }
}

template <int Capacity, typename Index>
__global__ void refine_kernel_(
    ForwardNeighborQueryDeviceView query,
    Index index,
    ForwardNeighborAnnListDeviceView lists,
    ForwardNeighborAnnSearchConfig config,
    ForwardNeighborAnnWorkspaceDeviceView workspace,
    ForwardNeighborResultDeviceView result) {
    static_assert(Capacity > 0 && Capacity <= kForwardNeighborMaxTopK);
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= query.rows) return;
    const unsigned mask = __activemask();
    __shared__ RoutedCandidate merged_storage[kWarpsPerBlock * Capacity];
    RoutedCandidate *merged = merged_storage + warp * Capacity;
    RoutedCandidate local[Capacity];
    initialize_candidates_(local, config.top_k);
    const __half *query_row = query.latent + row * static_cast<std::int64_t>(query.latent_dim);
    const float lower = query.window_lower[row];
    const float upper = query.window_upper[row];
    const std::int64_t query_embryo = query.embryo_ids != nullptr ? query.embryo_ids[row] : -1;
    const std::size_t selected_base = static_cast<std::size_t>(row) * config.probe_count;
    for (int probe = 0; probe < config.probe_count; ++probe) {
        const std::int32_t list_offset = workspace.selected_list_offsets[selected_base + probe];
        if (list_offset < 0 || list_offset >= lists.count) continue;
        const std::int64_t begin = lists.row_begin[list_offset];
        const std::int64_t end = lists.row_end[list_offset];
        if (begin < 0 || end < begin || end > index.rows) continue;
        for (std::int64_t index_row = begin + lane; index_row < end; index_row += kWarpThreads) {
            const float time = index.developmental_time[index_row];
            if (!(time > lower) || (isfinite(upper) && time > upper)) continue;
            if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only
                && query_embryo >= 0 && index.embryo_ids[index_row] != query_embryo) continue;
            insert_routed_(RoutedCandidate{
                score_row_(index, query_row, index_row),
                time,
                index.embryo_ids[index_row],
                index.cell_indices[index_row]}, local, config.top_k);
        }
    }
    if (lane == 0) initialize_candidates_(merged, config.top_k);
    for (int source_lane = 0; source_lane < kWarpThreads; ++source_lane) {
        for (int slot = 0; slot < config.top_k; ++slot) {
            const RoutedCandidate candidate = shfl_routed_(mask, local[slot], source_lane);
            if (lane == 0) insert_routed_(candidate, merged, config.top_k);
        }
    }
    if (lane == 0) {
        const std::size_t base = static_cast<std::size_t>(row) * config.top_k;
        for (int slot = 0; slot < config.top_k; ++slot) {
            result.cell_indices[base + slot] = merged[slot].cell_index;
            result.shard_indices[base + slot] = merged[slot].cell_index >= 0 ? index.shard_index : -1;
            result.developmental_time[base + slot] = merged[slot].developmental_time;
            result.embryo_ids[base + slot] = merged[slot].embryo_id;
            result.similarity[base + slot] = merged[slot].similarity;
        }
    }
}

template <typename Launch>
inline void dispatch_capacity_(int count, Launch &&launch) {
    if (count <= 1) launch(std::integral_constant<int, 1>{});
    else if (count <= 2) launch(std::integral_constant<int, 2>{});
    else if (count <= 4) launch(std::integral_constant<int, 4>{});
    else if (count <= 8) launch(std::integral_constant<int, 8>{});
    else if (count <= 16) launch(std::integral_constant<int, 16>{});
    else launch(std::integral_constant<int, 32>{});
}

template <typename Index>
void refine_(
    const ForwardNeighborQueryDeviceView &query,
    const Index &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream) {
    validate_query_(query);
    validate_lists_(lists, query.latent_dim);
    validate_config_(config);
    validate_policy_(query, lists, config);
    validate_index_(index, query.latent_dim);
    validate_workspace_(workspace, query.rows, config.probe_count);
    validate_result_(result, query.rows, config.top_k);
    if (query.rows == 0) return;
    dispatch_capacity_(config.top_k, [&](auto capacity) {
        refine_kernel_<decltype(capacity)::value><<<blocks_for_rows_(query.rows), kThreadsPerBlock, 0, stream>>>(
            query, index, lists, config, workspace, result);
    });
    cuda_require_(cudaGetLastError(), "forward-neighbor ANN refine launch");
}

} // namespace detail

std::size_t forward_neighbor_result_elements(std::int64_t query_rows, int top_k) {
    if (top_k > kForwardNeighborMaxTopK) throw std::invalid_argument("forward-neighbor top-k exceeds native capacity");
    return detail::checked_elements_(query_rows, top_k, "forward-neighbor result");
}

std::size_t forward_neighbor_ann_workspace_elements(std::int64_t query_rows, int probe_count) {
    if (probe_count > kForwardNeighborMaxProbeCount) {
        throw std::invalid_argument("forward-neighbor probe count exceeds native capacity");
    }
    return detail::checked_elements_(query_rows, probe_count, "forward-neighbor workspace");
}

void initialize_forward_neighbor_result(
    const ForwardNeighborResultDeviceView &result,
    std::int64_t query_rows,
    int top_k,
    cudaStream_t stream) {
    if (top_k <= 0 || top_k > kForwardNeighborMaxTopK) {
        throw std::invalid_argument("forward-neighbor top-k exceeds native capacity");
    }
    detail::validate_result_(result, query_rows, top_k);
    const std::size_t count = detail::checked_elements_(query_rows, top_k, "forward-neighbor result");
    if (count == 0) return;
    constexpr unsigned int threads = 128;
    const std::size_t block_count = (count + threads - 1u) / threads;
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("forward-neighbor result launch exceeds CUDA grid.x capacity");
    }
    detail::initialize_result_kernel_<<<static_cast<unsigned int>(block_count), threads, 0, stream>>>(result, count);
    detail::cuda_require_(cudaGetLastError(), "forward-neighbor result initialization launch");
}

void select_forward_neighbor_ann_lists(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    cudaStream_t stream) {
    detail::validate_query_(query);
    detail::validate_lists_(lists, query.latent_dim);
    detail::validate_config_(config);
    detail::validate_policy_(query, lists, config);
    detail::validate_workspace_(workspace, query.rows, config.probe_count);
    if (query.rows == 0) return;
    detail::dispatch_capacity_(config.probe_count, [&](auto capacity) {
        detail::select_lists_kernel_<decltype(capacity)::value>
            <<<detail::blocks_for_rows_(query.rows), detail::kThreadsPerBlock, 0, stream>>>(query, lists, config, workspace);
    });
    detail::cuda_require_(cudaGetLastError(), "forward-neighbor ANN probe launch");
}

void refine_forward_neighbors_dense(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborDenseIndexDeviceView &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream) {
    detail::refine_(query, index, lists, config, workspace, result, stream);
}

void refine_forward_neighbors_blocked_ell(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborBlockedEllIndexDeviceView &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream) {
    detail::refine_(query, index, lists, config, workspace, result, stream);
}

void refine_forward_neighbors_sliced_ell(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborSlicedEllIndexDeviceView &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream) {
    detail::refine_(query, index, lists, config, workspace, result, stream);
}

} // namespace cellerator::compute::neighbors::forward_neighbors
