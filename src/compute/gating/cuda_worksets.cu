#include <Cellerator/compute/gating/cuda_worksets.cuh>

#include <cuda_runtime.h>

#include <limits>

namespace cellerator::compute::gating {
namespace {

std::uint64_t host_words_per_cell(const condition_program_view_v1 &p) {
    return p.edge_count / 64u + (p.edge_count % 64u != 0u);
}

__device__ __forceinline__ std::uint64_t words_per_cell(
    const condition_program_view_v1 &p) {
    return p.edge_count / 64u + (p.edge_count % 64u != 0u);
}

__device__ __forceinline__ bool eval_nodes(
    const condition_program_view_v1 &p, const std::uint8_t *predicates,
    std::uint8_t *scratch) {
    for (std::uint64_t i = 0; i < p.node_count; ++i) {
        const auto &n = p.nodes[i];
        switch (n.opcode) {
        case condition_opcode_v1::predicate:
            scratch[i] = predicates[n.predicate_index] == n.expected_value;
            break;
        case condition_opcode_v1::all_of: {
            bool value = true;
            for (std::uint32_t j = 0; j < n.child_count; ++j)
                value = value && scratch[n.first_child + j] != 0;
            scratch[i] = value;
            break;
        }
        case condition_opcode_v1::any_of: {
            bool value = false;
            for (std::uint32_t j = 0; j < n.child_count; ++j)
                value = value || scratch[n.first_child + j] != 0;
            scratch[i] = value;
            break;
        }
        case condition_opcode_v1::logical_not:
            scratch[i] = scratch[n.first_child] == 0;
            break;
        default:
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ const edge_condition_binding_v1 *find_binding(
    const condition_program_view_v1 &p, std::uint64_t ordinal) {
    std::uint64_t lo = 0, hi = p.binding_count;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (p.edges[mid].logical_edge_ordinal < ordinal) lo = mid + 1;
        else hi = mid;
    }
    return lo < p.binding_count && p.edges[lo].logical_edge_ordinal == ordinal
        ? p.edges + lo : nullptr;
}

template <bool use_prepared_map>
__device__ __forceinline__ bool condition_root_for_edge(
    const condition_program_view_v1 &p,
    const std::uint32_t *root_node_by_edge_ordinal,
    std::uint64_t ordinal, std::uint32_t *root,
    cuda_workset_status_v1 *status) {
    if constexpr (use_prepared_map) {
        *root = root_node_by_edge_ordinal[ordinal];
        if (*root != unbound_condition_root_v1 && *root >= p.node_count) {
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_workset_code_v1::invalid_argument));
            return false;
        }
        return true;
    } else {
        const auto *binding = find_binding(p, ordinal);
        *root = binding == nullptr ? unbound_condition_root_v1 : binding->root_node;
        return true;
    }
}

__device__ __forceinline__ bool eligible(
    const condition_program_view_v1 &p, const std::uint64_t *bitmap,
    const indexing::indexed_work_item_v1 &item) {
    return ((bitmap[item.logical_edge_ordinal / 64u]
             >> (item.logical_edge_ordinal % 64u)) & 1u) != 0;
}

__global__ void clear_status(cuda_workset_status_v1 *status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *status = {};
}

template <bool use_prepared_map>
__global__ void bitmap_full_kernel(
    condition_program_view_v1 p, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t *bitmaps,
    std::uint8_t *node_scratch, cuda_workset_status_v1 *status,
    const std::uint32_t *root_node_by_edge_ordinal) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    __shared__ int valid;
    auto *nodes = node_scratch == nullptr ? nullptr : node_scratch + cell * p.node_count;
    const auto *pred = predicates == nullptr ? nullptr : predicates + cell * p.predicate_count;
    if (threadIdx.x == 0) {
        valid = eval_nodes(p, pred, nodes);
        if (!valid)
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_workset_code_v1::invalid_argument));
    }
    __syncthreads();
    if (!valid) return;
    const auto words = words_per_cell(p);
    auto *bitmap = bitmaps == nullptr ? nullptr : bitmaps + cell * words;
    for (std::uint64_t w = threadIdx.x; w < words; w += blockDim.x) bitmap[w] = 0;
    __syncthreads();
    for (std::uint64_t edge = threadIdx.x; edge < p.edge_count; edge += blockDim.x) {
        std::uint32_t root = unbound_condition_root_v1;
        if (!condition_root_for_edge<use_prepared_map>(p,
                root_node_by_edge_ordinal, edge, &root, status)) continue;
        const bool pass = root == unbound_condition_root_v1 || nodes[root] != 0;
        if (pass)
            atomicOr(reinterpret_cast<unsigned long long *>(&bitmap[edge / 64u]),
                     static_cast<unsigned long long>(std::uint64_t{1} << (edge % 64u)));
    }
}

template <bool use_prepared_map>
__global__ void bitmap_dirty_kernel(
    condition_program_view_v1 p, std::uint64_t cells,
    const std::uint8_t *predicates, const std::uint64_t *dirty_offsets,
    const std::uint64_t *dirty_edges, std::uint64_t *bitmaps,
    std::uint8_t *node_scratch, cuda_workset_status_v1 *status,
    const std::uint32_t *root_node_by_edge_ordinal) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    __shared__ int valid;
    const auto begin = dirty_offsets[cell], end = dirty_offsets[cell + 1];
    auto *nodes = node_scratch == nullptr ? nullptr : node_scratch + cell * p.node_count;
    const auto *pred = predicates == nullptr ? nullptr : predicates + cell * p.predicate_count;
    if (threadIdx.x == 0) {
        valid = end >= begin;
        if (valid && begin != end) valid = eval_nodes(p, pred, nodes);
        if (valid)
            for (auto i = begin; i < end; ++i)
                if (dirty_edges[i] >= p.edge_count) { valid = 0; break; }
        if (!valid)
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_workset_code_v1::invalid_argument));
    }
    __syncthreads();
    if (!valid || begin == end) return;
    auto *bitmap = bitmaps == nullptr ? nullptr
        : bitmaps + cell * words_per_cell(p);
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const auto edge = dirty_edges[i];
        std::uint32_t root = unbound_condition_root_v1;
        if (!condition_root_for_edge<use_prepared_map>(p,
                root_node_by_edge_ordinal, edge, &root, status)) continue;
        const bool pass = root == unbound_condition_root_v1 || nodes[root] != 0;
        const auto bit = std::uint64_t{1} << (edge % 64u);
        auto *word = reinterpret_cast<unsigned long long *>(&bitmap[edge / 64u]);
        if (pass) atomicOr(word, static_cast<unsigned long long>(bit));
        else atomicAnd(word, static_cast<unsigned long long>(~bit));
    }
}

__global__ void stage_worksets_kernel(
    condition_program_view_v1 p, std::uint64_t cells,
    const std::uint64_t *bitmaps, cuda_candidate_csr_v1 primal,
    cuda_candidate_csr_v1 derivative, std::uint64_t *p_stage_offsets,
    indexing::indexed_work_item_v1 *p_stage, std::uint64_t *d_stage_offsets,
    indexing::indexed_work_item_v1 *d_stage, std::uint64_t *p_counts,
    std::uint64_t *d_counts, std::uint64_t p_capacity,
    std::uint64_t d_capacity, std::uint64_t p_offsets_capacity,
    std::uint64_t d_offsets_capacity, cuda_workset_status_v1 *status) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells || threadIdx.x != 0) return;
    const auto pb = primal.offsets[cell], pe = primal.offsets[cell + 1];
    const auto db = derivative.offsets[cell], de = derivative.offsets[cell + 1];
    const auto ps = p_stage_offsets[cell], pnext = p_stage_offsets[cell + 1];
    const auto ds = d_stage_offsets[cell], dnext = d_stage_offsets[cell + 1];
    if (pb > pe || pe > primal.item_count || db > de || de > derivative.item_count
        || ps > pnext || pnext > p_capacity || ds > dnext || dnext > d_capacity
        || cell + 1 >= p_offsets_capacity || cell + 1 >= d_offsets_capacity
        || pnext - ps < pe - pb || (de - db) > UINT64_MAX - (pe - pb)
        || dnext - ds < (pe - pb) + (de - db)) {
        atomicCAS(&status->code, 0u,
                  static_cast<std::uint32_t>(cuda_workset_code_v1::insufficient_scratch));
        p_counts[cell] = d_counts[cell] = 0;
        return;
    }
    for (auto i = pb; i < pe; ++i)
        if (primal.items[i].logical_edge_id == 0
            || primal.items[i].logical_edge_ordinal >= p.edge_count
            || (i != pb && primal.items[i - 1].logical_edge_id
                            >= primal.items[i].logical_edge_id)) {
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_workset_code_v1::invalid_argument));
            p_counts[cell] = d_counts[cell] = 0;
            return;
        }
    for (auto i = db; i < de; ++i)
        if (derivative.items[i].logical_edge_id == 0
            || derivative.items[i].logical_edge_ordinal >= p.edge_count
            || (i != db && derivative.items[i - 1].logical_edge_id
                              >= derivative.items[i].logical_edge_id)) {
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_workset_code_v1::invalid_argument));
            p_counts[cell] = d_counts[cell] = 0;
            return;
        }
    const auto *bitmap = bitmaps == nullptr ? nullptr
        : bitmaps + cell * words_per_cell(p);
    auto *prow = p_stage + ps;
    auto *drow = d_stage + ds;
    std::uint64_t pn = 0, dn = 0;
    for (auto i = pb; i < pe; ++i) {
        const auto x = primal.items[i];
        if (x.logical_edge_id != 0 && x.logical_edge_ordinal < p.edge_count
            && eligible(p, bitmap, x))
            prow[pn++] = x;
    }
    auto pi = pb, di = db;
    for (;;) {
        while (pi < pe && (primal.items[pi].logical_edge_id == 0
               || primal.items[pi].logical_edge_ordinal >= p.edge_count
               || !eligible(p, bitmap, primal.items[pi]))) ++pi;
        while (di < de && (derivative.items[di].logical_edge_id == 0
               || derivative.items[di].logical_edge_ordinal >= p.edge_count
               || !eligible(p, bitmap, derivative.items[di]))) ++di;
        if (pi == pe && di == de) break;
        if (di == de || (pi < pe && primal.items[pi].logical_edge_id
                                      < derivative.items[di].logical_edge_id)) {
            drow[dn++] = primal.items[pi++];
        } else if (pi == pe || derivative.items[di].logical_edge_id
                                      < primal.items[pi].logical_edge_id) {
            drow[dn++] = derivative.items[di++];
        } else {
            drow[dn++] = primal.items[pi++];
            ++di;
        }
    }
    p_counts[cell] = pn;
    d_counts[cell] = dn;
    atomicAdd(reinterpret_cast<unsigned long long *>(&status->primal_required),
              static_cast<unsigned long long>(pn));
    atomicAdd(reinterpret_cast<unsigned long long *>(&status->derivative_required),
              static_cast<unsigned long long>(dn));
}

__global__ void prepare_offsets_kernel(
    std::uint64_t cells, const std::uint64_t *p_counts,
    const std::uint64_t *d_counts, cuda_output_csr_v1 p_out,
    cuda_output_csr_v1 d_out, cuda_workset_status_v1 *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (status->primal_required > p_out.item_capacity) {
        status->code = static_cast<std::uint32_t>(cuda_workset_code_v1::primal_overflow);
        return;
    }
    if (status->derivative_required > d_out.item_capacity) {
        status->code = static_cast<std::uint32_t>(cuda_workset_code_v1::derivative_overflow);
        return;
    }
    p_out.offsets[0] = 0;
    d_out.offsets[0] = 0;
    std::uint64_t ps = 0, ds = 0;
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        ps += p_counts[cell];
        ds += d_counts[cell];
        p_out.offsets[cell + 1] = ps;
        d_out.offsets[cell + 1] = ds;
    }
}

__global__ void scatter_worksets_kernel(
    std::uint64_t cells, const std::uint64_t *p_counts,
    const std::uint64_t *d_counts, const std::uint64_t *p_stage_offsets,
    const indexing::indexed_work_item_v1 *p_stage,
    const std::uint64_t *d_stage_offsets,
    const indexing::indexed_work_item_v1 *d_stage,
    cuda_output_csr_v1 p_out, cuda_output_csr_v1 d_out,
    const cuda_workset_status_v1 *status) {
    if (status->code != 0) return;
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells || threadIdx.x != 0) return;
    for (std::uint64_t i = 0; i < p_counts[cell]; ++i)
        p_out.items[p_out.offsets[cell] + i] = p_stage[p_stage_offsets[cell] + i];
    for (std::uint64_t i = 0; i < d_counts[cell]; ++i)
        d_out.items[d_out.offsets[cell] + i] = d_stage[d_stage_offsets[cell] + i];
}

bool mul_overflow(std::uint64_t a, std::uint64_t b, std::uint64_t *out) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return true;
    *out = a * b;
    return false;
}

template <bool use_prepared_map>
cudaError_t evaluate_condition_bitmaps_impl(
    const condition_program_view_v1 &p,
    const std::uint32_t *root_node_by_edge_ordinal,
    std::uint64_t root_map_capacity, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t predicate_count,
    std::uint64_t *bitmaps, std::uint64_t bitmap_capacity,
    std::uint8_t *node_scratch, std::uint64_t scratch_capacity,
    cuda_workset_status_v1 *device_status, cudaStream_t stream) noexcept {
    std::uint64_t expected_predicates = 0, words = 0, expected_words = 0,
                  expected_scratch = 0;
    if (cells > 0 && p.predicate_count > 0
        && mul_overflow(cells, p.predicate_count, &expected_predicates))
        return cudaErrorInvalidValue;
    words = host_words_per_cell(p);
    if (mul_overflow(cells, words, &expected_words)
        || mul_overflow(cells, p.node_count, &expected_scratch))
        return cudaErrorInvalidValue;
    if (p.structure_epoch == 0 || predicate_count != expected_predicates
        || (predicate_count && predicates == nullptr)
        || (expected_words && bitmaps == nullptr) || bitmap_capacity < expected_words
        || (expected_scratch && node_scratch == nullptr)
        || scratch_capacity < expected_scratch || device_status == nullptr
        || (use_prepared_map && (root_map_capacity < p.edge_count
            || (p.edge_count != 0u && root_node_by_edge_ordinal == nullptr)))
        || (!use_prepared_map && root_node_by_edge_ordinal != nullptr))
        return cudaErrorInvalidValue;
    clear_status<<<1, 1, 0, stream>>>(device_status);
    if (cells == 0) return cudaGetLastError();
    if (cells > 2147483647u) return cudaErrorInvalidValue;
    bitmap_full_kernel<use_prepared_map><<<static_cast<unsigned>(cells), 128, 0, stream>>>(
        p, cells, predicates, bitmaps, node_scratch, device_status,
        root_node_by_edge_ordinal);
    return cudaGetLastError();
}

template <bool use_prepared_map>
cudaError_t update_condition_bitmaps_impl(
    const condition_program_view_v1 &p,
    const std::uint32_t *root_node_by_edge_ordinal,
    std::uint64_t root_map_capacity, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t predicate_count,
    const std::uint64_t *dirty_offsets, const std::uint64_t *dirty_edges,
    std::uint64_t dirty_count, std::uint64_t *bitmaps,
    std::uint64_t bitmap_capacity, std::uint8_t *node_scratch,
    std::uint64_t scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept {
    std::uint64_t expected_predicates = 0, words = host_words_per_cell(p),
                  expected_words = 0, expected_scratch = 0;
    if ((cells && p.predicate_count
         && mul_overflow(cells, p.predicate_count, &expected_predicates))
        || mul_overflow(cells, words, &expected_words)
        || mul_overflow(cells, p.node_count, &expected_scratch))
        return cudaErrorInvalidValue;
    if (p.structure_epoch == 0 || predicate_count != expected_predicates
        || (predicate_count && predicates == nullptr)
        || (cells && dirty_offsets == nullptr) || (dirty_count && dirty_edges == nullptr)
        || (expected_words && bitmaps == nullptr) || bitmap_capacity < expected_words
        || (expected_scratch && node_scratch == nullptr)
        || scratch_capacity < expected_scratch || device_status == nullptr
        || (use_prepared_map && (root_map_capacity < p.edge_count
            || (p.edge_count != 0u && root_node_by_edge_ordinal == nullptr)))
        || (!use_prepared_map && root_node_by_edge_ordinal != nullptr))
        return cudaErrorInvalidValue;
    clear_status<<<1, 1, 0, stream>>>(device_status);
    if (cells == 0) return cudaGetLastError();
    if (cells > 2147483647u) return cudaErrorInvalidValue;
    bitmap_dirty_kernel<use_prepared_map><<<static_cast<unsigned>(cells), 128, 0, stream>>>(
        p, cells, predicates, dirty_offsets, dirty_edges, bitmaps, node_scratch,
        device_status, root_node_by_edge_ordinal);
    return cudaGetLastError();
}

} // namespace

cudaError_t cuda_evaluate_condition_bitmaps_v1(
    const condition_program_view_v1 &p, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t predicate_count,
    std::uint64_t *bitmaps, std::uint64_t bitmap_capacity,
    std::uint8_t *node_scratch, std::uint64_t scratch_capacity,
    cuda_workset_status_v1 *device_status, cudaStream_t stream) noexcept {
    return evaluate_condition_bitmaps_impl<false>(p, nullptr, 0u, cells,
        predicates, predicate_count, bitmaps, bitmap_capacity, node_scratch,
        scratch_capacity, device_status, stream);
}

cudaError_t cuda_evaluate_condition_bitmaps_v1(
    const cuda_condition_program_v1 &p, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t predicate_count,
    std::uint64_t *bitmaps, std::uint64_t bitmap_capacity,
    std::uint8_t *node_scratch, std::uint64_t scratch_capacity,
    cuda_workset_status_v1 *device_status, cudaStream_t stream) noexcept {
    return evaluate_condition_bitmaps_impl<true>(p.program,
        p.root_node_by_edge_ordinal, p.root_map_capacity, cells, predicates,
        predicate_count, bitmaps, bitmap_capacity, node_scratch,
        scratch_capacity, device_status, stream);
}

cudaError_t cuda_update_condition_bitmaps_v1(
    const condition_program_view_v1 &p, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t predicate_count,
    const std::uint64_t *dirty_offsets, const std::uint64_t *dirty_edges,
    std::uint64_t dirty_count, std::uint64_t *bitmaps,
    std::uint64_t bitmap_capacity, std::uint8_t *node_scratch,
    std::uint64_t scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept {
    return update_condition_bitmaps_impl<false>(p, nullptr, 0u, cells,
        predicates, predicate_count, dirty_offsets, dirty_edges, dirty_count,
        bitmaps, bitmap_capacity, node_scratch, scratch_capacity, device_status,
        stream);
}

cudaError_t cuda_update_condition_bitmaps_v1(
    const cuda_condition_program_v1 &p, std::uint64_t cells,
    const std::uint8_t *predicates, std::uint64_t predicate_count,
    const std::uint64_t *dirty_offsets, const std::uint64_t *dirty_edges,
    std::uint64_t dirty_count, std::uint64_t *bitmaps,
    std::uint64_t bitmap_capacity, std::uint8_t *node_scratch,
    std::uint64_t scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept {
    return update_condition_bitmaps_impl<true>(p.program,
        p.root_node_by_edge_ordinal, p.root_map_capacity, cells, predicates,
        predicate_count, dirty_offsets, dirty_edges, dirty_count, bitmaps,
        bitmap_capacity, node_scratch, scratch_capacity, device_status, stream);
}

cudaError_t cuda_build_condition_worksets_v1(
    const condition_program_view_v1 &p, std::uint64_t cells,
    const std::uint64_t *bitmaps, std::uint64_t bitmap_count,
    const cuda_candidate_csr_v1 &primal, const cuda_candidate_csr_v1 &derivative,
    cuda_output_csr_v1 p_out, cuda_output_csr_v1 d_out,
    cuda_workset_scratch_v1 scratch, cudaStream_t stream) noexcept {
    const auto words = host_words_per_cell(p);
    std::uint64_t expected_bitmap_words = 0;
    if (mul_overflow(cells, words, &expected_bitmap_words)) return cudaErrorInvalidValue;
    if (p.structure_epoch == 0 || bitmap_count != expected_bitmap_words
        || (bitmap_count && bitmaps == nullptr)
        || (cells && (primal.offsets == nullptr || derivative.offsets == nullptr
            || p_out.offsets == nullptr || d_out.offsets == nullptr
            || scratch.primal_offsets == nullptr || scratch.derivative_offsets == nullptr
            || scratch.primal_counts == nullptr || scratch.derivative_counts == nullptr))
        || (primal.item_count && primal.items == nullptr)
        || (derivative.item_count && derivative.items == nullptr)
        || cells == std::numeric_limits<std::uint64_t>::max()
        || (cells + 1 > p_out.offsets_capacity) || (cells + 1 > d_out.offsets_capacity)
        || scratch.status == nullptr
        || scratch.primal_offsets_capacity < cells + 1
        || scratch.derivative_offsets_capacity < cells + 1
        || (scratch.primal_item_capacity && scratch.primal_items == nullptr)
        || (scratch.derivative_item_capacity && scratch.derivative_items == nullptr)
        || (cells && (p_out.item_capacity && p_out.items == nullptr
                   || d_out.item_capacity && d_out.items == nullptr)))
        return cudaErrorInvalidValue;
    if (cells == 0) {
        clear_status<<<1, 1, 0, stream>>>(scratch.status);
        return cudaGetLastError();
    }
    clear_status<<<1, 1, 0, stream>>>(scratch.status);
    if (cells > 2147483647u) return cudaErrorInvalidValue;
    stage_worksets_kernel<<<static_cast<unsigned>(cells), 1, 0, stream>>>(
        p, cells, bitmaps, primal, derivative, scratch.primal_offsets,
        scratch.primal_items, scratch.derivative_offsets, scratch.derivative_items,
        scratch.primal_counts, scratch.derivative_counts,
        scratch.primal_item_capacity, scratch.derivative_item_capacity,
        scratch.primal_offsets_capacity, scratch.derivative_offsets_capacity,
        scratch.status);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    prepare_offsets_kernel<<<1, 1, 0, stream>>>(cells, scratch.primal_counts,
        scratch.derivative_counts, p_out, d_out, scratch.status);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    scatter_worksets_kernel<<<static_cast<unsigned>(cells), 1, 0, stream>>>(
        cells, scratch.primal_counts, scratch.derivative_counts,
        scratch.primal_offsets, scratch.primal_items,
        scratch.derivative_offsets, scratch.derivative_items, p_out, d_out,
        scratch.status);
    return cudaGetLastError();
}

} // namespace cellerator::compute::gating
