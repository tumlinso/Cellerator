#include <Cellerator/compute/indexing/cuda_postings.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

namespace cellerator::compute::indexing {
namespace {

constexpr unsigned posting_scatter_threads = 128u;

__global__ void clear_status(cuda_posting_status_v1 *status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *status = {};
}

__device__ __forceinline__ bool term_less(const posting_term_v1 &a,
                                           const posting_term_v1 &b) {
    return a.role < b.role || (a.role == b.role && a.entity_id < b.entity_id);
}

__device__ __forceinline__ bool same_term(const posting_term_v1 &a,
                                           const posting_term_v1 &b) {
    return a.role == b.role && a.entity_id == b.entity_id;
}

__device__ __forceinline__ std::uint64_t find_term(
    cuda_reverse_posting_view_v1 index, const posting_term_v1 &query) {
    std::uint64_t lo = 0, hi = index.term_count;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (term_less(index.terms[mid], query)) lo = mid + 1;
        else hi = mid;
    }
    return lo < index.term_count && same_term(index.terms[lo], query)
        ? lo : index.term_count;
}

__device__ void merge_cell(
    cuda_reverse_posting_view_v1 index, const std::uint64_t *query_offsets,
    const posting_term_v1 *queries, std::uint64_t cell,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    indexed_work_item_v1 *destination, std::uint64_t *count,
    cuda_posting_status_v1 *status) {
    const auto qb = query_offsets[cell], qe = query_offsets[cell + 1];
    if (qb > qe) {
        atomicCAS(&status->code, 0u,
                  static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
        *count = 0;
        return;
    }
    for (auto qi = qb; qi < qe; ++qi) {
        const auto query = queries[qi];
        if (query.reserved != 0u) {
            atomicCAS(&status->code, 0u,
                      static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
            *count = 0;
            return;
        }
        const auto term = find_term(index, query);
        if (term == index.term_count) {
            cursors[qi] = {0, 0};
        } else {
            cursors[qi] = {index.term_offsets[term], index.term_offsets[term + 1]};
        }
    }

    std::uint64_t written = 0;
    for (;;) {
        bool have_value = false;
        std::uint64_t minimum_id = 0;
        indexed_work_item_v1 minimum{};
        for (auto qi = qb; qi < qe; ++qi) {
            const auto c = cursors[qi];
            if (c.position >= c.end) continue;
            const auto item = index.items[c.position];
            if (!have_value || item.logical_edge_id < minimum_id) {
                minimum = item;
                minimum_id = item.logical_edge_id;
                have_value = true;
            }
        }
        if (!have_value) break;
        if (destination != nullptr) destination[written] = minimum;
        ++written;
        // Advance all lists at this identity. This also removes duplicate
        // query terms without changing incidence multiplicity in support.
        for (auto qi = qb; qi < qe; ++qi) {
            auto &c = cursors[qi];
            while (c.position < c.end
                   && index.items[c.position].logical_edge_id == minimum_id)
                ++c.position;
        }
    }
    *count = written;
}

__global__ void query_counts_kernel(
    cuda_reverse_posting_view_v1 index, std::uint64_t cells,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    std::uint64_t *counts, cuda_posting_status_v1 *status) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells || threadIdx.x != 0) return;
    const auto qb = query_offsets[cell];
    const auto qe = query_offsets[cell + 1];
    if (qb > qe) {
        atomicCAS(&status->code, 0u,
                  static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
        counts[cell] = 0;
        return;
    }
    if (qb == qe) {
        counts[cell] = 0;
        return;
    }
    if (qe - qb == 1u) {
        const auto query = queries[qb];
        if (query.reserved != 0u) {
            atomicCAS(&status->code, 0u,
                      static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
            counts[cell] = 0;
            return;
        }
        const auto term = find_term(index, query);
        const auto begin = term == index.term_count ? 0u : index.term_offsets[term];
        const auto end = term == index.term_count ? 0u : index.term_offsets[term + 1u];
        cursors[qb] = {begin, end};
        counts[cell] = end - begin;
    } else {
        merge_cell(index, query_offsets, queries, cell, cursors, nullptr,
                   counts + cell, status);
    }
    if (status->code == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(&status->required),
                  static_cast<unsigned long long>(counts[cell]));
}

__global__ void prepare_offsets_kernel(
    std::uint64_t cells, cuda_posting_output_v1 output,
    const std::uint64_t *counts, cuda_posting_status_v1 *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != 0) return;
    if (status->required > output.item_capacity) {
        status->code = static_cast<std::uint32_t>(cuda_posting_code_v1::output_overflow);
        return;
    }
    output.offsets[0] = 0;
    std::uint64_t prefix = 0;
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        prefix += counts[cell];
        output.offsets[cell + 1] = prefix;
    }
}

__global__ void query_scatter_kernel(
    cuda_reverse_posting_view_v1 index, std::uint64_t cells,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    const std::uint64_t *counts, cuda_posting_output_v1 output,
    cuda_posting_status_v1 *status) {
    if (status->code != 0) return;
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    const auto qb = query_offsets[cell], qe = query_offsets[cell + 1];
    if (qb <= qe && qe - qb == 1u) {
        const auto cursor = cursors[qb];
        const auto expected = counts[cell];
        if (cursor.end < cursor.position || cursor.end - cursor.position != expected) {
            if (threadIdx.x == 0)
                atomicCAS(&status->code, 0u,
                          static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
            return;
        }
        if (output.items == nullptr || expected == 0u) return;
        auto *destination = output.items + output.offsets[cell];
        for (std::uint64_t i = threadIdx.x; i < expected; i += blockDim.x)
            destination[i] = index.items[cursor.position + i];
        return;
    }
    if (threadIdx.x != 0) return;
    std::uint64_t written = 0;
    merge_cell(index, query_offsets, queries, cell, cursors,
        output.items == nullptr ? nullptr : output.items + output.offsets[cell],
        &written, status);
    // An immutable index/query snapshot must produce the same count in both
    // passes. If caller mutates borrowed data concurrently, fail closed.
    if (written != counts[cell])
        atomicCAS(&status->code, 0u,
                  static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
}

} // namespace

cuda_posting_prepare_requirements_v1 cuda_posting_projection_requirements_v1(
    const reverse_posting_view_v1 &host) noexcept {
    cuda_posting_prepare_requirements_v1 result{};
    if (host.relation_id == 0 || host.structure_id == 0
        || host.structure_epoch == 0 || (host.entry_count && host.entries == nullptr))
        return result;
    result.item_count = host.entry_count;
    for (std::uint64_t i = 0; i < host.entry_count; ++i)
        if (i == 0 || host.entries[i].role != host.entries[i - 1].role
            || host.entries[i].entity_id != host.entries[i - 1].entity_id)
            ++result.term_count;
    return result;
}

cuda_posting_prepare_status_v1 build_cuda_posting_projection_v1(
    const reverse_posting_view_v1 &host, posting_term_v1 *terms,
    std::uint64_t term_capacity, std::uint64_t *term_offsets,
    std::uint64_t term_offsets_capacity, indexed_work_item_v1 *items,
    std::uint64_t item_capacity,
    cuda_posting_prepare_requirements_v1 *required) noexcept {
    if (required == nullptr) return cuda_posting_prepare_status_v1::invalid_host_index;
    *required = cuda_posting_projection_requirements_v1(host);
    if (host.relation_id == 0 || host.structure_id == 0
        || host.structure_epoch == 0 || (host.entry_count && host.entries == nullptr))
        return cuda_posting_prepare_status_v1::invalid_host_index;
    if (term_capacity < required->term_count
        || term_offsets_capacity < required->term_count + 1
        || item_capacity < required->item_count
        || (required->term_count && terms == nullptr) || term_offsets == nullptr
        || (required->item_count && items == nullptr))
        return cuda_posting_prepare_status_v1::insufficient_capacity;
    std::uint64_t term = 0;
    term_offsets[0] = 0;
    for (std::uint64_t i = 0; i < host.entry_count; ++i) {
        const auto &entry = host.entries[i];
        if (i == 0 || entry.role != host.entries[i - 1].role
            || entry.entity_id != host.entries[i - 1].entity_id) {
            if (i != 0) term_offsets[term] = i;
            terms[term++] = {entry.role, 0u, entry.entity_id};
        }
        items[i] = {entry.logical_edge_id, entry.logical_edge_ordinal};
    }
    term_offsets[term] = host.entry_count;
    return cuda_posting_prepare_status_v1::success;
}

cudaError_t cuda_query_reverse_postings_any_v1(
    const cuda_reverse_posting_view_v1 &index,
    std::uint64_t expected_epoch, std::uint64_t cells,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    std::uint64_t query_count, cuda_posting_output_v1 output,
    cuda_posting_scratch_v1 scratch, cudaStream_t stream) noexcept {
    if (index.relation_id == 0 || index.structure_id == 0
        || index.structure_epoch == 0 || index.structure_epoch != expected_epoch)
        return cudaErrorInvalidValue;
    if ((index.term_count && (index.terms == nullptr || index.term_offsets == nullptr))
        || (index.item_count && index.items == nullptr)
        || (cells && query_offsets == nullptr)
        || (query_count && queries == nullptr)
        || (query_count && scratch.cursors == nullptr)
        || scratch.cursor_capacity < query_count
        || (cells && scratch.counts == nullptr) || scratch.status == nullptr
        || cells == std::numeric_limits<std::uint64_t>::max()
        || cells + 1 > output.offsets_capacity
        || (output.item_capacity && output.items == nullptr)
        || (cells && output.offsets == nullptr))
        return cudaErrorInvalidValue;
    if (cells > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidValue;

    clear_status<<<1, 1, 0, stream>>>(scratch.status);
    auto err = cudaGetLastError();
    if (err != cudaSuccess || cells == 0) return err;
    query_counts_kernel<<<static_cast<unsigned>(cells), 1, 0, stream>>>(
        index, cells, query_offsets, queries, scratch.cursors,
        scratch.counts, scratch.status);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    prepare_offsets_kernel<<<1, 1, 0, stream>>>(
        cells, output, scratch.counts, scratch.status);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    query_scatter_kernel<<<static_cast<unsigned>(cells), posting_scatter_threads, 0, stream>>>(
        index, cells, query_offsets, queries, scratch.cursors,
        scratch.counts, output, scratch.status);
    return cudaGetLastError();
}

} // namespace cellerator::compute::indexing
