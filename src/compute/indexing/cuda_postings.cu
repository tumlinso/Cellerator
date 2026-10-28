#include <Cellerator/compute/indexing/cuda_postings.cuh>

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <limits>

namespace cellerator::compute::indexing {
namespace {

constexpr unsigned posting_scatter_threads = 128u;
constexpr unsigned posting_query_threads = 128u;
constexpr unsigned posting_scan_items_per_thread = 16u;

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

// Prove the narrow layout where union-by-concatenation is exact. The canonical
// host projection guarantees each individual posting slice is strictly
// increasing by stable edge ID. Here we additionally prove the query keys are
// unique and ordered and that selected slices form one increasing, disjoint
// sequence. Any unsupported row falls back to merge_cell.
__device__ bool try_ordered_disjoint_cell(
    cuda_reverse_posting_view_v1 index, const std::uint64_t *query_offsets,
    const posting_term_v1 *queries, std::uint64_t cell,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    std::uint64_t *count, cuda_posting_status_v1 *status,
    bool query_cursors_prepared) {
    const auto qb = query_offsets[cell], qe = query_offsets[cell + 1];
    bool have_previous_term = false;
    bool have_previous_posting = false;
    posting_term_v1 previous_term{};
    std::uint64_t previous_last_id = 0;
    std::uint64_t prefix = 0;

    for (auto qi = qb; qi < qe; ++qi) {
        const auto query = queries[qi];
        if (query.reserved != 0u) {
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
            *count = 0;
            return false;
        }
        if (have_previous_term && !term_less(previous_term, query))
            return false;
        previous_term = query;
        have_previous_term = true;

        std::uint64_t begin = 0u, end = 0u;
        if (query_cursors_prepared) {
            begin = cursors[qi].position;
            end = cursors[qi].end;
        } else {
            const auto term = find_term(index, query);
            begin = term == index.term_count ? 0u : index.term_offsets[term];
            end = term == index.term_count ? 0u : index.term_offsets[term + 1u];
        }
        cursors[qi] = {begin, end, prefix, 0u, 0u};
        const auto length = end - begin;
        if (length != 0u) {
            const auto first_id = index.items[begin].logical_edge_id;
            const auto last_id = index.items[end - 1u].logical_edge_id;
            if (have_previous_posting && first_id <= previous_last_id)
                return false;
            previous_last_id = last_id;
            have_previous_posting = true;
        }
        if (prefix > ~std::uint64_t{0} - length)
            return false;
        prefix += length;
    }
    if (qb < qe) cursors[qb].row_mode = 1u;
    *count = prefix;
    return true;
}

__global__ void query_term_lookup_kernel(
    cuda_reverse_posting_view_v1 index, const posting_term_v1 *queries,
    std::uint64_t query_count,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    cuda_posting_status_v1 *status) {
    const auto qi = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (qi >= query_count) return;
    const auto query = queries[qi];
    if (query.reserved != 0u) {
        atomicCAS(&status->code, 0u,
            static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
        cursors[qi] = {0u, 0u, 0u, 0u, 0u};
        return;
    }
    const auto term = find_term(index, query);
    const auto begin = term == index.term_count ? 0u : index.term_offsets[term];
    const auto end = term == index.term_count ? 0u : index.term_offsets[term + 1u];
    cursors[qi] = {begin, end, 0u, 0u, 0u};
}

__global__ void query_counts_kernel(
    cuda_reverse_posting_view_v1 index, std::uint64_t cells,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    std::uint64_t query_count,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    std::uint64_t *counts, cuda_posting_status_v1 *status) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells || threadIdx.x != 0) return;
    const auto qb = query_offsets[cell];
    const auto qe = query_offsets[cell + 1];
    if (qb > qe || qe > query_count) {
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
        std::uint64_t begin = 0u, end = 0u;
        if (query_count > cells) {
            begin = cursors[qb].position;
            end = cursors[qb].end;
            cursors[qb].output_offset = 0u;
            cursors[qb].row_mode = 1u;
            cursors[qb].reserved = 0u;
        } else {
            const auto term = find_term(index, query);
            begin = term == index.term_count ? 0u : index.term_offsets[term];
            end = term == index.term_count ? 0u : index.term_offsets[term + 1u];
            cursors[qb] = {begin, end, 0u, 1u, 0u};
        }
        counts[cell] = end - begin;
    } else {
        if (!try_ordered_disjoint_cell(index, query_offsets, queries, cell,
                cursors, counts + cell, status, query_count > cells)) {
            if (status->code == 0u) {
                if (qb < qe) cursors[qb].row_mode = 0u;
                merge_cell(index, query_offsets, queries, cell, cursors, nullptr,
                           counts + cell, status);
            }
        }
    }
    if (status->code == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(&status->required),
                  static_cast<unsigned long long>(counts[cell]));
}

// For larger batches, term lookup runs in parallel and this row kernel proves
// the ordered-disjoint precondition while scanning each row's posting lengths.
// One CTA handles a row; 128 threads each own a contiguous chunk of terms so
// BlockScan yields the exact row-relative destination prefix for concatenation.
__global__ void query_counts_parallel_rows_kernel(
    cuda_reverse_posting_view_v1 index, std::uint64_t cells,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    std::uint64_t query_count,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    std::uint64_t *counts, cuda_posting_status_v1 *status) {
    using block_scan = cub::BlockScan<std::uint64_t, posting_query_threads>;
    __shared__ typename block_scan::TempStorage scan_storage;
    __shared__ std::uint32_t eligible;
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    const auto qb = query_offsets[cell];
    const auto qe = query_offsets[cell + 1u];
    if (qb > qe || qe > query_count) {
        if (threadIdx.x == 0u) {
            atomicCAS(&status->code, 0u,
                static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
            counts[cell] = 0u;
        }
        return;
    }
    if (qb == qe) {
        if (threadIdx.x == 0u) counts[cell] = 0u;
        return;
    }
    if (qe - qb == 1u) {
        if (threadIdx.x == 0u) {
            const auto query = queries[qb];
            if (query.reserved != 0u) {
                atomicCAS(&status->code, 0u,
                    static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
                counts[cell] = 0u;
            } else if (cursors[qb].end < cursors[qb].position
                || cursors[qb].end > index.item_count) {
                atomicCAS(&status->code, 0u,
                    static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
                counts[cell] = 0u;
            } else {
                const auto cursor = cursors[qb];
                cursors[qb].output_offset = 0u;
                cursors[qb].row_mode = 1u;
                cursors[qb].reserved = 0u;
                counts[cell] = cursor.end - cursor.position;
            }
        }
    } else {
        if (threadIdx.x == 0u) eligible = 1u;
        __syncthreads();
        if (threadIdx.x == 0u
            && qe - qb > posting_query_threads * posting_scan_items_per_thread)
            eligible = 0u;
        __syncthreads();
        std::uint64_t local_lengths[posting_scan_items_per_thread]{};
        std::uint64_t local_sum = 0u;
        for (unsigned item = 0u; item < posting_scan_items_per_thread; ++item) {
            const auto qi = qb
                + static_cast<std::uint64_t>(threadIdx.x)
                    * posting_scan_items_per_thread + item;
            if (qi >= qe) break;
            const auto query = queries[qi];
            const auto cursor = cursors[qi];
            if (query.reserved != 0u || cursor.end <= cursor.position
                || cursor.end > index.item_count) {
                atomicExch(&eligible, 0u);
                continue;
            }
            if (qi != qb) {
                const auto previous = queries[qi - 1u];
                const auto previous_cursor = cursors[qi - 1u];
                if (!term_less(previous, query)
                    || previous_cursor.end <= previous_cursor.position
                    || previous_cursor.end > index.item_count
                    || index.items[previous_cursor.end - 1u].logical_edge_id
                        >= index.items[cursor.position].logical_edge_id)
                    atomicExch(&eligible, 0u);
            }
            const auto length = cursor.end - cursor.position;
            local_lengths[item] = length;
            local_sum += length;
        }
        std::uint64_t thread_prefix = 0u, aggregate = 0u;
        block_scan(scan_storage).ExclusiveSum(local_sum, thread_prefix, aggregate);
        __syncthreads();
        if (eligible != 0u) {
            auto prefix = thread_prefix;
            for (unsigned item = 0u; item < posting_scan_items_per_thread; ++item) {
                const auto qi = qb
                    + static_cast<std::uint64_t>(threadIdx.x)
                        * posting_scan_items_per_thread + item;
                if (qi >= qe) break;
                cursors[qi].output_offset = prefix;
                prefix += local_lengths[item];
            }
            if (threadIdx.x == 0u) {
                cursors[qb].row_mode = 1u;
                counts[cell] = aggregate;
            }
        } else if (threadIdx.x == 0u) {
            cursors[qb].row_mode = 0u;
            merge_cell(index, query_offsets, queries, cell, cursors, nullptr,
                       counts + cell, status);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u && status->code == 0u)
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

__device__ std::uint64_t query_cell_for_index(
    const std::uint64_t *query_offsets, std::uint64_t cells,
    std::uint64_t query_index) {
    std::uint64_t lo = 0u, hi = cells;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2u;
        if (query_offsets[mid + 1u] <= query_index) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

__global__ void query_scatter_kernel(
    cuda_reverse_posting_view_v1 index, std::uint64_t cells,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    std::uint64_t query_count,
    cuda_posting_scratch_v1::cursor_v1 *cursors,
    const std::uint64_t *counts, cuda_posting_output_v1 output,
    cuda_posting_status_v1 *status) {
    if (status->code != 0) return;
    const auto qi = static_cast<std::uint64_t>(blockIdx.x);
    if (qi >= query_count) return;
    const auto cell = query_cell_for_index(query_offsets, cells, qi);
    if (cell >= cells) return;
    const auto qb = query_offsets[cell], qe = query_offsets[cell + 1];
    if (qb >= qe || qi < qb || qi >= qe) return;
    if (cursors[qb].row_mode != 0u) {
        const auto cursor = cursors[qi];
        const auto expected = counts[cell];
        if (cursor.end < cursor.position
            || cursor.output_offset > expected
            || cursor.end - cursor.position > expected - cursor.output_offset) {
            if (threadIdx.x == 0)
                atomicCAS(&status->code, 0u,
                          static_cast<std::uint32_t>(cuda_posting_code_v1::invalid_query));
            return;
        }
        const auto length = cursor.end - cursor.position;
        if (output.items == nullptr || length == 0u) return;
        auto *destination = output.items + output.offsets[cell] + cursor.output_offset;
        for (std::uint64_t i = threadIdx.x; i < length; i += blockDim.x)
            destination[i] = index.items[cursor.position + i];
        return;
    }
    if (qi != qb || threadIdx.x != 0) return;
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
    // Both status.required's atomic total and the output-offset prefix sum
    // aggregate all cell counts in uint64. A cell can emit at most every
    // indexed posting item, so reject launch shapes whose conservative bound
    // cannot be represented. This is deliberately checked before scheduling
    // any kernels (and without adding hot-path device arithmetic).
    if (cells != 0 && query_count != 0 && index.item_count != 0
        && index.item_count > std::numeric_limits<std::uint64_t>::max() / cells)
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
    if (cells > std::numeric_limits<unsigned>::max()
        || query_count > std::numeric_limits<unsigned>::max())
        return cudaErrorInvalidValue;

    clear_status<<<1, 1, 0, stream>>>(scratch.status);
    auto err = cudaGetLastError();
    if (err != cudaSuccess || cells == 0) return err;
    if (query_count > cells) {
        const auto lookup_blocks = static_cast<unsigned>(
            (query_count + posting_query_threads - 1u) / posting_query_threads);
        query_term_lookup_kernel<<<lookup_blocks, posting_query_threads, 0, stream>>>(
            index, queries, query_count, scratch.cursors, scratch.status);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    if (query_count > cells) {
        query_counts_parallel_rows_kernel<<<static_cast<unsigned>(cells),
            posting_query_threads, 0, stream>>>(index, cells, query_offsets,
            queries, query_count, scratch.cursors, scratch.counts, scratch.status);
    } else {
        query_counts_kernel<<<static_cast<unsigned>(cells), 1, 0, stream>>>(
            index, cells, query_offsets, queries, query_count, scratch.cursors,
            scratch.counts, scratch.status);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    prepare_offsets_kernel<<<1, 1, 0, stream>>>(
        cells, output, scratch.counts, scratch.status);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    if (query_count == 0u) return cudaSuccess;
    query_scatter_kernel<<<static_cast<unsigned>(query_count), posting_scatter_threads, 0, stream>>>(
        index, cells, query_offsets, queries, query_count, scratch.cursors,
        scratch.counts, output, scratch.status);
    return cudaGetLastError();
}

} // namespace cellerator::compute::indexing
