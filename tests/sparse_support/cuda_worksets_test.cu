#include <Cellerator/compute/gating/cuda_worksets.cuh>
#include <Cellerator/compute/indexing/cuda_postings.cuh>

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace cellerator::compute::gating;
using cellerator::compute::indexing::indexed_work_item_v1;

namespace {
template <class T> T *device_copy(const T *src, std::size_t n) {
    T *dst = nullptr;
    if (n == 0) return dst;
    assert(cudaMalloc(&dst, n * sizeof(T)) == cudaSuccess);
    assert(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return dst;
}

template <class T> std::vector<T> host_copy(const T *src, std::size_t n) {
    std::vector<T> result(n);
    if (n) assert(cudaMemcpy(result.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost)
                  == cudaSuccess);
    return result;
}

template <class T> void release(T *ptr) { if (ptr) assert(cudaFree(ptr) == cudaSuccess); }
}

int main() {
    // Edge 0 requires predicate 0, edge 1 is unbound, edge 2 requires NOT
    // predicate 0. The cell-major bitmap therefore changes across cells.
    const condition_node_v1 nodes[] = {
        {condition_opcode_v1::predicate, 1, 0, 0, 0, 0, 0},
        {condition_opcode_v1::logical_not, 1, 0, 0, 1, 0, 0},
    };
    const edge_condition_binding_v1 bindings[] = {{0, 0, 0}, {2, 1, 0}};
    condition_program_view_v1 host_program{17, nodes, 2, bindings, 2, 3, 1, 0};
    std::uint32_t host_root_map[3]{};
    std::uint64_t root_map_required = 0u;
    assert(prepare_condition_root_map_v1(host_program, host_root_map, 3,
        &root_map_required) == condition_status_v1::success);
    assert(root_map_required == 3
        && host_root_map[0] == 0
        && host_root_map[1] == unbound_condition_root_v1
        && host_root_map[2] == 1);
    auto *d_nodes = device_copy(nodes, 2);
    auto *d_bindings = device_copy(bindings, 2);
    condition_program_view_v1 program{17, d_nodes, 2, d_bindings, 2, 3, 1, 0};
    auto *d_root_map = device_copy(host_root_map, 3);
    cuda_condition_program_v1 prepared_program{program, d_root_map, 3};
    const std::uint8_t predicates[] = {1, 0};
    auto *d_predicates = device_copy(predicates, 2);
    std::uint64_t *d_bitmaps = nullptr;
    assert(cudaMalloc(&d_bitmaps, 2 * sizeof(std::uint64_t)) == cudaSuccess);
    std::uint8_t *d_node_scratch = nullptr;
    assert(cudaMalloc(&d_node_scratch, 4) == cudaSuccess);
    auto *d_status = device_copy<cuda_workset_status_v1>(nullptr, 0);
    assert(cudaMalloc(&d_status, sizeof(cuda_workset_status_v1)) == cudaSuccess);
    assert(cuda_evaluate_condition_bitmaps_v1(program, 2, d_predicates, 2,
        d_bitmaps, 2, d_node_scratch, 4, d_status, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    const auto bits = host_copy(d_bitmaps, 2);
    assert(bits[0] == 0b011);
    assert(bits[1] == 0b110);
    assert(cuda_evaluate_condition_bitmaps_v1(prepared_program, 2, d_predicates, 2,
        d_bitmaps, 2, d_node_scratch, 4, d_status, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    const auto prepared_bits = host_copy(d_bitmaps, 2);
    assert(prepared_bits == bits);
    cuda_condition_program_v1 short_map_program{program, d_root_map, 2};
    assert(cuda_evaluate_condition_bitmaps_v1(short_map_program, 2, d_predicates, 2,
        d_bitmaps, 2, d_node_scratch, 4, d_status, nullptr) == cudaErrorInvalidValue);

    const std::uint64_t candidate_offsets[] = {0, 3, 5};
    const indexed_work_item_v1 primal_items[] = {{10, 0}, {20, 1}, {30, 2},
                                                  {10, 0}, {30, 2}};
    const std::uint64_t derivative_offsets[] = {0, 2, 4};
    const indexed_work_item_v1 derivative_items[] = {{20, 1}, {30, 2},
                                                      {10, 0}, {20, 1}};
    auto *d_candidate_offsets = device_copy(candidate_offsets, 3);
    auto *d_primal_items = device_copy(primal_items, 5);
    auto *d_derivative_offsets = device_copy(derivative_offsets, 3);
    auto *d_derivative_items = device_copy(derivative_items, 4);
    std::uint64_t *d_p_stage_offsets = nullptr, *d_d_stage_offsets = nullptr;
    assert(cudaMalloc(&d_p_stage_offsets, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMalloc(&d_d_stage_offsets, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    const std::uint64_t p_stage_offsets[] = {0, 3, 5};
    const std::uint64_t d_stage_offsets[] = {0, 5, 9};
    assert(cudaMemcpy(d_p_stage_offsets, p_stage_offsets, sizeof(p_stage_offsets),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_d_stage_offsets, d_stage_offsets, sizeof(d_stage_offsets),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    indexed_work_item_v1 *d_p_stage = nullptr, *d_d_stage = nullptr;
    assert(cudaMalloc(&d_p_stage, 5 * sizeof(*d_p_stage)) == cudaSuccess);
    assert(cudaMalloc(&d_d_stage, 9 * sizeof(*d_d_stage)) == cudaSuccess);
    std::uint64_t *d_p_counts = nullptr, *d_d_counts = nullptr;
    assert(cudaMalloc(&d_p_counts, 2 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMalloc(&d_d_counts, 2 * sizeof(std::uint64_t)) == cudaSuccess);
    std::uint64_t *d_p_out_offsets = nullptr, *d_d_out_offsets = nullptr;
    assert(cudaMalloc(&d_p_out_offsets, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMalloc(&d_d_out_offsets, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    indexed_work_item_v1 *d_p_out = nullptr, *d_d_out = nullptr;
    assert(cudaMalloc(&d_p_out, 5 * sizeof(*d_p_out)) == cudaSuccess);
    assert(cudaMalloc(&d_d_out, 8 * sizeof(*d_d_out)) == cudaSuccess);
    cuda_candidate_csr_v1 primal{d_candidate_offsets, d_primal_items, 5};
    cuda_candidate_csr_v1 derivative{d_derivative_offsets, d_derivative_items, 4};
    cuda_workset_scratch_v1 scratch{};
    scratch.primal_offsets = d_p_stage_offsets;
    scratch.primal_items = d_p_stage;
    scratch.primal_item_capacity = 5;
    scratch.derivative_offsets = d_d_stage_offsets;
    scratch.derivative_items = d_d_stage;
    scratch.derivative_item_capacity = 9;
    scratch.primal_counts = d_p_counts;
    scratch.derivative_counts = d_d_counts;
    scratch.status = d_status;
    scratch.primal_offsets_capacity = scratch.derivative_offsets_capacity = 3;
    cuda_output_csr_v1 p_out{d_p_out_offsets, 3, d_p_out, 5};
    cuda_output_csr_v1 d_out{d_d_out_offsets, 3, d_d_out, 8};
    assert(cuda_build_condition_worksets_v1(program, 2, d_bitmaps, 2,
        primal, derivative, p_out, d_out, scratch, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    cuda_workset_status_v1 status{};
    assert(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost)
           == cudaSuccess);
    assert(status.code == 0 && status.primal_required == 3
           && status.derivative_required == 4);
    const auto po = host_copy(d_p_out_offsets, 3);
    const auto d_o = host_copy(d_d_out_offsets, 3);
    assert((po == std::vector<std::uint64_t>{0, 2, 3}));
    assert((d_o == std::vector<std::uint64_t>{0, 2, 4}));
    const auto pv = host_copy(d_p_out, 3);
    const auto dv = host_copy(d_d_out, 4);
    assert(pv[0].logical_edge_id == 10 && pv[1].logical_edge_id == 20
           && pv[2].logical_edge_id == 30);
    assert(dv[0].logical_edge_id == 10 && dv[1].logical_edge_id == 20
           && dv[2].logical_edge_id == 20 && dv[3].logical_edge_id == 30);

    // The device postings path searches only the selected entity lists and
    // merges them in stable-ID order; it does not inspect unrelated edges.
    using namespace cellerator::compute::indexing;
    const reverse_posting_record_v1 host_records[] = {
        {1, 0, 100, 10, 0}, {1, 0, 100, 30, 2},
        {1, 0, 200, 20, 1}, {1, 0, 200, 30, 2},
    };
    reverse_posting_view_v1 host_index{5, 7, 17, host_records, 4};
    const auto requirements = cuda_posting_projection_requirements_v1(host_index);
    assert(requirements.term_count == 2 && requirements.item_count == 4);
    posting_term_v1 host_terms[2]{};
    std::uint64_t host_term_offsets[3]{};
    indexed_work_item_v1 host_items[4]{};
    cuda_posting_prepare_requirements_v1 prepared{};
    assert(build_cuda_posting_projection_v1(host_index, host_terms, 2,
        host_term_offsets, 3, host_items, 4, &prepared)
        == cuda_posting_prepare_status_v1::success);
    auto *d_terms = device_copy(host_terms, 2);
    auto *d_term_offsets = device_copy(host_term_offsets, 3);
    auto *d_posting_items = device_copy(host_items, 4);
    cuda_reverse_posting_view_v1 posting_index{
        5, 7, 17, d_terms, d_term_offsets, 2, d_posting_items, 4};
    const std::uint64_t query_offsets[] = {0, 1, 3};
    const posting_term_v1 host_queries[] = {{1, 0, 100}, {1, 0, 100}, {1, 0, 200}};
    auto *d_query_offsets = device_copy(query_offsets, 3);
    auto *d_queries = device_copy(host_queries, 3);
    cuda_posting_scratch_v1 posting_scratch{};
    assert(cudaMalloc(&posting_scratch.cursors, 3 * sizeof(*posting_scratch.cursors))
           == cudaSuccess);
    posting_scratch.cursor_capacity = 3;
    assert(cudaMalloc(&posting_scratch.counts, 2 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMalloc(&posting_scratch.status, sizeof(cuda_posting_status_v1))
           == cudaSuccess);
    std::uint64_t *d_posting_offsets = nullptr;
    indexed_work_item_v1 *d_union_items = nullptr;
    assert(cudaMalloc(&d_posting_offsets, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMalloc(&d_union_items, 5 * sizeof(*d_union_items)) == cudaSuccess);
    cuda_posting_output_v1 posting_output{d_posting_offsets, 3, d_union_items, 5};
    assert(cuda_query_reverse_postings_any_v1(posting_index, 17, 2,
        d_query_offsets, d_queries, 3, posting_output, posting_scratch, nullptr)
        == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    cuda_posting_status_v1 posting_status{};
    assert(cudaMemcpy(&posting_status, posting_scratch.status, sizeof(posting_status),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(posting_status.code == 0 && posting_status.required == 5);
    const auto posting_result_offsets = host_copy(d_posting_offsets, 3);
    const auto posting_result = host_copy(d_union_items, 5);
    assert((posting_result_offsets == std::vector<std::uint64_t>{0, 2, 5}));
    assert(posting_result[0].logical_edge_id == 10
           && posting_result[1].logical_edge_id == 30
           && posting_result[2].logical_edge_id == 10
           && posting_result[3].logical_edge_id == 20
           && posting_result[4].logical_edge_id == 30);

    indexed_work_item_v1 sentinel{999, 999};
    std::uint64_t offset_sentinel = 999;
    indexed_work_item_v1 *d_small_items = nullptr;
    std::uint64_t *d_small_offsets = nullptr;
    assert(cudaMalloc(&d_small_items, 4 * sizeof(*d_small_items)) == cudaSuccess);
    assert(cudaMalloc(&d_small_offsets, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMemset(d_small_items, 0, 4 * sizeof(*d_small_items)) == cudaSuccess);
    assert(cudaMemset(d_small_offsets, 0, 3 * sizeof(std::uint64_t)) == cudaSuccess);
    assert(cudaMemcpy(d_small_items, &sentinel, sizeof(sentinel),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_small_offsets, &offset_sentinel, sizeof(offset_sentinel),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    cuda_posting_output_v1 too_small{d_small_offsets, 3, d_small_items, 4};
    assert(cuda_query_reverse_postings_any_v1(posting_index, 17, 2,
        d_query_offsets, d_queries, 3, too_small, posting_scratch, nullptr)
        == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(&posting_status, posting_scratch.status, sizeof(posting_status),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(posting_status.code
           == static_cast<std::uint32_t>(cuda_posting_code_v1::output_overflow));
    assert(posting_status.required == 5);
    indexed_work_item_v1 unchanged{};
    std::uint64_t unchanged_offset = 0;
    assert(cudaMemcpy(&unchanged, d_small_items, sizeof(unchanged),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&unchanged_offset, d_small_offsets, sizeof(unchanged_offset),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(unchanged.logical_edge_id == 999 && unchanged_offset == 999);

    // Sparse per-cell changes preserve the two untouched middle bits while
    // recomputing only the four dirty edge ordinals.
    const std::uint8_t changed_predicates[] = {0, 1};
    assert(cudaMemcpy(d_predicates, changed_predicates, sizeof(changed_predicates),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    const std::uint64_t dirty_offsets[] = {0, 2, 4};
    const std::uint64_t dirty_edges[] = {0, 2, 0, 2};
    auto *d_dirty_offsets = device_copy(dirty_offsets, 3);
    auto *d_dirty_edges = device_copy(dirty_edges, 4);
    assert(cuda_update_condition_bitmaps_v1(program, 2, d_predicates, 2,
        d_dirty_offsets, d_dirty_edges, 4, d_bitmaps, 2, d_node_scratch, 4,
        d_status, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost)
           == cudaSuccess);
    assert(status.code == 0);
    const auto changed_bits = host_copy(d_bitmaps, 2);
    assert(changed_bits[0] == 0b110 && changed_bits[1] == 0b011);
    assert(cuda_update_condition_bitmaps_v1(prepared_program, 2, d_predicates, 2,
        d_dirty_offsets, d_dirty_edges, 4, d_bitmaps, 2, d_node_scratch, 4,
        d_status, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(host_copy(d_bitmaps, 2) == changed_bits);

    release(d_nodes); release(d_bindings); release(d_root_map); release(d_predicates); release(d_bitmaps);
    release(d_node_scratch); release(d_status); release(d_candidate_offsets);
    release(d_primal_items); release(d_derivative_offsets); release(d_derivative_items);
    release(d_p_stage_offsets); release(d_d_stage_offsets); release(d_p_stage);
    release(d_d_stage); release(d_p_counts); release(d_d_counts); release(d_p_out_offsets);
    release(d_d_out_offsets); release(d_p_out); release(d_d_out);
    release(d_terms); release(d_term_offsets); release(d_posting_items);
    release(d_query_offsets); release(d_queries); release(posting_scratch.cursors);
    release(posting_scratch.counts); release(posting_scratch.status);
    release(d_posting_offsets); release(d_union_items);
    release(d_small_items); release(d_small_offsets);
    release(d_dirty_offsets); release(d_dirty_edges);
    return 0;
}
