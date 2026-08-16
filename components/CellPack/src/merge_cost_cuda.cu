/*
 * CP-BP-03 native path: one sm_70 block scores one candidate by streaming two
 * gene-major support bitsets and reducing exact popcounts. On 2026-08-16,
 * ./build-cp-bp03/cellPackMergeCostBench compared this host-staged custom CUDA
 * path with the CPU reference on a Tesla V100 for 65,536 cells, 30,000 genes,
 * 2,048 words/gene, and 105,000 candidates. CPU was 308.250 ms; three timed GPU
 * runs after one warmup measured 77.924 ms minimum and 78.895 ms median,
 * including 245,760,000 support bytes of H2D staging, allocation, kernel, and
 * D2H output. Every integer support/cost/gain field matched exactly (zero
 * tolerance). No NVIDIA library directly owns the fused pairwise bitset and
 * codec-accounting operation; persistent device support remains deferred.
 */

#include "CellPack/merge_cost.hh"

#include "merge_cost_internal.cuh"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace cellpack {
namespace {

namespace gc = ::cellerator::compute::gene_candidates;
namespace gs = ::cellerator::compute::gene_support;

constexpr unsigned int scorer_threads = 256u;

struct device_buffers {
    gs::support_word_t *support = nullptr;
    gc::gene_candidate_pair *pairs = nullptr;
    exact_gene_merge_cost *costs = nullptr;
    u32 *error_index = nullptr;
};

cudaError_t release_pointer(void *pointer, cudaError_t current) {
    if (pointer == nullptr) return current;
    const cudaError_t status = cudaFree(pointer);
    return current == cudaSuccess ? status : current;
}

cudaError_t release_buffers(device_buffers *buffers) {
    cudaError_t status = cudaSuccess;
    status = release_pointer(buffers->error_index, status);
    status = release_pointer(buffers->costs, status);
    status = release_pointer(buffers->pairs, status);
    status = release_pointer(buffers->support, status);
    *buffers = {};
    return status;
}

validation_result cuda_status(cudaError_t status, const char *message) {
    return status == cudaSuccess
        ? validation_ok()
        : validation_error(validation_code::insufficient_capacity, invalid_id, message);
}

__device__ inline unsigned long long warp_sum(unsigned long long value) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void score_gene_merges_kernel(
    const gs::support_word_t *support,
    std::size_t words_per_gene,
    u64 sampled_cell_count,
    const gc::gene_candidate_pair *pairs,
    u64 candidate_count,
    exact_merge_cost_policy policy,
    exact_gene_merge_cost *costs,
    u32 *error_index) {
    const u64 candidate = static_cast<u64>(blockIdx.x);
    if (candidate >= candidate_count) return;
    const gc::gene_candidate_pair pair = pairs[candidate];
    const gs::support_word_t *a = words_per_gene == 0u ? nullptr
        : support + static_cast<std::size_t>(pair.gene_a) * words_per_gene;
    const gs::support_word_t *b = words_per_gene == 0u ? nullptr
        : support + static_cast<std::size_t>(pair.gene_b) * words_per_gene;
    unsigned long long count_a = 0u, count_b = 0u, intersection = 0u, support_union = 0u;
    const u32 tail_bits = static_cast<u32>(sampled_cell_count % 32u);
    const u32 tail_mask = tail_bits == 0u ? UINT32_MAX
        : static_cast<u32>((1ull << tail_bits) - 1ull);
    for (std::size_t word = threadIdx.x; word < words_per_gene; word += blockDim.x) {
        u32 lhs = a[word], rhs = b[word];
        if (word + 1u == words_per_gene) {
            lhs &= tail_mask;
            rhs &= tail_mask;
        }
        count_a += __popc(lhs);
        count_b += __popc(rhs);
        intersection += __popc(lhs & rhs);
        support_union += __popc(lhs | rhs);
    }
    count_a = warp_sum(count_a);
    count_b = warp_sum(count_b);
    intersection = warp_sum(intersection);
    support_union = warp_sum(support_union);

    __shared__ unsigned long long warp_a[scorer_threads / 32u];
    __shared__ unsigned long long warp_b[scorer_threads / 32u];
    __shared__ unsigned long long warp_intersection[scorer_threads / 32u];
    __shared__ unsigned long long warp_union[scorer_threads / 32u];
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    if (lane == 0u) {
        warp_a[warp] = count_a;
        warp_b[warp] = count_b;
        warp_intersection[warp] = intersection;
        warp_union[warp] = support_union;
    }
    __syncthreads();
    if (warp != 0u) return;
    count_a = lane < scorer_threads / 32u ? warp_a[lane] : 0u;
    count_b = lane < scorer_threads / 32u ? warp_b[lane] : 0u;
    intersection = lane < scorer_threads / 32u ? warp_intersection[lane] : 0u;
    support_union = lane < scorer_threads / 32u ? warp_union[lane] : 0u;
    count_a = warp_sum(count_a);
    count_b = warp_sum(count_b);
    intersection = warp_sum(intersection);
    support_union = warp_sum(support_union);
    if (lane == 0u) {
        if (!detail::calculate_merge_cost(
                count_a, count_b, intersection, policy, &costs[candidate])
            || costs[candidate].support_union != support_union) {
            atomicCAS(error_index, 0u,
                      candidate >= static_cast<u64>(UINT32_MAX)
                          ? UINT32_MAX : static_cast<u32>(candidate) + 1u);
        }
    }
}

} // namespace

validation_result score_gene_merges_cuda(
    const gs::gene_support_bitset_view &support,
    const gc::gene_candidate_pair_view &candidates,
    const exact_merge_cost_policy &policy,
    int device,
    owned_exact_gene_merge_scores *out) {
    static_assert(std::is_trivially_copyable<exact_gene_merge_cost>::value,
                  "exact merge costs must remain device-copyable");
    static_assert(sizeof(gc::gene_candidate_pair) == sizeof(u64),
                  "candidate pairs must remain packed 64-bit endpoints");
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "CUDA exact merge-score output is null");
    }
    const validation_result input_status = detail::validate_scoring_inputs(
        support, candidates, policy);
    if (!input_status) return input_status;
    if (candidates.count > static_cast<u64>(INT_MAX)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "CUDA exact merge candidate count exceeds grid bounds");
    }
    if (candidates.count == 0u) {
        try {
            exact_merge_scoring_provenance provenance;
            provenance.algorithm_version = policy.version;
            provenance.policy = policy;
            provenance.candidates = *candidates.provenance;
            *out = owned_exact_gene_merge_scores({}, {}, 0u, std::move(provenance));
        } catch (const std::bad_alloc &) {
            return validation_error(validation_code::insufficient_capacity, invalid_id,
                                    "failed to copy CUDA exact merge-score provenance");
        }
        return validation_ok();
    }

    if (candidates.count > std::numeric_limits<std::size_t>::max()
            / sizeof(gc::gene_candidate_pair)
        || candidates.count > std::numeric_limits<std::size_t>::max()
            / sizeof(exact_gene_merge_cost)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "CUDA exact merge staging size overflows host size_t");
    }
    const std::size_t pair_bytes = static_cast<std::size_t>(candidates.count)
        * sizeof(gc::gene_candidate_pair);
    const std::size_t cost_bytes = static_cast<std::size_t>(candidates.count)
        * sizeof(exact_gene_merge_cost);

    int device_count = 0, previous_device = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) return cuda_status(status, "cudaGetDeviceCount failed for exact merge scoring");
    if (device < 0 || device >= device_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                "requested exact merge-score CUDA device is unavailable");
    }
    status = cudaGetDevice(&previous_device);
    if (status != cudaSuccess) return cuda_status(status, "cudaGetDevice failed for exact merge scoring");
    status = cudaSetDevice(device);
    if (status != cudaSuccess) return cuda_status(status, "cudaSetDevice failed for exact merge scoring");

    device_buffers buffers;
    if (support.layout.support_bytes != 0u) {
        status = cudaMalloc(reinterpret_cast<void **>(&buffers.support), support.layout.support_bytes);
    }
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void **>(&buffers.pairs), pair_bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void **>(&buffers.costs), cost_bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void **>(&buffers.error_index), sizeof(u32));
    if (status == cudaSuccess && support.layout.support_bytes != 0u) {
        status = cudaMemcpy(buffers.support, support.gene_support, support.layout.support_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(buffers.pairs, candidates.pairs, pair_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) status = cudaMemset(buffers.error_index, 0, sizeof(u32));
    if (status != cudaSuccess) {
        release_buffers(&buffers);
        cudaSetDevice(previous_device);
        return cuda_status(status, "CUDA exact merge-score staging failed");
    }

    score_gene_merges_kernel<<<static_cast<unsigned int>(candidates.count), scorer_threads>>>(
        buffers.support, support.layout.words_per_gene, support.layout.sampled_cell_count,
        buffers.pairs, candidates.count, policy, buffers.costs, buffers.error_index);
    status = cudaGetLastError();
    std::unique_ptr<exact_gene_merge_cost[]> costs(
        new (std::nothrow) exact_gene_merge_cost[static_cast<std::size_t>(candidates.count)]);
    std::unique_ptr<candidate_relation[]> relations(
        new (std::nothrow) candidate_relation[static_cast<std::size_t>(candidates.count)]);
    if (costs == nullptr || relations == nullptr) {
        release_buffers(&buffers);
        cudaSetDevice(previous_device);
        return validation_error(validation_code::insufficient_capacity, invalid_id,
                                "failed to allocate CUDA exact merge-score host output");
    }
    u32 device_error = 0u;
    if (status == cudaSuccess) {
        status = cudaMemcpy(costs.get(), buffers.costs, cost_bytes, cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(&device_error, buffers.error_index, sizeof(u32), cudaMemcpyDeviceToHost);
    }
    const cudaError_t release_status = release_buffers(&buffers);
    const cudaError_t restore_status = cudaSetDevice(previous_device);
    if (status == cudaSuccess) status = release_status;
    if (status == cudaSuccess) status = restore_status;
    if (status != cudaSuccess) {
        return cuda_status(status, "CUDA exact merge-score execution or cleanup failed");
    }
    if (device_error != 0u) {
        return validation_error(validation_code::integer_overflow,
                                device_error == UINT32_MAX ? invalid_id : device_error - 1u,
                                "CUDA exact merge byte accounting overflows");
    }
    for (u64 index = 0u; index < candidates.count; ++index) {
        const gc::gene_candidate_pair &pair = candidates.pairs[index];
        if (costs[index].support_a != support.detected_cell_counts[pair.gene_a]
            || costs[index].support_b != support.detected_cell_counts[pair.gene_b]) {
            return validation_error(validation_code::invalid_plan_geometry,
                                    index > invalid_id ? invalid_id : static_cast<u32>(index),
                                    "CUDA exact merge bitset/count evidence disagrees");
        }
        relations[index] = detail::make_exact_relation(
            pair.gene_a, pair.gene_b, costs[index]);
    }
    try {
        exact_merge_scoring_provenance provenance;
        provenance.algorithm_version = policy.version;
        provenance.policy = policy;
        provenance.candidates = *candidates.provenance;
        *out = owned_exact_gene_merge_scores(
            std::move(relations), std::move(costs), candidates.count, std::move(provenance));
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
                                "failed to copy CUDA exact merge-score provenance");
    }
    return validation_ok();
}

} // namespace cellpack
