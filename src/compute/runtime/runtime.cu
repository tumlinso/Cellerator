#include "runtime.hh"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cellerator::compute::runtime {

namespace {

constexpr int kAddThreads = 256;
constexpr int kNoPeerRank = std::numeric_limits<int>::max();
constexpr int kInvalidPairScore = -1000000000;

struct pair_reduction_plan {
    unsigned int peer0_index = 0u;
    unsigned int leader1_index = 0u;
    unsigned int peer1_index = 0u;
    unsigned int peer0 = 0u;
    unsigned int leader1 = 0u;
    unsigned int peer1 = 0u;
    bool valid = false;
};

__global__ void dense_add_inplace_kernel_(float *dst, const float *src, std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] += src[idx];
}

int peer_rank_index_(const fleet_context &fleet, unsigned int src_slot, unsigned int dst_slot) {
    return static_cast<int>(src_slot * fleet.local.device_count + dst_slot);
}

int best_peer_rank_(const fleet_context &fleet) {
    if (fleet.peer_performance_rank == nullptr) return -1;
    int best_rank = kNoPeerRank;
    for (unsigned int src = 0; src < fleet.local.device_count; ++src) {
        for (unsigned int dst = 0; dst < fleet.local.device_count; ++dst) {
            if (src == dst) continue;
            const int rank = fleet.peer_performance_rank[peer_rank_index_(fleet, src, dst)];
            if (rank >= 0 && rank < best_rank) best_rank = rank;
        }
    }
    return best_rank == kNoPeerRank ? -1 : best_rank;
}

bool rank_strictly_worse_(const fleet_context &fleet, unsigned int src_slot, unsigned int dst_slot, int best_rank) {
    const int rank = fleet_peer_performance_rank(fleet, src_slot, dst_slot);
    return rank >= 0 && rank > best_rank;
}

void detect_topology_profile_(fleet_context *fleet) {
    fleet->topology = fleet_topology_descriptor{};
    fleet->topology.best_peer_rank = best_peer_rank_(*fleet);
    if (fleet->local.device_count < 4u || fleet->topology.best_peer_rank < 0) return;

    const int best_rank = fleet->topology.best_peer_rank;
    const bool pair02_best =
        fleet_peer_performance_rank(*fleet, 0u, 2u) == best_rank
        && fleet_peer_performance_rank(*fleet, 2u, 0u) == best_rank;
    const bool pair13_best =
        fleet_peer_performance_rank(*fleet, 1u, 3u) == best_rank
        && fleet_peer_performance_rank(*fleet, 3u, 1u) == best_rank;
    const bool leaders_worse =
        rank_strictly_worse_(*fleet, 0u, 1u, best_rank)
        && rank_strictly_worse_(*fleet, 1u, 0u, best_rank)
        && rank_strictly_worse_(*fleet, 2u, 3u, best_rank)
        && rank_strictly_worse_(*fleet, 3u, 2u, best_rank);
    const bool cross_worse =
        rank_strictly_worse_(*fleet, 0u, 3u, best_rank)
        && rank_strictly_worse_(*fleet, 3u, 0u, best_rank)
        && rank_strictly_worse_(*fleet, 1u, 2u, best_rank)
        && rank_strictly_worse_(*fleet, 2u, 1u, best_rank);

    if (pair02_best && pair13_best && leaders_worse && cross_worse) {
        fleet->topology.active_profile = fleet_topology_descriptor::profile::native_v100_pairs;
        fleet->topology.native_pair_count = 2u;
    }
}

void require_slots_(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slots == nullptr && slot_count != 0) throw std::invalid_argument("distributed launch requires slot storage");
    for (unsigned int i = 0; i < slot_count; ++i) {
        if (!fleet_slot_available(fleet, slots[i])) throw std::out_of_range("distributed launch slot is unavailable");
    }
}

int slot_index_in_list_(const unsigned int *slots, unsigned int slot_count, unsigned int slot) {
    if (slots == nullptr) return -1;
    for (unsigned int i = 0u; i < slot_count; ++i) {
        if (slots[i] == slot) return (int) i;
    }
    return -1;
}

void copy_or_alias_to_leader_(
    fleet_context *fleet,
    unsigned int leader_slot,
    const float *src,
    int src_device,
    std::size_t count,
    float *leader_out) {
    const int leader_device = fleet_device_id(*fleet, leader_slot);
    cuda_require(cudaSetDevice(leader_device), "cudaSetDevice(copy_or_alias_to_leader)");
    if (src == leader_out && src_device == leader_device) return;
    const std::size_t bytes = count * sizeof(float);
    if (src_device == leader_device) {
        cuda_require(cudaMemcpyAsync(leader_out, src, bytes, cudaMemcpyDeviceToDevice, fleet_stream(*fleet, leader_slot)), "cudaMemcpyAsync(copy_or_alias_to_leader)");
        return;
    }
    cuda_require(cudaMemcpyPeerAsync(leader_out, leader_device, src, src_device, bytes, fleet_stream(*fleet, leader_slot)), "cudaMemcpyPeerAsync(copy_or_alias_to_leader)");
}

int pair_link_score_(const fleet_context &fleet, unsigned int lhs_slot, unsigned int rhs_slot) {
    const int lhs_rank = fleet_peer_performance_rank(fleet, lhs_slot, rhs_slot);
    const int rhs_rank = fleet_peer_performance_rank(fleet, rhs_slot, lhs_slot);
    if (lhs_rank < 0 || rhs_rank < 0) return kInvalidPairScore;
    return -(lhs_rank + rhs_rank);
}

bool matches_native_slot_order_(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slot_count != 4u || !fleet_has_native_v100_topology(fleet)) return false;
    unsigned int native_slots[4] = {};
    if (default_native_fleet_slots(fleet, native_slots, 4u) != 4u) return false;
    for (unsigned int i = 0u; i < 4u; ++i) {
        if (slots[i] != native_slots[i]) return false;
    }
    return true;
}

void warn_nccl_fallback_once_(const char *reason) {
    static bool warned = false;
    if (warned) return;
    warned = true;
    std::fprintf(
        stderr,
        "CelleratorDist warning: %s; using local CUDA peer-copy reduction fallback for reduce_sum_to_leader_f32. "
        "This path is for correctness and tests, not steady-state multi-GPU scaling.\n",
        reason != nullptr ? reason : "NCCL collective path is unavailable");
}

pair_reduction_plan build_generic_pair_plan_(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    pair_reduction_plan best_plan;
    if (slot_count != 4u || slots == nullptr) return best_plan;

    int best_score = kInvalidPairScore;
    for (unsigned int peer_index = 1u; peer_index < 4u; ++peer_index) {
        const unsigned int leader0 = slots[0];
        const unsigned int peer0 = slots[peer_index];
        unsigned int remaining[2] = {};
        unsigned int cursor = 0u;
        for (unsigned int i = 1u; i < 4u; ++i) {
            if (i == peer_index) continue;
            remaining[cursor++] = slots[i];
        }
        const int score0 = pair_link_score_(fleet, leader0, peer0);
        const int score1 = pair_link_score_(fleet, remaining[0], remaining[1]);
        if (score0 == kInvalidPairScore || score1 == kInvalidPairScore) continue;
        const int total_score = score0 + score1;
        if (!best_plan.valid || total_score > best_score) {
            best_score = total_score;
            best_plan.peer0_index = peer_index;
            best_plan.leader1_index = 0u;
            best_plan.peer1_index = 0u;
            best_plan.peer0 = peer0;
            best_plan.leader1 = remaining[0];
            best_plan.peer1 = remaining[1];
            for (unsigned int i = 1u; i < 4u; ++i) {
                if (slots[i] == remaining[0]) best_plan.leader1_index = i;
                if (slots[i] == remaining[1]) best_plan.peer1_index = i;
            }
            best_plan.valid = true;
        }
    }
    return best_plan;
}

} // namespace

void init(fleet_context *fleet) {
    if (fleet == nullptr) throw std::invalid_argument("init(fleet_context) requires a fleet");
    cdist::init(&fleet->local);
    fleet->reduce_scratch = nullptr;
    fleet->reduce_scratch_bytes = nullptr;
    fleet->peer_performance_rank = nullptr;
    fleet->topology = fleet_topology_descriptor{};
}

void clear(fleet_context *fleet) {
    if (fleet == nullptr) return;
    if (fleet->reduce_scratch != nullptr) {
        for (unsigned int i = 0; i < fleet->local.device_count; ++i) {
            if (fleet->reduce_scratch[i] != nullptr) {
                cudaSetDevice(fleet->local.device_ids[i]);
                cudaFree(fleet->reduce_scratch[i]);
            }
        }
    }
    std::free(fleet->reduce_scratch);
    std::free(fleet->reduce_scratch_bytes);
    std::free(fleet->peer_performance_rank);
    fleet->reduce_scratch = nullptr;
    fleet->reduce_scratch_bytes = nullptr;
    fleet->peer_performance_rank = nullptr;
    fleet->topology = fleet_topology_descriptor{};
    cdist::clear(&fleet->local);
}

void discover_fleet(
    fleet_context *fleet,
    bool create_streams,
    unsigned int stream_flags,
    bool enable_peer_access) {
    if (fleet == nullptr) throw std::invalid_argument("discover_fleet requires a fleet");
    clear(fleet);
    init(fleet);
    cuda_require(
        cdist::discover_local(&fleet->local, create_streams ? 1 : 0, stream_flags),
        "discover_local(runtime fleet)");
    if (enable_peer_access && fleet->local.device_count != 0) {
        cuda_require(cdist::enable_peer_access(&fleet->local), "enable_peer_access(runtime fleet)");
    }
#if CELLERATOR_DIST_HAS_NCCL
    if (fleet->local.device_count != 0) (void) cdist::init_local_nccl(&fleet->local);
#endif
    if (fleet->local.device_count != 0) {
        fleet->reduce_scratch = static_cast<void **>(std::calloc(fleet->local.device_count, sizeof(void *)));
        fleet->reduce_scratch_bytes = static_cast<std::size_t *>(std::calloc(fleet->local.device_count, sizeof(std::size_t)));
        fleet->peer_performance_rank = static_cast<int *>(
            std::malloc(static_cast<std::size_t>(fleet->local.device_count) * fleet->local.device_count * sizeof(int)));
        if (fleet->reduce_scratch == nullptr || fleet->reduce_scratch_bytes == nullptr || fleet->peer_performance_rank == nullptr) {
            clear(fleet);
            throw std::bad_alloc();
        }
        for (unsigned int src = 0; src < fleet->local.device_count; ++src) {
            for (unsigned int dst = 0; dst < fleet->local.device_count; ++dst) {
                const int index = peer_rank_index_(*fleet, src, dst);
                if (src == dst) {
                    fleet->peer_performance_rank[index] = 0;
                    continue;
                }
                if (!cdist::peer_access_supported(&fleet->local, src, dst)) {
                    fleet->peer_performance_rank[index] = -1;
                    continue;
                }
                int rank = -1;
                cuda_require(
                    cudaDeviceGetP2PAttribute(
                        &rank,
                        cudaDevP2PAttrPerformanceRank,
                        fleet->local.device_ids[src],
                        fleet->local.device_ids[dst]),
                    "cudaDeviceGetP2PAttribute(runtime fleet)");
                fleet->peer_performance_rank[index] = rank;
            }
        }
        detect_topology_profile_(fleet);
    }
}

void *request_fleet_scratch(fleet_context *fleet, unsigned int slot, std::size_t bytes) {
    if (fleet == nullptr) throw std::invalid_argument("request_fleet_scratch requires a fleet");
    if (!fleet_slot_available(*fleet, slot)) throw std::out_of_range("request_fleet_scratch slot is unavailable");
    if (bytes <= fleet->reduce_scratch_bytes[slot]) return fleet->reduce_scratch[slot];
    cuda_require(cudaSetDevice(fleet->local.device_ids[slot]), "cudaSetDevice(request_fleet_scratch)");
    if (fleet->reduce_scratch[slot] != nullptr) {
        cudaFree(fleet->reduce_scratch[slot]);
        fleet->reduce_scratch[slot] = nullptr;
        fleet->reduce_scratch_bytes[slot] = 0;
    }
    if (bytes == 0) return nullptr;
    cuda_require(cudaMalloc(fleet->reduce_scratch + slot, bytes), "cudaMalloc(runtime fleet scratch)");
    fleet->reduce_scratch_bytes[slot] = bytes;
    return fleet->reduce_scratch[slot];
}

bool fleet_slot_available(const fleet_context &fleet, unsigned int slot) {
    return slot < fleet.local.device_count && fleet.local.device_ids != nullptr;
}

int fleet_device_id(const fleet_context &fleet, unsigned int slot) {
    if (!fleet_slot_available(fleet, slot)) throw std::out_of_range("fleet_device_id slot is unavailable");
    return fleet.local.device_ids[slot];
}

cudaStream_t fleet_stream(const fleet_context &fleet, unsigned int slot) {
    if (!fleet_slot_available(fleet, slot)) throw std::out_of_range("fleet_stream slot is unavailable");
    return fleet.local.streams != nullptr ? fleet.local.streams[slot] : nullptr;
}

int fleet_peer_performance_rank(const fleet_context &fleet, unsigned int src_slot, unsigned int dst_slot) {
    if (!fleet_slot_available(fleet, src_slot) || !fleet_slot_available(fleet, dst_slot)) {
        throw std::out_of_range("fleet_peer_performance_rank slot is unavailable");
    }
    if (fleet.peer_performance_rank == nullptr) return -1;
    return fleet.peer_performance_rank[peer_rank_index_(fleet, src_slot, dst_slot)];
}

bool fleet_has_native_v100_topology(const fleet_context &fleet) {
    return fleet.topology.active_profile == fleet_topology_descriptor::profile::native_v100_pairs;
}

void synchronize_slots(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slots == nullptr && slot_count != 0) throw std::invalid_argument("synchronize_slots requires slot storage");
    for (unsigned int i = 0; i < slot_count; ++i) {
        const unsigned int slot = slots[i];
        if (!fleet_slot_available(fleet, slot)) throw std::out_of_range("synchronize_slots slot is unavailable");
        if (fleet.local.streams == nullptr || fleet.local.streams[slot] == nullptr) continue;
        cuda_require(cudaSetDevice(fleet.local.device_ids[slot]), "cudaSetDevice(synchronize_slots)");
        cuda_require(cudaStreamSynchronize(fleet.local.streams[slot]), "cudaStreamSynchronize(synchronize_slots)");
    }
}

void dense_add_inplace_f32(const execution_context &ctx, float *dst, const float *src, std::size_t count) {
    if (count == 0) return;
    if (dst == nullptr || src == nullptr) throw std::invalid_argument("dense_add_inplace_f32 requires source and destination buffers");
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(dense_add_inplace_f32)");
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kAddThreads) - 1u) / static_cast<std::size_t>(kAddThreads));
    dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, ctx.stream>>>(dst, src, count);
    cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(dense_add_inplace_f32)");
}

void reduce_sum_to_leader_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const float *const *partials,
    std::size_t count,
    unsigned int leader_slot,
    float *leader_out,
    const reduce_sum_to_leader_f32_options *options) {
    if (fleet == nullptr) throw std::invalid_argument("reduce_sum_to_leader_f32 requires a fleet");
    if (partials == nullptr && slot_count != 0) throw std::invalid_argument("reduce_sum_to_leader_f32 requires partial storage");
    if (leader_out == nullptr && count != 0) throw std::invalid_argument("reduce_sum_to_leader_f32 requires a leader output");
    require_slots_(*fleet, slots, slot_count);
    if (slot_count == 0 || count == 0) return;
    const int leader_index = slot_index_in_list_(slots, slot_count, leader_slot);
    if (leader_index < 0) throw std::invalid_argument("reduce_sum_to_leader_f32 leader slot is not in the slot list");

    synchronize_slots(*fleet, slots, slot_count);

#if CELLERATOR_DIST_HAS_NCCL
    if (options != nullptr && options->ranked_nccl != nullptr && options->ranked_nccl->ready != 0u) {
        const cdist::nccl_communicator *comm = options->ranked_nccl;
        std::unique_ptr<const void *[]> sendbufs(new const void *[comm->device_count]);
        std::unique_ptr<void *[]> recvbufs(new void *[comm->device_count]);
        std::unique_ptr<cudaStream_t[]> streams(new cudaStream_t[comm->device_count]);
        for (unsigned int rank = 0u; rank < comm->device_count; ++rank) {
            const unsigned int slot = comm->local_slots[rank];
            const int selected = slot_index_in_list_(slots, slot_count, slot);
            if (selected < 0) throw std::invalid_argument("ranked NCCL communicator does not match reduction slots");
            sendbufs[rank] = partials[selected];
            recvbufs[rank] = slot == leader_slot
                ? static_cast<void *>(leader_out)
                : request_fleet_scratch(fleet, slot, count * sizeof(float));
            streams[rank] = fleet_stream(*fleet, slot);
        }
        const ncclResult_t result = cdist::communicator_allreduce(
            comm,
            sendbufs.get(),
            recvbufs.get(),
            count,
            ncclFloat32,
            ncclSum,
            streams.get());
        if (result == ncclSuccess) return;
        throw std::runtime_error("reduce_sum_to_leader_f32 ranked NCCL allreduce failed");
    }

    if (slot_count == 1u) {
        copy_or_alias_to_leader_(fleet, leader_slot, partials[leader_index], fleet_device_id(*fleet, leader_slot), count, leader_out);
        return;
    }
    {
        std::unique_ptr<void *[]> recvbufs(new void *[slot_count]);
        for (unsigned int i = 0u; i < slot_count; ++i) {
            recvbufs[i] = slots[i] == leader_slot
                ? static_cast<void *>(leader_out)
                : request_fleet_scratch(fleet, slots[i], count * sizeof(float));
        }
        const ncclResult_t result = cdist::local_allreduce(
            &fleet->local,
            slots,
            slot_count,
            reinterpret_cast<const void *const *>(partials),
            recvbufs.get(),
            count,
            ncclFloat32,
            ncclSum);
        if (result == ncclSuccess) return;
        warn_nccl_fallback_once_("NCCL communicator is unavailable");
    }
#else
    if (slot_count > 1u) warn_nccl_fallback_once_("NCCL support was not built");
#endif

    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kAddThreads) - 1u) / static_cast<std::size_t>(kAddThreads));
    const int leader_device = fleet_device_id(*fleet, leader_slot);
    copy_or_alias_to_leader_(fleet, leader_slot, partials[leader_index], leader_device, count, leader_out);

    pair_reduction_plan pair_plan;
    if (leader_index == 0 && slot_count == 4u) {
        if (!::cellerator::build::cuda_mode_is_generic && matches_native_slot_order_(*fleet, slots, slot_count)) {
            pair_plan.peer0_index = 1u;
            pair_plan.leader1_index = 2u;
            pair_plan.peer1_index = 3u;
            pair_plan.peer0 = slots[1];
            pair_plan.leader1 = slots[2];
            pair_plan.peer1 = slots[3];
            pair_plan.valid = true;
        } else {
            pair_plan = build_generic_pair_plan_(*fleet, slots, slot_count);
        }
    }

    if (pair_plan.valid) {
        const unsigned int peer0 = pair_plan.peer0;
        const unsigned int leader1 = pair_plan.leader1;
        const unsigned int peer1 = pair_plan.peer1;

        float *leader_scratch = static_cast<float *>(request_fleet_scratch(fleet, leader_slot, count * sizeof(float)));
        cuda_require(
            cudaMemcpyPeerAsync(
                leader_scratch,
                leader_device,
                partials[pair_plan.peer0_index],
                fleet_device_id(*fleet, peer0),
                count * sizeof(float),
                fleet_stream(*fleet, leader_slot)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader pair0)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader_slot)>>>(leader_out, leader_scratch, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader pair0)");

        const int leader1_device = fleet_device_id(*fleet, leader1);
        float *leader1_scratch = static_cast<float *>(request_fleet_scratch(fleet, leader1, count * sizeof(float) * 2u));
        float *pair1_accum = leader1_scratch;
        float *pair1_tmp = leader1_scratch + count;
        copy_or_alias_to_leader_(fleet, leader1, partials[pair_plan.leader1_index], leader1_device, count, pair1_accum);
        cuda_require(
            cudaMemcpyPeerAsync(
                pair1_tmp,
                leader1_device,
                partials[pair_plan.peer1_index],
                fleet_device_id(*fleet, peer1),
                count * sizeof(float),
                fleet_stream(*fleet, leader1)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader pair1)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader1)>>>(pair1_accum, pair1_tmp, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader pair1)");
        cuda_require(cudaStreamSynchronize(fleet_stream(*fleet, leader1)), "cudaStreamSynchronize(reduce_sum_to_leader pair1)");

        cuda_require(cudaSetDevice(leader_device), "cudaSetDevice(reduce_sum_to_leader leaders)");
        cuda_require(
            cudaMemcpyPeerAsync(leader_scratch, leader_device, pair1_accum, leader1_device, count * sizeof(float), fleet_stream(*fleet, leader_slot)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader leaders)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader_slot)>>>(leader_out, leader_scratch, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader leaders)");
        return;
    }

    for (unsigned int i = 0; i < slot_count; ++i) {
        if (i == static_cast<unsigned int>(leader_index)) continue;
        float *scratch = static_cast<float *>(request_fleet_scratch(fleet, leader_slot, count * sizeof(float)));
        cuda_require(cudaSetDevice(leader_device), "cudaSetDevice(reduce_sum_to_leader direct)");
        cuda_require(
            cudaMemcpyPeerAsync(scratch, leader_device, partials[i], fleet_device_id(*fleet, slots[i]), count * sizeof(float), fleet_stream(*fleet, leader_slot)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader direct)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader_slot)>>>(leader_out, scratch, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader direct)");
    }
}

} // namespace cellerator::compute::runtime
