#include "runtime.hh"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cellerator::compute::runtime {

namespace {

constexpr int kNoPeerRank = std::numeric_limits<int>::max();

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

} // namespace cellerator::compute::runtime
