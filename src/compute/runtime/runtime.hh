#pragma once

#include <CellShard/CellShard.hh>
#include <Cellerator/core/runtime/runtime.cuh>
#include <Cellerator/dist/distributed.cuh>
#include "cellerator_cuda_mode.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::runtime {

namespace cs = ::cellshard;
namespace cdist = ::cellerator::dist;

using ::cellerator::core::runtime::cuda_require;
using ::cellerator::core::runtime::device_buffer;
using ::cellerator::core::runtime::allocate_device_buffer;
using ::cellerator::core::runtime::upload_device_buffer;
using ::cellerator::core::runtime::download_device_buffer;
using ::cellerator::core::runtime::execution_context;
using ::cellerator::core::runtime::scratch_arena;
using ::cellerator::core::runtime::cusparse_cache;
using ::cellerator::core::runtime::cublas_cache;
using ::cellerator::core::runtime::request_scratch;
using ::cellerator::core::runtime::acquire_cusparse;
using ::cellerator::core::runtime::acquire_csr_f32_descriptor;
using ::cellerator::core::runtime::acquire_blocked_ell_f16_descriptor;
using ::cellerator::core::runtime::cached_spmv_bytes;
using ::cellerator::core::runtime::cached_spmm_bytes;
using ::cellerator::core::runtime::cached_blocked_ell_spmm_bytes;
using ::cellerator::core::runtime::acquire_cublas;

struct fleet_topology_descriptor {
    enum class profile {
        generic = 0,
        native_v100_pairs = 1
    };

    profile active_profile = profile::generic;
    int best_peer_rank = -1;
    unsigned int native_pair_count = 0u;
};

struct fleet_context {
    cdist::local_context local;
    void **reduce_scratch = nullptr;
    std::size_t *reduce_scratch_bytes = nullptr;
    int *peer_performance_rank = nullptr;
    fleet_topology_descriptor topology;
};


using ::cellerator::core::runtime::init;
using ::cellerator::core::runtime::clear;

void init(fleet_context *fleet);
void clear(fleet_context *fleet);
void discover_fleet(
    fleet_context *fleet,
    bool create_streams = true,
    unsigned int stream_flags = cudaStreamNonBlocking,
    bool enable_peer_access = true);
void *request_fleet_scratch(fleet_context *fleet, unsigned int slot, std::size_t bytes);
bool fleet_slot_available(const fleet_context &fleet, unsigned int slot);
int fleet_device_id(const fleet_context &fleet, unsigned int slot);
cudaStream_t fleet_stream(const fleet_context &fleet, unsigned int slot);
int fleet_peer_performance_rank(const fleet_context &fleet, unsigned int src_slot, unsigned int dst_slot);
bool fleet_has_native_v100_topology(const fleet_context &fleet);
void synchronize_slots(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count);

inline unsigned int default_generic_pair_slots(
    const fleet_context &fleet,
    unsigned int pair_index,
    unsigned int *slots,
    unsigned int capacity) {
    if (slots == nullptr || capacity < 2u) return 0u;
    const unsigned int begin = pair_index * 2u;
    if (begin + 1u >= fleet.local.device_count) return 0u;
    slots[0] = begin;
    slots[1] = begin + 1u;
    return 2u;
}

inline unsigned int default_generic_fleet_slots(
    const fleet_context &fleet,
    unsigned int *slots,
    unsigned int capacity) {
    if (slots == nullptr) return 0u;
    const unsigned int use_count = fleet.local.device_count < capacity ? fleet.local.device_count : capacity;
    for (unsigned int i = 0; i < use_count; ++i) slots[i] = i;
    return use_count;
}

inline unsigned int default_native_pair_slots(
    const fleet_context &fleet,
    unsigned int pair_index,
    unsigned int *slots,
    unsigned int capacity) {
    if (!fleet_has_native_v100_topology(fleet) || slots == nullptr || capacity < 2u) return 0u;
    if (pair_index >= fleet.topology.native_pair_count) return 0u;
    if (pair_index == 0u) {
        slots[0] = 0u;
        slots[1] = 2u;
        return 2u;
    }
    slots[0] = 1u;
    slots[1] = 3u;
    return 2u;
}

inline unsigned int default_native_fleet_slots(
    const fleet_context &fleet,
    unsigned int *slots,
    unsigned int capacity) {
    if (!fleet_has_native_v100_topology(fleet) || slots == nullptr || capacity < 4u) return 0u;
    slots[0] = 0u;
    slots[1] = 2u;
    slots[2] = 1u;
    slots[3] = 3u;
    return 4u;
}

inline unsigned int default_mode_pair_slots(
    const fleet_context &fleet,
    unsigned int pair_index,
    unsigned int *slots,
    unsigned int capacity) {
    if (!build::cuda_mode_is_generic) {
        const unsigned int native_count = default_native_pair_slots(fleet, pair_index, slots, capacity);
        if (native_count != 0u) return native_count;
    }
    return default_generic_pair_slots(fleet, pair_index, slots, capacity);
}

inline unsigned int default_mode_fleet_slots(
    const fleet_context &fleet,
    unsigned int *slots,
    unsigned int capacity) {
    if (!build::cuda_mode_is_generic) {
        const unsigned int native_count = default_native_fleet_slots(fleet, slots, capacity);
        if (native_count != 0u) return native_count;
    }
    return default_generic_fleet_slots(fleet, slots, capacity);
}

} // namespace cellerator::compute::runtime
