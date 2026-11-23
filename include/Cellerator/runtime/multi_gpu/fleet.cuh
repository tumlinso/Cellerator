#pragma once

#include <Cellerator/runtime/runtime.cuh>
#include <Cellerator/runtime/multi_gpu/context.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::runtime {

namespace cdist = ::cellerator::dist;

struct fleet_context {
    cdist::local_context local;
    void **reduce_scratch = nullptr;
    std::size_t *reduce_scratch_bytes = nullptr;
    int *peer_performance_rank = nullptr;
};

struct reduce_sum_to_leader_f32_options {
#if CELLERATOR_DIST_HAS_NCCL
    const cdist::nccl_communicator *ranked_nccl = nullptr;
#endif
};

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
void synchronize_slots(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count);
void dense_add_inplace_f32(const execution_context &ctx, float *dst, const float *src, std::size_t count);
void reduce_sum_to_leader_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const float *const *partials,
    std::size_t count,
    unsigned int leader_slot,
    float *leader_out,
    const reduce_sum_to_leader_f32_options *options = nullptr);

inline unsigned int default_pair_slots(
    const fleet_context &fleet,
    unsigned int pair_index,
    unsigned int *slots,
    unsigned int capacity) {
    if (slots == nullptr || capacity < 2u || pair_index >= fleet.local.device_count / 2u) return 0u;
    const unsigned int begin = pair_index * 2u;
    slots[0] = begin;
    slots[1] = begin + 1u;
    return 2u;
}

inline unsigned int default_fleet_slots(
    const fleet_context &fleet,
    unsigned int *slots,
    unsigned int capacity) {
    if (slots == nullptr) return 0u;
    const unsigned int use_count = fleet.local.device_count < capacity ? fleet.local.device_count : capacity;
    for (unsigned int i = 0; i < use_count; ++i) slots[i] = i;
    return use_count;
}

} // namespace cellerator::runtime
