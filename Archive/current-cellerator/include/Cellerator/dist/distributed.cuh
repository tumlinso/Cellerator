#pragma once

#include <cstddef>
#include <cstdlib>

#include <cuda_runtime.h>

#ifndef CELLERATOR_DIST_HAS_NCCL
#define CELLERATOR_DIST_HAS_NCCL 0
#endif

#if CELLERATOR_DIST_HAS_NCCL
#if defined(__has_include)
#if __has_include(<nccl.h>)
#include <nccl.h>
#elif __has_include("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include/nccl.h")
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include/nccl.h"
#else
#error "CELLERATOR_DIST_HAS_NCCL requires nccl.h"
#endif
#else
#include <nccl.h>
#endif
#endif

namespace cellerator {
namespace dist {

#if CELLERATOR_DIST_HAS_NCCL
struct nccl_communicator {
    unsigned int device_count;
    int *device_ids;
    unsigned int *local_slots;
    int *world_ranks;
    int world_size;
    ncclComm_t *comms;
    unsigned char ready;
    unsigned char owns_storage;
    unsigned long long slot_mask;
};
#endif

// local_context is intentionally minimal:
// - one visible device id per local GPU
// - one optional stream per device
// - a dense peer-access capability table
// - optional NCCL communicators
//
// This is not a scheduler and not a hidden runtime. It only holds enough state
// to make explicit multi-GPU execution cheap for the caller.
struct local_context {
    unsigned int device_count;
    int *device_ids;
    cudaStream_t *streams;
    unsigned char *peer_access;
#if CELLERATOR_DIST_HAS_NCCL
    nccl_communicator full_nccl;
    nccl_communicator *subset_nccl;
    unsigned int subset_nccl_count;
    unsigned int subset_nccl_capacity;
    ncclComm_t *comms;
    unsigned char nccl_ready;
#endif
};

#if CELLERATOR_DIST_HAS_NCCL
inline void init(nccl_communicator *comm) {
    comm->device_count = 0;
    comm->device_ids = 0;
    comm->local_slots = 0;
    comm->world_ranks = 0;
    comm->world_size = 0;
    comm->comms = 0;
    comm->ready = 0u;
    comm->owns_storage = 0u;
    comm->slot_mask = 0u;
}

inline void clear(nccl_communicator *comm) {
    unsigned int i = 0;
    if (comm == 0) return;
    if (comm->comms != 0) {
        for (i = 0; i < comm->device_count; ++i) {
            if (comm->comms[i] != 0) ncclCommDestroy(comm->comms[i]);
        }
    }
    std::free(comm->comms);
    if (comm->owns_storage != 0u) {
        std::free(comm->device_ids);
        std::free(comm->local_slots);
        std::free(comm->world_ranks);
    }
    init(comm);
}

inline int communicator_rank_for_slot(const nccl_communicator *comm, unsigned int slot) {
    unsigned int i = 0;
    if (comm == 0 || comm->local_slots == 0) return -1;
    for (i = 0; i < comm->device_count; ++i) {
        if (comm->local_slots[i] == slot) return (int) i;
    }
    return -1;
}

inline int slot_index_in_list_(const unsigned int *slots, unsigned int slot_count, unsigned int slot) {
    unsigned int i = 0;
    if (slots == 0) return -1;
    for (i = 0; i < slot_count; ++i) {
        if (slots[i] == slot) return (int) i;
    }
    return -1;
}

inline ncclResult_t copy_communicator_layout_(nccl_communicator *comm,
                                              const int *device_ids,
                                              const unsigned int *local_slots,
                                              unsigned int device_count,
                                              const int *world_ranks,
                                              int world_size) {
    unsigned int i = 0;

    if (comm == 0 || device_count == 0 || device_ids == 0) return ncclInvalidArgument;
    clear(comm);

    comm->device_ids = (int *) std::calloc((std::size_t) device_count, sizeof(int));
    comm->local_slots = (unsigned int *) std::calloc((std::size_t) device_count, sizeof(unsigned int));
    comm->comms = (ncclComm_t *) std::calloc((std::size_t) device_count, sizeof(ncclComm_t));
    if (comm->device_ids == 0 || comm->local_slots == 0 || comm->comms == 0) {
        clear(comm);
        return ncclSystemError;
    }
    if (world_ranks != 0) {
        comm->world_ranks = (int *) std::calloc((std::size_t) device_count, sizeof(int));
        if (comm->world_ranks == 0) {
            clear(comm);
            return ncclSystemError;
        }
    }

    comm->device_count = device_count;
    comm->world_size = world_size;
    comm->owns_storage = 1u;
    for (i = 0; i < device_count; ++i) {
        comm->device_ids[i] = device_ids[i];
        comm->local_slots[i] = local_slots != 0 ? local_slots[i] : i;
        if (world_ranks != 0) comm->world_ranks[i] = world_ranks[i];
        if (comm->local_slots[i] < 64u) comm->slot_mask |= (1ull << comm->local_slots[i]);
    }
    return ncclSuccess;
}

inline ncclResult_t init_local_nccl_communicator(nccl_communicator *comm,
                                                 const int *device_ids,
                                                 const unsigned int *local_slots,
                                                 unsigned int device_count) {
    ncclResult_t result = ncclSuccess;

    result = copy_communicator_layout_(comm, device_ids, local_slots, device_count, 0, (int) device_count);
    if (result != ncclSuccess) return result;
    result = ncclCommInitAll(comm->comms, (int) device_count, comm->device_ids);
    if (result == ncclSuccess) {
        comm->ready = 1u;
        return result;
    }
    clear(comm);
    return result;
}

inline ncclResult_t init_ranked_nccl_communicator(nccl_communicator *comm,
                                                  const int *device_ids,
                                                  const unsigned int *local_slots,
                                                  unsigned int device_count,
                                                  const int *world_ranks,
                                                  int world_size,
                                                  const ncclUniqueId *unique_id) {
    unsigned int i = 0;
    ncclResult_t result = ncclSuccess;

    if (unique_id == 0 || world_ranks == 0 || world_size <= 0) return ncclInvalidArgument;
    result = copy_communicator_layout_(comm, device_ids, local_slots, device_count, world_ranks, world_size);
    if (result != ncclSuccess) return result;

    ncclGroupStart();
    for (i = 0; i < device_count; ++i) {
        cudaSetDevice(comm->device_ids[i]);
        result = ncclCommInitRank(comm->comms + i, world_size, *unique_id, comm->world_ranks[i]);
        if (result != ncclSuccess) break;
    }
    {
        const ncclResult_t group_result = ncclGroupEnd();
        if (result == ncclSuccess) result = group_result;
    }
    if (result == ncclSuccess) {
        comm->ready = 1u;
        return result;
    }
    clear(comm);
    return result;
}

inline void sync_local_nccl_views_(local_context *ctx) {
    if (ctx == 0) return;
    ctx->comms = ctx->full_nccl.comms;
    ctx->nccl_ready = ctx->full_nccl.ready;
}

inline int reserve_local_nccl_cache_(local_context *ctx, unsigned int capacity) {
    unsigned int i = 0;
    nccl_communicator *next = 0;

    if (ctx == 0) return 0;
    if (capacity <= ctx->subset_nccl_capacity) return 1;
    next = (nccl_communicator *) std::realloc(ctx->subset_nccl, (std::size_t) capacity * sizeof(nccl_communicator));
    if (next == 0) return 0;
    ctx->subset_nccl = next;
    for (i = ctx->subset_nccl_capacity; i < capacity; ++i) init(ctx->subset_nccl + i);
    ctx->subset_nccl_capacity = capacity;
    return 1;
}

inline unsigned int canonicalize_slots_(const local_context *ctx,
                                        const unsigned int *slots,
                                        unsigned int slot_count,
                                        unsigned int *ordered,
                                        unsigned long long *mask) {
    unsigned int i = 0;
    unsigned int out_count = 0;
    unsigned long long out_mask = 0u;

    if (ordered == 0 || mask == 0 || ctx == 0) return 0u;
    for (i = 0; i < slot_count; ++i) {
        unsigned int value = slots[i];
        unsigned int pos = out_count;
        if (value >= ctx->device_count) return 0u;
        if (value < 64u && (out_mask & (1ull << value)) != 0u) continue;
        while (pos > 0u && ordered[pos - 1u] > value) {
            ordered[pos] = ordered[pos - 1u];
            --pos;
        }
        ordered[pos] = value;
        ++out_count;
        if (value < 64u) out_mask |= (1ull << value);
    }
    *mask = out_mask;
    return out_count;
}

inline int mask_is_full_(const local_context *ctx, unsigned long long mask) {
    unsigned int i = 0;
    unsigned long long full = 0u;
    if (ctx == 0) return 0;
    for (i = 0; i < ctx->device_count && i < 64u; ++i) full |= (1ull << i);
    return mask == full;
}

inline ncclResult_t init_local_nccl(local_context *ctx) {
    if (ctx == 0 || ctx->device_count == 0 || ctx->device_ids == 0) return ncclInvalidArgument;
    if (ctx->full_nccl.ready != 0u) {
        sync_local_nccl_views_(ctx);
        return ncclSuccess;
    }
    {
        const ncclResult_t result = init_local_nccl_communicator(&ctx->full_nccl, ctx->device_ids, 0, ctx->device_count);
        sync_local_nccl_views_(ctx);
        return result;
    }
}

inline ncclResult_t acquire_local_nccl_subset(local_context *ctx,
                                              const unsigned int *slots,
                                              unsigned int slot_count,
                                              nccl_communicator **out) {
    std::size_t i = 0;
    unsigned long long mask = 0u;
    unsigned int *ordered = 0;
    unsigned int ordered_count = 0u;

    if (out == 0) return ncclInvalidArgument;
    *out = 0;
    if (ctx == 0 || slots == 0 || slot_count == 0u) return ncclInvalidArgument;

    ordered = (unsigned int *) std::calloc((std::size_t) slot_count, sizeof(unsigned int));
    if (ordered == 0) return ncclSystemError;
    ordered_count = canonicalize_slots_(ctx, slots, slot_count, ordered, &mask);
    if (ordered_count == 0u) {
        std::free(ordered);
        return ncclInvalidArgument;
    }

    if (ordered_count == ctx->device_count && mask_is_full_(ctx, mask)) {
        std::free(ordered);
        {
            const ncclResult_t result = init_local_nccl(ctx);
            if (result == ncclSuccess) *out = &ctx->full_nccl;
            return result;
        }
    }

    for (i = 0; i < ctx->subset_nccl_count; ++i) {
        nccl_communicator *candidate = ctx->subset_nccl + i;
        if (candidate->ready == 0u) continue;
        if (candidate->slot_mask == mask && candidate->device_count == ordered_count) {
            std::free(ordered);
            *out = candidate;
            return ncclSuccess;
        }
    }

    if (!reserve_local_nccl_cache_(ctx, ctx->subset_nccl_count + 1u)) {
        std::free(ordered);
        return ncclSystemError;
    }

    {
        nccl_communicator *entry = ctx->subset_nccl + ctx->subset_nccl_count;
        int *device_ids = (int *) std::calloc((std::size_t) ordered_count, sizeof(int));
        ncclResult_t result = ncclSuccess;
        if (device_ids == 0) {
            std::free(ordered);
            return ncclSystemError;
        }
        for (i = 0; i < ordered_count; ++i) device_ids[i] = ctx->device_ids[ordered[i]];
        result = init_local_nccl_communicator(entry, device_ids, ordered, ordered_count);
        std::free(device_ids);
        std::free(ordered);
        if (result != ncclSuccess) return result;
        ctx->subset_nccl_count += 1u;
        *out = entry;
        return ncclSuccess;
    }
}

inline ncclResult_t communicator_allreduce(const nccl_communicator *comm,
                                           const void *const *sendbufs,
                                           void *const *recvbufs,
                                           std::size_t count,
                                           ncclDataType_t datatype,
                                           ncclRedOp_t op,
                                           const cudaStream_t *streams) {
    unsigned int rank = 0;
    if (comm == 0 || sendbufs == 0 || recvbufs == 0 || comm->ready == 0u) return ncclInvalidArgument;
    ncclGroupStart();
    for (rank = 0; rank < comm->device_count; ++rank) {
        const ncclResult_t result = ncclAllReduce(
            sendbufs[rank],
            recvbufs[rank],
            count,
            datatype,
            op,
            comm->comms[rank],
            streams != 0 ? streams[rank] : (cudaStream_t) 0);
        if (result != ncclSuccess) {
            ncclGroupEnd();
            return result;
        }
    }
    return ncclGroupEnd();
}

inline ncclResult_t communicator_reduce(const nccl_communicator *comm,
                                        const void *const *sendbufs,
                                        void *const *recvbufs,
                                        std::size_t count,
                                        ncclDataType_t datatype,
                                        ncclRedOp_t op,
                                        int root_rank,
                                        const cudaStream_t *streams) {
    unsigned int rank = 0;
    if (comm == 0 || sendbufs == 0 || recvbufs == 0 || comm->ready == 0u) return ncclInvalidArgument;
    if (root_rank < 0 || root_rank >= (int) comm->device_count) return ncclInvalidArgument;
    ncclGroupStart();
    for (rank = 0; rank < comm->device_count; ++rank) {
        const ncclResult_t result = ncclReduce(
            sendbufs[rank],
            recvbufs[rank],
            count,
            datatype,
            op,
            root_rank,
            comm->comms[rank],
            streams != 0 ? streams[rank] : (cudaStream_t) 0);
        if (result != ncclSuccess) {
            ncclGroupEnd();
            return result;
        }
    }
    return ncclGroupEnd();
}

inline ncclResult_t communicator_broadcast(const nccl_communicator *comm,
                                           const void *const *buffers,
                                           std::size_t count,
                                           ncclDataType_t datatype,
                                           int root_rank,
                                           const cudaStream_t *streams) {
    unsigned int rank = 0;
    if (comm == 0 || buffers == 0 || comm->ready == 0u) return ncclInvalidArgument;
    if (root_rank < 0 || root_rank >= (int) comm->device_count) return ncclInvalidArgument;
    ncclGroupStart();
    for (rank = 0; rank < comm->device_count; ++rank) {
        const ncclResult_t result = ncclBroadcast(
            buffers[root_rank],
            (void *) buffers[rank],
            count,
            datatype,
            root_rank,
            comm->comms[rank],
            streams != 0 ? streams[rank] : (cudaStream_t) 0);
        if (result != ncclSuccess) {
            ncclGroupEnd();
            return result;
        }
    }
    return ncclGroupEnd();
}

inline ncclResult_t communicator_allgather(const nccl_communicator *comm,
                                           const void *const *sendbufs,
                                           void *const *recvbufs,
                                           std::size_t count,
                                           ncclDataType_t datatype,
                                           const cudaStream_t *streams) {
    unsigned int rank = 0;
    if (comm == 0 || sendbufs == 0 || recvbufs == 0 || comm->ready == 0u) return ncclInvalidArgument;
    ncclGroupStart();
    for (rank = 0; rank < comm->device_count; ++rank) {
        const ncclResult_t result = ncclAllGather(
            sendbufs[rank],
            recvbufs[rank],
            count,
            datatype,
            comm->comms[rank],
            streams != 0 ? streams[rank] : (cudaStream_t) 0);
        if (result != ncclSuccess) {
            ncclGroupEnd();
            return result;
        }
    }
    return ncclGroupEnd();
}

inline ncclResult_t local_allreduce(local_context *ctx,
                                    const unsigned int *slots,
                                    unsigned int slot_count,
                                    const void *const *sendbufs,
                                    void *const *recvbufs,
                                    std::size_t count,
                                    ncclDataType_t datatype,
                                    ncclRedOp_t op) {
    unsigned int rank = 0;
    nccl_communicator *comm = 0;
    const void **ordered_send = 0;
    void **ordered_recv = 0;
    cudaStream_t *ordered_streams = 0;
    ncclResult_t result = ncclSuccess;

    result = acquire_local_nccl_subset(ctx, slots, slot_count, &comm);
    if (result != ncclSuccess) return result;
    ordered_send = (const void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_recv = (void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_streams = (cudaStream_t *) std::calloc((std::size_t) comm->device_count, sizeof(cudaStream_t));
    if (ordered_send == 0 || ordered_recv == 0 || ordered_streams == 0) {
        std::free(ordered_send);
        std::free(ordered_recv);
        std::free(ordered_streams);
        return ncclSystemError;
    }
    for (rank = 0; rank < comm->device_count; ++rank) {
        int source_index = slot_index_in_list_(slots, slot_count, comm->local_slots[rank]);
        if (source_index < 0) {
            std::free(ordered_send);
            std::free(ordered_recv);
            std::free(ordered_streams);
            return ncclInvalidArgument;
        }
        ordered_send[rank] = sendbufs[source_index];
        ordered_recv[rank] = recvbufs[source_index];
        ordered_streams[rank] = ctx->streams != 0 ? ctx->streams[comm->local_slots[rank]] : (cudaStream_t) 0;
    }
    result = communicator_allreduce(comm, ordered_send, ordered_recv, count, datatype, op, ordered_streams);
    std::free(ordered_send);
    std::free(ordered_recv);
    std::free(ordered_streams);
    return result;
}

inline ncclResult_t local_reduce(local_context *ctx,
                                 const unsigned int *slots,
                                 unsigned int slot_count,
                                 const void *const *sendbufs,
                                 void *const *recvbufs,
                                 std::size_t count,
                                 ncclDataType_t datatype,
                                 ncclRedOp_t op,
                                 unsigned int root_slot) {
    unsigned int rank = 0;
    nccl_communicator *comm = 0;
    const void **ordered_send = 0;
    void **ordered_recv = 0;
    cudaStream_t *ordered_streams = 0;
    int root_rank = -1;
    ncclResult_t result = acquire_local_nccl_subset(ctx, slots, slot_count, &comm);
    if (result != ncclSuccess) return result;
    root_rank = communicator_rank_for_slot(comm, root_slot);
    if (root_rank < 0) return ncclInvalidArgument;
    ordered_send = (const void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_recv = (void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_streams = (cudaStream_t *) std::calloc((std::size_t) comm->device_count, sizeof(cudaStream_t));
    if (ordered_send == 0 || ordered_recv == 0 || ordered_streams == 0) {
        std::free(ordered_send);
        std::free(ordered_recv);
        std::free(ordered_streams);
        return ncclSystemError;
    }
    for (rank = 0; rank < comm->device_count; ++rank) {
        int source_index = slot_index_in_list_(slots, slot_count, comm->local_slots[rank]);
        if (source_index < 0) {
            std::free(ordered_send);
            std::free(ordered_recv);
            std::free(ordered_streams);
            return ncclInvalidArgument;
        }
        ordered_send[rank] = sendbufs[source_index];
        ordered_recv[rank] = recvbufs[source_index];
        ordered_streams[rank] = ctx->streams != 0 ? ctx->streams[comm->local_slots[rank]] : (cudaStream_t) 0;
    }
    result = communicator_reduce(comm, ordered_send, ordered_recv, count, datatype, op, root_rank, ordered_streams);
    std::free(ordered_send);
    std::free(ordered_recv);
    std::free(ordered_streams);
    return result;
}

inline ncclResult_t local_broadcast(local_context *ctx,
                                    const unsigned int *slots,
                                    unsigned int slot_count,
                                    const void *const *buffers,
                                    std::size_t count,
                                    ncclDataType_t datatype,
                                    unsigned int root_slot) {
    unsigned int rank = 0;
    nccl_communicator *comm = 0;
    const void **ordered_buffers = 0;
    cudaStream_t *ordered_streams = 0;
    int root_rank = -1;
    ncclResult_t result = acquire_local_nccl_subset(ctx, slots, slot_count, &comm);
    if (result != ncclSuccess) return result;
    root_rank = communicator_rank_for_slot(comm, root_slot);
    if (root_rank < 0) return ncclInvalidArgument;
    ordered_buffers = (const void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_streams = (cudaStream_t *) std::calloc((std::size_t) comm->device_count, sizeof(cudaStream_t));
    if (ordered_buffers == 0 || ordered_streams == 0) {
        std::free(ordered_buffers);
        std::free(ordered_streams);
        return ncclSystemError;
    }
    for (rank = 0; rank < comm->device_count; ++rank) {
        int source_index = slot_index_in_list_(slots, slot_count, comm->local_slots[rank]);
        if (source_index < 0) {
            std::free(ordered_buffers);
            std::free(ordered_streams);
            return ncclInvalidArgument;
        }
        ordered_buffers[rank] = buffers[source_index];
        ordered_streams[rank] = ctx->streams != 0 ? ctx->streams[comm->local_slots[rank]] : (cudaStream_t) 0;
    }
    result = communicator_broadcast(comm, ordered_buffers, count, datatype, root_rank, ordered_streams);
    std::free(ordered_buffers);
    std::free(ordered_streams);
    return result;
}

inline ncclResult_t local_allgather(local_context *ctx,
                                    const unsigned int *slots,
                                    unsigned int slot_count,
                                    const void *const *sendbufs,
                                    void *const *recvbufs,
                                    std::size_t count,
                                    ncclDataType_t datatype) {
    unsigned int rank = 0;
    nccl_communicator *comm = 0;
    const void **ordered_send = 0;
    void **ordered_recv = 0;
    cudaStream_t *ordered_streams = 0;
    ncclResult_t result = acquire_local_nccl_subset(ctx, slots, slot_count, &comm);
    if (result != ncclSuccess) return result;
    ordered_send = (const void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_recv = (void **) std::calloc((std::size_t) comm->device_count, sizeof(void *));
    ordered_streams = (cudaStream_t *) std::calloc((std::size_t) comm->device_count, sizeof(cudaStream_t));
    if (ordered_send == 0 || ordered_recv == 0 || ordered_streams == 0) {
        std::free(ordered_send);
        std::free(ordered_recv);
        std::free(ordered_streams);
        return ncclSystemError;
    }
    for (rank = 0; rank < comm->device_count; ++rank) {
        int source_index = slot_index_in_list_(slots, slot_count, comm->local_slots[rank]);
        if (source_index < 0) {
            std::free(ordered_send);
            std::free(ordered_recv);
            std::free(ordered_streams);
            return ncclInvalidArgument;
        }
        ordered_send[rank] = sendbufs[source_index];
        ordered_recv[rank] = recvbufs[source_index];
        ordered_streams[rank] = ctx->streams != 0 ? ctx->streams[comm->local_slots[rank]] : (cudaStream_t) 0;
    }
    result = communicator_allgather(comm, ordered_send, ordered_recv, count, datatype, ordered_streams);
    std::free(ordered_send);
    std::free(ordered_recv);
    std::free(ordered_streams);
    return result;
}
#endif

inline void init(local_context *ctx) {
    ctx->device_count = 0;
    ctx->device_ids = 0;
    ctx->streams = 0;
    ctx->peer_access = 0;
#if CELLERATOR_DIST_HAS_NCCL
    init(&ctx->full_nccl);
    ctx->subset_nccl = 0;
    ctx->subset_nccl_count = 0u;
    ctx->subset_nccl_capacity = 0u;
    ctx->comms = 0;
    ctx->nccl_ready = 0;
#endif
}

inline void clear(local_context *ctx) {
    unsigned int i = 0;

    if (ctx->streams != 0) {
        for (i = 0; i < ctx->device_count; ++i) {
            if (ctx->streams[i] != 0) {
                cudaSetDevice(ctx->device_ids != 0 ? ctx->device_ids[i] : (int) i);
                cudaStreamDestroy(ctx->streams[i]);
            }
        }
    }
#if CELLERATOR_DIST_HAS_NCCL
    if (ctx->subset_nccl != 0) {
        for (i = 0; i < ctx->subset_nccl_count; ++i) clear(ctx->subset_nccl + i);
    }
    std::free(ctx->subset_nccl);
    clear(&ctx->full_nccl);
#endif
    std::free(ctx->peer_access);
    std::free(ctx->streams);
    std::free(ctx->device_ids);
    init(ctx);
}

inline cudaError_t discover_local(local_context *ctx, int create_streams, unsigned int stream_flags) {
    int count = 0;
    unsigned int i = 0;
    cudaError_t err = cudaSuccess;

    clear(ctx);
    // Enumerate only CUDA-visible devices. The caller controls visibility with
    // CUDA_VISIBLE_DEVICES or equivalent before process launch.
    err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return err;
    if (count <= 0) return cudaSuccess;

    ctx->device_ids = (int *) std::calloc((std::size_t) count, sizeof(int));
    ctx->peer_access = (unsigned char *) std::calloc((std::size_t) count * (std::size_t) count, sizeof(unsigned char));
    if (ctx->device_ids == 0 || ctx->peer_access == 0) {
        clear(ctx);
        return cudaErrorMemoryAllocation;
    }
    if (create_streams) {
        ctx->streams = (cudaStream_t *) std::calloc((std::size_t) count, sizeof(cudaStream_t));
        if (ctx->streams == 0) {
            clear(ctx);
            return cudaErrorMemoryAllocation;
        }
    }
    ctx->device_count = (unsigned int) count;
    for (i = 0; i < ctx->device_count; ++i) {
        ctx->device_ids[i] = (int) i;
        if (ctx->streams != 0) {
            // One stream per device keeps staging explicit and graph-friendly.
            err = cudaSetDevice(ctx->device_ids[i]);
            if (err != cudaSuccess) {
                clear(ctx);
                return err;
            }
            err = cudaStreamCreateWithFlags(ctx->streams + i, stream_flags);
            if (err != cudaSuccess) {
                clear(ctx);
                return err;
            }
        }
    }

    for (i = 0; i < ctx->device_count; ++i) {
        unsigned int j = 0;
        err = cudaSetDevice(ctx->device_ids[i]);
        if (err != cudaSuccess) {
            clear(ctx);
            return err;
        }
        for (j = 0; j < ctx->device_count; ++j) {
            int can_access = 0;
            err = cudaDeviceCanAccessPeer(&can_access, ctx->device_ids[i], ctx->device_ids[j]);
            if (err != cudaSuccess) {
                clear(ctx);
                return err;
            }
            ctx->peer_access[(std::size_t) i * ctx->device_count + j] = (unsigned char) (can_access != 0);
        }
    }
    return cudaSuccess;
}

inline int peer_access_supported(const local_context *ctx, unsigned int src_slot, unsigned int dst_slot) {
    if (ctx == 0 || ctx->peer_access == 0) return 0;
    if (src_slot >= ctx->device_count || dst_slot >= ctx->device_count) return 0;
    return ctx->peer_access[(std::size_t) src_slot * ctx->device_count + dst_slot] != 0;
}

inline cudaError_t enable_peer_access(local_context *ctx) {
    unsigned int i = 0;
    cudaError_t err = cudaSuccess;

    if (ctx == 0) return cudaErrorInvalidValue;
    for (i = 0; i < ctx->device_count; ++i) {
        unsigned int j = 0;
        err = cudaSetDevice(ctx->device_ids[i]);
        if (err != cudaSuccess) return err;
        for (j = 0; j < ctx->device_count; ++j) {
            if (i == j) continue;
            if (!peer_access_supported(ctx, i, j)) continue;
            // Peer access only enables direct addressability. It does not move
            // any bytes by itself.
            err = cudaDeviceEnablePeerAccess(ctx->device_ids[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
                err = cudaSuccess;
            }
            if (err != cudaSuccess) return err;
        }
    }
    return cudaSuccess;
}

#if CELLERATOR_DIST_HAS_NCCL
inline int local_nccl_ready(const local_context *ctx) {
    return ctx != 0 && ctx->full_nccl.ready != 0u;
}
#endif

inline cudaError_t synchronize(const local_context *ctx) {
    unsigned int i = 0;
    cudaError_t err = cudaSuccess;

    if (ctx == 0) return cudaErrorInvalidValue;
    for (i = 0; i < ctx->device_count; ++i) {
        err = cudaSetDevice(ctx->device_ids[i]);
        if (err != cudaSuccess) return err;
        if (ctx->streams != 0 && ctx->streams[i] != 0) err = cudaStreamSynchronize(ctx->streams[i]);
        else err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

} // namespace dist
} // namespace cellerator
