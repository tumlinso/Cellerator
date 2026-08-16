#pragma once

#include <cstddef>
#include <cstdlib>

#include <cuda_runtime.h>

#include "Cellerator/dist/nccl_select.cuh"
#include "Cellerator/dist/nccl_communicator.cuh"

namespace cellerator {
namespace dist {

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
