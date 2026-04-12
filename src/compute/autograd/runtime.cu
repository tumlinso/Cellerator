#include "autograd.hh"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace cellerator::compute::autograd {

void init(execution_context *ctx, int device, cudaStream_t stream) {
    if (ctx == nullptr) throw std::invalid_argument("init(execution_context) requires a context");
    ctx->device = -1;
    ctx->stream = nullptr;
    ctx->owns_stream = false;

    if (device >= 0) {
        ctx->device = device;
    } else {
        cuda_require(cudaGetDevice(&ctx->device), "cudaGetDevice(autograd)");
    }
    cuda_require(cudaSetDevice(ctx->device), "cudaSetDevice(autograd)");
    if (stream != nullptr) {
        ctx->stream = stream;
        ctx->owns_stream = false;
        return;
    }
    cuda_require(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(autograd)");
    ctx->owns_stream = true;
}

void clear(execution_context *ctx) {
    if (ctx == nullptr) return;
    if (ctx->stream != nullptr && ctx->owns_stream) {
        cudaSetDevice(ctx->device);
        cudaStreamDestroy(ctx->stream);
    }
    ctx->device = -1;
    ctx->stream = nullptr;
    ctx->owns_stream = false;
}

void init(scratch_arena *arena) {
    if (arena == nullptr) throw std::invalid_argument("init(scratch_arena) requires an arena");
    arena->data = nullptr;
    arena->bytes = 0;
}

void clear(scratch_arena *arena) {
    if (arena == nullptr) return;
    if (arena->data != nullptr) cudaFree(arena->data);
    arena->data = nullptr;
    arena->bytes = 0;
}

void *request_scratch(scratch_arena *arena, std::size_t bytes) {
    if (arena == nullptr) throw std::invalid_argument("request_scratch requires an arena");
    if (bytes <= arena->bytes) return arena->data;
    if (arena->data != nullptr) {
        cudaFree(arena->data);
        arena->data = nullptr;
        arena->bytes = 0;
    }
    if (bytes == 0) return nullptr;
    cuda_require(cudaMalloc(&arena->data, bytes), "cudaMalloc(autograd scratch)");
    arena->bytes = bytes;
    return arena->data;
}

void init(cusparse_cache *cache) {
    if (cache == nullptr) throw std::invalid_argument("init(cusparse_cache) requires a cache");
    std::memset(cache, 0, sizeof(*cache));
    cache->device = -1;
}

void clear(cusparse_cache *cache) {
    if (cache == nullptr) return;
    if (cache->csr_f32 != nullptr) {
        cusparseDestroySpMat(cache->csr_f32);
        cache->csr_f32 = nullptr;
    }
    if (cache->handle != nullptr && cache->owns_handle) {
        cusparseDestroy(cache->handle);
        cache->handle = nullptr;
    }
    cache->device = -1;
    cache->owns_handle = false;
    cache->matrix_token = nullptr;
    cache->spmv_bytes_non_transpose = 0;
    cache->spmv_bytes_transpose = 0;
    cache->spmm_bytes_non_transpose = 0;
    cache->spmm_bytes_transpose = 0;
}

cusparseHandle_t acquire_cusparse(cusparse_cache *cache, const execution_context &ctx) {
    if (cache == nullptr) throw std::invalid_argument("acquire_cusparse requires a cache");
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(acquire_cusparse)");
    if (cache->handle == nullptr) {
        if (cusparseCreate(&cache->handle) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("cusparseCreate(autograd) failed");
        }
        cache->owns_handle = true;
        cache->device = ctx.device;
    }
    if (cache->device != ctx.device) {
        clear(cache);
        if (cusparseCreate(&cache->handle) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("cusparseCreate(autograd device switch) failed");
        }
        cache->owns_handle = true;
        cache->device = ctx.device;
    }
    if (cusparseSetStream(cache->handle, ctx.stream) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseSetStream(autograd) failed");
    }
    return cache->handle;
}

cusparseSpMatDescr_t acquire_csr_f32_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values) {
    if (cache == nullptr) throw std::invalid_argument("acquire_csr_f32_descriptor requires a cache");
    acquire_cusparse(cache, ctx);
    if (cache->matrix_token == matrix_token && cache->csr_f32 != nullptr) return cache->csr_f32;
    if (cache->csr_f32 != nullptr) {
        cusparseDestroySpMat(cache->csr_f32);
        cache->csr_f32 = nullptr;
    }
    if (cusparseCreateCsr(
            &cache->csr_f32,
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
            static_cast<std::int64_t>(nnz),
            const_cast<std::uint32_t *>(major_ptr),
            const_cast<std::uint32_t *>(minor_idx),
            const_cast<float *>(values),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseCreateCsr(autograd f32 cache) failed");
    }
    cache->matrix_token = matrix_token;
    cache->spmv_bytes_non_transpose = 0;
    cache->spmv_bytes_transpose = 0;
    cache->spmm_bytes_non_transpose = 0;
    cache->spmm_bytes_transpose = 0;
    return cache->csr_f32;
}

std::size_t &cached_spmv_bytes(cusparse_cache *cache, cusparseOperation_t op) {
    if (cache == nullptr) throw std::invalid_argument("cached_spmv_bytes requires a cache");
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE
        ? cache->spmv_bytes_non_transpose
        : cache->spmv_bytes_transpose;
}

std::size_t &cached_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op) {
    if (cache == nullptr) throw std::invalid_argument("cached_spmm_bytes requires a cache");
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE
        ? cache->spmm_bytes_non_transpose
        : cache->spmm_bytes_transpose;
}

void init(fleet_context *fleet) {
    if (fleet == nullptr) throw std::invalid_argument("init(fleet_context) requires a fleet");
    csdist::init(&fleet->local);
    fleet->reduce_scratch = nullptr;
    fleet->reduce_scratch_bytes = nullptr;
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
    fleet->reduce_scratch = nullptr;
    fleet->reduce_scratch_bytes = nullptr;
    csdist::clear(&fleet->local);
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
        csdist::discover_local(&fleet->local, create_streams ? 1 : 0, stream_flags),
        "discover_local(autograd fleet)");
    if (enable_peer_access && fleet->local.device_count != 0) {
        cuda_require(csdist::enable_peer_access(&fleet->local), "enable_peer_access(autograd fleet)");
    }
    if (fleet->local.device_count != 0) {
        fleet->reduce_scratch = static_cast<void **>(std::calloc(fleet->local.device_count, sizeof(void *)));
        fleet->reduce_scratch_bytes = static_cast<std::size_t *>(std::calloc(fleet->local.device_count, sizeof(std::size_t)));
        if (fleet->reduce_scratch == nullptr || fleet->reduce_scratch_bytes == nullptr) {
            clear(fleet);
            throw std::bad_alloc();
        }
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
    cuda_require(cudaMalloc(fleet->reduce_scratch + slot, bytes), "cudaMalloc(autograd fleet scratch)");
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

} // namespace cellerator::compute::autograd
