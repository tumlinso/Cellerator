#include <Cellerator/compute/math/runtime.hh>

#include <stdexcept>

namespace cellerator::compute::math {

WorkspacePool::~WorkspacePool() {
    clear(this);
}

void init(WorkspacePool *pool, int device_ordinal) {
    if (pool == nullptr) {
        throw std::invalid_argument("init(WorkspacePool) requires a pool");
    }
    clear(pool);
    if (device_ordinal < 0) {
        runtime::cuda_require(
            cudaGetDevice(&device_ordinal),
            "cudaGetDevice(WorkspacePool)");
    }
    runtime::cuda_require(
        cudaSetDevice(device_ordinal),
        "cudaSetDevice(WorkspacePool)");
    runtime::init(&pool->storage);
    pool->device_ordinal = device_ordinal;
}

void clear(WorkspacePool *pool) noexcept {
    if (pool == nullptr) return;
    if (pool->device_ordinal >= 0) {
        (void) cudaSetDevice(pool->device_ordinal);
    }
    runtime::clear(&pool->storage);
    pool->device_ordinal = -1;
    pool->allocation_count = 0u;
    pool->high_watermark_bytes = 0u;
}

void *request_workspace(WorkspacePool *pool, std::size_t bytes) {
    if (pool == nullptr) {
        throw std::invalid_argument("request_workspace requires a pool");
    }
    if (pool->device_ordinal < 0) {
        throw std::logic_error("WorkspacePool is not initialized");
    }
    runtime::cuda_require(
        cudaSetDevice(pool->device_ordinal),
        "cudaSetDevice(request_workspace)");
    const std::size_t previous_capacity = pool->storage.bytes;
    void *const out = runtime::request_scratch(&pool->storage, bytes);
    if (pool->storage.bytes > previous_capacity) {
        ++pool->allocation_count;
        pool->high_watermark_bytes = pool->storage.bytes;
    }
    return out;
}

} // namespace cellerator::compute::math
