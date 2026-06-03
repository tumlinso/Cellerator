#include "benchmark_mutex.hh"

#include <Cellerator/dist/distributed.cuh>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace cdist = ::cellerator::dist;

struct config {
    unsigned int cols = 32768u;
    unsigned int devices = 4u;
    unsigned int warmup = 10u;
    unsigned int repeats = 100u;
};

struct buffers {
    int device = -1;
    cudaStream_t stream = nullptr;
    float *sum = nullptr;
    float *sq = nullptr;
    float *det = nullptr;
    float *active = nullptr;
    float *packed = nullptr;
    float *out_sum = nullptr;
    float *out_sq = nullptr;
    float *out_det = nullptr;
    float *out_active = nullptr;
    float *out_packed = nullptr;
};

int check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

int check(ncclResult_t err, const char *label) {
    if (err == ncclSuccess) return 1;
    std::fprintf(stderr, "%s: %s\n", label, ncclGetErrorString(err));
    return 0;
}

int parse_u32(const char *text, unsigned int *out) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != '\0' || parsed > 0xfffffffful) return 0;
    *out = (unsigned int) parsed;
    return 1;
}

int parse_args(int argc, char **argv, config *cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->cols)) return 0;
        } else if (std::strcmp(argv[i], "--devices") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->devices)) return 0;
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->warmup)) return 0;
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->repeats)) return 0;
        } else {
            return 0;
        }
    }
    return cfg->cols != 0u && cfg->devices != 0u && cfg->repeats != 0u;
}

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--cols N] [--devices N] [--warmup N] [--repeats N]\n",
                 argv0);
}

__global__ void fill_kernel(float *sum,
                            float *sq,
                            float *det,
                            float *active,
                            float *packed,
                            unsigned int cols,
                            unsigned int rank) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    const float base = (float) (rank + 1u);
    for (unsigned int i = tid; i < cols; i += stride) {
        sum[i] = base;
        sq[i] = 2.0f * base;
        det[i] = 3.0f * base;
        packed[i] = base;
        packed[cols + i] = 2.0f * base;
        packed[2u * cols + i] = 3.0f * base;
    }
    if (tid == 0u) {
        active[0] = base;
        packed[3u * cols] = base;
    }
}

int alloc_one(buffers *b, unsigned int device, unsigned int cols) {
    const std::size_t metric_bytes = (std::size_t) cols * sizeof(float);
    const std::size_t packed_bytes = (std::size_t) (3u * cols + 1u) * sizeof(float);
    b->device = (int) device;
    if (!check(cudaSetDevice(b->device), "cudaSetDevice alloc")) return 0;
    if (!check(cudaStreamCreateWithFlags(&b->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) return 0;
    if (!check(cudaMalloc((void **) &b->sum, metric_bytes), "cudaMalloc sum")) return 0;
    if (!check(cudaMalloc((void **) &b->sq, metric_bytes), "cudaMalloc sq")) return 0;
    if (!check(cudaMalloc((void **) &b->det, metric_bytes), "cudaMalloc det")) return 0;
    if (!check(cudaMalloc((void **) &b->active, sizeof(float)), "cudaMalloc active")) return 0;
    if (!check(cudaMalloc((void **) &b->packed, packed_bytes), "cudaMalloc packed")) return 0;
    if (!check(cudaMalloc((void **) &b->out_sum, metric_bytes), "cudaMalloc out_sum")) return 0;
    if (!check(cudaMalloc((void **) &b->out_sq, metric_bytes), "cudaMalloc out_sq")) return 0;
    if (!check(cudaMalloc((void **) &b->out_det, metric_bytes), "cudaMalloc out_det")) return 0;
    if (!check(cudaMalloc((void **) &b->out_active, sizeof(float)), "cudaMalloc out_active")) return 0;
    if (!check(cudaMalloc((void **) &b->out_packed, packed_bytes), "cudaMalloc out_packed")) return 0;
    fill_kernel<<<128, 256, 0, b->stream>>>(b->sum, b->sq, b->det, b->active, b->packed, cols, device);
    return check(cudaGetLastError(), "fill_kernel");
}

void free_one(buffers *b) {
    if (b->device >= 0) (void) cudaSetDevice(b->device);
    if (b->sum) (void) cudaFree(b->sum);
    if (b->sq) (void) cudaFree(b->sq);
    if (b->det) (void) cudaFree(b->det);
    if (b->active) (void) cudaFree(b->active);
    if (b->packed) (void) cudaFree(b->packed);
    if (b->out_sum) (void) cudaFree(b->out_sum);
    if (b->out_sq) (void) cudaFree(b->out_sq);
    if (b->out_det) (void) cudaFree(b->out_det);
    if (b->out_active) (void) cudaFree(b->out_active);
    if (b->out_packed) (void) cudaFree(b->out_packed);
    if (b->stream) (void) cudaStreamDestroy(b->stream);
}

int sync_all(std::vector<buffers> &bufs) {
    for (buffers &b : bufs) {
        if (!check(cudaSetDevice(b.device), "cudaSetDevice sync")) return 0;
        if (!check(cudaStreamSynchronize(b.stream), "cudaStreamSynchronize")) return 0;
    }
    return 1;
}

int separate(cdist::nccl_communicator *comm, std::vector<buffers> &bufs, unsigned int cols) {
    std::vector<const void *> send(comm->device_count);
    std::vector<void *> recv(comm->device_count);
    std::vector<cudaStream_t> streams(comm->device_count);
    for (unsigned int i = 0u; i < comm->device_count; ++i) streams[i] = bufs[i].stream;
    for (unsigned int metric = 0u; metric < 4u; ++metric) {
        for (unsigned int i = 0u; i < comm->device_count; ++i) {
            if (metric == 0u) {
                send[i] = bufs[i].sum; recv[i] = bufs[i].out_sum;
            } else if (metric == 1u) {
                send[i] = bufs[i].sq; recv[i] = bufs[i].out_sq;
            } else if (metric == 2u) {
                send[i] = bufs[i].det; recv[i] = bufs[i].out_det;
            } else {
                send[i] = bufs[i].active; recv[i] = bufs[i].out_active;
            }
        }
        if (!check(cdist::communicator_allreduce(comm,
                                                 send.data(),
                                                 recv.data(),
                                                 metric == 3u ? 1u : (std::size_t) cols,
                                                 ncclFloat32,
                                                 ncclSum,
                                                 streams.data()),
                   "separate allreduce")) return 0;
    }
    return 1;
}

int grouped(cdist::nccl_communicator *comm, std::vector<buffers> &bufs, unsigned int cols) {
    ncclGroupStart();
    for (unsigned int metric = 0u; metric < 4u; ++metric) {
        const std::size_t count = metric == 3u ? 1u : (std::size_t) cols;
        for (unsigned int i = 0u; i < comm->device_count; ++i) {
            const void *send = metric == 0u ? (const void *) bufs[i].sum
                : metric == 1u ? (const void *) bufs[i].sq
                : metric == 2u ? (const void *) bufs[i].det
                               : (const void *) bufs[i].active;
            void *recv = metric == 0u ? (void *) bufs[i].out_sum
                : metric == 1u ? (void *) bufs[i].out_sq
                : metric == 2u ? (void *) bufs[i].out_det
                               : (void *) bufs[i].out_active;
            const ncclResult_t result = ncclAllReduce(send, recv, count, ncclFloat32, ncclSum, comm->comms[i], bufs[i].stream);
            if (result != ncclSuccess) {
                ncclGroupEnd();
                return check(result, "grouped ncclAllReduce");
            }
        }
    }
    return check(ncclGroupEnd(), "grouped ncclGroupEnd");
}

int contiguous(cdist::nccl_communicator *comm, std::vector<buffers> &bufs, unsigned int cols) {
    std::vector<const void *> send(comm->device_count);
    std::vector<void *> recv(comm->device_count);
    std::vector<cudaStream_t> streams(comm->device_count);
    for (unsigned int i = 0u; i < comm->device_count; ++i) {
        send[i] = bufs[i].packed;
        recv[i] = bufs[i].out_packed;
        streams[i] = bufs[i].stream;
    }
    return check(cdist::communicator_allreduce(comm,
                                               send.data(),
                                               recv.data(),
                                               (std::size_t) 3u * cols + 1u,
                                               ncclFloat32,
                                               ncclSum,
                                               streams.data()),
                 "contiguous allreduce");
}

double time_mode(const char *label,
                 int (*fn)(cdist::nccl_communicator *, std::vector<buffers> &, unsigned int),
                 cdist::nccl_communicator *comm,
                 std::vector<buffers> &bufs,
                 const config &cfg) {
    for (unsigned int i = 0u; i < cfg.warmup; ++i) {
        if (!fn(comm, bufs, cfg.cols)) return -1.0;
    }
    if (!sync_all(bufs)) return -1.0;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned int i = 0u; i < cfg.repeats; ++i) {
        if (!fn(comm, bufs, cfg.cols)) return -1.0;
    }
    if (!sync_all(bufs)) return -1.0;
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count() / (double) cfg.repeats;
    std::printf("preprocess_nccl_reduce: mode=%s devices=%u cols=%u floats=%u avg_ms=%.6f\n",
                label,
                comm->device_count,
                cfg.cols,
                3u * cfg.cols + 1u,
                ms);
    return ms;
}

int check_outputs(buffers &b, unsigned int devices, unsigned int cols) {
    const float expected = 0.5f * (float) (devices * (devices + 1u));
    float got[4] = {};
    if (!check(cudaSetDevice(b.device), "cudaSetDevice check")) return 0;
    if (!check(cudaMemcpy(got + 0, b.out_sum, sizeof(float), cudaMemcpyDeviceToHost), "copy sum")) return 0;
    if (!check(cudaMemcpy(got + 1, b.out_sq, sizeof(float), cudaMemcpyDeviceToHost), "copy sq")) return 0;
    if (!check(cudaMemcpy(got + 2, b.out_det, sizeof(float), cudaMemcpyDeviceToHost), "copy det")) return 0;
    if (!check(cudaMemcpy(got + 3, b.out_active, sizeof(float), cudaMemcpyDeviceToHost), "copy active")) return 0;
    if (got[0] != expected || got[1] != 2.0f * expected || got[2] != 3.0f * expected || got[3] != expected) return 0;
    if (!check(cudaMemcpy(got + 0, b.out_packed, sizeof(float), cudaMemcpyDeviceToHost), "copy packed sum")) return 0;
    if (!check(cudaMemcpy(got + 1, b.out_packed + cols, sizeof(float), cudaMemcpyDeviceToHost), "copy packed sq")) return 0;
    if (!check(cudaMemcpy(got + 2, b.out_packed + 2u * cols, sizeof(float), cudaMemcpyDeviceToHost), "copy packed det")) return 0;
    if (!check(cudaMemcpy(got + 3, b.out_packed + 3u * cols, sizeof(float), cudaMemcpyDeviceToHost), "copy packed active")) return 0;
    return got[0] == expected && got[1] == 2.0f * expected && got[2] == 3.0f * expected && got[3] == expected;
}

} // namespace

int main(int argc, char **argv) {
#if CELLERATOR_DIST_HAS_NCCL
    config cfg;
    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 1;
    }
    int visible = 0;
    if (!check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount")) return 1;
    if (visible == 0) return 0;
    if (cfg.devices > (unsigned int) visible) cfg.devices = (unsigned int) visible;

    std::vector<int> device_ids(cfg.devices);
    std::vector<unsigned int> slots(cfg.devices);
    std::vector<int> ranks(cfg.devices);
    for (unsigned int i = 0u; i < cfg.devices; ++i) {
        device_ids[i] = (int) i;
        slots[i] = i;
        ranks[i] = (int) i;
    }

    cdist::nccl_communicator comm;
    cdist::init(&comm);
    std::vector<buffers> bufs(cfg.devices);
    int ok = 0;
    {
        const cellerator::preprocess::bench::benchmark_mutex_guard guard("CelleratorPreprocessNcclReduceBench",
                                                                         device_ids.data(),
                                                                         device_ids.size());
        ncclUniqueId id;
        if (!check(ncclGetUniqueId(&id), "ncclGetUniqueId")) goto done;
        if (!check(cdist::init_ranked_nccl_communicator(&comm,
                                                        device_ids.data(),
                                                        slots.data(),
                                                        cfg.devices,
                                                        ranks.data(),
                                                        (int) cfg.devices,
                                                        &id),
                   "init_ranked_nccl_communicator")) goto done;
        for (unsigned int i = 0u; i < cfg.devices; ++i) {
            if (!alloc_one(bufs.data() + i, i, cfg.cols)) goto done;
        }
        if (!sync_all(bufs)) goto done;
        if (time_mode("separate", separate, &comm, bufs, cfg) < 0.0) goto done;
        if (time_mode("grouped", grouped, &comm, bufs, cfg) < 0.0) goto done;
        if (time_mode("contiguous", contiguous, &comm, bufs, cfg) < 0.0) goto done;
        ok = check_outputs(bufs[0], cfg.devices, cfg.cols);
    }

done:
    for (buffers &b : bufs) free_one(&b);
    cdist::clear(&comm);
    return ok ? 0 : 1;
#else
    (void) argc;
    (void) argv;
    std::fprintf(stderr, "CELLERATOR_DIST_HAS_NCCL=0; skipping NCCL reduction benchmark\n");
    return 0;
#endif
}
