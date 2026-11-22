#include "sequence_fixture.hh"
#include <Cellerator/layout/sequence/dna2_cuda.hh>
#include <Cellerator/ops/sequence/dna2_cuda.hh>
#include <cuda_runtime.h>

namespace {

using namespace sequence_test;

void cuda_check(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
    exit(1);
}

struct DeviceFixture {
    uint64_t words[word_count], validity[2];
    uint32_t low[word_count], high[word_count], valid[word_count];
    uint32_t cytosine[word_count], gc[word_count], cpg[word_count];
    unsigned long long motif_count;
    seq::sequence_emit_counts counts;
    seq::sequence_event events[base_count + 1u];
};

void check_hits(const Fixture &f, const DeviceFixture &result,
                const uint32_t *positions, uint32_t expected, uint32_t capacity) {
    require(result.motif_count == expected, "CUDA count differs from text oracle");
    check_counts(result.counts, expected, capacity);
    bool found[base_count]{};
    for (uint32_t i = 0; i < result.counts.stored_records; ++i) {
        const auto hit = result.events[i];
        const uint32_t p = hit.local_position;
        require(p + uint64_t(3u) <= base_count && !found[p] && hit.predicate_id == 17u,
                "CUDA emit has an invalid or duplicate anchor");
        const int distance = mismatches(f, p);
        require(distance >= 0 && distance <= 1 && hit.payload == distance, "CUDA emit differs from text oracle");
        found[p] = true;
    }
    require(result.events[capacity].local_position == UINT32_MAX, "CUDA emit wrote beyond capacity");
    if (capacity >= expected) {
        for (uint32_t i = 0; i < expected; ++i) require(found[positions[i]], "CUDA emit lost an expected anchor");
    }
}

} // namespace

int main() {
    Fixture f{};
    make_fixture(&f);
    uint32_t positions[base_count], cytosine[word_count], gc[word_count], cpg[word_count];
    const uint32_t expected = oracle_hits(f, positions);
    oracle_masks(f, cytosine, gc, cpg);

    DeviceFixture *device = nullptr;
    cudaStream_t stream;
    cuda_check(cudaMalloc(&device, sizeof(*device)), "allocate device fixture");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create sequence stream");
    cuda_check(cudaMemsetAsync(device, 0, sizeof(*device), stream), "initialize device fixture");
    cuda_check(cudaMemcpyAsync(device->words, f.words, sizeof(f.words), cudaMemcpyHostToDevice, stream),
               "upload packed words");
    cuda_check(cudaMemcpyAsync(device->validity, f.validity, sizeof(f.validity), cudaMemcpyHostToDevice, stream),
               "upload packed validity");

    const tensor::PackedView packed{device->words, device->validity, word_count, base_count, 2u};
    const tensor::BitPlanesView bits{device->low, device->high, device->valid, word_count, base_count};
    const tensor::MutableBitPlanesView output{device->low, device->high, device->valid, word_count};
    cuda_check(layout::pack_planes_cuda(packed, output, stream), "CUDA word-to-plane conversion");
    cuda_check(seq::base_mask_cuda(bits, 1u, device->cytosine, word_count, stream), "CUDA cytosine mask");
    cuda_check(seq::gc_mask_cuda(bits, device->gc, word_count, stream), "CUDA GC mask");
    cuda_check(seq::cpg_start_mask_cuda(bits, device->cpg, word_count, stream), "CUDA CpG mask");

    const uint32_t capacities[] = {0u, 2u, expected, base_count};
    for (uint32_t capacity : capacities) {
        cuda_check(cudaMemsetAsync(device->events + capacity, 0xff, sizeof(seq::sequence_event), stream),
                   "set event sentinel");
        cuda_check(seq::count_motif_cuda(packed, f.motif, &device->motif_count, stream), "CUDA motif count");
        cuda_check(seq::emit_motif_cuda(packed, f.motif, capacity ? device->events : nullptr, capacity,
                                       &device->counts, stream), "CUDA motif emit");
        DeviceFixture result{};
        cuda_check(cudaMemcpyAsync(&result, device, sizeof(result), cudaMemcpyDeviceToHost, stream), "download oracle result");
        cuda_check(cudaStreamSynchronize(stream), "complete sequence CUDA test cases");
        check_hits(f, result, positions, expected, capacity);
        for (uint32_t i = 0; i < word_count; ++i) {
            require(result.low[i] == f.low[i] && result.high[i] == f.high[i] && result.valid[i] == f.valid[i],
                    "CUDA planes differ from checked host planes");
            require(result.cytosine[i] == cytosine[i] && result.gc[i] == gc[i] && result.cpg[i] == cpg[i],
                    "CUDA masks differ from text oracle at seam or tail");
        }
    }

    auto malformed = packed; --malformed.word_count;
    require(seq::count_motif_cuda(malformed, f.motif, &device->motif_count, stream) == cudaErrorInvalidValue,
            "CUDA scanner accepted short input");
    cuda_check(seq::count_motif_cuda({}, f.motif, &device->motif_count, stream), "CUDA empty motif count");
    unsigned long long zero;
    cuda_check(cudaMemcpyAsync(&zero, &device->motif_count, sizeof(zero), cudaMemcpyDeviceToHost, stream),
               "download empty sequence count");
    cuda_check(cudaStreamSynchronize(stream), "complete empty sequence count");
    require(zero == 0u, "CUDA empty input did not clear a reused counter");
    cuda_check(cudaStreamDestroy(stream), "destroy sequence stream");
    cuda_check(cudaFree(device), "release device fixture");
    puts("sequence CUDA contracts passed");
    return 0;
}
