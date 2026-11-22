#include "workload.hh"
#include <Cellerator/ops/sequence/dna2_cuda.hh>
#include <cuda_runtime.h>

namespace {
void cuda_check(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
    exit(1);
}
} // namespace

int main(int argc, char **argv) {
    using namespace sequence_bench;
    const Settings s = settings(argc, argv);
    const Input input = make_input(s.bases);
    const double pack_start = now_ms();
    const auto view = pack(input, s.bases);
    const double pack_ms = now_ms() - pack_start;
    const double prepare_start = now_ms();
    const auto motif = prepare_predicate();
    const double prepare_ms = now_ms() - prepare_start;

    const size_t bytes = (size_t(input.word_count) + input.validity_count)*sizeof(uint64_t);
    uint64_t *words = nullptr;
    unsigned long long *device_count = nullptr;
    cudaStream_t stream;
    cudaEvent_t begin, end;
    cuda_check(cudaMalloc(&words, bytes), "allocate packed words and validity");
    cuda_check(cudaMalloc(&device_count, sizeof(*device_count)), "allocate count");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    cuda_check(cudaEventCreate(&begin), "create begin event");
    cuda_check(cudaEventCreate(&end), "create end event");

    const double upload_start = now_ms();
    cuda_check(cudaMemcpyAsync(words, view.words, bytes, cudaMemcpyHostToDevice, stream),
               "upload packed words and validity");
    cuda_check(cudaStreamSynchronize(stream), "complete upload");
    const double upload_ms = now_ms() - upload_start;
    const tensor::PackedView device_view{words, words + input.word_count, input.word_count,
                                         s.bases, input.validity_count};

    for (uint32_t i = 0; i < s.warmup; ++i)
        cuda_check(seq::count_motif_cuda(device_view, motif, device_count, stream), "warmup motif count");
    cuda_check(cudaEventRecord(begin, stream), "record begin");
    for (uint32_t i = 0; i < s.repeats; ++i)
        cuda_check(seq::count_motif_cuda(device_view, motif, device_count, stream), "repeated motif count");
    cuda_check(cudaEventRecord(end, stream), "record end");
    cuda_check(cudaEventSynchronize(end), "complete resident batch");
    float repeat_ms;
    cuda_check(cudaEventElapsedTime(&repeat_ms, begin, end), "time resident batch");
    unsigned long long count;
    cuda_check(cudaMemcpy(&count, device_count, sizeof(count), cudaMemcpyDeviceToHost), "download count");
    require(count == input.expected, "CUDA count differs from text oracle");
    print_result(s, "cuda", count, pack_ms, prepare_ms, upload_ms, repeat_ms);

    cuda_check(cudaEventDestroy(end), "release end event");
    cuda_check(cudaEventDestroy(begin), "release begin event");
    cuda_check(cudaStreamDestroy(stream), "release stream");
    cuda_check(cudaFree(device_count), "release counter");
    cuda_check(cudaFree(words), "release packed words");
    free(input.words); free(input.bases);
    return 0;
}
