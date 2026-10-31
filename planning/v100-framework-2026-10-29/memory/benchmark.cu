#include <Cellerator/modeling/cuda/memory.hh>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace cellerator::modeling::cuda;

template<class T> T* device_copy(const std::vector<T>& host) {
    T* result = nullptr;
    if (!host.empty()) {
        assert(cudaMalloc(&result, host.size() * sizeof(T)) == cudaSuccess);
        assert(cudaMemcpy(result, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    return result;
}

int main() {
    assert(cudaSetDevice(0) == cudaSuccess);
    constexpr int warmups=100, repetitions=1000, samples=7;
    std::cout << "reads,channels,fp32_us_per_call\n";
    for (int reads : {1, 32, 512, 4096}) for (int channels : {1, 33, 257, 1024}) {
        std::vector<float> values(static_cast<std::size_t>(reads * channels), 1.f);
        std::vector<float> anchors(static_cast<std::size_t>(reads), 0.f);
        std::vector<float> tau(static_cast<std::size_t>(channels), 1.f);
        std::vector<std::int64_t> entities(static_cast<std::size_t>(reads));
        std::iota(entities.begin(), entities.end(), std::int64_t{0});
        auto* d_values=device_copy(values); auto* d_anchors=device_copy(anchors);
        auto* d_tau=device_copy(tau); auto* d_entities=device_copy(entities);
        float* d_output=nullptr;
        assert(cudaMalloc(&d_output, values.size()*sizeof(float))==cudaSuccess);
        EventMemoryView<float> view{d_values,values.size(),d_anchors,anchors.size(),d_tau,
                                    tau.size(),reads,channels};
        cudaEvent_t start{}, stop{};
        assert(cudaEventCreate(&start)==cudaSuccess && cudaEventCreate(&stop)==cudaSuccess);
        std::vector<float> times; times.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            for(int i=0;i<warmups;++i)
                assert(read_traces_fp32(view,d_entities,reads,0.5f,d_output,values.size(),nullptr)==cudaSuccess);
            assert(cudaEventRecord(start,nullptr)==cudaSuccess);
            for(int i=0;i<repetitions;++i)
                assert(read_traces_fp32(view,d_entities,reads,0.5f,d_output,values.size(),nullptr)==cudaSuccess);
            assert(cudaEventRecord(stop,nullptr)==cudaSuccess && cudaEventSynchronize(stop)==cudaSuccess);
            float elapsed=0.f; assert(cudaEventElapsedTime(&elapsed,start,stop)==cudaSuccess);
            times.push_back(elapsed*1000.f/static_cast<float>(repetitions));
        }
        assert(cudaMemcpy(values.data(),d_output,values.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
        const float expected=std::exp(-0.5f);
        for(float actual:values) assert(std::abs(actual-expected)<2e-5f);
        std::sort(times.begin(),times.end());
        std::cout<<reads<<','<<channels<<','<<times[times.size()/2]<<'\n';
        cudaEventDestroy(stop); cudaEventDestroy(start);
        cudaFree(d_output); cudaFree(d_entities); cudaFree(d_tau); cudaFree(d_anchors); cudaFree(d_values);
    }
}
