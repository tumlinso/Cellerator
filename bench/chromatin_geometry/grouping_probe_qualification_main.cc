#include "backends/consumer_fixture.hh"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <string>

int main() {
    cudaStream_t stream{};
    const auto create = cudaStreamCreate(&stream);
    if (create != cudaSuccess) {
        std::fprintf(stderr,"qualification stream: %s\n",cudaGetErrorString(create)); return 1;
    }
    std::string error;
    const bool ok = cgs::run_grouping_probe_qualification(stream,&error);
    const auto destroy = cudaStreamDestroy(stream);
    if (!ok) std::fprintf(stderr,"grouping probe qualification: %s\n",error.c_str());
    if (destroy != cudaSuccess) std::fprintf(stderr,"destroy stream: %s\n",cudaGetErrorString(destroy));
    if (!ok || destroy != cudaSuccess) return 1;
    std::puts("{\"schema\":\"cgs.grouping-probe-qualification.v1\",\"status\":\"pass\",\"arms\":[\"masked-direct-two-tile\",\"cohort-direct-two-tile\"],\"body\":\"supplied-product-fp16\",\"tile\":32,\"cases\":[\"tail-65-repeated-zero-arity32-33\",\"zero-arity\",\"poisoned-excluded-forward\",\"empty-batch\",\"stale-accessibility-and-mapping\"],\"validation\":\"independent-product-reference-and-exact-descriptors\"}");
    return 0;
}
