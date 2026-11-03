#include "backends/consumer_fixture.hh"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <string>
int main() {
    cudaStream_t stream{};
    auto e=cudaStreamCreate(&stream);
    if(e!=cudaSuccess){std::fprintf(stderr,"stream: %s\n",cudaGetErrorString(e));return 1;}
    std::string error; const bool ok=cgs::run_consumer_tests(stream,&error);
    e=cudaStreamDestroy(stream);
    if(!ok)std::fprintf(stderr,"consumer tests: %s\n",error.c_str());
    if(e!=cudaSuccess)std::fprintf(stderr,"destroy stream: %s\n",cudaGetErrorString(e));
    return ok && e==cudaSuccess ? 0 : 1;
}
