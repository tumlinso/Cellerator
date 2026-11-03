#include "backends/conditions.hh"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <string>
int main() {
    cudaStream_t stream{}; auto status=cudaStreamCreate(&stream);
    if(status!=cudaSuccess){std::fprintf(stderr,"stream: %s\n",cudaGetErrorString(status));return 1;}
    std::string error; const bool ok=cgs::run_condition_tests(stream,&error);
    status=cudaStreamDestroy(stream);
    if(!ok)std::fprintf(stderr,"condition tests: %s\n",error.c_str());
    return ok && status==cudaSuccess ? 0 : 1;
}
