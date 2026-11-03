#include "backends/grouped_linear_library.hh"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <string>
int main(){cudaStream_t stream{};auto e=cudaStreamCreate(&stream);if(e!=cudaSuccess){std::fprintf(stderr,"stream: %s\n",cudaGetErrorString(e));return 1;}std::string error;bool ok=cgs::backends::run_grouped_linear_library_tests(stream,&error);e=cudaStreamDestroy(stream);if(!ok)std::fprintf(stderr,"linear control: %s\n",error.c_str());return ok&&e==cudaSuccess?0:1;}
