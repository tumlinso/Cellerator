#include "learning/ste.hh"
#include "learning/teacher.hh"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <string>

int run_teacher_chain_tests(cudaStream_t stream);

// This executable is for an admitted Scheduler job; it initializes CUDA.
int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "ste") &&
                      std::strcmp(argv[1], "teacher"))) {
        std::fputs("usage: cgs_learning_tests ste|teacher\n", stderr);
        return 2;
    }
    cudaStream_t stream{};
    cudaError_t result = cudaStreamCreate(&stream);
    if (result != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate: %s\n", cudaGetErrorString(result));
        return 1;
    }
    int status = 0;
    if (!std::strcmp(argv[1], "ste")) {
        std::string error;
        if (!cgs::run_ste_tests(stream, &error)) {
            std::fprintf(stderr, "STE: %s\n", error.c_str());
            status = 1;
        }
    } else {
        status = run_teacher_tests(stream);
        if (!status) status = run_teacher_chain_tests(stream);
    }
    result = cudaStreamDestroy(stream);
    if (result != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamDestroy: %s\n", cudaGetErrorString(result));
        status = 1;
    }
    return status;
}
