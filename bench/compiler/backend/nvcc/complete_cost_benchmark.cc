#include <Cellerator/compiler/backend/nvcc/benchmark_nvcc_backend_complete_cost_v1.hh>

#include <iostream>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;
    const nvcc_complete_cost_sample measured{
        nvcc_candidate_kind::generated,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 1, true};
    const auto result = complete_nvcc_cost(measured, 1);
    if (!result) {
        return 1;
    }
    std::cout << "cold_ns,warm_ns,total_reuse_ns,reuse_count\n"
              << result->cold_ns << ',' << result->warm_ns << ','
              << result->total_reuse_ns << ',' << result->reuse_count << '\n';
}
