#include <Cellerator/compiler/backend/nvcc/benchmark_nvcc_backend_complete_cost_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;

    nvcc_complete_cost_sample generated{
        nvcc_candidate_kind::generated,
        100, 20, 1000, 40, 30, 20, 10, 50, 20,
        4096, 32, 0, 1, true};
    const auto cost = complete_nvcc_cost(generated, 3);
    assert(cost);
    assert(cost->warm_ns == 170);
    assert(cost->cold_ns == 1290);
    assert(cost->total_reuse_ns == 1630);

    auto native = generated;
    native.candidate = nvcc_candidate_kind::prelinked_native;
    native.planning_ns = 10;
    native.source_emission_ns = 0;
    native.nvcc_ns = 0;
    native.kernel_ns = 30;
    const auto comparison = compare_nvcc_candidates({generated, native}, 3);
    assert(comparison);
    assert(comparison->selected == nvcc_candidate_kind::prelinked_native);
    assert(!comparison->generated_promoted); // Negative results are legitimate.

    generated.exact_output = false;
    assert(!complete_nvcc_cost(generated, 3));
}
