#include <Cellerator/compiler/backend/nvcc/integrate_cuda_runtime_and_library_linkage_v1.hh>

#include <cassert>
#include <string>
#include <vector>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;

    const auto simple = select_cuda_linkage({cuda_library_requirement::runtime});
    assert(simple.link_libraries == std::vector<std::string>{"CUDA::cudart"});
    assert(simple.header_dependencies.empty());

    const auto host_only = select_cuda_linkage({});
    assert(host_only.link_libraries.empty());
    assert(host_only.header_dependencies.empty());

    const auto selected = select_cuda_linkage({
        cuda_library_requirement::driver_api |
        cuda_library_requirement::sparse |
        cuda_library_requirement::cub |
        cuda_library_requirement::provider_sm70});
    assert((selected.link_libraries == std::vector<std::string>{
        "CUDA::cuda_driver", "CUDA::cusparse", "Cellerator::provider_sm70"}));
    assert(selected.header_dependencies ==
           std::vector<std::string>{"cub/cub.cuh"});

    const auto runtime_only = select_cuda_linkage(
        {cuda_library_requirement::cellerator_runtime});
    assert(runtime_only.link_libraries ==
           std::vector<std::string>{"Cellerator::runtime"});
}
