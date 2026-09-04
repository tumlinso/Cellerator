#include <Cellerator/compiler/backend/nvcc/integrate_cuda_runtime_and_library_linkage_v1.hh>

namespace cellerator::compiler::backend::nvcc::v1 {
namespace {

bool includes(cuda_library_requirement requirements,
              cuda_library_requirement candidate) noexcept {
    return (static_cast<std::uint32_t>(requirements) &
            static_cast<std::uint32_t>(candidate)) != 0U;
}

} // namespace

cuda_linkage select_cuda_linkage(const cuda_linkage_request& request) {
    cuda_linkage result;
    const auto requirements = request.requirements;

    // This order is the backend ABI: platform libraries first, then
    // Cellerator targets whose transitive dependencies are supplied by CMake.
    if (includes(requirements, cuda_library_requirement::runtime)) {
        result.link_libraries.emplace_back("CUDA::cudart");
    }
    if (includes(requirements, cuda_library_requirement::driver_api)) {
        result.link_libraries.emplace_back("CUDA::cuda_driver");
    }
    if (includes(requirements, cuda_library_requirement::sparse)) {
        result.link_libraries.emplace_back("CUDA::cusparse");
    }
    if (includes(requirements, cuda_library_requirement::blas)) {
        result.link_libraries.emplace_back("CUDA::cublas");
    }
    if (includes(requirements, cuda_library_requirement::nccl)) {
        result.link_libraries.emplace_back("NCCL::NCCL");
    }
    if (includes(requirements, cuda_library_requirement::cellerator_runtime)) {
        result.link_libraries.emplace_back("Cellerator::runtime");
    }
    if (includes(requirements, cuda_library_requirement::provider_sm70)) {
        result.link_libraries.emplace_back("Cellerator::provider_sm70");
    }
    if (includes(requirements, cuda_library_requirement::cub)) {
        result.header_dependencies.emplace_back("cub/cub.cuh");
    }
    return result;
}

} // namespace cellerator::compiler::backend::nvcc::v1
