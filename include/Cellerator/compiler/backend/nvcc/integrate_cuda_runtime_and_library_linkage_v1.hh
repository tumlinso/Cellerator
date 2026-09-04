#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

enum class cuda_library_requirement : std::uint32_t {
    none = 0,
    runtime = 1U << 0U,
    driver_api = 1U << 1U,
    sparse = 1U << 2U,
    blas = 1U << 3U,
    cub = 1U << 4U,
    nccl = 1U << 5U,
    provider_sm70 = 1U << 6U,
    cellerator_runtime = 1U << 7U,
};

[[nodiscard]] constexpr cuda_library_requirement operator|(
    cuda_library_requirement lhs,
    cuda_library_requirement rhs) noexcept {
    return static_cast<cuda_library_requirement>(
        static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

struct cuda_linkage_request {
    cuda_library_requirement requirements = cuda_library_requirement::none;
};

struct cuda_linkage {
    std::vector<std::string> link_libraries;
    std::vector<std::string> header_dependencies;
};

// Returns a canonical, duplicate-free dependency list. Header-only CUB is
// deliberately represented separately and never becomes a link dependency.
[[nodiscard]] cuda_linkage select_cuda_linkage(
    const cuda_linkage_request& request);

} // namespace cellerator::compiler::backend::nvcc::v1
