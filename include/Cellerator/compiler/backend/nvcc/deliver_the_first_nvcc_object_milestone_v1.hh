#pragma once

#include <Cellerator/compiler/backend/nvcc/implement_cuda_source_emission_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

struct nvcc_object_milestone_request {
    std::uint64_t profile = 0;
    std::uint64_t planning_ir = 0;
    std::uint64_t realization_ir = 0;
    std::uint32_t compute_capability = 0;
    realized_cuda_entity relation_kernel;
};

struct nvcc_object_milestone {
    std::string cuda_source;
    std::vector<std::string> compile_arguments;
    std::vector<std::string> link_arguments;
    bool conventional_fallback_retained = true;
};

[[nodiscard]] std::optional<nvcc_object_milestone>
make_first_nvcc_object_milestone(const nvcc_object_milestone_request& request);

} // namespace cellerator::compiler::backend::nvcc::v1
